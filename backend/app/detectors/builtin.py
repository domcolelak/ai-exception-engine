"""The five MVP detectors.

Each looks for a different shape of abnormality, and they disagree usefully:
robust deviation catches a single extreme value, categorical rarity catches a
combination nobody has seen, isolation forest catches points that are odd only
in combination, rolling quantile catches drift against recent history, and
change-point catches an entity whose own behaviour shifted.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from app.baselines.model import MIN_INFORMATIVE, ZERO_INFLATION_THRESHOLD

#: Minimum spread for a rolling comparison, as a fraction of the median.
MIN_RELATIVE_SPREAD = 0.05
from app.detectors.base import (
    DetectionContext,
    Detector,
    DetectorResult,
    Reason,
    normalise_z,
    number,
    numbers,
)


class RobustDeviationDetector(Detector):
    """Median/MAD deviation against the contextual baseline.

    The workhorse. Robust statistics matter here: with mean and standard
    deviation, a few large refunds inflate the spread enough to hide themselves.
    """

    name = "robust_deviation"
    version = "v1"
    default_weight = 1.0

    def __init__(self, knee: float = 3.0) -> None:
        self.knee = knee

    def score(self, context: DetectionContext) -> DetectorResult:
        baseline = context.baseline
        if baseline is None or not baseline.numeric:
            return self._abstain("no numeric baseline for this scope")

        reasons: list[Reason] = []
        worst = 0.0
        for feature in context.numeric_features:
            stats = baseline.numeric.get(feature)
            observed = number(context.event.get(feature))
            if stats is None or observed is None:
                continue

            z = stats.robust_z(observed)
            deviation = normalise_z(z, knee=self.knee)
            worst = max(worst, deviation)
            if deviation < 0.4:
                continue

            low, high = stats.expected_range(self.knee)
            direction = "above" if z > 0 else "below"
            reasons.append(
                Reason(
                    feature=feature,
                    observed=observed,
                    expected_range=(low, high),
                    deviation_score=round(deviation, 4),
                    explanation=(
                        f"{feature} is {observed:,.2f}, {abs(z):.1f} robust standard "
                        f"deviations {direction} the median of {stats.median:,.2f} for "
                        f"{baseline.describe_scope()}"
                    ),
                    scope=baseline.scope_key,
                )
            )

        reasons.sort(key=lambda r: -r.deviation_score)
        return DetectorResult(
            detector=self.name,
            version=self.version,
            raw_score=worst,
            score=worst,
            reasons=reasons,
            evidence={
                "scope": baseline.scope_key,
                "scope_samples": baseline.sample_count,
                "used_fallback_scope": baseline.is_fallback,
            },
        )


class CategoricalRarityDetector(Detector):
    """Values, and combinations of values, that are rare in this scope."""

    name = "categorical_rarity"
    version = "v1"
    default_weight = 0.8

    #: Below this rarity a value is ordinary and not worth a reason.
    threshold = 0.90

    def score(self, context: DetectionContext) -> DetectorResult:
        baseline = context.baseline
        if baseline is None or not baseline.categorical:
            return self._abstain("no categorical baseline for this scope")

        reasons: list[Reason] = []
        worst = 0.0
        for feature in context.categorical_features:
            stats = baseline.categorical.get(feature)
            raw = context.event.get(feature)
            if stats is None or raw in (None, ""):
                continue

            value = str(raw)
            rarity = stats.rarity(value)
            worst = max(worst, rarity)
            if rarity < self.threshold:
                continue

            share = stats.frequencies.get(value, 0.0)
            seen = "never seen" if share == 0 else f"seen in {share:.2%} of cases"
            reasons.append(
                Reason(
                    feature=feature,
                    observed=value,
                    expected_range=None,
                    deviation_score=round(rarity, 4),
                    explanation=(
                        f"{feature}={value} is {seen} among the "
                        f"{stats.count:,} events for {baseline.describe_scope()}"
                    ),
                    scope=baseline.scope_key,
                )
            )

        reasons.sort(key=lambda r: -r.deviation_score)
        return DetectorResult(
            detector=self.name,
            version=self.version,
            raw_score=worst,
            score=worst,
            reasons=reasons,
            evidence={"scope": baseline.scope_key, "scope_samples": baseline.sample_count},
        )


class IsolationForestDetector(Detector):
    """Points that are unremarkable per feature but odd in combination.

    Fitted per scope rather than per event. Refitting a forest for every event
    is O(n) model fits over a batch -- on a few thousand events that turned a
    detection run into minutes of pure retraining, and it produces the same
    answer as fitting once on the scope's history. The model is fitted from the
    training slice only, so no event contributes to the model that judges it.
    """

    name = "isolation_forest"
    version = "v1"
    default_weight = 0.7

    #: Fewer rows than this and the forest memorises rather than generalises.
    min_history = 50

    def __init__(self, random_state: int = 20260830) -> None:
        self.random_state = random_state
        self._models: dict[str, tuple[object, list[str]]] = {}

    def fit(self, scope_key: str, history: Sequence[dict], features: Sequence[str]) -> None:
        """Fit and cache one forest for a scope. Safe to call repeatedly."""
        if scope_key in self._models or len(history) < self.min_history:
            return
        usable = [f for f in features]
        matrix = []
        for event in history:
            row = [number(event.get(f)) for f in usable]
            if all(v is not None for v in row):
                matrix.append(row)
        if len(matrix) < self.min_history or len(usable) < 2:
            return

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:  # pragma: no cover - declared dependency
            return

        forest = IsolationForest(
            n_estimators=100, contamination="auto", random_state=self.random_state
        )
        forest.fit(matrix)
        self._models[scope_key] = (forest, usable)

    def score(self, context: DetectionContext) -> DetectorResult:
        scope = context.baseline.scope_key if context.baseline else "*"
        model = self._models.get(scope)
        if model is None:
            # Fit lazily from the scope history when the pipeline has not
            # pre-fitted this scope.
            self.fit(scope, context.scope_history, context.numeric_features)
            model = self._models.get(scope)
        if model is None:
            return self._abstain(
                f"no fitted model for scope {scope}; needs {self.min_history} prior events"
            )

        forest, features = model
        point = [number(context.event.get(f)) for f in features]
        if any(v is None for v in point):
            return self._abstain("event is missing one of the modelled features")

        # decision_function: positive is normal, negative is an outlier.
        margin = float(forest.decision_function([point])[0])
        score = float(max(0.0, min(1.0, -margin * 2.5)))

        reasons: list[Reason] = []
        if score >= 0.4:
            reasons.append(
                Reason(
                    feature="+".join(features),
                    observed={f: context.event.get(f) for f in features},
                    expected_range=None,
                    deviation_score=round(score, 4),
                    explanation=(
                        "The combination of "
                        + ", ".join(features)
                        + " is unusual for "
                        + (context.baseline.describe_scope() if context.baseline else "this population")
                        + ", even though no single value is extreme"
                    ),
                    scope=scope,
                )
            )

        return DetectorResult(
            detector=self.name,
            version=self.version,
            raw_score=margin,
            score=score,
            reasons=reasons,
            evidence={"features": features, "scope": scope, "margin": round(margin, 6)},
        )


class RollingQuantileDetector(Detector):
    """Deviation from the entity's own recent behaviour.

    Catches drift a static baseline misses: a seller whose refund rate is
    normal for the market but has doubled against its own recent history.
    """

    name = "rolling_quantile"
    version = "v1"
    default_weight = 0.9

    min_history = 12

    def __init__(self, upper_q: float = 0.9) -> None:
        self.upper_q = upper_q

    def score(self, context: DetectionContext) -> DetectorResult:
        history = context.scoped_entity_history(self.min_history)
        if len(history) < self.min_history:
            return self._abstain(
                f"needs {self.min_history} prior comparable events for this entity, "
                f"has {len(history)}"
            )

        reasons: list[Reason] = []
        worst = 0.0
        for feature in context.numeric_features:
            observed = number(context.event.get(feature))
            raw_values = numbers(history, feature)
            if observed is None or len(raw_values) < self.min_history:
                continue

            # Zero-inflated features (a refund amount is 0 on every order that
            # was not refunded) would otherwise give p90 ~ median ~ 0, making
            # every ordinary occurrence look like an extreme outlier.
            zero_share = sum(1 for v in raw_values if v == 0) / len(raw_values)
            if zero_share >= ZERO_INFLATION_THRESHOLD:
                if observed == 0:
                    continue
                values = [v for v in raw_values if v != 0]
                if len(values) < MIN_INFORMATIVE:
                    continue
            else:
                values = raw_values

            ordered = sorted(values)
            upper = _quantile(ordered, self.upper_q)
            median = _quantile(ordered, 0.5)
            # A near-constant history has no spread of its own, which would
            # either make every wobble look enormous (spread ~ 0) or make a
            # genuine jump invisible (spread exactly 0). A floor proportional
            # to the level gives a sane scale in both directions -- the same
            # approach the baselines take.
            spread = max(upper - median, abs(median) * MIN_RELATIVE_SPREAD)
            if spread <= 0:
                continue

            ratio = (observed - upper) / spread
            if ratio <= 0:
                continue
            deviation = float(ratio / (ratio + 2.0))
            worst = max(worst, deviation)
            if deviation < 0.35:
                continue

            reasons.append(
                Reason(
                    feature=feature,
                    observed=observed,
                    expected_range=(median, upper),
                    deviation_score=round(deviation, 4),
                    explanation=(
                        f"{feature} is {observed:,.2f}, above this entity's own recent "
                        f"p{int(self.upper_q * 100)} of {upper:,.2f} "
                        f"(median {median:,.2f} over its last {len(values)} events)"
                    ),
                    scope="entity_history",
                )
            )

        reasons.sort(key=lambda r: -r.deviation_score)
        return DetectorResult(
            detector=self.name,
            version=self.version,
            raw_score=worst,
            score=worst,
            reasons=reasons,
            evidence={"history_length": len(history)},
        )


class ChangePointDetector(Detector):
    """A sustained shift in an entity's own level, not a single spike.

    Implemented as binary segmentation with a mean-shift cost: find the split
    that most reduces within-segment variance, and report it when the level
    difference is large relative to the noise. A single outlier moves the mean
    of one side very little, so this stays quiet for spikes -- which is the
    point, since the deviation detectors already cover those.
    """

    name = "change_point"
    version = "v1"
    default_weight = 0.8

    min_history = 20
    min_segment = 5

    def score(self, context: DetectionContext) -> DetectorResult:
        history = context.scoped_entity_history(self.min_history)
        if len(history) < self.min_history:
            return self._abstain(
                f"needs {self.min_history} prior comparable events for this entity, "
                f"has {len(history)}"
            )

        reasons: list[Reason] = []
        worst = 0.0
        details: dict = {}

        for feature in context.numeric_features:
            observed = number(context.event.get(feature))
            series = numbers(history, feature)
            if observed is None or len(series) < self.min_history:
                continue
            series = series + [observed]

            split, strength, before, after = _best_split(series, self.min_segment)
            if split is None or strength < 0.35:
                continue

            worst = max(worst, strength)
            details[feature] = {"split_index": split, "before_mean": before, "after_mean": after}
            direction = "increased" if after > before else "decreased"
            reasons.append(
                Reason(
                    feature=feature,
                    observed=observed,
                    expected_range=(min(before, after), max(before, after)),
                    deviation_score=round(strength, 4),
                    explanation=(
                        f"{feature} {direction} from an average of {before:,.2f} to "
                        f"{after:,.2f} for this entity, a sustained shift rather than a "
                        f"single unusual value"
                    ),
                    scope="entity_history",
                )
            )

        reasons.sort(key=lambda r: -r.deviation_score)
        return DetectorResult(
            detector=self.name,
            version=self.version,
            raw_score=worst,
            score=worst,
            reasons=reasons,
            evidence={"history_length": len(history), "splits": details},
        )


def _best_split(
    series: Sequence[float], min_segment: int
) -> tuple[int | None, float, float, float]:
    """Binary segmentation for a single mean shift.

    Returns ``(split_index, strength, before_mean, after_mean)``. Strength is
    the absolute mean difference expressed in pooled standard deviations and
    squashed into ``[0, 1)``.
    """
    n = len(series)
    if n < 2 * min_segment:
        return None, 0.0, 0.0, 0.0

    total = sum(series)
    best_index: int | None = None
    best_cost = math.inf

    prefix = 0.0
    for index in range(1, n):
        prefix += series[index - 1]
        if index < min_segment or n - index < min_segment:
            continue
        left_mean = prefix / index
        right_mean = (total - prefix) / (n - index)
        # Within-segment sum of squares; minimising it maximises the split's
        # explanatory power.
        cost = sum((v - left_mean) ** 2 for v in series[:index]) + sum(
            (v - right_mean) ** 2 for v in series[index:]
        )
        if cost < best_cost:
            best_cost = cost
            best_index = index

    if best_index is None:
        return None, 0.0, 0.0, 0.0

    left = series[:best_index]
    right = series[best_index:]
    before = sum(left) / len(left)
    after = sum(right) / len(right)

    pooled = math.sqrt(
        (
            sum((v - before) ** 2 for v in left)
            + sum((v - after) ** 2 for v in right)
        )
        / max(n - 2, 1)
    )
    if pooled <= 0:
        # A perfectly clean step is a real change point, not a division by zero.
        return (best_index, 1.0, before, after) if before != after else (None, 0.0, 0.0, 0.0)

    effect = abs(after - before) / pooled
    return best_index, float(effect / (effect + 2.0)), before, after


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


#: Instantiated in a fixed order so ensemble output is reproducible.
DEFAULT_DETECTORS: tuple[Detector, ...] = (
    RobustDeviationDetector(),
    CategoricalRarityDetector(),
    IsolationForestDetector(),
    RollingQuantileDetector(),
    ChangePointDetector(),
)
