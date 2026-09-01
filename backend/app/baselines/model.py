"""Contextual baselines.

The product's whole claim is that "normal" depends on context. A 2,000 EUR
refund is unremarkable for an electronics seller and extraordinary for a
stationery seller, so a single global threshold produces either noise or
silence.

A baseline is therefore computed *per scope*: a combination of context
dimensions such as ``(seller_category=electronics, market=SK)``. Every scope
also has a parent, and a scope with too little data falls back to it rather
than pretending to know. That fallback is recorded on the baseline, so an
exception can always say which population it was judged against.

Statistics are robust by default -- median and MAD rather than mean and
standard deviation. A handful of extreme values is exactly what we are looking
for, and they would otherwise inflate the very spread used to detect them.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

#: Scaling factor making MAD a consistent estimator of the standard deviation
#: for normally distributed data.
MAD_TO_SIGMA = 1.4826

#: Below this many observations a scope cannot support its own baseline.
MIN_SCOPE_SAMPLES = 30

#: Floor for the spread estimate, as a fraction of the median. Without it a
#: perfectly constant series yields a spread of zero and every subsequent value
#: scores as infinitely anomalous.
MIN_SPREAD_RATIO = 0.01


#: Above this share of exact zeros a feature is treated as zero-inflated.
ZERO_INFLATION_THRESHOLD = 0.40

#: Minimum informative (non-zero) observations before a zero-inflated feature
#: can be judged at all.
MIN_INFORMATIVE = 12


@dataclass
class NumericStats:
    count: int
    median: float
    mad: float
    sigma: float
    p05: float
    p25: float
    p75: float
    p95: float
    minimum: float
    maximum: float
    #: Share of observations that were exactly zero.
    zero_share: float = 0.0
    #: How many observations the statistics above were actually computed over.
    informative_count: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def zero_inflated(self) -> bool:
        """True when the feature mixes 'did not happen' with 'how much'.

        A refund amount is zero on every order that was not refunded. Left
        alone, those structural zeros dominate the distribution: the median is
        0, the spread collapses, and every ordinary refund scores as a wild
        outlier while genuinely large refunds are indistinguishable from small
        ones. The statistics are therefore computed over the non-zero
        observations only, and a zero observation is treated as "the thing
        simply did not happen" rather than as an extreme low value.
        """
        return self.zero_share >= ZERO_INFLATION_THRESHOLD

    def robust_z(self, value: float) -> float:
        """Deviation in robust standard deviations. Signed."""
        if self.sigma <= 0:
            return 0.0
        if self.zero_inflated:
            if value == 0:
                return 0.0
            if self.informative_count < MIN_INFORMATIVE:
                return 0.0
        return (value - self.median) / self.sigma

    def expected_range(self, z: float = 3.0) -> tuple[float, float]:
        return (self.median - z * self.sigma, self.median + z * self.sigma)


@dataclass
class CategoricalStats:
    count: int
    frequencies: dict[str, float]
    observed_values: int

    def as_dict(self) -> dict:
        return asdict(self)

    def rarity(self, value: str) -> float:
        """0 for a common value, approaching 1 for a never-seen one.

        Uses additive smoothing so an unseen category gets a high but finite
        rarity that still depends on how much data the scope has -- a value not
        seen in 40 observations is far weaker evidence than one not seen in
        40,000.
        """
        if self.count <= 0:
            return 0.0
        seen = self.frequencies.get(value, 0.0)
        smoothed = (seen * self.count + 1) / (self.count + self.observed_values + 1)
        return max(0.0, min(1.0, 1.0 - smoothed * (self.observed_values + 1)))


@dataclass
class Baseline:
    """Statistics for one scope, over one feature set."""

    scope_key: str
    scope: dict[str, str]
    sample_count: int
    numeric: dict[str, NumericStats] = field(default_factory=dict)
    categorical: dict[str, CategoricalStats] = field(default_factory=dict)
    #: Set when this scope was too small and a parent scope's numbers are used.
    fallback_from: str | None = None
    version: str = "v1"

    def as_dict(self) -> dict:
        return {
            "scope_key": self.scope_key,
            "scope": self.scope,
            "sample_count": self.sample_count,
            "numeric": {k: v.as_dict() for k, v in self.numeric.items()},
            "categorical": {k: v.as_dict() for k, v in self.categorical.items()},
            "fallback_from": self.fallback_from,
            "version": self.version,
        }

    @property
    def is_fallback(self) -> bool:
        return self.fallback_from is not None

    def describe_scope(self) -> str:
        if not self.scope:
            return "all data"
        return ", ".join(f"{k}={v}" for k, v in sorted(self.scope.items()))


@dataclass
class BaselineSet:
    """Every scope's baseline, plus the lookup that handles fallback."""

    dimensions: tuple[str, ...]
    baselines: dict[str, Baseline] = field(default_factory=dict)
    global_baseline: Baseline | None = None
    min_samples: int = MIN_SCOPE_SAMPLES

    def scope_key_for(self, event: dict[str, Any]) -> str:
        return scope_key({dim: str(event.get(dim, "")) for dim in self.dimensions})

    def lookup(self, event: dict[str, Any]) -> Baseline | None:
        """Most specific baseline with enough data, else a broader one.

        Dimensions are dropped from the right, so ordering them
        most-significant-first in the schema matters: the last dimension is the
        first thing given up when data is thin.
        """
        values = {dim: str(event.get(dim, "")) for dim in self.dimensions}
        for depth in range(len(self.dimensions), 0, -1):
            partial = {dim: values[dim] for dim in self.dimensions[:depth]}
            candidate = self.baselines.get(scope_key(partial))
            if candidate is not None and candidate.sample_count >= self.min_samples:
                return candidate
        return self.global_baseline

    def as_dict(self) -> dict:
        return {
            "dimensions": list(self.dimensions),
            "min_samples": self.min_samples,
            "scopes": {k: v.as_dict() for k, v in self.baselines.items()},
            "global": self.global_baseline.as_dict() if self.global_baseline else None,
        }


def scope_key(scope: dict[str, str]) -> str:
    """Stable, readable identity for a scope."""
    if not scope:
        return "*"
    return "|".join(f"{k}={scope[k]}" for k in sorted(scope))


def build_baselines(
    events: Sequence[dict[str, Any]],
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    dimensions: Sequence[str] = (),
    min_samples: int = MIN_SCOPE_SAMPLES,
    version: str = "v1",
) -> BaselineSet:
    """Compute a baseline for the global population and every context scope.

    Baselines are built for *every prefix* of the dimension list, so a scope
    that is too thin can fall back to a broader one that is not.
    """
    dims = tuple(dimensions)
    result = BaselineSet(dimensions=dims, min_samples=min_samples)

    result.global_baseline = _compute(
        events, {}, numeric_features, categorical_features, version
    )

    for depth in range(1, len(dims) + 1):
        prefix = dims[:depth]
        grouped: defaultdict[str, list[dict]] = defaultdict(list)
        scopes: dict[str, dict[str, str]] = {}
        for event in events:
            scope = {dim: str(event.get(dim, "")) for dim in prefix}
            key = scope_key(scope)
            grouped[key].append(event)
            scopes[key] = scope

        for key, group in grouped.items():
            baseline = _compute(
                group, scopes[key], numeric_features, categorical_features, version
            )
            # A thin scope keeps its own sample count -- lookup uses that to
            # decide fallback -- but borrows the broader numbers so a caller
            # that addresses it directly still gets usable statistics.
            if baseline.sample_count < min_samples:
                parent = _parent_baseline(result, prefix, scopes[key])
                if parent is not None:
                    baseline.numeric = parent.numeric
                    baseline.categorical = parent.categorical
                    baseline.fallback_from = parent.scope_key
            result.baselines[key] = baseline

    return result


def _parent_baseline(
    result: BaselineSet, prefix: tuple[str, ...], scope: dict[str, str]
) -> Baseline | None:
    for depth in range(len(prefix) - 1, 0, -1):
        parent_scope = {dim: scope[dim] for dim in prefix[:depth]}
        parent = result.baselines.get(scope_key(parent_scope))
        if parent is not None and parent.sample_count >= result.min_samples:
            return parent
    return result.global_baseline


def _compute(
    events: Sequence[dict[str, Any]],
    scope: dict[str, str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    version: str,
) -> Baseline:
    baseline = Baseline(
        scope_key=scope_key(scope),
        scope=scope,
        sample_count=len(events),
        version=version,
    )

    for feature in numeric_features:
        values = [
            v for v in (_number(event.get(feature)) for event in events) if v is not None
        ]
        if len(values) >= 3:
            baseline.numeric[feature] = _numeric_stats(values)

    for feature in categorical_features:
        values = [
            str(event[feature])
            for event in events
            if event.get(feature) not in (None, "")
        ]
        if values:
            counts = Counter(values)
            total = len(values)
            baseline.categorical[feature] = CategoricalStats(
                count=total,
                frequencies={k: v / total for k, v in counts.items()},
                observed_values=len(counts),
            )

    return baseline


def _numeric_stats(values: list[float]) -> NumericStats:
    total = len(values)
    zeros = sum(1 for v in values if v == 0)
    zero_share = zeros / total if total else 0.0

    # For a zero-inflated feature the statistics describe the non-zero
    # population; the zero share is kept separately so callers can tell the two
    # questions apart.
    population = [v for v in values if v != 0] if zero_share >= ZERO_INFLATION_THRESHOLD else values
    if len(population) < 3:
        population = values

    ordered = sorted(population)
    med = _percentile(ordered, 0.5)
    mad = _percentile(sorted(abs(v - med) for v in ordered), 0.5)
    sigma = mad * MAD_TO_SIGMA

    # A constant or near-constant series would otherwise make every future
    # value infinitely anomalous.
    floor = abs(med) * MIN_SPREAD_RATIO
    if sigma < floor:
        sigma = floor
    if sigma <= 0:
        spread = ordered[-1] - ordered[0]
        sigma = spread / 4 if spread > 0 else 0.0

    return NumericStats(
        count=total,
        zero_share=round(zero_share, 4),
        informative_count=len(ordered),
        median=round(med, 6),
        mad=round(mad, 6),
        sigma=round(sigma, 6),
        p05=round(_percentile(ordered, 0.05), 6),
        p25=round(_percentile(ordered, 0.25), 6),
        p75=round(_percentile(ordered, 0.75), 6),
        p95=round(_percentile(ordered, 0.95), 6),
        minimum=round(ordered[0], 6),
        maximum=round(ordered[-1], 6),
    )


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile on an already-sorted sequence."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def rolling_windows(
    events: Sequence[dict[str, Any]], *, timestamp_field: str, window: int
) -> Iterable[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Yield ``(event, preceding_window)`` in chronological order.

    The window never includes the event itself -- scoring a point against a
    baseline that already contains it dilutes exactly the deviation being
    measured, which is the single easiest way to make an anomaly detector look
    calibrated and find nothing.
    """
    ordered = sorted(events, key=lambda e: e.get(timestamp_field) or 0)
    for index, event in enumerate(ordered):
        start = max(0, index - window)
        yield event, ordered[start:index]
