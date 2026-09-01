"""Deterministic ensemble scoring.

The overall anomaly score must be reproducible from the stored detector
outputs alone. That is what lets an analyst re-open a six-month-old exception
and see exactly why it fired, and what lets a disputed score be recomputed
rather than argued about.

Two decisions carry most of the weight here:

* **Abstaining detectors are excluded from the denominator.** A detector that
  lacked history did not vote "normal"; treating it as a zero would quietly
  drag every score down whenever data is thin, which is precisely when new
  entities look suspicious.
* **The score is the strongest detector, raised by agreement.** Detectors are
  independent tests of abnormality, not voters. A weighted mean is the wrong
  model: with six detectors, five of which have nothing to say about a
  particular kind of problem, a single certain detector is diluted into
  silence -- a seller refunding 60% of its orders against a peer median of 12%
  scored 29 out of 100 under a mean. The strongest detector therefore sets the
  floor, and agreement from the others lifts the score toward 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from app.detectors.base import DetectionContext, Detector, DetectorResult, Reason

#: How much unanimous agreement from the other detectors can add to whatever
#: the strongest one already found. At 0.5, full agreement closes half the
#: remaining distance to 1.0.
AGREEMENT_WEIGHT = 0.5

SEVERITY_BANDS = (
    (85.0, "critical"),
    (70.0, "high"),
    (50.0, "medium"),
    (0.0, "low"),
)


@dataclass
class EnsembleResult:
    score: float
    severity: str
    confidence: float
    weighted_mean: float
    strongest: str | None
    detector_results: list[DetectorResult] = field(default_factory=list)
    reasons: list[Reason] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "weighted_mean": round(self.weighted_mean, 6),
            "strongest_detector": self.strongest,
            "weights": self.weights,
            "detectors": [d.as_dict() for d in self.detector_results],
            "reasons": [r.as_dict() for r in self.reasons],
        }

    @property
    def applicable_detectors(self) -> list[DetectorResult]:
        return [d for d in self.detector_results if d.applicable]


def score_event(
    context: DetectionContext,
    detectors: Sequence[Detector],
    *,
    weights: dict[str, float] | None = None,
) -> EnsembleResult:
    """Run every detector and combine the results."""
    configured = dict(weights or {})
    results: list[DetectorResult] = []

    for detector in detectors:
        try:
            results.append(detector.score(context))
        except Exception as exc:  # pragma: no cover - a detector must never break scoring
            results.append(
                DetectorResult(
                    detector=detector.name,
                    version=detector.version,
                    raw_score=0.0,
                    score=0.0,
                    applicable=False,
                    evidence={"error": str(exc)[:300]},
                )
            )

    used = {d.name: configured.get(d.name, d.default_weight) for d in detectors}
    applicable = [r for r in results if r.applicable]

    if not applicable:
        return EnsembleResult(
            score=0.0,
            severity="low",
            confidence=0.0,
            weighted_mean=0.0,
            strongest=None,
            detector_results=results,
            weights=used,
        )

    strongest_result = max(applicable, key=lambda r: r.score)
    combined = combine(
        [(r.detector, r.score) for r in applicable], used
    )
    weighted_mean = _support_of([(r.detector, r.score) for r in applicable], used)
    score = round(100 * combined, 2)

    reasons: list[Reason] = []
    for result in sorted(applicable, key=lambda r: -r.score):
        for reason in result.reasons:
            reason.detector = result.detector
            reasons.append(reason)

    return EnsembleResult(
        score=score,
        severity=severity_for(score),
        confidence=confidence_for(applicable, len(results)),
        weighted_mean=weighted_mean,
        strongest=strongest_result.detector if strongest_result.score > 0 else None,
        detector_results=results,
        reasons=reasons,
        weights=used,
    )


def severity_for(score: float) -> str:
    for threshold, label in SEVERITY_BANDS:
        if score >= threshold:
            return label
    return "low"


def confidence_for(applicable: Sequence[DetectorResult], total: int) -> float:
    """How much the ensemble trusts its own score.

    Driven by coverage (how many detectors could judge) and agreement (how many
    of those actually saw something). One detector firing alone out of five is
    a weaker signal than three firing together, even at the same score.
    """
    if total == 0 or not applicable:
        return 0.0
    coverage = len(applicable) / total
    firing = sum(1 for r in applicable if r.score >= 0.4)
    agreement = firing / len(applicable)
    return round(min(0.95, 0.35 + 0.35 * coverage + 0.30 * agreement), 4)


def recompute(stored: dict, weights: dict[str, float] | None = None) -> float:
    """Recompute the score from a persisted ensemble payload.

    Used by tests and by the audit path: a stored exception must be able to
    prove its own score without re-running detection against data that has
    since changed.
    """
    detectors = [d for d in stored.get("detectors", []) if d.get("applicable")]
    if not detectors:
        return 0.0
    used = weights or stored.get("weights", {})
    return round(
        100 * combine([(d["detector"], d["score"]) for d in detectors], used), 2
    )


def combine(scores: Sequence[tuple[str, float]], weights: dict[str, float]) -> float:
    """The strongest detector, lifted by weighted agreement from the rest."""
    if not scores:
        return 0.0
    strongest = max(score for _, score in scores)
    support = _support_of(scores, weights)
    return float(min(1.0, strongest + (1.0 - strongest) * support * AGREEMENT_WEIGHT))


def _support_of(scores: Sequence[tuple[str, float]], weights: dict[str, float]) -> float:
    """Weighted mean of every detector other than the strongest.

    Excluding the strongest keeps agreement and strength separate: without it a
    lone confident detector would count itself as its own corroboration.
    """
    if len(scores) < 2:
        return 0.0
    peak = max(score for _, score in scores)
    others: list[tuple[str, float]] = []
    dropped = False
    for name, score in scores:
        if not dropped and score == peak:
            dropped = True
            continue
        others.append((name, score))
    total = sum(weights.get(name, 1.0) for name, _ in others)
    if total <= 0:
        return 0.0
    return sum(score * weights.get(name, 1.0) for name, score in others) / total
