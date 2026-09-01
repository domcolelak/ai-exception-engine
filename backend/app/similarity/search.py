"""Similar historical exceptions.

The first question an analyst asks about an exception is "have we seen this
before, and what did we do?". Answering it well is what turns a queue into
institutional memory.

Similarity is structured, not embedded. These objects are numbers and
categories with known meanings; a vector embedding would throw that structure
away and make the ranking unexplainable. Text embeddings are reserved for the
free-text notes, where they actually add something -- and that path is optional
so the feature works with no provider configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class SimilarityWeights:
    #: Same kind of problem (which detector, which features) matters most: two
    #: exceptions about different things are not similar just because they
    #: concern the same seller.
    reason_overlap: float = 0.40
    same_entity: float = 0.15
    same_scope: float = 0.15
    severity_match: float = 0.10
    numeric_proximity: float = 0.20


@dataclass
class SimilarExample:
    exception_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    status: str = ""
    feedback_label: str | None = None
    resolution_note: str = ""
    title: str = ""

    def as_dict(self) -> dict:
        return {
            "exception_id": self.exception_id,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "status": self.status,
            "feedback_label": self.feedback_label,
            "resolution_note": self.resolution_note,
            "title": self.title,
        }


@dataclass
class ExceptionView:
    """The subset of an exception that similarity needs."""

    id: str
    entity_id: str
    baseline_scope: str
    severity: str
    anomaly_score: float
    reasons: list[dict]
    status: str = ""
    feedback_label: str | None = None
    resolution_note: str = ""
    title: str = ""


def find_similar(
    target: ExceptionView,
    candidates: Sequence[ExceptionView],
    *,
    limit: int = 5,
    weights: SimilarityWeights | None = None,
    min_score: float = 0.25,
) -> list[SimilarExample]:
    """Rank historical exceptions by structured similarity to ``target``."""
    w = weights or SimilarityWeights()
    scored: list[SimilarExample] = []

    target_reasons = _reason_keys(target.reasons)
    target_values = _numeric_values(target.reasons)

    for candidate in candidates:
        if candidate.id == target.id:
            continue

        components = {
            "reason_overlap": _jaccard(target_reasons, _reason_keys(candidate.reasons)),
            "same_entity": 1.0 if candidate.entity_id == target.entity_id else 0.0,
            "same_scope": 1.0 if candidate.baseline_scope == target.baseline_scope else 0.0,
            "severity_match": 1.0 if candidate.severity == target.severity else 0.0,
            "numeric_proximity": _numeric_proximity(
                target_values, _numeric_values(candidate.reasons)
            ),
        }
        total = (
            components["reason_overlap"] * w.reason_overlap
            + components["same_entity"] * w.same_entity
            + components["same_scope"] * w.same_scope
            + components["severity_match"] * w.severity_match
            + components["numeric_proximity"] * w.numeric_proximity
        )
        if total < min_score:
            continue

        scored.append(
            SimilarExample(
                exception_id=candidate.id,
                score=total,
                components=components,
                status=candidate.status,
                feedback_label=candidate.feedback_label,
                resolution_note=candidate.resolution_note,
                title=candidate.title,
            )
        )

    # Resolved cases first at equal score: a similar case somebody already
    # settled is more useful than another open one.
    scored.sort(key=lambda s: (-s.score, 0 if s.status == "resolved" else 1))
    return scored[:limit]


def _reason_keys(reasons: Sequence[dict]) -> set[str]:
    """What the exception is *about*: detector plus feature."""
    return {
        f"{r.get('detector', '')}:{r.get('feature', '')}"
        for r in reasons
        if r.get("feature")
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _numeric_values(reasons: Sequence[dict]) -> dict[str, float]:
    values: dict[str, float] = {}
    for reason in reasons:
        observed = reason.get("observed")
        feature = reason.get("feature")
        if not feature or not isinstance(observed, (int, float)) or isinstance(observed, bool):
            continue
        values[str(feature)] = float(observed)
    return values


def _numeric_proximity(left: dict[str, float], right: dict[str, float]) -> float:
    """How close the observed values are, on features both exceptions share.

    Scale-free: compared as a ratio, so a pair of 1,000-EUR refunds is as close
    as a pair of 10-EUR ones. Comparing absolute differences would make every
    large-value pair look dissimilar simply because the numbers are big.
    """
    shared = set(left) & set(right)
    if not shared:
        return 0.0
    closeness = []
    for feature in shared:
        a, b = abs(left[feature]), abs(right[feature])
        if a == 0 and b == 0:
            closeness.append(1.0)
            continue
        closeness.append(min(a, b) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(closeness) / len(closeness)
