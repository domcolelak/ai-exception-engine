"""Detector interface and the shared result shape.

Every detector answers the same question -- *how unusual is this event, and
why* -- and returns the same structure, so the ensemble never needs to know
which detector produced what.

Two rules hold for all of them:

* the normalised score is in ``[0, 1]`` and comparable across detectors, so a
  weighted sum is meaningful;
* every score is accompanied by feature-level reasons carrying the observed
  value and the range that was expected. A score without a reason is not
  actionable, and the AI layer is only ever given these reasons -- never the
  raw data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from app.baselines.model import Baseline


@dataclass
class Reason:
    """Why one feature contributed to the score."""

    feature: str
    observed: Any
    expected_range: tuple[float, float] | None
    deviation_score: float
    explanation: str
    scope: str = ""
    #: Filled in by the ensemble so a reason can be traced to its detector.
    detector: str = ""

    def as_dict(self) -> dict:
        payload = asdict(self)
        if self.expected_range is not None:
            payload["expected_range"] = [
                round(self.expected_range[0], 4),
                round(self.expected_range[1], 4),
            ]
        return payload


@dataclass
class DetectorResult:
    detector: str
    version: str
    raw_score: float
    score: float
    reasons: list[Reason] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    #: False when the detector had too little data to judge. Abstaining is not
    #: the same as scoring zero, and the ensemble must not treat it as such.
    applicable: bool = True

    def as_dict(self) -> dict:
        return {
            "detector": self.detector,
            "version": self.version,
            "raw_score": round(self.raw_score, 6),
            "score": round(self.score, 6),
            "applicable": self.applicable,
            "reasons": [r.as_dict() for r in self.reasons],
            "evidence": self.evidence,
        }


@dataclass
class DetectionContext:
    """Everything a detector may look at for one event."""

    event: dict[str, Any]
    baseline: Baseline | None
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    #: Prior events in the same entity's history, oldest first, excluding this one.
    entity_history: list[dict[str, Any]] = field(default_factory=list)
    #: Prior events in the same baseline scope, oldest first, excluding this one.
    scope_history: list[dict[str, Any]] = field(default_factory=list)
    #: Context dimensions defining the baseline scope, for scoped comparisons.
    scope_dimensions: tuple[str, ...] = ()

    def scoped_entity_history(self, min_events: int = 12) -> list[dict[str, Any]]:
        """This entity's prior events *in the closest comparable context*.

        Comparing an entity against its own unfiltered history mixes contexts:
        a seller's luxury order looks enormous next to its books, and the
        detector reports a category difference as an anomaly.

        But filtering on every context dimension at once usually leaves too
        little to judge -- one entity rarely has much history in one exact
        (category, market) cell. Dimensions are therefore dropped from the
        right until enough comparable events remain, the same fallback the
        baselines use. Returns the unfiltered history only as a last resort.
        """
        if not self.scope_dimensions:
            return self.entity_history

        for depth in range(len(self.scope_dimensions), 0, -1):
            dims = self.scope_dimensions[:depth]
            target = tuple(str(self.event.get(dim, "")) for dim in dims)
            subset = [
                event
                for event in self.entity_history
                if tuple(str(event.get(dim, "")) for dim in dims) == target
            ]
            if len(subset) >= min_events:
                return subset
            coarsest = subset

        # Never fall back to the entity's unfiltered history. Comparing a
        # luxury order against the same seller's books is not a weaker
        # comparison, it is a wrong one, and it manufactures exactly the
        # false positives this detector exists to avoid. The caller sees a
        # short history and abstains, which is the honest answer.
        return coarsest


class Detector(ABC):
    """A single way of noticing that something is unusual."""

    name: str = "detector"
    version: str = "v1"
    #: Default weight in the ensemble. Overridable per schema.
    default_weight: float = 1.0

    @abstractmethod
    def score(self, context: DetectionContext) -> DetectorResult:
        """Score one event. Must never raise on malformed input."""

    def _abstain(self, note: str) -> DetectorResult:
        return DetectorResult(
            detector=self.name,
            version=self.version,
            raw_score=0.0,
            score=0.0,
            applicable=False,
            evidence={"note": note},
        )


def normalise_z(z: float, *, knee: float = 3.0) -> float:
    """Map a robust z-score onto ``[0, 1]``.

    ``knee`` is the deviation treated as clearly anomalous; it maps to ~0.5,
    and larger deviations approach 1 with diminishing returns so that a single
    enormous outlier cannot dominate the ensemble by an unbounded amount.
    """
    magnitude = abs(z)
    if magnitude <= 0:
        return 0.0
    return float(magnitude / (magnitude + knee))


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def numbers(events: Sequence[dict], feature: str) -> list[float]:
    return [v for v in (number(e.get(feature)) for e in events) if v is not None]
