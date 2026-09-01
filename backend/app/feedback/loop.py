"""The feedback loop.

Human labels are the most valuable data this product collects and the easiest
thing to mishandle. The rule enforced here:

> A label never changes production behaviour on its own.

Feedback accumulates. When it accumulates enough, a retraining job produces a
*candidate* model with adjusted thresholds and a report of what would change.
Somebody then promotes it, or does not. One analyst clicking "false positive"
must not be able to silently blind the system.

The second rule: a suppression policy proposed from feedback is capped by
score, so accepting "these are fine" can never hide a genuine crisis of the
same shape.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Sequence

#: The vocabulary. Kept small on purpose -- a label set nobody can remember
#: gets used inconsistently, and inconsistent labels are worse than none.
LABELS = (
    "true_exception",
    "false_positive",
    "expected_behavior",
    "duplicate",
    "data_quality_issue",
)

#: Labels that mean "this should not have fired".
NEGATIVE_LABELS = frozenset({"false_positive", "expected_behavior"})


@dataclass
class LabelledException:
    """An exception plus the verdict it received."""

    exception_id: str
    label: str
    entity_id: str
    baseline_scope: str
    severity: str
    anomaly_score: float
    strongest_detector: str
    reasons: list[dict] = field(default_factory=list)


@dataclass
class DetectorQuality:
    detector: str
    total: int
    confirmed: int
    rejected: int

    @property
    def precision(self) -> float:
        judged = self.confirmed + self.rejected
        return round(self.confirmed / judged, 4) if judged else 0.0

    def as_dict(self) -> dict:
        return asdict(self) | {"precision": self.precision}


@dataclass
class ProposedPolicy:
    name: str
    match: dict
    detectors: list[str]
    max_score: float
    reason: str
    supporting_exceptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrainingProposal:
    """What a retraining job suggests. Never applied automatically."""

    labelled_count: int
    detector_quality: list[DetectorQuality] = field(default_factory=list)
    suggested_weights: dict[str, float] = field(default_factory=dict)
    suggested_min_score: float | None = None
    proposed_policies: list[ProposedPolicy] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Always false. Promotion is a separate, explicit action.
    applied: bool = False

    def as_dict(self) -> dict:
        return {
            "labelled_count": self.labelled_count,
            "detector_quality": [d.as_dict() for d in self.detector_quality],
            "suggested_weights": {k: round(v, 4) for k, v in self.suggested_weights.items()},
            "suggested_min_score": self.suggested_min_score,
            "proposed_policies": [p.as_dict() for p in self.proposed_policies],
            "notes": self.notes,
            "applied": self.applied,
        }


#: Below this many labels for a detector, its precision is noise.
MIN_LABELS_PER_DETECTOR = 8

#: Below this many labels overall, do not propose anything at all.
MIN_LABELS_TO_PROPOSE = 15

#: A detector's weight is never driven below this, so a bad week cannot switch
#: a whole detection method off.
MIN_WEIGHT = 0.25


def summarise_quality(labelled: Sequence[LabelledException]) -> list[DetectorQuality]:
    """Per-detector precision, from the labels humans actually gave."""
    totals: Counter[str] = Counter()
    confirmed: Counter[str] = Counter()
    rejected: Counter[str] = Counter()

    for item in labelled:
        detector = item.strongest_detector or "unknown"
        totals[detector] += 1
        if item.label == "true_exception":
            confirmed[detector] += 1
        elif item.label in NEGATIVE_LABELS:
            rejected[detector] += 1

    return sorted(
        (
            DetectorQuality(
                detector=detector,
                total=totals[detector],
                confirmed=confirmed[detector],
                rejected=rejected[detector],
            )
            for detector in totals
        ),
        key=lambda d: -d.total,
    )


def propose_retraining(
    labelled: Sequence[LabelledException],
    *,
    current_weights: dict[str, float] | None = None,
    current_min_score: float = 60.0,
) -> RetrainingProposal:
    """Turn accumulated feedback into a reviewable proposal."""
    proposal = RetrainingProposal(labelled_count=len(labelled))
    if len(labelled) < MIN_LABELS_TO_PROPOSE:
        proposal.notes.append(
            f"Only {len(labelled)} labelled exceptions; at least "
            f"{MIN_LABELS_TO_PROPOSE} are needed before adjusting anything."
        )
        return proposal

    quality = summarise_quality(labelled)
    proposal.detector_quality = quality

    weights = dict(current_weights or {})
    for entry in quality:
        if entry.total < MIN_LABELS_PER_DETECTOR:
            proposal.notes.append(
                f"{entry.detector}: {entry.total} labels is too few to judge; weight unchanged."
            )
            continue
        current = weights.get(entry.detector, 1.0)
        # Nudge toward observed precision rather than jumping to it: one review
        # round is evidence, not proof.
        target = max(MIN_WEIGHT, min(1.0, entry.precision))
        adjusted = round(current + 0.5 * (target - current), 4)
        if abs(adjusted - current) >= 0.01:
            weights[entry.detector] = max(MIN_WEIGHT, adjusted)
            proposal.notes.append(
                f"{entry.detector}: precision {entry.precision:.0%} over {entry.total} "
                f"labels; weight {current:.2f} -> {weights[entry.detector]:.2f}."
            )
    proposal.suggested_weights = weights

    negatives = [item for item in labelled if item.label in NEGATIVE_LABELS]
    if negatives:
        share = len(negatives) / len(labelled)
        if share > 0.5:
            # Too much noise reaching humans: raise the bar, but only to just
            # above the loudest thing they rejected.
            ceiling = max(item.anomaly_score for item in negatives)
            suggested = min(90.0, max(current_min_score, round(ceiling + 1, 1)))
            if suggested > current_min_score:
                proposal.suggested_min_score = suggested
                proposal.notes.append(
                    f"{share:.0%} of reviewed exceptions were rejected; raising the "
                    f"threshold from {current_min_score} to {suggested} would have "
                    f"excluded them."
                )

    proposal.proposed_policies = _propose_policies(negatives)
    proposal.notes.append(
        "Nothing here has been applied. Promote the proposal to make it take effect."
    )
    return proposal


def _propose_policies(negatives: Sequence[LabelledException]) -> list[ProposedPolicy]:
    """Suggest suppression for repeated, agreed-upon non-problems."""
    grouped: defaultdict[tuple[str, str], list[LabelledException]] = defaultdict(list)
    for item in negatives:
        grouped[(item.baseline_scope, item.strongest_detector)].append(item)

    policies: list[ProposedPolicy] = []
    for (scope, detector), items in grouped.items():
        if len(items) < 3:
            continue
        ceiling = max(item.anomaly_score for item in items)
        policies.append(
            ProposedPolicy(
                name=f"Known-good: {detector} in {scope or 'all data'}",
                match=_scope_to_match(scope),
                detectors=[detector],
                # Capped just above what was actually rejected, so the policy
                # cannot hide a more severe version of the same shape later.
                max_score=min(95.0, round(ceiling + 2, 1)),
                reason=(
                    f"{len(items)} exceptions from {detector} in this scope were "
                    f"reviewed and found not to be problems."
                ),
                supporting_exceptions=[item.exception_id for item in items[:10]],
            )
        )
    policies.sort(key=lambda p: -len(p.supporting_exceptions))
    return policies


def _scope_to_match(scope: str) -> dict:
    """Turn a baseline scope key back into field equality conditions."""
    if not scope or scope == "*":
        return {}
    match: dict[str, str] = {}
    for part in scope.split("|"):
        key, _, value = part.partition("=")
        if key:
            match[key] = value
    return match
