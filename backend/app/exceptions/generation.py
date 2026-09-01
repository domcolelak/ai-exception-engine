"""Turning scores into exceptions.

Scoring every event is easy; deciding which scores deserve a human is the part
that determines whether anyone still reads the queue in month two.

An exception is created only when all five hold:

1. the score clears the threshold,
2. the ensemble is confident enough to be worth someone's time,
3. at least one detector produced a specific, citable reason,
4. no suppression policy covers it,
5. it is not a repeat of something already open for the same entity.

Rule 4 matters more than it looks. Without it a seller having a bad week
generates one exception per order, and the queue becomes unreadable exactly
when it is most needed.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from app.scoring.ensemble import EnsembleResult


#: Detectors whose finding is a property of the entity rather than of one
#: event. "This seller refunds far more than its peers" does not become new
#: information because another order arrived, so while such an exception is
#: open a repeat is always folded into it -- the time window does not apply.
ENTITY_LEVEL_DETECTORS = frozenset({"peer_group", "change_point"})


@dataclass
class GenerationConfig:
    min_score: float = 60.0
    min_confidence: float = 0.5
    #: A repeat for the same entity and reason within this window is folded
    #: into the existing exception instead of creating a new one.
    dedup_window_hours: float = 72.0
    #: Cap per entity per window, so one misbehaving entity cannot flood the queue.
    max_open_per_entity: int = 3


@dataclass
class SuppressionPolicy:
    """A known-and-accepted pattern that should not raise exceptions."""

    id: str
    name: str
    #: Exact-match conditions on event fields.
    match: dict[str, Any] = field(default_factory=dict)
    #: Only suppress below this score, so a policy cannot hide a genuine crisis.
    max_score: float = 100.0
    detectors: list[str] = field(default_factory=list)
    reason: str = ""
    active: bool = True

    def covers(self, event: dict[str, Any], result: EnsembleResult) -> bool:
        if not self.active:
            return False
        if result.score > self.max_score:
            return False
        if self.detectors and result.strongest not in self.detectors:
            return False
        for key, expected in self.match.items():
            if str(event.get(key, "")) != str(expected):
                return False
        return True


@dataclass
class ExceptionCandidate:
    entity_type: str
    entity_id: str
    event_id: str
    detected_at: datetime
    score: float
    severity: str
    confidence: float
    fingerprint: str
    title: str
    reasons: list[dict] = field(default_factory=list)
    detector_versions: dict[str, str] = field(default_factory=dict)
    baseline_scope: str = ""
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["detected_at"] = self.detected_at.isoformat()
        return payload


@dataclass
class GenerationDecision:
    created: ExceptionCandidate | None
    suppressed_by: str | None = None
    duplicate_of: str | None = None
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.created is not None


@dataclass
class OpenException:
    """The minimum an existing exception must expose for deduplication."""

    id: str
    entity_id: str
    fingerprint: str
    detected_at: datetime
    status: str


def fingerprint_for(
    entity_id: str, reasons: Sequence[dict], strongest: str | None = None
) -> str:
    """Identity of *what kind of problem* this is, for one entity.

    Built from the leading detector and the features it fired on -- values are
    excluded, so the same problem recurring on the same entity collapses while
    a genuinely different problem still surfaces.

    Only the leading detector's reasons count. Using every reason made the
    fingerprint unstable: whenever some unrelated detector happened to chime in
    on one event, the hash changed and an already-open problem was raised
    again. That is how a single misbehaving entity produced eleven identical
    exceptions.
    """
    relevant = (
        [r for r in reasons if r.get("detector") == strongest] if strongest else []
    ) or list(reasons)
    features = sorted({str(r.get("feature", "")) for r in relevant if r.get("feature")})
    raw = f"{entity_id}|{strongest or ''}|" + "|".join(features)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]


def build_title(entity_type: str, entity_id: str, result: EnsembleResult) -> str:
    """A deterministic, human-readable headline.

    Generated here rather than by the model: the queue must be readable and
    sortable whether or not an AI provider is configured. The phrasing is
    detector-aware because the raw feature name is often not a sentence -- a
    peer-group finding on ``rate:refund_amount`` is "refunds more often than
    its peers", and an isolation-forest finding spans several features at once.
    """
    if not result.reasons:
        return f"Unusual behaviour for {entity_type} {entity_id}"

    primary = result.reasons[0]
    feature = primary.feature
    detector = primary.detector or result.strongest or ""

    if detector == "peer_group":
        kind, _, name = feature.partition(":")
        if kind == "rate":
            return f"{entity_type} {entity_id} records {name} far more often than its peers"
        return f"{entity_type} {entity_id} has an unusually high average {name} for its peers"

    if detector == "isolation_forest":
        return f"Unusual combination of values for {entity_type} {entity_id}"

    if detector == "change_point":
        return f"{feature} shifted to a new level for {entity_type} {entity_id}"

    if detector == "rolling_quantile":
        return f"{feature} is far above this {entity_type}'s own recent range"

    if detector == "categorical_rarity":
        return f"Rare {feature} for {entity_type} {entity_id}"

    return f"{feature} is far outside the normal range for {entity_type} {entity_id}"


def evaluate(
    *,
    event: dict[str, Any],
    result: EnsembleResult,
    entity_type: str,
    entity_id: str,
    event_id: str,
    detected_at: datetime | None = None,
    open_exceptions: Sequence[OpenException] = (),
    policies: Sequence[SuppressionPolicy] = (),
    config: GenerationConfig | None = None,
) -> GenerationDecision:
    """Decide whether this scored event becomes an exception."""
    cfg = config or GenerationConfig()
    now = detected_at or datetime.now(timezone.utc)

    if result.score < cfg.min_score:
        return GenerationDecision(None, reason=f"score {result.score} below {cfg.min_score}")
    if result.confidence < cfg.min_confidence:
        return GenerationDecision(
            None, reason=f"confidence {result.confidence} below {cfg.min_confidence}"
        )

    for policy in policies:
        if policy.covers(event, result):
            return GenerationDecision(
                None,
                suppressed_by=policy.id,
                reason=f"suppressed by policy '{policy.name}': {policy.reason}",
            )

    reasons = [r.as_dict() for r in result.reasons]
    # A detector reports a score even when no single feature crossed its
    # reporting threshold, and agreement between several such near-misses can
    # push the total over the bar. The result is a case with a number and no
    # explanation, which is worse than no case at all: nobody can action it and
    # nobody can check it.
    if not reasons:
        return GenerationDecision(
            None,
            reason=(
                "score cleared the threshold but no detector produced a specific "
                "reason; an exception nobody can explain is not raised"
            ),
        )

    fingerprint = fingerprint_for(entity_id, reasons, result.strongest)
    window_start = now - timedelta(hours=cfg.dedup_window_hours)
    entity_level = result.strongest in ENTITY_LEVEL_DETECTORS

    open_for_entity = [
        e
        for e in open_exceptions
        if e.entity_id == entity_id
        and e.status in ("open", "acknowledged", "in_progress")
    ]

    # An entity-level finding stays a duplicate for as long as it is open. An
    # event-level one can legitimately recur once the window has passed.
    for existing in open_for_entity:
        if existing.fingerprint != fingerprint:
            continue
        if entity_level or _aware(existing.detected_at) >= window_start:
            return GenerationDecision(
                None,
                duplicate_of=existing.id,
                reason=(
                    "the same entity-level problem is already open"
                    if entity_level
                    else "the same problem is already open for this entity"
                ),
            )

    # The cap counts everything still open for this entity, not just recent
    # arrivals. Windowing it defeats the purpose: an entity that keeps
    # producing findings simply waits out the window and files the next one,
    # which is how a single seller came to own eleven of eighteen exceptions.
    if len(open_for_entity) >= cfg.max_open_per_entity:
        return GenerationDecision(
            None,
            reason=(
                f"{entity_id} already has {len(open_for_entity)} open exceptions; "
                f"capped at {cfg.max_open_per_entity} so one entity cannot flood the "
                f"queue. Resolve one to see the next."
            ),
        )

    candidate = ExceptionCandidate(
        entity_type=entity_type,
        entity_id=entity_id,
        event_id=event_id,
        detected_at=now,
        score=result.score,
        severity=result.severity,
        confidence=result.confidence,
        fingerprint=fingerprint,
        title=build_title(entity_type, entity_id, result),
        reasons=reasons,
        detector_versions={
            d.detector: d.version for d in result.detector_results if d.applicable
        },
        baseline_scope=_scope_of(result),
        evidence={
            "weights": result.weights,
            "weighted_mean": round(result.weighted_mean, 6),
            "strongest_detector": result.strongest,
            "detectors": [d.as_dict() for d in result.detector_results],
        },
    )
    return GenerationDecision(candidate)


def _scope_of(result: EnsembleResult) -> str:
    for detector in result.detector_results:
        scope = detector.evidence.get("scope")
        if scope:
            return str(scope)
    return ""


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
