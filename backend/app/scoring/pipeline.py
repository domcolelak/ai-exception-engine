"""Scoring pipeline.

Ties baselines, detectors, the ensemble and exception generation into one
deterministic pass over a batch of events.

The ordering rule that makes results trustworthy: an event is always scored
against history that precedes it. Baselines are fitted on a training slice, and
per-entity/per-scope history passed to a detector never contains the event
being scored. Getting this wrong produces a detector that looks well-calibrated
in testing and finds nothing in production.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from app.baselines.model import BaselineSet, build_baselines
from app.detectors.base import DetectionContext, Detector
from app.detectors.builtin import DEFAULT_DETECTORS
from app.detectors.peer_group import PeerGroupDetector, build_peer_groups
from app.exceptions.generation import (
    GenerationConfig,
    GenerationDecision,
    OpenException,
    SuppressionPolicy,
    evaluate,
)
from app.scoring.ensemble import EnsembleResult, score_event


@dataclass
class SchemaSpec:
    """What the tenant told us their events mean."""

    name: str
    entity_type: str
    entity_key: str
    event_key: str
    timestamp_field: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    context_dimensions: tuple[str, ...] = ()
    #: Numeric columns whose *occurrence* is meaningful, not just their size --
    #: a refund amount is one: the share of non-zero values is the refund rate.
    rate_features: tuple[str, ...] = ()
    detector_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> SchemaSpec:
        return cls(
            name=payload["name"],
            entity_type=payload.get("entity_type", "entity"),
            entity_key=payload.get("entity_key", "entity_id"),
            event_key=payload.get("event_key", "event_id"),
            timestamp_field=payload.get("timestamp_field", "occurred_at"),
            numeric_features=tuple(payload.get("numeric_features", [])),
            categorical_features=tuple(payload.get("categorical_features", [])),
            context_dimensions=tuple(payload.get("context_dimensions", [])),
            rate_features=tuple(payload.get("rate_features", [])),
            detector_weights=dict(payload.get("detector_weights", {})),
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "entity_key": self.entity_key,
            "event_key": self.event_key,
            "timestamp_field": self.timestamp_field,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "context_dimensions": list(self.context_dimensions),
            "rate_features": list(self.rate_features),
            "detector_weights": self.detector_weights,
        }


@dataclass
class ScoredEvent:
    event: dict[str, Any]
    result: EnsembleResult
    decision: GenerationDecision


@dataclass
class PipelineResult:
    baselines: BaselineSet
    scored: list[ScoredEvent] = field(default_factory=list)
    training_count: int = 0
    peer_groups: dict = field(default_factory=dict)

    @property
    def exceptions(self) -> list[ScoredEvent]:
        return [s for s in self.scored if s.decision.accepted]

    @property
    def suppressed(self) -> list[ScoredEvent]:
        return [s for s in self.scored if s.decision.suppressed_by]

    @property
    def duplicates(self) -> list[ScoredEvent]:
        return [s for s in self.scored if s.decision.duplicate_of]


def fit_baselines(
    events: Sequence[dict[str, Any]], schema: SchemaSpec, *, version: str = "v1"
) -> BaselineSet:
    return build_baselines(
        events,
        numeric_features=schema.numeric_features,
        categorical_features=schema.categorical_features,
        dimensions=schema.context_dimensions,
        version=version,
    )


def run_pipeline(
    events: Sequence[dict[str, Any]],
    schema: SchemaSpec,
    *,
    detectors: Sequence[Detector] = DEFAULT_DETECTORS,
    baselines: BaselineSet | None = None,
    train_ratio: float = 0.6,
    policies: Sequence[SuppressionPolicy] = (),
    config: GenerationConfig | None = None,
    entity_history_limit: int = 400,
    scope_history_limit: int = 400,
) -> PipelineResult:
    """Fit baselines on the earlier part of the log, score the rest."""
    ordered = sorted(events, key=lambda e: _timestamp(e, schema))
    if not ordered:
        return PipelineResult(baselines=BaselineSet(dimensions=schema.context_dimensions))

    split = int(len(ordered) * train_ratio)
    training = ordered[:split] if baselines is None else ordered
    fitted = baselines or fit_baselines(training, schema)

    scoring_slice = ordered[split:] if baselines is None else ordered

    # Peer groups are aggregates, so they are built from the same training
    # slice the baselines use -- never from the events being scored.
    peer_groups = build_peer_groups(
        training,
        entity_key=schema.entity_key,
        numeric_features=schema.numeric_features,
        rate_features=schema.rate_features,
        dimensions=schema.context_dimensions,
    )
    active = list(detectors) + [
        PeerGroupDetector(
            peer_groups,
            entity_key=schema.entity_key,
            dimensions=schema.context_dimensions,
        )
    ]

    # Pre-fit the forest once per scope from the training slice. Left to fit
    # lazily it would refit on every event, which dominates the run time.
    training_by_scope: defaultdict[str, list[dict]] = defaultdict(list)
    for train_event in training:
        training_by_scope[fitted.scope_key_for(train_event)].append(train_event)
        # The global scope is fitted too: a thin scope falls back to the global
        # baseline, and without a model there it would refit lazily per event.
        training_by_scope["*"].append(train_event)
    for detector in active:
        fit = getattr(detector, "fit", None)
        if fit is None:
            continue
        for scope_name, scope_events in training_by_scope.items():
            fit(scope_name, scope_events, schema.numeric_features)

    # Histories are accumulated as we walk forward, so an event never sees
    # itself or anything after it.
    entity_history: defaultdict[str, list[dict]] = defaultdict(list)
    scope_history: defaultdict[str, list[dict]] = defaultdict(list)
    for event in ordered[:split]:
        entity_history[str(event.get(schema.entity_key, ""))].append(event)
        scope_history[fitted.scope_key_for(event)].append(event)

    open_exceptions: list[OpenException] = []
    result = PipelineResult(
        baselines=fitted, training_count=len(training), peer_groups=peer_groups
    )

    for event in scoring_slice:
        entity_id = str(event.get(schema.entity_key, ""))
        scope = fitted.scope_key_for(event)

        context = DetectionContext(
            event=event,
            baseline=fitted.lookup(event),
            numeric_features=schema.numeric_features,
            categorical_features=schema.categorical_features,
            entity_history=entity_history[entity_id][-entity_history_limit:],
            scope_history=scope_history[scope][-scope_history_limit:],
            scope_dimensions=schema.context_dimensions,
        )

        ensemble = score_event(context, active, weights=schema.detector_weights)
        decision = evaluate(
            event=event,
            result=ensemble,
            entity_type=schema.entity_type,
            entity_id=entity_id,
            event_id=str(event.get(schema.event_key, "")),
            detected_at=_timestamp(event, schema),
            open_exceptions=open_exceptions,
            policies=policies,
            config=config,
        )

        if decision.created is not None:
            open_exceptions.append(
                OpenException(
                    id=decision.created.event_id,
                    entity_id=entity_id,
                    fingerprint=decision.created.fingerprint,
                    detected_at=decision.created.detected_at,
                    status="open",
                )
            )

        result.scored.append(ScoredEvent(event=event, result=ensemble, decision=decision))

        entity_history[entity_id].append(event)
        scope_history[scope].append(event)

    return result


def _timestamp(event: dict, schema: SchemaSpec) -> datetime:
    value = event.get(schema.timestamp_field)
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    from datetime import timezone

    return datetime(1970, 1, 1, tzinfo=timezone.utc)
