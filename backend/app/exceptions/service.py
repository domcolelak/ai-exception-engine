"""Service layer: database <-> detection pipeline.

Loads events for a schema, runs the pipeline, persists baselines, exceptions
and the audit trail. Every query is tenant scoped.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.exceptions.generation import GenerationConfig, SuppressionPolicy
from app.feedback.loop import LabelledException
from app.models import (
    BaselineModel,
    Event,
    EventSchema,
    ExceptionCase,
    Feedback,
    ScoringRun,
    SuppressionPolicyRow,
)
from app.scoring.pipeline import SchemaSpec, run_pipeline
from app.similarity.search import ExceptionView


def get_schema(db: Session, tenant_id: uuid.UUID, schema_id: uuid.UUID) -> EventSchema | None:
    return db.scalar(
        select(EventSchema).where(
            EventSchema.tenant_id == tenant_id, EventSchema.id == schema_id
        )
    )


def spec_for(schema: EventSchema) -> SchemaSpec:
    return SchemaSpec.from_dict(
        {
            "name": schema.name,
            "entity_type": schema.entity_type,
            "entity_key": schema.entity_key,
            "event_key": schema.event_key,
            "timestamp_field": schema.timestamp_field,
            "numeric_features": schema.numeric_features or [],
            "categorical_features": schema.categorical_features or [],
            "context_dimensions": schema.context_dimensions or [],
            "rate_features": schema.rate_features or [],
            "detector_weights": schema.detector_weights or {},
        }
    )


def load_events(
    db: Session, tenant_id: uuid.UUID, schema_id: uuid.UUID, *, limit: int | None = None
) -> list[dict]:
    stmt = (
        select(Event)
        .where(Event.tenant_id == tenant_id, Event.schema_id == schema_id)
        .order_by(Event.occurred_at)
    )
    if limit:
        stmt = stmt.limit(limit)
    events = []
    for row in db.scalars(stmt):
        payload = dict(row.payload or {})
        # The stored timestamp is authoritative; the payload copy may be a
        # string from whatever the caller sent.
        payload["occurred_at"] = _aware(row.occurred_at)
        events.append(payload)
    return events


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def load_policies(
    db: Session, tenant_id: uuid.UUID, schema_id: uuid.UUID
) -> list[SuppressionPolicy]:
    rows = db.scalars(
        select(SuppressionPolicyRow).where(
            SuppressionPolicyRow.tenant_id == tenant_id,
            SuppressionPolicyRow.active.is_(True),
        )
    )
    return [
        SuppressionPolicy(
            id=str(row.id),
            name=row.name,
            match=row.match or {},
            max_score=row.max_score,
            detectors=row.detectors or [],
            reason=row.reason,
            active=row.active,
        )
        for row in rows
        if row.schema_id in (None, schema_id)
    ]


def run_detection(
    db: Session,
    tenant_id: uuid.UUID,
    schema_id: uuid.UUID,
    *,
    train_ratio: float = 0.6,
) -> ScoringRun:
    """Fit baselines, score events, persist the exceptions that survive."""
    schema = get_schema(db, tenant_id, schema_id)
    if schema is None:
        raise LookupError("schema not found for this tenant")

    run = ScoringRun(
        tenant_id=tenant_id, schema_id=schema_id, status="running"
    )
    db.add(run)
    db.flush()

    try:
        events = load_events(db, tenant_id, schema_id)
        spec = spec_for(schema)
        result = run_pipeline(
            events,
            spec,
            train_ratio=train_ratio,
            policies=load_policies(db, tenant_id, schema_id),
            config=GenerationConfig(min_score=schema.min_score),
        )

        baseline = _persist_baseline(db, tenant_id, schema_id, result, len(events))

        # Existing open exceptions must take part in deduplication, so the
        # pipeline's decisions are re-checked against what is already stored.
        existing = {
            (row.entity_id, row.fingerprint)
            for row in db.scalars(
                select(ExceptionCase).where(
                    ExceptionCase.tenant_id == tenant_id,
                    ExceptionCase.schema_id == schema_id,
                    ExceptionCase.status.in_(("open", "acknowledged", "in_progress")),
                )
            )
        }

        created = 0
        for scored in result.exceptions:
            candidate = scored.decision.created
            if (candidate.entity_id, candidate.fingerprint) in existing:
                continue
            existing.add((candidate.entity_id, candidate.fingerprint))
            db.add(
                ExceptionCase(
                    tenant_id=tenant_id,
                    schema_id=schema_id,
                    scoring_run_id=run.id,
                    entity_type=candidate.entity_type,
                    entity_id=candidate.entity_id,
                    event_external_id=candidate.event_id,
                    fingerprint=candidate.fingerprint,
                    title=candidate.title,
                    detected_at=candidate.detected_at,
                    anomaly_score=candidate.score,
                    severity=candidate.severity,
                    confidence=candidate.confidence,
                    baseline_scope=candidate.baseline_scope,
                    detector_versions=candidate.detector_versions,
                    reasons=candidate.reasons,
                    evidence=candidate.evidence,
                    event_payload=_serialisable(scored.event),
                )
            )
            created += 1

        run.baseline_id = baseline.id
        run.status = "completed"
        run.scored_count = len(result.scored)
        run.exception_count = created
        run.suppressed_count = len(result.suppressed)
        run.duplicate_count = len(result.duplicates)
    except Exception as exc:  # pragma: no cover - surfaced through the API
        run.status = "failed"
        run.error = str(exc)[:2000]
        raise
    finally:
        run.finished_at = datetime.now(timezone.utc)
        db.flush()

    return run


def _serialisable(event: dict) -> dict:
    return {
        k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in event.items()
    }


def _persist_baseline(
    db: Session,
    tenant_id: uuid.UUID,
    schema_id: uuid.UUID,
    result,
    event_count: int,
) -> BaselineModel:
    """Store the fitted baseline and make it the active one."""
    version = f"v{db.scalar(select(func.count(BaselineModel.id)).where(BaselineModel.tenant_id == tenant_id, BaselineModel.schema_id == schema_id)) + 1}"

    db.execute(
        update(BaselineModel)
        .where(
            BaselineModel.tenant_id == tenant_id,
            BaselineModel.schema_id == schema_id,
        )
        .values(active=False)
    )

    baseline = BaselineModel(
        tenant_id=tenant_id,
        schema_id=schema_id,
        version=version,
        active=True,
        trained_on=result.training_count,
        payload=result.baselines.as_dict(),
        peer_groups={k: v.as_dict() for k, v in (result.peer_groups or {}).items()},
    )
    db.add(baseline)
    db.flush()
    return baseline


def exception_views(
    db: Session, tenant_id: uuid.UUID, *, schema_id: uuid.UUID | None = None
) -> list[ExceptionView]:
    """Load exceptions in the shape the similarity search needs."""
    stmt = select(ExceptionCase).where(ExceptionCase.tenant_id == tenant_id)
    if schema_id:
        stmt = stmt.where(ExceptionCase.schema_id == schema_id)

    views = []
    for row in db.scalars(stmt):
        label = db.scalar(
            select(Feedback.label)
            .where(Feedback.tenant_id == tenant_id, Feedback.exception_id == row.id)
            .order_by(Feedback.created_at.desc())
            .limit(1)
        )
        views.append(
            ExceptionView(
                id=str(row.id),
                entity_id=row.entity_id,
                baseline_scope=row.baseline_scope,
                severity=row.severity,
                anomaly_score=row.anomaly_score,
                reasons=row.reasons or [],
                status=row.status,
                feedback_label=label,
                resolution_note=row.resolution_note,
                title=row.title,
            )
        )
    return views


def labelled_exceptions(
    db: Session, tenant_id: uuid.UUID, *, schema_id: uuid.UUID | None = None
) -> list[LabelledException]:
    """Every exception that carries a human verdict."""
    stmt = (
        select(ExceptionCase, Feedback)
        .join(Feedback, Feedback.exception_id == ExceptionCase.id)
        .where(ExceptionCase.tenant_id == tenant_id)
    )
    if schema_id:
        stmt = stmt.where(ExceptionCase.schema_id == schema_id)

    out = []
    for row, feedback in db.execute(stmt):
        out.append(
            LabelledException(
                exception_id=str(row.id),
                label=feedback.label,
                entity_id=row.entity_id,
                baseline_scope=row.baseline_scope,
                severity=row.severity,
                anomaly_score=row.anomaly_score,
                strongest_detector=(row.evidence or {}).get("strongest_detector", ""),
                reasons=row.reasons or [],
            )
        )
    return out
