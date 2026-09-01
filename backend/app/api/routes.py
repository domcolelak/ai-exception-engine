"""HTTP API.

Every endpoint resolves a :class:`TenantContext` first and filters every query
by ``ctx.tenant_id``. Cross-tenant access returns 404, not 403.

Route ordering note: literal paths are declared before parameterised ones on
the same prefix, because FastAPI matches in declaration order and a literal
declared afterwards is unreachable.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.explain import explain_exception
from app.audit.log import record_audit
from app.core.db import get_db
from app.core.security import TenantContext, current_tenant
from app.exceptions.service import (
    exception_views,
    get_schema,
    labelled_exceptions,
    run_detection,
)
from app.feedback.loop import propose_retraining, summarise_quality
from app.models import (
    BaselineModel,
    Event,
    EventSchema,
    ExceptionCase,
    Feedback,
    ScoringRun,
    SuppressionPolicyRow,
)
from app.notifications.dispatch import notify
from app.scoring.ensemble import recompute
from app.schemas import (
    AssignRequest,
    BaselineDetail,
    BaselineOut,
    DetectionRequest,
    EventBatchRequest,
    EventBatchResponse,
    ExceptionDetail,
    ExceptionOut,
    FeedbackOut,
    FeedbackRequest,
    OverviewResponse,
    PolicyCreate,
    PolicyOut,
    PromoteRequest,
    ResolveRequest,
    RetrainingProposalOut,
    SchemaCreate,
    SchemaListItem,
    SchemaOut,
    ScoringRunOut,
    SimilarResponse,
)
from app.similarity.search import find_similar

router = APIRouter()

SIMILARITY_NOTE = (
    "Similarity is computed from the structured evidence -- which detector "
    "fired, on which features, in which scope -- not from a text embedding, so "
    "every component of the score is shown."
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


@router.get("/schemas", response_model=list[SchemaListItem])
def list_schemas(
    ctx: TenantContext = Depends(current_tenant), db: Session = Depends(get_db)
) -> list[SchemaListItem]:
    schemas = db.scalars(
        select(EventSchema)
        .where(EventSchema.tenant_id == ctx.tenant_id)
        .order_by(EventSchema.name)
    ).all()

    items = []
    for schema in schemas:
        events = db.scalar(
            select(func.count(Event.id)).where(
                Event.tenant_id == ctx.tenant_id, Event.schema_id == schema.id
            )
        )
        open_count = db.scalar(
            select(func.count(ExceptionCase.id)).where(
                ExceptionCase.tenant_id == ctx.tenant_id,
                ExceptionCase.schema_id == schema.id,
                ExceptionCase.status == "open",
            )
        )
        items.append(
            SchemaListItem(
                **SchemaOut.model_validate(schema).model_dump(),
                event_count=events or 0,
                open_exceptions=open_count or 0,
            )
        )
    return items


@router.post("/schemas", response_model=SchemaOut, status_code=status.HTTP_201_CREATED)
def create_schema(
    body: SchemaCreate,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> EventSchema:
    if not body.numeric_features and not body.categorical_features:
        raise HTTPException(
            status_code=422,
            detail="a schema needs at least one numeric or categorical feature",
        )
    existing = db.scalar(
        select(EventSchema).where(
            EventSchema.tenant_id == ctx.tenant_id, EventSchema.name == body.name
        )
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="a schema with that name already exists")

    schema = EventSchema(tenant_id=ctx.tenant_id, **body.model_dump())
    db.add(schema)
    db.flush()
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="schema.created",
        object_type="schema",
        object_id=schema.id,
    )
    return schema


@router.get("/schemas/{schema_id}", response_model=SchemaOut)
def get_schema_detail(
    schema_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> EventSchema:
    return _require_schema(db, ctx, schema_id)


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


@router.post("/events/batch", response_model=EventBatchResponse)
def ingest_events(
    body: EventBatchRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> EventBatchResponse:
    """Idempotent batch ingestion, validated against the tenant's schema."""
    schema = _require_schema(db, ctx, body.schema_id)

    known = {
        row
        for row in db.scalars(
            select(Event.external_id).where(
                Event.tenant_id == ctx.tenant_id, Event.schema_id == schema.id
            )
        )
    }

    accepted = duplicates = 0
    errors: list[dict] = []
    seen: set[str] = set()

    for index, item in enumerate(body.events):
        payload = item.payload
        entity_id = str(payload.get(schema.entity_key, "")).strip()
        occurred_at = item.occurred_at or _parse_time(payload.get(schema.timestamp_field))

        problems = []
        if not entity_id:
            problems.append(f"missing entity key '{schema.entity_key}'")
        if occurred_at is None:
            problems.append(f"missing or unparseable '{schema.timestamp_field}'")
        # A feature the schema declares but the event omits is a data-quality
        # problem worth reporting, not a silent zero.
        missing = [
            f
            for f in (schema.numeric_features or [])
            if f not in payload
        ]
        if missing:
            problems.append(f"missing declared numeric features: {', '.join(missing[:5])}")
        if problems:
            errors.append({"index": index, "problems": problems})
            continue

        external_id = (
            item.external_id
            or str(payload.get(schema.event_key, ""))
            or f"{entity_id}:{occurred_at.isoformat()}"
        )
        if external_id in known or external_id in seen:
            duplicates += 1
            continue
        seen.add(external_id)

        db.add(
            Event(
                tenant_id=ctx.tenant_id,
                schema_id=schema.id,
                external_id=external_id,
                entity_id=entity_id,
                occurred_at=occurred_at,
                payload=payload,
            )
        )
        accepted += 1

    db.flush()
    return EventBatchResponse(
        schema_id=schema.id,
        accepted=accepted,
        duplicates=duplicates,
        rejected=len(errors),
        errors=errors[:50],
    )


def _parse_time(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@router.post("/detection-runs", response_model=ScoringRunOut, status_code=status.HTTP_201_CREATED)
def create_detection_run(
    body: DetectionRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> ScoringRun:
    _require_schema(db, ctx, body.schema_id)
    try:
        run = run_detection(db, ctx.tenant_id, body.schema_id, train_ratio=body.train_ratio)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    for case in db.scalars(
        select(ExceptionCase).where(
            ExceptionCase.tenant_id == ctx.tenant_id,
            ExceptionCase.scoring_run_id == run.id,
        )
    ):
        notify(db, ctx.tenant_id, case)

    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="detection.run",
        object_type="scoring_run",
        object_id=run.id,
        payload={"exceptions": run.exception_count, "scored": run.scored_count},
    )
    return run


@router.get("/detection-runs/{run_id}", response_model=ScoringRunOut)
def get_detection_run(
    run_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> ScoringRun:
    run = db.scalar(
        select(ScoringRun).where(
            ScoringRun.tenant_id == ctx.tenant_id, ScoringRun.id == run_id
        )
    )
    if run is None:
        raise HTTPException(status_code=404, detail="detection run not found")
    return run


@router.get("/baselines", response_model=list[BaselineOut])
def list_baselines(
    schema_id: uuid.UUID | None = None,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> list[BaselineModel]:
    stmt = select(BaselineModel).where(BaselineModel.tenant_id == ctx.tenant_id)
    if schema_id:
        stmt = stmt.where(BaselineModel.schema_id == schema_id)
    return list(db.scalars(stmt.order_by(BaselineModel.created_at.desc())))


@router.get("/baselines/{baseline_id}", response_model=BaselineDetail)
def get_baseline(
    baseline_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> BaselineModel:
    baseline = db.scalar(
        select(BaselineModel).where(
            BaselineModel.tenant_id == ctx.tenant_id, BaselineModel.id == baseline_id
        )
    )
    if baseline is None:
        raise HTTPException(status_code=404, detail="baseline not found")
    return baseline


# --------------------------------------------------------------------------
# Exception queue
# --------------------------------------------------------------------------


@router.get("/exceptions", response_model=list[ExceptionOut])
def list_exceptions(
    schema_id: uuid.UUID | None = None,
    exception_status: str | None = Query(None, alias="status"),
    severity: str | None = None,
    entity_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> list[ExceptionCase]:
    """The work queue, most severe first."""
    stmt = select(ExceptionCase).where(ExceptionCase.tenant_id == ctx.tenant_id)
    if schema_id:
        stmt = stmt.where(ExceptionCase.schema_id == schema_id)
    if exception_status:
        stmt = stmt.where(ExceptionCase.status == exception_status)
    if severity:
        stmt = stmt.where(ExceptionCase.severity == severity)
    if entity_id:
        stmt = stmt.where(ExceptionCase.entity_id == entity_id)
    stmt = stmt.order_by(ExceptionCase.anomaly_score.desc()).limit(limit)
    return list(db.scalars(stmt))


@router.get("/exceptions/{exception_id}", response_model=ExceptionDetail)
def get_exception(
    exception_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> ExceptionDetail:
    case = _require_exception(db, ctx, exception_id)
    feedback = db.scalars(
        select(Feedback).where(
            Feedback.tenant_id == ctx.tenant_id, Feedback.exception_id == exception_id
        )
    ).all()
    return ExceptionDetail(
        **ExceptionOut.model_validate(case).model_dump(),
        evidence=case.evidence or {},
        event_payload=case.event_payload or {},
        # Proves the stored score follows from the stored detector outputs.
        recomputed_score=recompute(case.evidence or {}),
        feedback=[
            {
                "label": f.label,
                "note": f.note,
                "author": f.author,
                "created_at": f.created_at.isoformat(),
            }
            for f in feedback
        ],
    )


@router.get("/exceptions/{exception_id}/similar", response_model=SimilarResponse)
def similar_exceptions(
    exception_id: uuid.UUID,
    limit: int = Query(5, ge=1, le=25),
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> SimilarResponse:
    case = _require_exception(db, ctx, exception_id)
    views = exception_views(db, ctx.tenant_id, schema_id=case.schema_id)
    target = next((v for v in views if v.id == str(exception_id)), None)
    if target is None:
        raise HTTPException(status_code=404, detail="exception not found")
    matches = find_similar(target, views, limit=limit)
    return SimilarResponse(
        exception_id=exception_id,
        similar=[m.as_dict() for m in matches],
        note=SIMILARITY_NOTE,
    )


@router.post("/exceptions/{exception_id}/explain", response_model=ExceptionOut)
def explain(
    exception_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> ExceptionCase:
    case = _require_exception(db, ctx, exception_id)
    views = exception_views(db, ctx.tenant_id, schema_id=case.schema_id)
    target = next((v for v in views if v.id == str(exception_id)), None)
    similar = (
        [m.as_dict() for m in find_similar(target, views, limit=3)] if target else []
    )

    narrative = explain_exception(
        db,
        ctx.tenant_id,
        title=case.title,
        entity_type=case.entity_type,
        entity_id=case.entity_id,
        score=case.anomaly_score,
        severity=case.severity,
        confidence=case.confidence,
        baseline_scope=case.baseline_scope,
        reasons=case.reasons or [],
        similar=similar,
    )
    if narrative is not None:
        case.narrative = narrative.model_dump()
    db.flush()
    return case


@router.post("/exceptions/{exception_id}/assign", response_model=ExceptionOut)
def assign_exception(
    exception_id: uuid.UUID,
    body: AssignRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> ExceptionCase:
    case = _require_exception(db, ctx, exception_id)
    case.assignee = body.assignee
    if case.status == "open":
        case.status = "acknowledged"
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="exception.assigned",
        object_type="exception",
        object_id=exception_id,
        payload={"assignee": body.assignee},
    )
    return case


@router.post("/exceptions/{exception_id}/resolve", response_model=ExceptionOut)
def resolve_exception(
    exception_id: uuid.UUID,
    body: ResolveRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> ExceptionCase:
    case = _require_exception(db, ctx, exception_id)
    case.status = body.status
    case.resolution_note = body.resolution_note
    case.resolved_at = (
        datetime.now(timezone.utc) if body.status in ("resolved", "dismissed") else None
    )
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="exception.status_changed",
        object_type="exception",
        object_id=exception_id,
        payload={"status": body.status},
    )
    return case


@router.post(
    "/exceptions/{exception_id}/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
)
def add_feedback(
    exception_id: uuid.UUID,
    body: FeedbackRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> Feedback:
    """Record a human verdict.

    This does not change any model. It accumulates until somebody asks for a
    retraining proposal and explicitly promotes it.
    """
    _require_exception(db, ctx, exception_id)
    entry = Feedback(
        tenant_id=ctx.tenant_id,
        exception_id=exception_id,
        label=body.label,
        note=body.note,
        author=body.author,
    )
    db.add(entry)
    db.flush()
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="feedback.recorded",
        object_type="exception",
        object_id=exception_id,
        payload={"label": body.label},
    )
    return entry


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------


@router.get("/policies", response_model=list[PolicyOut])
def list_policies(
    ctx: TenantContext = Depends(current_tenant), db: Session = Depends(get_db)
) -> list[SuppressionPolicyRow]:
    return list(
        db.scalars(
            select(SuppressionPolicyRow)
            .where(SuppressionPolicyRow.tenant_id == ctx.tenant_id)
            .order_by(SuppressionPolicyRow.created_at.desc())
        )
    )


@router.post("/policies", response_model=PolicyOut, status_code=status.HTTP_201_CREATED)
def create_policy(
    body: PolicyCreate,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> SuppressionPolicyRow:
    if body.schema_id is not None:
        _require_schema(db, ctx, body.schema_id)
    policy = SuppressionPolicyRow(tenant_id=ctx.tenant_id, **body.model_dump())
    db.add(policy)
    db.flush()
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="policy.created",
        object_type="policy",
        object_id=policy.id,
        payload={"name": policy.name, "max_score": policy.max_score},
    )
    return policy


@router.post("/policies/{policy_id}/deactivate", response_model=PolicyOut)
def deactivate_policy(
    policy_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> SuppressionPolicyRow:
    policy = db.scalar(
        select(SuppressionPolicyRow).where(
            SuppressionPolicyRow.tenant_id == ctx.tenant_id,
            SuppressionPolicyRow.id == policy_id,
        )
    )
    if policy is None:
        raise HTTPException(status_code=404, detail="policy not found")
    policy.active = False
    return policy


# --------------------------------------------------------------------------
# Feedback loop
# --------------------------------------------------------------------------


@router.get("/retraining/proposal", response_model=RetrainingProposalOut)
def retraining_proposal(
    schema_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> RetrainingProposalOut:
    """What accumulated feedback suggests changing. Applies nothing."""
    schema = _require_schema(db, ctx, schema_id)
    labelled = labelled_exceptions(db, ctx.tenant_id, schema_id=schema_id)
    proposal = propose_retraining(
        labelled,
        current_weights=schema.detector_weights or {},
        current_min_score=schema.min_score,
    )
    return RetrainingProposalOut(schema_id=schema_id, **proposal.as_dict())


@router.post("/retraining/promote", response_model=SchemaOut)
def promote_proposal(
    schema_id: uuid.UUID,
    body: PromoteRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> EventSchema:
    """Explicitly apply a retraining proposal.

    Separate from generating it on purpose: feedback must never reconfigure
    detection on its own.
    """
    schema = _require_schema(db, ctx, schema_id)
    labelled = labelled_exceptions(db, ctx.tenant_id, schema_id=schema_id)
    proposal = propose_retraining(
        labelled,
        current_weights=schema.detector_weights or {},
        current_min_score=schema.min_score,
    )

    applied: dict = {}
    if body.apply_weights and proposal.suggested_weights:
        schema.detector_weights = proposal.suggested_weights
        applied["weights"] = proposal.suggested_weights
    if body.apply_min_score and proposal.suggested_min_score is not None:
        schema.min_score = proposal.suggested_min_score
        applied["min_score"] = proposal.suggested_min_score
    if body.create_policies:
        for proposed in proposal.proposed_policies:
            db.add(
                SuppressionPolicyRow(
                    tenant_id=ctx.tenant_id,
                    schema_id=schema_id,
                    name=proposed.name,
                    match=proposed.match,
                    detectors=proposed.detectors,
                    max_score=proposed.max_score,
                    reason=proposed.reason,
                    created_by="retraining_proposal",
                )
            )
        applied["policies"] = len(proposal.proposed_policies)

    db.flush()
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="retraining.promoted",
        object_type="schema",
        object_id=schema_id,
        payload=applied,
    )
    return schema


@router.get("/overview", response_model=OverviewResponse)
def overview(
    ctx: TenantContext = Depends(current_tenant), db: Session = Depends(get_db)
) -> OverviewResponse:
    def count(model, *conditions):
        return (
            db.scalar(select(func.count(model.id)).where(model.tenant_id == ctx.tenant_id, *conditions))
            or 0
        )

    top = list(
        db.scalars(
            select(ExceptionCase)
            .where(
                ExceptionCase.tenant_id == ctx.tenant_id,
                ExceptionCase.status.in_(("open", "acknowledged", "in_progress")),
            )
            .order_by(ExceptionCase.anomaly_score.desc())
            .limit(10)
        )
    )
    quality = summarise_quality(labelled_exceptions(db, ctx.tenant_id))

    return OverviewResponse(
        schema_count=count(EventSchema),
        event_count=count(Event),
        open_exceptions=count(ExceptionCase, ExceptionCase.status == "open"),
        critical_exceptions=count(ExceptionCase, ExceptionCase.severity == "critical"),
        resolved_exceptions=count(ExceptionCase, ExceptionCase.status == "resolved"),
        labelled_exceptions=count(Feedback),
        detector_precision=[q.as_dict() for q in quality],
        top_exceptions=[ExceptionOut.model_validate(e) for e in top],
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _require_schema(db: Session, ctx: TenantContext, schema_id: uuid.UUID) -> EventSchema:
    schema = get_schema(db, ctx.tenant_id, schema_id)
    if schema is None:
        raise HTTPException(status_code=404, detail="schema not found")
    return schema


def _require_exception(
    db: Session, ctx: TenantContext, exception_id: uuid.UUID
) -> ExceptionCase:
    case = db.scalar(
        select(ExceptionCase).where(
            ExceptionCase.tenant_id == ctx.tenant_id, ExceptionCase.id == exception_id
        )
    )
    if case is None:
        raise HTTPException(status_code=404, detail="exception not found")
    return case
