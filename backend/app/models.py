"""SQLAlchemy models.

Every tenant-owned table carries ``tenant_id`` and indexes it first.

Model versioning is deliberate throughout: baselines, detector configurations
and the ensemble weights are all stored with a version, and an exception keeps
the versions that produced it. Without that, "why did this fire?" becomes
unanswerable the moment anything is retrained -- and retraining is the whole
point of the feedback loop.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base, GUID


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    api_key_hash: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(320))
    display_name: Mapped[str] = mapped_column(String(200), default="")
    role: Mapped[str] = mapped_column(String(32), default="analyst")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_user_email_per_tenant"),)


class EventSchema(Base):
    """The tenant's description of what their events mean."""

    __tablename__ = "event_schemas"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    entity_type: Mapped[str] = mapped_column(String(64), default="entity")
    entity_key: Mapped[str] = mapped_column(String(200), default="entity_id")
    event_key: Mapped[str] = mapped_column(String(200), default="event_id")
    timestamp_field: Mapped[str] = mapped_column(String(200), default="occurred_at")
    numeric_features: Mapped[list] = mapped_column(JSON, default=list)
    categorical_features: Mapped[list] = mapped_column(JSON, default=list)
    context_dimensions: Mapped[list] = mapped_column(JSON, default=list)
    rate_features: Mapped[list] = mapped_column(JSON, default=list)
    detector_weights: Mapped[dict] = mapped_column(JSON, default=dict)
    min_score: Mapped[float] = mapped_column(Float, default=60.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_schema_name_per_tenant"),)


class Event(Base):
    """One ingested event, stored as validated JSON against its schema."""

    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    schema_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("event_schemas.id"), index=True)
    #: Caller-supplied identity; ingestion is idempotent on this.
    external_id: Mapped[str] = mapped_column(String(200), index=True)
    entity_id: Mapped[str] = mapped_column(String(200), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("tenant_id", "schema_id", "external_id", name="uq_event_identity"),
        Index("ix_event_tenant_schema_time", "tenant_id", "schema_id", "occurred_at"),
    )


class BaselineModel(Base):
    """A fitted, versioned set of contextual baselines."""

    __tablename__ = "baseline_models"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    schema_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("event_schemas.id"), index=True)
    version: Mapped[str] = mapped_column(String(32))
    #: Only one baseline per schema is active; the rest are history.
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    trained_on: Mapped[int] = mapped_column(Integer, default=0)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    peer_groups: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    #: Set when a retraining job produced this as a candidate rather than
    #: promoting it. Feedback must never silently replace a live model.
    promoted_from_feedback: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "schema_id", "version", name="uq_baseline_version"),
    )


class ScoringRun(Base):
    __tablename__ = "scoring_runs"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    schema_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("event_schemas.id"), index=True)
    baseline_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("baseline_models.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), default="completed")
    scored_count: Mapped[int] = mapped_column(Integer, default=0)
    exception_count: Mapped[int] = mapped_column(Integer, default=0)
    suppressed_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExceptionCase(Base):
    """One thing a human should look at."""

    __tablename__ = "exceptions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    schema_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("event_schemas.id"), index=True)
    scoring_run_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("scoring_runs.id"), nullable=True
    )
    entity_type: Mapped[str] = mapped_column(String(64), default="entity")
    entity_id: Mapped[str] = mapped_column(String(200), index=True)
    event_external_id: Mapped[str] = mapped_column(String(200), default="")
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    anomaly_score: Mapped[float] = mapped_column(Float, index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_scope: Mapped[str] = mapped_column(String(400), default="")
    detector_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    event_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    assignee: Mapped[str | None] = mapped_column(String(200), nullable=True)
    resolution_note: Mapped[str] = mapped_column(Text, default="")
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    narrative: Mapped[dict] = mapped_column(JSON, default=dict)

    feedback: Mapped[list["Feedback"]] = relationship(
        back_populates="exception", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_exception_tenant_status_score", "tenant_id", "status", "anomaly_score"),
        Index("ix_exception_tenant_entity", "tenant_id", "entity_id"),
    )


class Feedback(Base):
    """A human verdict on an exception.

    Stored separately from every model so that a label can never directly
    mutate production behaviour -- it feeds a retraining job that produces a
    *candidate*, which somebody then has to promote.
    """

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    exception_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("exceptions.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(32), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    author: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    exception: Mapped[ExceptionCase] = relationship(back_populates="feedback")


class SuppressionPolicyRow(Base):
    __tablename__ = "policies"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    schema_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("event_schemas.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200))
    match: Mapped[dict] = mapped_column(JSON, default=dict)
    detectors: Mapped[list] = mapped_column(JSON, default=list)
    max_score: Mapped[float] = mapped_column(Float, default=100.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_by: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Notification(Base):
    """An outbound alert. Delivery is pluggable; the record is not."""

    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    exception_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("exceptions.id"), nullable=True
    )
    channel: Mapped[str] = mapped_column(String(32), default="webhook")
    target: Mapped[str] = mapped_column(String(500), default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    actor: Mapped[str] = mapped_column(String(200), default="system")
    action: Mapped[str] = mapped_column(String(120), index=True)
    object_type: Mapped[str] = mapped_column(String(64), default="")
    object_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class AILogEntry(Base):
    __tablename__ = "ai_log"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    purpose: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(120), default="")
    prompt_version: Mapped[str] = mapped_column(String(64), default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
