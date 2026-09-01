"""Pydantic request/response models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.feedback.loop import LABELS


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- schemas ---------------------------------------------------------------


class SchemaCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    entity_type: str = "entity"
    entity_key: str = "entity_id"
    event_key: str = "event_id"
    timestamp_field: str = "occurred_at"
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    context_dimensions: list[str] = Field(default_factory=list)
    rate_features: list[str] = Field(default_factory=list)
    detector_weights: dict[str, float] = Field(default_factory=dict)
    min_score: float = Field(default=60.0, ge=0.0, le=100.0)


class SchemaOut(ORMModel):
    id: uuid.UUID
    name: str
    entity_type: str
    entity_key: str
    event_key: str
    timestamp_field: str
    numeric_features: list[str]
    categorical_features: list[str]
    context_dimensions: list[str]
    rate_features: list[str]
    detector_weights: dict[str, float]
    min_score: float
    created_at: datetime


class SchemaListItem(SchemaOut):
    event_count: int = 0
    open_exceptions: int = 0


# --- ingestion -------------------------------------------------------------


class EventIn(BaseModel):
    external_id: str | None = None
    occurred_at: datetime | None = None
    payload: dict[str, Any]


class EventBatchRequest(BaseModel):
    schema_id: uuid.UUID
    events: list[EventIn] = Field(min_length=1, max_length=20_000)


class EventBatchResponse(BaseModel):
    schema_id: uuid.UUID
    accepted: int
    duplicates: int
    rejected: int
    errors: list[dict[str, Any]] = Field(default_factory=list)


# --- detection -------------------------------------------------------------


class DetectionRequest(BaseModel):
    schema_id: uuid.UUID
    train_ratio: float = Field(default=0.6, gt=0.0, lt=1.0)


class ScoringRunOut(ORMModel):
    id: uuid.UUID
    schema_id: uuid.UUID
    baseline_id: uuid.UUID | None
    status: str
    scored_count: int
    exception_count: int
    suppressed_count: int
    duplicate_count: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class BaselineOut(ORMModel):
    id: uuid.UUID
    schema_id: uuid.UUID
    version: str
    active: bool
    trained_on: int
    created_at: datetime


class BaselineDetail(BaselineOut):
    payload: dict[str, Any] = Field(default_factory=dict)
    peer_groups: dict[str, Any] = Field(default_factory=dict)


# --- exceptions ------------------------------------------------------------


class ExceptionOut(ORMModel):
    id: uuid.UUID
    schema_id: uuid.UUID
    entity_type: str
    entity_id: str
    event_external_id: str
    fingerprint: str
    title: str
    detected_at: datetime
    anomaly_score: float
    severity: str
    confidence: float
    baseline_scope: str
    detector_versions: dict[str, Any]
    reasons: list[Any]
    status: str
    assignee: str | None
    resolution_note: str
    narrative: dict[str, Any] = Field(default_factory=dict)


class ExceptionDetail(ExceptionOut):
    evidence: dict[str, Any] = Field(default_factory=dict)
    event_payload: dict[str, Any] = Field(default_factory=dict)
    #: Recomputed from the stored detector outputs, to prove the score.
    recomputed_score: float = 0.0
    feedback: list[dict[str, Any]] = Field(default_factory=list)


class AssignRequest(BaseModel):
    assignee: str = Field(min_length=1, max_length=200)


class ResolveRequest(BaseModel):
    status: str = Field(pattern="^(open|acknowledged|in_progress|resolved|dismissed)$")
    resolution_note: str = ""


class FeedbackRequest(BaseModel):
    #: Constrained to the fixed vocabulary; a free-text label would make the
    #: precision statistics that drive retraining meaningless.
    label: Literal[
        "true_exception",
        "false_positive",
        "expected_behavior",
        "duplicate",
        "data_quality_issue",
    ]
    note: str = ""
    author: str = ""


class FeedbackOut(ORMModel):
    id: uuid.UUID
    exception_id: uuid.UUID
    label: str
    note: str
    author: str
    created_at: datetime


class SimilarResponse(BaseModel):
    exception_id: uuid.UUID
    similar: list[dict[str, Any]] = Field(default_factory=list)
    note: str


# --- policies --------------------------------------------------------------


class PolicyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    schema_id: uuid.UUID | None = None
    match: dict[str, Any] = Field(default_factory=dict)
    detectors: list[str] = Field(default_factory=list)
    max_score: float = Field(default=100.0, ge=0.0, le=100.0)
    reason: str = ""
    created_by: str = ""


class PolicyOut(ORMModel):
    id: uuid.UUID
    schema_id: uuid.UUID | None
    name: str
    match: dict[str, Any]
    detectors: list[str]
    max_score: float
    reason: str
    active: bool
    created_by: str
    created_at: datetime


# --- feedback loop ---------------------------------------------------------


class RetrainingProposalOut(BaseModel):
    schema_id: uuid.UUID
    labelled_count: int
    detector_quality: list[dict[str, Any]] = Field(default_factory=list)
    suggested_weights: dict[str, float] = Field(default_factory=dict)
    suggested_min_score: float | None = None
    proposed_policies: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    applied: bool = False


class PromoteRequest(BaseModel):
    """Explicitly apply a retraining proposal. Never automatic."""

    apply_weights: bool = True
    apply_min_score: bool = False
    create_policies: bool = False


class OverviewResponse(BaseModel):
    schema_count: int
    event_count: int
    open_exceptions: int
    critical_exceptions: int
    resolved_exceptions: int
    labelled_exceptions: int
    detector_precision: list[dict[str, Any]] = Field(default_factory=list)
    top_exceptions: list[ExceptionOut] = Field(default_factory=list)


VALID_LABELS = set(LABELS)
