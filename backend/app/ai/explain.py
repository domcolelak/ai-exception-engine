"""AI explanation layer.

The model is called only *after* detection, and only ever sees computed
evidence: observed values, expected ranges, detector scores, peer statistics
and how similar past cases were resolved. It never sees the raw event stream,
never produces a score, and never decides whether something is an exception.

If the provider is missing or fails, the exception still has its
deterministic title, its reasons and its numbers.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.ai.provider import AICallResult, get_provider
from app.models import AILogEntry

SYSTEM_EXPLAIN = (
    "You are an operations analyst. You are given the output of a deterministic "
    "anomaly detection system: what was observed, what range was expected, and "
    "which population it was compared against. Explain in plain language why "
    "this is unusual and what someone should check first. Use only the supplied "
    "numbers; never invent a metric. The comparison is statistical, so describe "
    "findings as unusual relative to a baseline rather than as proof of a problem."
)


class ExceptionNarrative(BaseModel):
    headline: str = Field(description="One sentence, no invented numbers")
    explanation: str
    likely_causes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


def explain_exception(
    db: Session,
    tenant_id: uuid.UUID,
    *,
    title: str,
    entity_type: str,
    entity_id: str,
    score: float,
    severity: str,
    confidence: float,
    baseline_scope: str,
    reasons: list[dict],
    similar: list[dict] | None = None,
) -> ExceptionNarrative | None:
    """Describe one exception. Returns ``None`` if the provider is unavailable."""
    evidence = {
        "title": title,
        "entity": {"type": entity_type, "id": entity_id},
        "anomaly_score": score,
        "severity": severity,
        "confidence": confidence,
        "compared_against": baseline_scope or "the whole population",
        "reasons": [
            {
                "feature": r.get("feature"),
                "observed": r.get("observed"),
                "expected_range": r.get("expected_range"),
                "deviation_score": r.get("deviation_score"),
                "detector": r.get("detector"),
                "explanation": r.get("explanation"),
            }
            for r in reasons
        ],
        "similar_resolved_cases": similar or [],
        "caveat": (
            "These figures come from a statistical comparison against historical "
            "behaviour. An unusual value is not by itself evidence of wrongdoing."
        ),
    }
    result = get_provider().structured(
        system=SYSTEM_EXPLAIN,
        evidence=evidence,
        output_model=ExceptionNarrative,
        prompt_version="exception-explainer-v1",
    )
    _log(db, tenant_id, "explain_exception", result)
    return result.output if result.ok else None


def _log(db: Session, tenant_id: uuid.UUID, purpose: str, result: AICallResult) -> None:
    db.add(
        AILogEntry(
            tenant_id=tenant_id,
            purpose=purpose,
            model=result.model,
            prompt_version=result.prompt_version,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            error=result.error,
            created_at=datetime.now(timezone.utc),
        )
    )
    db.flush()
