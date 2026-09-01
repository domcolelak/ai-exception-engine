"""Append-only audit trail helper."""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session


def record_audit(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    action: str,
    object_type: str,
    object_id: str | None = None,
    actor: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Write one audit entry. Never raises into the caller's happy path."""
    from app.models import AuditLog

    entry = AuditLog(
        tenant_id=tenant_id,
        action=action,
        object_type=object_type,
        object_id=str(object_id) if object_id is not None else None,
        actor=actor,
        payload=payload or {},
    )
    db.add(entry)
    db.flush()
