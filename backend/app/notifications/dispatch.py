"""Notification dispatch.

Delivery is intentionally an abstraction with a recording no-op default. What
matters for correctness is that every alert that *should* have gone out leaves
a row: an operations tool that silently fails to notify is worse than one that
does not notify at all, because nobody discovers the gap until it matters.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import ExceptionCase, Notification

#: Only these reach a human by default. Everything else sits in the queue.
NOTIFY_SEVERITIES = frozenset({"critical", "high"})


class Channel(ABC):
    name = "channel"

    @abstractmethod
    def deliver(self, target: str, payload: dict) -> None:
        """Send. Raise on failure so the row records the error."""


class RecordingChannel(Channel):
    """Default channel: records the intent without any network call."""

    name = "recording"

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def deliver(self, target: str, payload: dict) -> None:
        self.sent.append((target, payload))


_channel: Channel = RecordingChannel()


def set_channel(channel: Channel) -> None:
    global _channel
    _channel = channel


def get_channel() -> Channel:
    return _channel


def notify(
    db: Session,
    tenant_id: uuid.UUID,
    exception: ExceptionCase,
    *,
    target: str = "",
    force: bool = False,
) -> Notification | None:
    """Queue and attempt a notification for one exception."""
    if not force and exception.severity not in NOTIFY_SEVERITIES:
        return None

    payload = {
        "exception_id": str(exception.id),
        "title": exception.title,
        "entity_id": exception.entity_id,
        "severity": exception.severity,
        "score": exception.anomaly_score,
        "detected_at": exception.detected_at.isoformat(),
        "reasons": [r.get("explanation") for r in (exception.reasons or [])][:3],
    }
    record = Notification(
        tenant_id=tenant_id,
        exception_id=exception.id,
        channel=get_channel().name,
        target=target,
        payload=payload,
    )
    db.add(record)
    db.flush()

    try:
        get_channel().deliver(target, payload)
        record.status = "delivered"
        record.delivered_at = datetime.now(timezone.utc)
    except Exception as exc:  # pragma: no cover - depends on the channel
        record.status = "failed"
        record.error = str(exc)[:500]
    db.flush()
    return record
