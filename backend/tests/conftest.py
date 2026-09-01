from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

_TMP_DIR = Path(tempfile.mkdtemp(prefix="ai-exception-engine-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+pysqlite:///{_TMP_DIR / 'test.db'}")
os.environ.setdefault("SEED_DEMO_ON_STARTUP", "false")
os.environ.setdefault("AI_PROVIDER", "offline")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.demo.dataset import DEMO_SCHEMA, generate_events  # noqa: E402
from app.demo.seed import ensure_demo_tenant, seed_demo  # noqa: E402
from app.main import app  # noqa: E402
from app.scoring.pipeline import SchemaSpec, run_pipeline  # noqa: E402

BASE_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


@pytest.fixture()
def tenant(db):
    t = ensure_demo_tenant(db)
    db.commit()
    return t


@pytest.fixture()
def seeded(db, tenant):
    result = seed_demo(db, event_count=1500)
    db.commit()
    return result


@pytest.fixture()
def client(seeded):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def demo_events():
    """The full synthetic event log. Read-only, so built once."""
    return generate_events(count=4000)


@pytest.fixture(scope="session")
def demo_spec():
    return SchemaSpec.from_dict(DEMO_SCHEMA)


@pytest.fixture(scope="session")
def demo_pipeline(demo_events, demo_spec):
    """A full detection pass over the demo log. Expensive, so built once."""
    return run_pipeline(demo_events, demo_spec)


def event(
    entity_id: str,
    *,
    minutes: int = 0,
    category: str = "books",
    market: str = "SK",
    payment: str = "card",
    **numeric: Any,
) -> dict:
    """Compact event builder for hand-written fixtures."""
    payload = {
        "event_id": f"{entity_id}-{minutes}",
        "entity_id": entity_id,
        "occurred_at": BASE_TIME + timedelta(minutes=minutes),
        "seller_category": category,
        "market": market,
        "payment_method": payment,
        "order_value": 100.0,
        "refund_amount": 0.0,
        "refund_ratio": 0.0,
        "delivery_days": 3.0,
        "items": 2,
    }
    payload.update(numeric)
    return payload
