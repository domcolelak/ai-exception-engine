"""Demo workspace seeding."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import DEMO_TENANT_SLUG, hash_api_key
from app.demo.dataset import DEMO_SCHEMA, PLANTED, generate_events
from app.exceptions.service import run_detection
from app.models import Event, EventSchema, Tenant, User

DEMO_API_KEY = "pk_demo_ai_exception_engine"


def ensure_demo_tenant(db: Session) -> Tenant:
    tenant = db.scalar(select(Tenant).where(Tenant.slug == DEMO_TENANT_SLUG))
    if tenant is None:
        tenant = Tenant(
            slug=DEMO_TENANT_SLUG,
            name="Demo workspace",
            api_key_hash=hash_api_key(DEMO_API_KEY),
        )
        db.add(tenant)
        db.flush()
        db.add(
            User(
                tenant_id=tenant.id,
                email="ops@demo.local",
                display_name="Demo operations lead",
                role="admin",
            )
        )
        db.flush()
    return tenant


def seed_demo(db: Session, *, event_count: int = 4000, force: bool = False) -> dict:
    """Create the demo tenant, schema, events and a first detection run."""
    tenant = ensure_demo_tenant(db)

    schema = db.scalar(
        select(EventSchema).where(
            EventSchema.tenant_id == tenant.id, EventSchema.name == DEMO_SCHEMA["name"]
        )
    )
    if schema is not None and not force:
        existing = db.scalar(
            select(func.count(Event.id)).where(
                Event.tenant_id == tenant.id, Event.schema_id == schema.id
            )
        )
        if existing:
            return {
                "tenant_id": str(tenant.id),
                "schema_id": str(schema.id),
                "created": False,
                "event_count": existing,
            }

    if schema is None:
        schema = EventSchema(
            tenant_id=tenant.id,
            name=DEMO_SCHEMA["name"],
            entity_type=DEMO_SCHEMA["entity_type"],
            entity_key=DEMO_SCHEMA["entity_key"],
            event_key=DEMO_SCHEMA["event_key"],
            timestamp_field=DEMO_SCHEMA["timestamp_field"],
            numeric_features=DEMO_SCHEMA["numeric_features"],
            categorical_features=DEMO_SCHEMA["categorical_features"],
            context_dimensions=DEMO_SCHEMA["context_dimensions"],
            rate_features=DEMO_SCHEMA["rate_features"],
            detector_weights=DEMO_SCHEMA["detector_weights"],
        )
        db.add(schema)
        db.flush()

    events = generate_events(count=event_count)
    db.add_all(
        [
            Event(
                tenant_id=tenant.id,
                schema_id=schema.id,
                external_id=event["event_id"],
                entity_id=event["entity_id"],
                occurred_at=event["occurred_at"],
                payload={
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in event.items()
                },
            )
            for event in events
        ]
    )
    db.flush()

    run = run_detection(db, tenant.id, schema.id)

    return {
        "tenant_id": str(tenant.id),
        "schema_id": str(schema.id),
        "created": True,
        "event_count": len(events),
        "scoring_run_id": str(run.id),
        "exception_count": run.exception_count,
        "planted_anomalies": [p.description for p in PLANTED],
    }
