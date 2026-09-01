"""End-to-end API tests, including the feedback loop and tenant isolation."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from app.core.security import hash_api_key
from app.demo.seed import DEMO_API_KEY
from app.demo.dataset import DEMO_SCHEMA
from app.models import Event, EventSchema, ExceptionCase, Tenant


def demo_schema_id(client) -> str:
    """The seeded schema, by name.

    Not `schemas[0]`: the list is sorted by name and other tests create
    schemas that sort ahead of it, which silently pointed detection at an
    empty schema.
    """
    for schema in client.get("/v1/schemas").json():
        if schema["name"] == DEMO_SCHEMA["name"]:
            return schema["id"]
    raise AssertionError("the demo schema is missing")


class TestHealthAndOverview:
    def test_health(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_openapi(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_overview(self, client):
        body = client.get("/v1/overview").json()
        assert body["schema_count"] >= 1
        assert body["event_count"] > 0
        assert body["open_exceptions"] > 0
        assert body["top_exceptions"]


class TestSchemas:
    def test_demo_schema_is_listed(self, client):
        schemas = client.get("/v1/schemas").json()
        assert schemas
        schema = next(s for s in schemas if s["name"] == DEMO_SCHEMA["name"])
        assert schema["event_count"] > 0
        assert "order_value" in schema["numeric_features"]
        assert schema["context_dimensions"] == ["seller_category", "market"]

    def test_create_and_fetch(self, client):
        created = client.post(
            "/v1/schemas",
            json={
                "name": "invoices",
                "entity_type": "supplier",
                "numeric_features": ["amount"],
                "categorical_features": ["currency"],
                "context_dimensions": ["currency"],
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert client.get(f"/v1/schemas/{body['id']}").status_code == 200

    def test_a_schema_needs_features(self, client):
        response = client.post("/v1/schemas", json={"name": "empty"})
        assert response.status_code == 422

    def test_duplicate_name_is_rejected(self, client):
        payload = {"name": "dupes", "numeric_features": ["a"]}
        assert client.post("/v1/schemas", json=payload).status_code == 201
        assert client.post("/v1/schemas", json=payload).status_code == 409

    def test_unknown_schema_is_404(self, client):
        assert client.get(f"/v1/schemas/{uuid.uuid4()}").status_code == 404


class TestIngestion:
    def _schema(self, client) -> str:
        return client.post(
            "/v1/schemas",
            json={
                "name": f"ingest-{uuid.uuid4().hex[:6]}",
                "numeric_features": ["amount"],
                "categorical_features": ["currency"],
            },
        ).json()["id"]

    def test_accepts_a_batch(self, client):
        schema_id = self._schema(client)
        response = client.post(
            "/v1/events/batch",
            json={
                "schema_id": schema_id,
                "events": [
                    {
                        "external_id": "E1",
                        "payload": {
                            "entity_id": "S1",
                            "occurred_at": "2026-05-01T08:00:00+00:00",
                            "amount": 10.0,
                            "currency": "EUR",
                        },
                    }
                ],
            },
        ).json()
        assert response["accepted"] == 1
        assert response["rejected"] == 0

    def test_ingestion_is_idempotent(self, client):
        schema_id = self._schema(client)
        payload = {
            "schema_id": schema_id,
            "events": [
                {
                    "external_id": "SAME",
                    "payload": {
                        "entity_id": "S1",
                        "occurred_at": "2026-05-01T08:00:00+00:00",
                        "amount": 10.0,
                        "currency": "EUR",
                    },
                }
            ],
        }
        assert client.post("/v1/events/batch", json=payload).json()["accepted"] == 1
        second = client.post("/v1/events/batch", json=payload).json()
        assert second["accepted"] == 0 and second["duplicates"] == 1

    def test_missing_entity_key_is_reported_not_silently_dropped(self, client):
        schema_id = self._schema(client)
        response = client.post(
            "/v1/events/batch",
            json={
                "schema_id": schema_id,
                "events": [
                    {"payload": {"occurred_at": "2026-05-01T08:00:00+00:00", "amount": 1.0}}
                ],
            },
        ).json()
        assert response["accepted"] == 0
        assert response["rejected"] == 1
        assert "entity" in response["errors"][0]["problems"][0]

    def test_a_missing_declared_feature_is_a_data_quality_error(self, client):
        schema_id = self._schema(client)
        response = client.post(
            "/v1/events/batch",
            json={
                "schema_id": schema_id,
                "events": [
                    {
                        "payload": {
                            "entity_id": "S1",
                            "occurred_at": "2026-05-01T08:00:00+00:00",
                            "currency": "EUR",
                        }
                    }
                ],
            },
        ).json()
        assert response["rejected"] == 1
        assert "amount" in response["errors"][0]["problems"][0]

    def test_empty_batch_is_rejected(self, client):
        assert (
            client.post(
                "/v1/events/batch", json={"schema_id": self._schema(client), "events": []}
            ).status_code
            == 422
        )


class TestDetection:
    def test_run_produces_exceptions_and_a_baseline(self, client):
        schema_id = demo_schema_id(client)
        run = client.post("/v1/detection-runs", json={"schema_id": schema_id})
        assert run.status_code == 201
        body = run.json()
        assert body["status"] == "completed"
        assert body["scored_count"] > 0
        assert body["baseline_id"]
        assert client.get(f"/v1/detection-runs/{body['id']}").status_code == 200

    def test_baseline_exposes_its_scopes(self, client):
        schema_id = demo_schema_id(client)
        baselines = client.get("/v1/baselines", params={"schema_id": schema_id}).json()
        assert baselines
        detail = client.get(f"/v1/baselines/{baselines[0]['id']}").json()
        assert detail["payload"]["scopes"], "the baseline must show what it learned"
        assert detail["trained_on"] > 0

    def test_a_second_run_does_not_duplicate_open_exceptions(self, client):
        schema_id = demo_schema_id(client)
        before = len(client.get("/v1/exceptions").json())
        client.post("/v1/detection-runs", json={"schema_id": schema_id})
        after = len(client.get("/v1/exceptions").json())
        assert after == before, "re-running detection must not re-raise what is already open"


class TestExceptionQueue:
    def test_queue_is_ranked_by_score(self, client):
        exceptions = client.get("/v1/exceptions").json()
        assert exceptions
        scores = [e["anomaly_score"] for e in exceptions]
        assert scores == sorted(scores, reverse=True)

    def test_every_exception_carries_reasons(self, client):
        for case in client.get("/v1/exceptions").json():
            assert case["reasons"], "an exception without a reason is not actionable"
            assert case["detector_versions"]
            assert 0 < case["confidence"] <= 0.95

    def test_detail_recomputes_its_own_score(self, client):
        case_id = client.get("/v1/exceptions").json()[0]["id"]
        detail = client.get(f"/v1/exceptions/{case_id}").json()
        assert abs(detail["recomputed_score"] - detail["anomaly_score"]) < 0.01, (
            "a stored exception must be able to prove its score from stored evidence"
        )

    def test_filters(self, client):
        critical = client.get("/v1/exceptions", params={"severity": "critical"}).json()
        assert all(c["severity"] == "critical" for c in critical)
        opened = client.get("/v1/exceptions", params={"status": "open"}).json()
        assert all(c["status"] == "open" for c in opened)

    def test_assign_moves_it_out_of_the_open_pile(self, client):
        case_id = client.get("/v1/exceptions").json()[0]["id"]
        body = client.post(
            f"/v1/exceptions/{case_id}/assign", json={"assignee": "ana"}
        ).json()
        assert body["assignee"] == "ana"
        assert body["status"] == "acknowledged"

    def test_resolve_records_the_note(self, client):
        case_id = client.get("/v1/exceptions").json()[0]["id"]
        body = client.post(
            f"/v1/exceptions/{case_id}/resolve",
            json={"status": "resolved", "resolution_note": "supplier confirmed"},
        ).json()
        assert body["status"] == "resolved"
        assert body["resolution_note"] == "supplier confirmed"

    def test_invalid_status_is_rejected(self, client):
        case_id = client.get("/v1/exceptions").json()[0]["id"]
        assert (
            client.post(
                f"/v1/exceptions/{case_id}/resolve", json={"status": "whatever"}
            ).status_code
            == 422
        )

    def test_explain_uses_the_offline_provider(self, client):
        case_id = client.get("/v1/exceptions").json()[0]["id"]
        body = client.post(f"/v1/exceptions/{case_id}/explain").json()
        assert body["narrative"]["headline"]
        assert body["narrative"]["explanation"]

    def test_unknown_exception_is_404(self, client):
        assert client.get(f"/v1/exceptions/{uuid.uuid4()}").status_code == 404


class TestSimilarity:
    def test_similar_returns_scored_components(self, client):
        case_id = client.get("/v1/exceptions").json()[0]["id"]
        body = client.get(f"/v1/exceptions/{case_id}/similar").json()
        assert "structured evidence" in body["note"]
        for match in body["similar"]:
            assert match["exception_id"] != case_id
            assert set(match["components"]) >= {"reason_overlap", "same_entity"}
            assert 0.0 <= match["score"] <= 1.0


class TestPolicies:
    def test_create_and_deactivate(self, client):
        schema_id = demo_schema_id(client)
        created = client.post(
            "/v1/policies",
            json={
                "name": "known good books",
                "schema_id": schema_id,
                "match": {"seller_category": "books"},
                "max_score": 75.0,
                "reason": "reviewed",
            },
        )
        assert created.status_code == 201
        policy = created.json()
        assert policy["active"] is True
        assert client.post(f"/v1/policies/{policy['id']}/deactivate").json()["active"] is False

    def test_a_policy_reduces_the_queue_on_the_next_run(self, client):
        schema_id = demo_schema_id(client)
        for case in client.get("/v1/exceptions").json():
            client.post(
                f"/v1/exceptions/{case['id']}/resolve",
                json={"status": "resolved", "resolution_note": "clearing for the test"},
            )
        client.post(
            "/v1/policies",
            json={
                "name": "suppress everything books",
                "schema_id": schema_id,
                "match": {"seller_category": "books"},
                "reason": "test",
            },
        )
        client.post("/v1/detection-runs", json={"schema_id": schema_id})
        remaining = client.get("/v1/exceptions", params={"status": "open"}).json()
        assert not any(
            c["event_payload"].get("seller_category") == "books"
            for c in (client.get(f"/v1/exceptions/{r['id']}").json() for r in remaining)
        )


class TestFeedbackLoop:
    def _label_many(self, client, label: str, count: int = 20):
        cases = client.get("/v1/exceptions").json()
        labelled = 0
        for case in cases:
            if labelled >= count:
                break
            response = client.post(
                f"/v1/exceptions/{case['id']}/feedback",
                json={"label": label, "author": "tester"},
            )
            assert response.status_code == 201
            labelled += 1
        return labelled

    def test_label_vocabulary_is_enforced(self, client):
        case_id = client.get("/v1/exceptions").json()[0]["id"]
        assert (
            client.post(
                f"/v1/exceptions/{case_id}/feedback", json={"label": "made_up"}
            ).status_code
            == 422
        )

    def test_feedback_alone_changes_nothing(self, client):
        schema_id = demo_schema_id(client)
        before = client.get(f"/v1/schemas/{schema_id}").json()
        self._label_many(client, "false_positive", count=5)
        after = client.get(f"/v1/schemas/{schema_id}").json()
        assert after["detector_weights"] == before["detector_weights"]
        assert after["min_score"] == before["min_score"], (
            "one analyst clicking a button must never reconfigure detection"
        )

    def test_too_little_feedback_proposes_nothing(self, client):
        schema_id = demo_schema_id(client)
        self._label_many(client, "true_exception", count=2)
        proposal = client.get(
            "/v1/retraining/proposal", params={"schema_id": schema_id}
        ).json()
        assert proposal["suggested_weights"] == {}
        assert any("are needed" in note for note in proposal["notes"])

    def test_proposal_is_never_pre_applied(self, client):
        schema_id = demo_schema_id(client)
        proposal = client.get(
            "/v1/retraining/proposal", params={"schema_id": schema_id}
        ).json()
        assert proposal["applied"] is False

    def test_promotion_is_explicit(self, client):
        schema_id = demo_schema_id(client)
        promoted = client.post(
            "/v1/retraining/promote",
            params={"schema_id": schema_id},
            json={"apply_weights": True, "apply_min_score": False},
        )
        assert promoted.status_code == 200

    def test_overview_reports_detector_precision(self, client):
        self._label_many(client, "true_exception", count=3)
        body = client.get("/v1/overview").json()
        assert body["labelled_exceptions"] > 0
        assert body["detector_precision"]
        for entry in body["detector_precision"]:
            assert 0.0 <= entry["precision"] <= 1.0


class TestTenantIsolation:
    def test_other_tenant_sees_nothing(self, client, db):
        db.add(Tenant(slug="other", name="Other", api_key_hash=hash_api_key("pk_other_key")))
        db.commit()
        headers = {"X-API-Key": "pk_other_key"}
        assert client.get("/v1/schemas", headers=headers).json() == []
        assert client.get("/v1/exceptions", headers=headers).json() == []
        assert client.get("/v1/overview", headers=headers).json()["event_count"] == 0

    def test_cross_tenant_access_is_404(self, client, db):
        db.add(Tenant(slug="snoop", name="Snoop", api_key_hash=hash_api_key("pk_snoop_key")))
        db.commit()
        headers = {"X-API-Key": "pk_snoop_key"}
        schema = db.scalar(select(EventSchema))
        case = db.scalar(select(ExceptionCase))
        assert client.get(f"/v1/schemas/{schema.id}", headers=headers).status_code == 404
        assert client.get(f"/v1/exceptions/{case.id}", headers=headers).status_code == 404

    def test_cross_tenant_detection_is_refused(self, client, db):
        db.add(Tenant(slug="runner", name="Runner", api_key_hash=hash_api_key("pk_runner_key")))
        db.commit()
        schema = db.scalar(select(EventSchema))
        assert (
            client.post(
                "/v1/detection-runs",
                json={"schema_id": str(schema.id)},
                headers={"X-API-Key": "pk_runner_key"},
            ).status_code
            == 404
        )

    def test_cross_tenant_ingestion_is_refused(self, client, db):
        db.add(Tenant(slug="feeder", name="Feeder", api_key_hash=hash_api_key("pk_feeder_key")))
        db.commit()
        schema = db.scalar(select(EventSchema))
        assert (
            client.post(
                "/v1/events/batch",
                json={
                    "schema_id": str(schema.id),
                    "events": [{"payload": {"entity_id": "X", "occurred_at": "2026-05-01T00:00:00Z"}}],
                },
                headers={"X-API-Key": "pk_feeder_key"},
            ).status_code
            == 404
        )

    def test_invalid_key_is_401(self, client):
        assert client.get("/v1/schemas", headers={"X-API-Key": "pk_nope"}).status_code == 401

    def test_demo_key_works(self, client):
        assert client.get("/v1/schemas", headers={"X-API-Key": DEMO_API_KEY}).status_code == 200

    def test_all_events_belong_to_the_demo_tenant(self, db):
        demo = db.scalar(select(Tenant).where(Tenant.slug == "demo"))
        events = db.scalars(select(Event)).all()
        assert events
        assert all(e.tenant_id == demo.id for e in events)
