"""Tests for detectors, the ensemble, exception generation and the pipeline."""
from __future__ import annotations

from collections import Counter

import pytest

from app.baselines.model import build_baselines
from app.demo.dataset import (
    BURST_SELLER,
    CHANGE_ANOMALY_SELLER,
    NEW_SELLER,
    PEER_ANOMALY_SELLER,
)
from app.detectors.base import DetectionContext, Detector, DetectorResult, normalise_z
from app.detectors.builtin import (
    CategoricalRarityDetector,
    ChangePointDetector,
    RobustDeviationDetector,
    RollingQuantileDetector,
)
from app.detectors.peer_group import PeerGroupDetector, build_peer_groups
from app.exceptions.generation import (
    GenerationConfig,
    OpenException,
    SuppressionPolicy,
    evaluate,
    fingerprint_for,
)
from app.scoring.ensemble import combine, confidence_for, recompute, score_event, severity_for
from tests.conftest import BASE_TIME, event


def context(target, history=(), scope_history=(), baseline=None, dims=("seller_category",)):
    return DetectionContext(
        event=target,
        baseline=baseline,
        numeric_features=("order_value", "refund_amount"),
        categorical_features=("seller_category", "market", "payment_method"),
        entity_history=list(history),
        scope_history=list(scope_history),
        scope_dimensions=dims,
    )


def baseline_from(events, dims=("seller_category",)):
    return build_baselines(
        events,
        numeric_features=["order_value", "refund_amount"],
        categorical_features=["seller_category", "market", "payment_method"],
        dimensions=dims,
    )


class TestNormalisation:
    def test_maps_into_the_unit_interval(self):
        for z in (0, 1, 3, 10, 1000, -50):
            assert 0.0 <= normalise_z(z) <= 1.0

    def test_is_symmetric_and_monotonic(self):
        assert normalise_z(5) == normalise_z(-5)
        assert normalise_z(1) < normalise_z(3) < normalise_z(9)

    def test_the_knee_maps_to_a_half(self):
        assert normalise_z(3.0, knee=3.0) == pytest.approx(0.5)


class TestRobustDeviation:
    def test_flags_a_value_far_from_its_scope_median(self):
        history = [event(f"E{i}", minutes=i, order_value=20 + i % 4) for i in range(60)]
        result = RobustDeviationDetector().score(
            context(event("X", order_value=5000), baseline=baseline_from(history).global_baseline)
        )
        assert result.applicable and result.score > 0.8
        assert result.reasons[0].feature == "order_value"
        assert result.reasons[0].expected_range is not None

    def test_ignores_a_typical_value(self):
        history = [event(f"E{i}", minutes=i, order_value=20 + i % 4) for i in range(60)]
        result = RobustDeviationDetector().score(
            context(event("X", order_value=21), baseline=baseline_from(history).global_baseline)
        )
        assert result.score < 0.4
        assert result.reasons == []

    def test_abstains_without_a_baseline(self):
        assert RobustDeviationDetector().score(context(event("X"))).applicable is False

    def test_reason_names_the_population_it_compared_against(self):
        history = [
            event(f"E{i}", minutes=i, category="books", order_value=20 + i % 4)
            for i in range(60)
        ]
        baselines = baseline_from(history)
        target = event("X", category="books", order_value=5000)
        result = RobustDeviationDetector().score(
            context(target, baseline=baselines.lookup(target))
        )
        assert "seller_category=books" in result.reasons[0].explanation


class TestCategoricalRarity:
    def test_flags_an_unseen_value(self):
        history = [event(f"E{i}", minutes=i, payment="card") for i in range(100)]
        result = CategoricalRarityDetector().score(
            context(event("X", payment="crypto"), baseline=baseline_from(history).global_baseline)
        )
        assert result.score > 0.9
        assert "crypto" in result.reasons[0].explanation

    def test_ignores_a_common_value(self):
        history = [event(f"E{i}", minutes=i, payment="card") for i in range(100)]
        result = CategoricalRarityDetector().score(
            context(event("X", payment="card"), baseline=baseline_from(history).global_baseline)
        )
        assert result.reasons == []


class TestRollingQuantile:
    def test_flags_a_jump_against_the_entity_own_history(self):
        history = [event("S", minutes=i, order_value=100 + i % 10) for i in range(40)]
        result = RollingQuantileDetector().score(
            context(event("S", minutes=99, order_value=5000), history=history)
        )
        assert result.applicable and result.score > 0.5

    def test_abstains_on_a_short_history(self):
        history = [event("S", minutes=i, order_value=100) for i in range(3)]
        assert RollingQuantileDetector().score(
            context(event("S", minutes=9, order_value=5000), history=history)
        ).applicable is False

    def test_zero_inflated_feature_does_not_fire_on_an_ordinary_occurrence(self):
        """The bug this guards: p90 of a mostly-zero column is ~0."""
        history = [
            event("S", minutes=i, refund_amount=0.0 if i % 4 else 50.0) for i in range(60)
        ]
        result = RollingQuantileDetector().score(
            context(event("S", minutes=99, refund_amount=52.0), history=history)
        )
        assert result.score < 0.35, (
            "a typical refund must not be flagged just because most orders have none"
        )

    def test_a_genuinely_large_value_still_fires(self):
        history = [
            event("S", minutes=i, refund_amount=0.0 if i % 4 else 50.0) for i in range(60)
        ]
        result = RollingQuantileDetector().score(
            context(event("S", minutes=99, refund_amount=9000.0), history=history)
        )
        assert result.score > 0.4

    def test_history_is_scoped_so_categories_are_not_mixed(self):
        """A luxury order must not be compared against the same seller's books."""
        history = [event("S", minutes=i, category="books", order_value=20) for i in range(40)]
        history += [
            event("S", minutes=100 + i, category="luxury", order_value=4000 + i)
            for i in range(20)
        ]
        result = RollingQuantileDetector().score(
            context(event("S", minutes=999, category="luxury", order_value=4100), history=history)
        )
        assert result.score < 0.35


class TestChangePoint:
    def test_detects_a_sustained_shift(self):
        history = [event("S", minutes=i, order_value=100) for i in range(30)]
        history += [event("S", minutes=100 + i, order_value=900) for i in range(30)]
        result = ChangePointDetector().score(
            context(event("S", minutes=999, order_value=900), history=history)
        )
        assert result.applicable and result.score > 0.5
        assert "sustained shift" in result.reasons[0].explanation

    def test_stays_quiet_for_a_single_spike(self):
        """Spikes are the deviation detectors' job, not this one's."""
        history = [event("S", minutes=i, order_value=100 + i % 5) for i in range(59)]
        history.append(event("S", minutes=500, order_value=9000))
        result = ChangePointDetector().score(
            context(event("S", minutes=999, order_value=102), history=history)
        )
        assert result.score < 0.5

    def test_abstains_on_a_short_history(self):
        assert ChangePointDetector().score(
            context(event("S", order_value=100), history=[event("S", minutes=1)])
        ).applicable is False


class TestPeerGroup:
    @pytest.fixture()
    def events(self):
        """Twelve well-behaved sellers and one that refunds constantly."""
        out = []
        for seller in range(12):
            for i in range(40):
                out.append(
                    event(
                        f"S{seller}",
                        minutes=seller * 100 + i,
                        refund_amount=60.0 if i % 10 == 0 else 0.0,
                    )
                )
        for i in range(40):
            out.append(event("BAD", minutes=5000 + i, refund_amount=60.0 if i % 2 else 0.0))
        return out

    @pytest.fixture()
    def detector(self, events):
        groups = build_peer_groups(
            events,
            entity_key="entity_id",
            numeric_features=["order_value", "refund_amount"],
            rate_features=["refund_amount"],
            dimensions=["seller_category"],
        )
        return PeerGroupDetector(groups, entity_key="entity_id", dimensions=["seller_category"])

    def test_flags_an_entity_unlike_its_peers(self, detector):
        result = detector.score(context(event("BAD", refund_amount=60.0)))
        assert result.applicable and result.score > 0.45
        explanation = result.reasons[0].explanation
        assert "BAD" in explanation
        assert "comparable entities" in explanation, (
            "the reason must name the population the entity was measured against"
        )

    def test_leaves_a_conforming_entity_alone(self, detector):
        result = detector.score(context(event("S3", refund_amount=0.0)))
        assert result.reasons == []

    def test_abstains_for_an_entity_with_no_history(self, detector):
        result = detector.score(context(event("BRAND-NEW")))
        assert result.applicable is False, "being new is not the same as being abnormal"

    def test_only_flags_the_worse_side(self, detector):
        """An unusually *low* refund rate is not an operations exception."""
        result = detector.score(context(event("S3", refund_amount=0.0)))
        assert all(r.deviation_score >= 0 for r in result.reasons)


class TestEnsemble:
    def test_a_lone_confident_detector_is_not_diluted(self):
        """The failure this guards: a mean buried a certain detector under five quiet ones."""
        scores = [("a", 0.62), ("b", 0.05), ("c", 0.0), ("d", 0.0), ("e", 0.02), ("f", 0.0)]
        assert combine(scores, {}) > 0.6

    def test_agreement_raises_the_score(self):
        alone = combine([("a", 0.6), ("b", 0.0), ("c", 0.0)], {})
        together = combine([("a", 0.6), ("b", 0.6), ("c", 0.6)], {})
        assert together > alone

    def test_weak_signals_stay_weak(self):
        assert combine([("a", 0.2), ("b", 0.2), ("c", 0.2)], {}) < 0.35

    def test_a_single_detector_scores_itself(self):
        assert combine([("a", 0.9)], {}) == pytest.approx(0.9)

    def test_empty_is_zero(self):
        assert combine([], {}) == 0.0

    def test_abstaining_detectors_do_not_drag_the_score_down(self):
        class Quiet(Detector):
            name = "quiet"

            def score(self, ctx):
                return self._abstain("no data")

        class Loud(Detector):
            name = "loud"

            def score(self, ctx):
                return DetectorResult(detector="loud", version="v1", raw_score=0.9, score=0.9)

        result = score_event(context(event("X")), [Loud(), Quiet(), Quiet(), Quiet()])
        assert result.score == pytest.approx(90.0)

    def test_a_broken_detector_cannot_break_scoring(self):
        class Broken(Detector):
            name = "broken"

            def score(self, ctx):
                raise RuntimeError("boom")

        result = score_event(context(event("X")), [Broken()])
        assert result.score == 0.0
        assert result.detector_results[0].applicable is False
        assert "boom" in result.detector_results[0].evidence["error"]

    def test_score_is_reproducible_from_stored_output(self):
        class Fixed(Detector):
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def score(self, ctx):
                return DetectorResult(
                    detector=self.name, version="v1", raw_score=self.value, score=self.value
                )

        result = score_event(context(event("X")), [Fixed("a", 0.8), Fixed("b", 0.3)])
        assert recompute(result.as_dict()) == pytest.approx(result.score)

    def test_severity_bands(self):
        assert severity_for(90) == "critical"
        assert severity_for(75) == "high"
        assert severity_for(55) == "medium"
        assert severity_for(10) == "low"

    def test_confidence_rises_with_coverage_and_agreement(self):
        one = DetectorResult(detector="a", version="v1", raw_score=0.9, score=0.9)
        quiet = DetectorResult(detector="b", version="v1", raw_score=0.0, score=0.0)
        assert confidence_for([one, one, one], 3) > confidence_for([one, quiet], 6)

    def test_confidence_is_capped_below_certainty(self):
        loud = DetectorResult(detector="a", version="v1", raw_score=1.0, score=1.0)
        assert confidence_for([loud] * 10, 10) <= 0.95


class TestExceptionGeneration:
    def _result(self, score=80.0, confidence=0.8, strongest="robust_deviation", feature="x"):
        from app.detectors.base import Reason
        from app.scoring.ensemble import EnsembleResult

        return EnsembleResult(
            score=score,
            severity=severity_for(score),
            confidence=confidence,
            weighted_mean=0.5,
            strongest=strongest,
            reasons=[
                Reason(
                    feature=feature,
                    observed=1,
                    expected_range=None,
                    deviation_score=0.9,
                    explanation="because",
                    detector=strongest,
                )
            ],
        )

    def _evaluate(self, **kwargs):
        defaults = dict(
            event=event("S"),
            result=self._result(),
            entity_type="seller",
            entity_id="S",
            event_id="E1",
            detected_at=BASE_TIME,
        )
        defaults.update(kwargs)
        return evaluate(**defaults)

    def test_a_strong_confident_finding_becomes_an_exception(self):
        assert self._evaluate().accepted is True

    def test_a_low_score_is_rejected(self):
        decision = self._evaluate(result=self._result(score=30.0))
        assert decision.accepted is False and "below" in decision.reason

    def test_low_confidence_is_rejected(self):
        decision = self._evaluate(result=self._result(confidence=0.1))
        assert decision.accepted is False and "confidence" in decision.reason

    def test_a_matching_policy_suppresses(self):
        policy = SuppressionPolicy(
            id="p1", name="known good", match={"seller_category": "books"}, reason="expected"
        )
        decision = self._evaluate(policies=[policy])
        assert decision.suppressed_by == "p1"

    def test_a_policy_cannot_hide_a_severe_case(self):
        policy = SuppressionPolicy(
            id="p1",
            name="known good",
            match={"seller_category": "books"},
            max_score=70.0,
            reason="expected",
        )
        decision = self._evaluate(result=self._result(score=95.0), policies=[policy])
        assert decision.accepted is True, (
            "a suppression policy must not blind the system to a crisis of the same shape"
        )

    def test_the_same_problem_is_deduplicated(self):
        first = self._evaluate()
        existing = OpenException(
            id="X1",
            entity_id="S",
            fingerprint=first.created.fingerprint,
            detected_at=BASE_TIME,
            status="open",
        )
        second = self._evaluate(open_exceptions=[existing])
        assert second.duplicate_of == "X1"

    def test_a_different_problem_on_the_same_entity_still_surfaces(self):
        first = self._evaluate()
        existing = OpenException(
            id="X1",
            entity_id="S",
            fingerprint=first.created.fingerprint,
            detected_at=BASE_TIME,
            status="open",
        )
        other = self._evaluate(result=self._result(feature="something_else"), open_exceptions=[existing])
        assert other.accepted is True

    def test_an_entity_level_finding_stays_deduplicated_regardless_of_age(self):
        from datetime import timedelta

        first = self._evaluate(result=self._result(strongest="peer_group"))
        stale = OpenException(
            id="X1",
            entity_id="S",
            fingerprint=first.created.fingerprint,
            detected_at=BASE_TIME - timedelta(days=90),
            status="open",
        )
        again = self._evaluate(
            result=self._result(strongest="peer_group"), open_exceptions=[stale]
        )
        assert again.duplicate_of == "X1", (
            "an unchanged entity-level fact is not new information a month later"
        )

    def test_one_entity_cannot_flood_the_queue(self):
        existing = [
            OpenException(
                id=f"X{i}",
                entity_id="S",
                fingerprint=f"other-{i}",
                detected_at=BASE_TIME,
                status="open",
            )
            for i in range(3)
        ]
        decision = self._evaluate(open_exceptions=existing)
        assert decision.accepted is False and "capped" in decision.reason

    def test_a_resolved_exception_frees_the_slot(self):
        existing = [
            OpenException(
                id=f"X{i}",
                entity_id="S",
                fingerprint=f"other-{i}",
                detected_at=BASE_TIME,
                status="resolved",
            )
            for i in range(5)
        ]
        assert self._evaluate(open_exceptions=existing).accepted is True

    def test_fingerprint_is_stable_for_the_same_problem(self):
        reasons = [{"feature": "refund_amount", "detector": "peer_group"}]
        assert fingerprint_for("S", reasons, "peer_group") == fingerprint_for(
            "S", reasons, "peer_group"
        )

    def test_fingerprint_ignores_unrelated_detectors(self):
        """The bug this guards: an unrelated detector chiming in re-raised a closed matter."""
        core = [{"feature": "refund_amount", "detector": "peer_group"}]
        noisy = core + [{"feature": "delivery_days", "detector": "isolation_forest"}]
        assert fingerprint_for("S", core, "peer_group") == fingerprint_for(
            "S", noisy, "peer_group"
        )

    def test_fingerprint_differs_between_entities(self):
        reasons = [{"feature": "refund_amount", "detector": "peer_group"}]
        assert fingerprint_for("A", reasons, "peer_group") != fingerprint_for(
            "B", reasons, "peer_group"
        )


class TestPipelineOnDemoData:
    """The planted anomalies must be found, and the planted non-anomalies must not."""

    def test_the_queue_stays_readable(self, demo_pipeline):
        assert 0 < len(demo_pipeline.exceptions) <= 30, (
            "thousands of events must not produce a queue nobody can work through"
        )

    def test_events_are_scored_against_earlier_data_only(self, demo_pipeline, demo_events):
        assert demo_pipeline.training_count > 0
        assert len(demo_pipeline.scored) < len(demo_events)

    def test_every_detector_contributes(self, demo_pipeline):
        winners = {s.result.strongest for s in demo_pipeline.exceptions}
        assert len(winners) >= 4, (
            f"an ensemble where one detector wins everything is not an ensemble: {winners}"
        )

    def test_the_peer_group_anomaly_is_found(self, demo_pipeline):
        entities = {s.decision.created.entity_id for s in demo_pipeline.exceptions}
        assert PEER_ANOMALY_SELLER in entities

    def test_the_change_anomaly_is_found(self, demo_pipeline):
        entities = {s.decision.created.entity_id for s in demo_pipeline.exceptions}
        assert CHANGE_ANOMALY_SELLER in entities

    def test_the_burst_seller_is_found(self, demo_pipeline):
        entities = {s.decision.created.entity_id for s in demo_pipeline.exceptions}
        assert BURST_SELLER in entities

    def test_a_new_entity_is_never_flagged_merely_for_being_new(self, demo_pipeline):
        entities = {s.decision.created.entity_id for s in demo_pipeline.exceptions}
        assert NEW_SELLER not in entities

    def test_legitimately_large_luxury_orders_are_mostly_absorbed(self, demo_pipeline):
        luxury = [
            s for s in demo_pipeline.exceptions if s.event["seller_category"] == "luxury"
        ]
        assert len(luxury) <= 2, (
            "luxury orders are large by nature; flagging them means the baseline is not contextual"
        )

    def test_no_single_entity_dominates(self, demo_pipeline):
        counts = Counter(s.decision.created.entity_id for s in demo_pipeline.exceptions)
        assert counts.most_common(1)[0][1] <= 3

    def test_duplicates_are_folded_rather_than_raised(self, demo_pipeline):
        assert len(demo_pipeline.duplicates) > len(demo_pipeline.exceptions)

    def test_every_exception_carries_evidence(self, demo_pipeline):
        for scored in demo_pipeline.exceptions:
            candidate = scored.decision.created
            assert candidate.reasons, "an exception without a reason is not actionable"
            assert candidate.detector_versions
            assert 0 < candidate.confidence <= 0.95

    def test_scores_are_reproducible(self, demo_pipeline):
        for scored in demo_pipeline.exceptions:
            candidate = scored.decision.created
            assert recompute(candidate.evidence) == pytest.approx(candidate.score, abs=0.01)

    def test_the_run_is_deterministic(self, demo_events, demo_spec):
        from app.scoring.pipeline import run_pipeline

        first = run_pipeline(demo_events, demo_spec)
        ids = [s.decision.created.event_id for s in first.exceptions]
        second = run_pipeline(demo_events, demo_spec)
        assert ids == [s.decision.created.event_id for s in second.exceptions]

    def test_a_suppression_policy_removes_a_class_of_finding(self, demo_events, demo_spec):
        from app.scoring.pipeline import run_pipeline

        policy = SuppressionPolicy(
            id="p1",
            name="accepted books behaviour",
            match={"seller_category": "books"},
            reason="reviewed and accepted",
        )
        filtered = run_pipeline(demo_events, demo_spec, policies=[policy])
        assert filtered.suppressed, "the policy should have caught something"
        assert not any(
            s.event["seller_category"] == "books" for s in filtered.exceptions
        )

    def test_raising_the_threshold_reduces_the_queue(self, demo_events, demo_spec):
        from app.scoring.pipeline import run_pipeline

        strict = run_pipeline(demo_events, demo_spec, config=GenerationConfig(min_score=85.0))
        baseline = run_pipeline(demo_events, demo_spec)
        assert len(strict.exceptions) < len(baseline.exceptions)
