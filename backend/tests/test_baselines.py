"""Tests for contextual baselines and the statistics they rest on."""
from __future__ import annotations

import pytest

from app.baselines.model import (
    MIN_SCOPE_SAMPLES,
    ZERO_INFLATION_THRESHOLD,
    build_baselines,
    scope_key,
)
from tests.conftest import event


def numeric_events(values, **kwargs):
    return [event(f"E{i}", minutes=i, order_value=v, **kwargs) for i, v in enumerate(values)]


class TestScopeKey:
    def test_is_order_independent(self):
        assert scope_key({"b": "2", "a": "1"}) == scope_key({"a": "1", "b": "2"})

    def test_empty_scope_is_star(self):
        assert scope_key({}) == "*"


class TestNumericStats:
    def test_robust_centre_ignores_a_single_extreme(self):
        stats = build_baselines(
            numeric_events([10, 12, 11, 13, 9, 100]),
            numeric_features=["order_value"],
            categorical_features=[],
        ).global_baseline.numeric["order_value"]
        assert 10 <= stats.median <= 12
        assert stats.maximum == 100

    def test_constant_series_does_not_make_everything_anomalous(self):
        stats = build_baselines(
            numeric_events([50] * 40),
            numeric_features=["order_value"],
            categorical_features=[],
        ).global_baseline.numeric["order_value"]
        assert stats.sigma > 0, "a zero spread would score every later value as infinite"
        assert abs(stats.robust_z(50)) < 1

    def test_outlier_scores_high(self):
        stats = build_baselines(
            numeric_events([10, 11, 12, 9, 10, 11, 13, 10]),
            numeric_features=["order_value"],
            categorical_features=[],
        ).global_baseline.numeric["order_value"]
        assert stats.robust_z(500) > 10

    def test_too_few_values_yields_no_statistics(self):
        baseline = build_baselines(
            numeric_events([10, 11]),
            numeric_features=["order_value"],
            categorical_features=[],
        ).global_baseline
        assert "order_value" not in baseline.numeric


class TestZeroInflation:
    """Structural zeros are the most common way an anomaly detector goes wrong."""

    def _stats(self, values):
        return build_baselines(
            [event(f"E{i}", minutes=i, refund_amount=v) for i, v in enumerate(values)],
            numeric_features=["refund_amount"],
            categorical_features=[],
        ).global_baseline.numeric["refund_amount"]

    def test_mostly_zero_feature_is_detected(self):
        stats = self._stats([0.0] * 80 + [50.0, 60.0, 55.0, 45.0, 52.0] * 4)
        assert stats.zero_inflated is True
        assert stats.zero_share >= ZERO_INFLATION_THRESHOLD

    def test_statistics_describe_the_non_zero_population(self):
        stats = self._stats([0.0] * 80 + [50.0] * 20)
        assert stats.median == pytest.approx(50.0), (
            "structural zeros must not drag the median to 0"
        )

    def test_a_zero_observation_is_not_an_anomaly(self):
        stats = self._stats([0.0] * 80 + [50.0] * 20)
        assert stats.robust_z(0.0) == 0.0, (
            "'no refund happened' is not an extreme low refund"
        )

    def test_an_ordinary_occurrence_is_not_an_anomaly(self):
        stats = self._stats([0.0] * 80 + [50.0, 48.0, 52.0, 49.0, 51.0] * 4)
        assert abs(stats.robust_z(50.0)) < 3, (
            "a typical refund must not look extreme just because most orders have none"
        )

    def test_a_genuinely_large_value_still_scores(self):
        stats = self._stats([0.0] * 80 + [50.0, 48.0, 52.0, 49.0, 51.0] * 4)
        assert stats.robust_z(5000.0) > 10

    def test_a_dense_feature_is_untouched(self):
        stats = self._stats([10.0, 12.0, 11.0, 13.0] * 10)
        assert stats.zero_inflated is False
        assert stats.zero_share == 0.0

    def test_too_few_informative_values_abstains(self):
        stats = self._stats([0.0] * 90 + [50.0, 51.0, 49.0])
        assert stats.robust_z(500.0) == 0.0, (
            "three non-zero observations cannot support a judgement"
        )


class TestScopedBaselines:
    @pytest.fixture()
    def events(self):
        """Books are cheap, luxury is expensive. Same column, different normal."""
        out = []
        for i in range(60):
            out.append(event(f"B{i}", minutes=i, category="books", order_value=20 + i % 5))
        for i in range(60):
            out.append(
                event(f"L{i}", minutes=200 + i, category="luxury", order_value=4000 + i % 50)
            )
        return out

    @pytest.fixture()
    def baselines(self, events):
        return build_baselines(
            events,
            numeric_features=["order_value"],
            categorical_features=[],
            dimensions=["seller_category"],
        )

    def test_each_scope_gets_its_own_normal(self, baselines):
        books = baselines.baselines["seller_category=books"].numeric["order_value"]
        luxury = baselines.baselines["seller_category=luxury"].numeric["order_value"]
        assert books.median < 50
        assert luxury.median > 3000

    def test_a_luxury_price_is_normal_in_luxury_and_extreme_in_books(self, baselines):
        """The product thesis, as an assertion."""
        luxury_price = 4020
        books = baselines.baselines["seller_category=books"].numeric["order_value"]
        luxury = baselines.baselines["seller_category=luxury"].numeric["order_value"]
        assert abs(luxury.robust_z(luxury_price)) < 3
        assert books.robust_z(luxury_price) > 50

    def test_lookup_selects_the_matching_scope(self, baselines):
        found = baselines.lookup(event("X", category="luxury"))
        assert found.scope == {"seller_category": "luxury"}

    def test_describe_scope_is_readable(self, baselines):
        assert (
            baselines.baselines["seller_category=luxury"].describe_scope()
            == "seller_category=luxury"
        )
        assert baselines.global_baseline.describe_scope() == "all data"


class TestFallback:
    @pytest.fixture()
    def baselines(self):
        events = [event(f"B{i}", minutes=i, category="books") for i in range(60)]
        events.append(event("rare", minutes=999, category="antiques"))
        return build_baselines(
            events,
            numeric_features=["order_value"],
            categorical_features=[],
            dimensions=["seller_category"],
        )

    def test_a_thin_scope_falls_back_to_a_broader_one(self, baselines):
        thin = baselines.baselines["seller_category=antiques"]
        assert thin.sample_count < MIN_SCOPE_SAMPLES
        assert thin.is_fallback is True
        assert thin.fallback_from is not None

    def test_lookup_skips_a_thin_scope(self, baselines):
        found = baselines.lookup(event("X", category="antiques"))
        assert found is baselines.global_baseline, (
            "a scope with one sample must not be used as its own baseline"
        )

    def test_the_fallback_is_recorded_so_an_exception_can_say_so(self, baselines):
        payload = baselines.as_dict()["scopes"]["seller_category=antiques"]
        assert payload["fallback_from"] is not None

    def test_unknown_scope_returns_the_global_baseline(self, baselines):
        assert baselines.lookup(event("X", category="never_seen")) is baselines.global_baseline


class TestCategoricalRarity:
    def _stats(self, count, payment="card"):
        return build_baselines(
            [event(f"E{i}", minutes=i, payment=payment) for i in range(count)],
            numeric_features=[],
            categorical_features=["payment_method"],
        ).global_baseline.categorical["payment_method"]

    def test_common_values_are_not_rare(self):
        assert self._stats(100).rarity("card") < 0.2

    def test_unseen_values_are_rare(self):
        assert self._stats(100).rarity("crypto") > 0.9

    def test_rarity_accounts_for_sample_size(self):
        """Unseen in 20 observations is weaker evidence than unseen in 2000."""
        assert self._stats(2000).rarity("crypto") > self._stats(20).rarity("crypto")

    def test_empty_baseline_is_not_rare(self):
        baseline = build_baselines(
            [], numeric_features=[], categorical_features=["payment_method"]
        ).global_baseline
        assert baseline.categorical == {}
