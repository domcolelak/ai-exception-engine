"""Peer-group detector.

Some abnormalities do not exist in any single event. A seller that refunds 55%
of its orders looks entirely ordinary order by order -- each refund is a
perfectly normal refund. The abnormality is a property of the *entity*, visible
only when its aggregate behaviour is compared against comparable entities.

This detector therefore works on entity profiles: it summarises each entity
within a context scope, then measures how far one entity sits from its peers
using the same robust statistics used everywhere else.

It is deliberately separate from the per-event detectors. Trying to express a
rate as a per-event feature is how teams end up with a "refund_ratio" column
that means something different on every row.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.baselines.model import MAD_TO_SIGMA
from app.detectors.base import (
    DetectionContext,
    Detector,
    DetectorResult,
    Reason,
    normalise_z,
    number,
)


@dataclass
class EntityProfile:
    """Aggregate behaviour of one entity within a scope."""

    entity_id: str
    event_count: int
    means: dict[str, float] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "event_count": self.event_count,
            "means": {k: round(v, 6) for k, v in self.means.items()},
            "rates": {k: round(v, 6) for k, v in self.rates.items()},
        }


@dataclass
class PeerStats:
    """Distribution of one aggregate metric across peer entities."""

    metric: str
    median: float
    sigma: float
    peer_count: int

    def robust_z(self, value: float) -> float:
        if self.sigma <= 0:
            return 0.0
        return (value - self.median) / self.sigma


@dataclass
class PeerGroup:
    scope_key: str
    profiles: dict[str, EntityProfile] = field(default_factory=dict)
    stats: dict[str, PeerStats] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "scope_key": self.scope_key,
            "entity_count": len(self.profiles),
            "metrics": {
                k: {"median": round(v.median, 6), "sigma": round(v.sigma, 6), "peers": v.peer_count}
                for k, v in self.stats.items()
            },
        }


#: An entity needs this many events before its aggregate means anything.
MIN_EVENTS_PER_ENTITY = 25

#: A peer group needs this many entities before "unlike its peers" is a claim
#: worth making.
MIN_PEERS = 8


def build_peer_groups(
    events: Sequence[dict[str, Any]],
    *,
    entity_key: str,
    numeric_features: Sequence[str],
    rate_features: Sequence[str] = (),
    dimensions: Sequence[str] = (),
    min_events: int = MIN_EVENTS_PER_ENTITY,
) -> dict[str, PeerGroup]:
    """Summarise every entity, then describe the spread across peers.

    ``rate_features`` are numeric columns whose *occurrence* matters as much as
    their size -- a refund amount is one: the share of events where it is
    non-zero is the refund rate, which is exactly the peer-group signal.
    """
    # Peer groups are built at every prefix depth, plus globally. A peer group
    # needs many entities each with many events, so the finest context scope is
    # usually far too thin -- one seller rarely has 25 events in one exact
    # (category, market) cell. The detector then picks the finest group that
    # actually contains the entity.
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    dims = tuple(dimensions)
    for event in events:
        grouped["*"].append(event)
        for depth in range(1, len(dims) + 1):
            grouped[peer_scope_key(event, dims[:depth])].append(event)

    groups: dict[str, PeerGroup] = {}
    for scope, scope_events in grouped.items():
        by_entity: defaultdict[str, list[dict]] = defaultdict(list)
        for event in scope_events:
            by_entity[str(event.get(entity_key, ""))].append(event)

        profiles: dict[str, EntityProfile] = {}
        for entity_id, entity_events in by_entity.items():
            if len(entity_events) < min_events:
                continue
            profiles[entity_id] = _profile(
                entity_id, entity_events, numeric_features, rate_features
            )

        if len(profiles) < MIN_PEERS:
            continue

        group = PeerGroup(scope_key=scope, profiles=profiles)
        metrics = set()
        for profile in profiles.values():
            metrics.update(f"mean:{k}" for k in profile.means)
            metrics.update(f"rate:{k}" for k in profile.rates)

        for metric in sorted(metrics):
            values = [_metric_value(p, metric) for p in profiles.values()]
            values = [v for v in values if v is not None]
            if len(values) < MIN_PEERS:
                continue
            group.stats[metric] = _peer_stats(metric, values)

        groups[scope] = group
    return groups


def peer_scope_key(event: dict[str, Any], dimensions: Sequence[str]) -> str:
    """Scope identity for peer grouping. Mirrors the baseline scope format."""
    if not dimensions:
        return "*"
    return "|".join(f"{d}={event.get(d, '')}" for d in sorted(dimensions))


def _profile(
    entity_id: str,
    events: Sequence[dict],
    numeric_features: Sequence[str],
    rate_features: Sequence[str],
) -> EntityProfile:
    profile = EntityProfile(entity_id=entity_id, event_count=len(events))
    for feature in numeric_features:
        values = [v for v in (number(e.get(feature)) for e in events) if v is not None]
        if values:
            profile.means[feature] = sum(values) / len(values)
    for feature in rate_features:
        values = [number(e.get(feature)) for e in events]
        present = [v for v in values if v is not None]
        if present:
            profile.rates[feature] = sum(1 for v in present if v != 0) / len(present)
    return profile


def _metric_value(profile: EntityProfile, metric: str) -> float | None:
    kind, _, name = metric.partition(":")
    return (profile.means if kind == "mean" else profile.rates).get(name)


def _peer_stats(metric: str, values: list[float]) -> PeerStats:
    ordered = sorted(values)
    median = _median(ordered)
    mad = _median(sorted(abs(v - median) for v in ordered))
    sigma = mad * MAD_TO_SIGMA
    if sigma <= 0:
        spread = ordered[-1] - ordered[0]
        sigma = spread / 4 if spread > 0 else 0.0
    return PeerStats(metric=metric, median=median, sigma=sigma, peer_count=len(values))


def _median(ordered: Sequence[float]) -> float:
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


class PeerGroupDetector(Detector):
    """Flags an entity whose aggregate behaviour is unlike its peers."""

    name = "peer_group"
    version = "v1"
    default_weight = 1.0

    #: Peer deviation is a slower, structural signal, so the bar is higher than
    #: for a single surprising event.
    knee = 2.5
    threshold = 0.45

    def __init__(self, groups: dict[str, PeerGroup], *, entity_key: str, dimensions: Sequence[str]):
        self.groups = groups
        self.entity_key = entity_key
        self.dimensions = tuple(dimensions)

    def _resolve(self, event: dict, entity_id: str) -> tuple[str, PeerGroup, EntityProfile] | None:
        """Finest peer group that actually knows this entity.

        Falls back through coarser scopes, then global. An entity absent from
        every group has too little history to be compared -- and being new is
        not the same as being abnormal, so the detector abstains rather than
        scoring it.
        """
        for depth in range(len(self.dimensions), -1, -1):
            scope = peer_scope_key(event, self.dimensions[:depth]) if depth else "*"
            group = self.groups.get(scope)
            if group is None:
                continue
            profile = group.profiles.get(entity_id)
            if profile is not None:
                return scope, group, profile
        return None

    def score(self, context: DetectionContext) -> DetectorResult:
        entity_id = str(context.event.get(self.entity_key, ""))
        resolved = self._resolve(context.event, entity_id)
        if resolved is None:
            return self._abstain(
                f"{entity_id} has too little history to compare against peers"
            )
        scope, group, profile = resolved

        # When the entity's own context scope had too few comparable entities we
        # fall back to a broader group -- but only scale-free metrics survive
        # that move. A luxury seller's average refund is legitimately ten times
        # a bookseller's, so comparing means across a mixed group manufactures
        # false positives; the *share* of orders refunded stays comparable.
        own_scope = peer_scope_key(context.event, self.dimensions) if self.dimensions else "*"
        scale_free_only = scope != own_scope

        reasons: list[Reason] = []
        worst = 0.0
        for metric, stats in group.stats.items():
            if scale_free_only and not metric.startswith("rate:"):
                continue
            value = _metric_value(profile, metric)
            if value is None:
                continue
            z = stats.robust_z(value)
            # Only an entity that is *worse* than its peers is interesting here;
            # an unusually low refund rate is not an operations exception.
            if z <= 0:
                continue
            deviation = normalise_z(z, knee=self.knee)
            worst = max(worst, deviation)
            if deviation < self.threshold:
                continue

            kind, _, name = metric.partition(":")
            label = f"share of events with a non-zero {name}" if kind == "rate" else f"average {name}"
            shown = f"{value:.1%}" if kind == "rate" else f"{value:,.2f}"
            peer_shown = f"{stats.median:.1%}" if kind == "rate" else f"{stats.median:,.2f}"
            reasons.append(
                Reason(
                    feature=metric,
                    observed=round(value, 6),
                    expected_range=(
                        stats.median - self.knee * stats.sigma,
                        stats.median + self.knee * stats.sigma,
                    ),
                    deviation_score=round(deviation, 4),
                    explanation=(
                        f"{entity_id}: {label} is {shown} across its "
                        f"{profile.event_count} events, versus {peer_shown} for the "
                        f"{stats.peer_count} comparable entities in this group"
                    ),
                    scope=scope,
                )
            )

        reasons.sort(key=lambda r: -r.deviation_score)
        return DetectorResult(
            detector=self.name,
            version=self.version,
            raw_score=worst,
            score=worst,
            reasons=reasons,
            evidence={
                "scope": scope,
                "entity_scope": own_scope,
                "scale_free_metrics_only": scale_free_only,
                "peer_count": len(group.profiles),
                "entity_events": profile.event_count,
                "profile": profile.as_dict(),
            },
        )
