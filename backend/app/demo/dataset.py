"""Deterministic demo dataset with planted anomalies.

Synthetic e-commerce orders and refunds. The generator plants six specific
abnormalities, each shaped to be caught by a different detector, plus two
deliberate false positives — values that look extreme globally but are normal
in their own context.

Those false positives are the point of the whole exercise. A detector that
flags them is not contextual, it is just a threshold with extra steps.

Planted:

1. **Point anomaly** — a handful of refunds far above their category's normal.
2. **Contextual anomaly** — refunds normal for electronics, extreme for books.
3. **Peer-group anomaly** — one seller with a refund ratio unlike its peers.
4. **Change anomaly** — a seller whose basket value shifts partway through.
5. **Frequency anomaly** — a burst of refunds from one seller in a short window.
6. **Rare combination** — a country/payment-method pair that never occurs.

Deliberate non-anomalies:

* Luxury-category orders are genuinely large; extreme globally, normal in scope.
* A new seller with little history — thin data must not read as suspicious.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

CATEGORIES = ("books", "electronics", "clothing", "luxury")
MARKETS = ("SK", "CZ", "DE", "AT")
PAYMENTS = ("card", "transfer", "cod")

#: Typical order value per category. Luxury is legitimately an order of
#: magnitude larger -- the contextual baseline must absorb that.
CATEGORY_VALUE = {
    "books": (8, 40),
    "electronics": (80, 900),
    "clothing": (20, 150),
    "luxury": (1_500, 9_000),
}

CATEGORY_REFUND_RATE = {
    "books": 0.03,
    "electronics": 0.09,
    "clothing": 0.22,
    "luxury": 0.05,
}

START = datetime(2026, 5, 1, tzinfo=timezone.utc)

#: Sellers carrying a planted behaviour.
PEER_ANOMALY_SELLER = "S-014"
CHANGE_ANOMALY_SELLER = "S-021"
BURST_SELLER = "S-033"
NEW_SELLER = "S-090"


@dataclass
class PlantedAnomaly:
    key: str
    kind: str
    description: str
    #: Which detector should be able to see it.
    expected_detector: str


PLANTED = [
    PlantedAnomaly(
        "point_refund",
        "point",
        "A few refunds far above the normal amount for their category",
        "robust_deviation",
    ),
    PlantedAnomaly(
        "contextual_refund",
        "contextual",
        "Refund amounts that are ordinary for electronics but extreme for books",
        "robust_deviation",
    ),
    PlantedAnomaly(
        "peer_group",
        "peer_group",
        f"Seller {PEER_ANOMALY_SELLER} refunds far more often than comparable sellers",
        "robust_deviation",
    ),
    PlantedAnomaly(
        "change_point",
        "change",
        f"Seller {CHANGE_ANOMALY_SELLER} basket value shifts sharply partway through",
        "change_point",
    ),
    PlantedAnomaly(
        "frequency_burst",
        "frequency",
        f"Seller {BURST_SELLER} produces a burst of refunds in a short window",
        "rolling_quantile",
    ),
    PlantedAnomaly(
        "rare_combination",
        "rare_combination",
        "A country and payment-method pair that appears almost nowhere else",
        "categorical_rarity",
    ),
]

EXPECTED_NON_ANOMALIES = [
    "Luxury orders are large globally but normal within the luxury category",
    "A new seller with little history must not be flagged merely for being new",
]


def generate_events(*, count: int = 4000, seed: int = 20260830) -> list[dict[str, Any]]:
    """Build the synthetic event log. Pure function, no database."""
    rng = random.Random(seed)
    sellers = [f"S-{i:03d}" for i in range(1, 40)]
    events: list[dict[str, Any]] = []

    for index in range(count):
        occurred_at = START + timedelta(minutes=18 * index + rng.randint(0, 12))
        seller = rng.choice(sellers)
        category = rng.choices(CATEGORIES, weights=[35, 30, 30, 5])[0]
        market = rng.choices(MARKETS, weights=[40, 25, 25, 10])[0]
        payment = rng.choices(PAYMENTS, weights=[60, 25, 15])[0]

        low, high = CATEGORY_VALUE[category]
        order_value = round(rng.uniform(low, high), 2)
        refund_rate = CATEGORY_REFUND_RATE[category]

        # 3. Peer-group anomaly: this seller refunds far more than its peers.
        if seller == PEER_ANOMALY_SELLER:
            refund_rate = 0.55

        # 4. Change anomaly: basket value steps up halfway through the window.
        if seller == CHANGE_ANOMALY_SELLER and index > count * 0.6:
            order_value = round(order_value * 3.2, 2)

        refunded = rng.random() < refund_rate
        refund_amount = round(order_value * rng.uniform(0.3, 1.0), 2) if refunded else 0.0

        # 1 + 2. Point and contextual anomalies on refund amount.
        if refunded and rng.random() < 0.004:
            refund_amount = round(order_value * rng.uniform(6, 14), 2)
        if category == "books" and refunded and rng.random() < 0.01:
            # Normal for electronics, absurd for books.
            refund_amount = round(rng.uniform(300, 700), 2)

        # 5. Frequency burst: one seller floods refunds in a narrow window.
        if seller == BURST_SELLER and 0.42 < index / count < 0.46:
            refunded = True
            refund_amount = round(order_value * rng.uniform(0.8, 1.0), 2)

        # 6. Rare combination that appears essentially nowhere else.
        if rng.random() < 0.0015:
            market, payment = "AT", "cod"

        events.append(
            {
                "event_id": f"E-{100_000 + index}",
                "entity_id": seller,
                "entity_type": "seller",
                "occurred_at": occurred_at,
                "order_id": f"O-{500_000 + index}",
                "seller_category": category,
                "market": market,
                "payment_method": payment,
                "order_value": order_value,
                "refund_amount": refund_amount,
                "refund_ratio": round(refund_amount / order_value, 4) if order_value else 0.0,
                "items": rng.randint(1, 6),
                "delivery_days": round(rng.uniform(1, 9), 1),
            }
        )

    # The new seller appears only at the very end, with a short history.
    tail_start = START + timedelta(minutes=18 * count)
    for offset in range(8):
        category = "clothing"
        low, high = CATEGORY_VALUE[category]
        value = round(rng.uniform(low, high), 2)
        events.append(
            {
                "event_id": f"E-{200_000 + offset}",
                "entity_id": NEW_SELLER,
                "entity_type": "seller",
                "occurred_at": tail_start + timedelta(minutes=25 * offset),
                "order_id": f"O-{900_000 + offset}",
                "seller_category": category,
                "market": "SK",
                "payment_method": "card",
                "order_value": value,
                "refund_amount": 0.0,
                "refund_ratio": 0.0,
                "items": rng.randint(1, 4),
                "delivery_days": round(rng.uniform(1, 6), 1),
            }
        )

    return events


#: Schema the demo tenant registers, mirroring what a customer would configure.
DEMO_SCHEMA = {
    "name": "orders_and_refunds",
    "entity_type": "seller",
    "entity_key": "entity_id",
    "event_key": "event_id",
    "timestamp_field": "occurred_at",
    "numeric_features": ["order_value", "refund_amount", "refund_ratio", "delivery_days", "items"],
    "categorical_features": ["seller_category", "market", "payment_method"],
    # Ordered most significant first: the last dimension is dropped first when
    # a scope has too little data.
    "context_dimensions": ["seller_category", "market"],
    "rate_features": ["refund_amount"],
    "detector_weights": {},
}


def to_csv(events: list[dict[str, Any]]) -> str:
    """Render the events as the CSV a customer would upload."""
    header = [
        "event_id",
        "entity_id",
        "occurred_at",
        "order_id",
        "seller_category",
        "market",
        "payment_method",
        "order_value",
        "refund_amount",
        "refund_ratio",
        "items",
        "delivery_days",
    ]
    lines = [",".join(header)]
    for event in events:
        lines.append(
            ",".join(
                [
                    event["event_id"],
                    event["entity_id"],
                    event["occurred_at"].isoformat(),
                    event["order_id"],
                    event["seller_category"],
                    event["market"],
                    event["payment_method"],
                    f"{event['order_value']:.2f}",
                    f"{event['refund_amount']:.2f}",
                    f"{event['refund_ratio']:.4f}",
                    str(event["items"]),
                    f"{event['delivery_days']:.1f}",
                ]
            )
        )
    return "\n".join(lines)
