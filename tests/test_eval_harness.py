
import pandas as pd
import pytest

from src.metrics import compute_weekly_metrics


def make_shipments(rows):
    """Create a DataFrame from compact shipment test records."""
    return pd.DataFrame(
        rows,
        columns=[
            "origin",
            "destination",
            "quantity_tonnes",
            "distance_km",
            "freight_cost_inr",
            "shipment_date",
            "route_type",
            "shipment_id",
        ],
    )


def test_week_starts_on_monday():
    shipments = make_shipments([
        ("A", "B", 10, 10, 100, "2025-03-05", "Short", "S1"),
        ("A", "B", 10, 10, 100, "2025-03-09", "Short", "S2"),
    ])

    result = compute_weekly_metrics(shipments)

    # Wednesday March 5 and Sunday March 9 belong to
    # the week starting Monday March 3.
    assert len(result) == 1
    assert pd.Timestamp(result.iloc[0]["week_of"]) == pd.Timestamp("2025-03-03")


def test_cost_per_tonne_km_formula():
    shipments = make_shipments([
        ("A", "B", 10, 10, 100, "2025-03-03", "Short", "S1"),
        ("A", "B", 10, 10, 200, "2025-03-04", "Short", "S2"),
    ])

    result = compute_weekly_metrics(shipments)

    # Total freight cost = 300
    # Total tonne-km = (10 * 10) + (10 * 10) = 200
    # Cost per tonne-km = 300 / 200 = 1.5
    assert result.iloc[0]["cost_per_tonne_km"] == pytest.approx(1.5)


def test_own_history_does_not_look_ahead():
    shipments = make_shipments([
        ("A", "B", 10, 10, 100, "2025-03-03", "Short", "S1"),
        ("A", "B", 10, 10, 200, "2025-03-10", "Short", "S2"),
        ("A", "B", 10, 10, 900, "2025-03-17", "Short", "S3"),
    ])

    result = compute_weekly_metrics(shipments).sort_values("week_of")
    result = result.reset_index(drop=True)

    first_rate = result.loc[0, "cost_per_tonne_km"]
    second_rate = result.loc[1, "cost_per_tonne_km"]

    # First week has no historical baseline.
    assert pd.isna(result.loc[0, "own_history_avg"])

    # Second week's baseline uses only the first week's rate.
    assert result.loc[1, "own_history_avg"] == pytest.approx(first_rate)

    # Third week's baseline averages the two previous weekly rates,
    # never the third week's own rate.
    expected_third_baseline = (first_rate + second_rate) / 2
    assert result.loc[2, "own_history_avg"] == pytest.approx(
        expected_third_baseline
    )


def test_peer_baseline_excludes_current_route():
    shipments = make_shipments([
        # Route A: cost per tonne-km = 2
        ("A", "B", 10, 10, 200, "2025-03-03", "Short", "S1"),

        # Route C: cost per tonne-km = 4
        ("C", "D", 10, 10, 400, "2025-03-03", "Short", "S2"),

        # Route E: cost per tonne-km = 6
        ("E", "F", 10, 10, 600, "2025-03-03", "Short", "S3"),
    ])

    result = compute_weekly_metrics(shipments)
    rates = result.set_index("route")["similar_routes_avg"]

    # Each route's peer baseline is the average of the OTHER routes.
    assert rates["A-B"] == pytest.approx((4 + 6) / 2)
    assert rates["C-D"] == pytest.approx((2 + 6) / 2)
    assert rates["E-F"] == pytest.approx((2 + 4) / 2)
