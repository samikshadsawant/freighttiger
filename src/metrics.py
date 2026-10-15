
import pandas as pd


def compute_weekly_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    df = shipments.copy()

    # STEP 1: WEEKLY METRICS

    # 1. Create route name
    df["route"] = (
        df["origin"].astype(str)
        + "-"
        + df["destination"].astype(str)
    )

    # 2. Convert shipment date to datetime
    df["shipment_date"] = pd.to_datetime(
        df["shipment_date"], errors="raise"
    )

    # 3. Calculate Monday of each week
    df["week_of"] = (
        df["shipment_date"]
        - pd.to_timedelta(
            df["shipment_date"].dt.weekday,
            unit="D"
        )
    )

    # 4. Calculate tonne-km for each shipment
    df["tonne_km"] = (
        df["quantity_tonnes"] * df["distance_km"]
    )

    # 5. Group shipments by route, route type, and week
    weekly = (
        df.groupby(
            ["route", "route_type", "week_of"],
            as_index=False
        )
        .agg(
            total_cost_inr=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count")
        )
    )

    # 6. Calculate cost per tonne-km
    weekly["cost_per_tonne_km"] = (
        weekly["total_cost_inr"]
        / weekly["total_tonne_km"]
    )

    # STEP 2: BASELINE CALCULATIONS

    # 7. Sort chronologically before rolling calculations
    weekly = weekly.sort_values(
        ["route", "week_of"]
    ).reset_index(drop=True)

    # 8. Own-history baseline:
    # Previous 8 available weeks, excluding current week
    weekly["own_history_avg"] = (
        weekly.groupby("route")["cost_per_tonne_km"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=8,
                min_periods=1
            ).mean()
        )
    )

    # 9. Peer baseline:
    # Same week, same route type, excluding itself
    peer_groups = weekly.groupby(
        ["week_of", "route_type"]
    )["cost_per_tonne_km"]

    peer_sum = peer_groups.transform("sum")
    peer_count = peer_groups.transform("count")

    weekly["similar_routes_avg"] = (
        (peer_sum - weekly["cost_per_tonne_km"])
        / (peer_count - 1).where(peer_count > 1)
    )

    # 10. Percentage difference from own history
    weekly["pct_vs_own_history"] = (
        (
            weekly["cost_per_tonne_km"]
            - weekly["own_history_avg"]
        )
        / weekly["own_history_avg"]
        * 100
    )

    # 11. Percentage difference from similar routes
    weekly["pct_vs_similar_routes"] = (
        (
            weekly["cost_per_tonne_km"]
            - weekly["similar_routes_avg"]
        )
        / weekly["similar_routes_avg"]
        * 100
    )

    return weekly
