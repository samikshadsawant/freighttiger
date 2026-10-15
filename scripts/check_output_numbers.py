from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "shipment_records.csv"
OUTPUT_PATH = ROOT / "outputs" / "final_output.csv"


def independent_weekly_metrics(shipments):
    records = shipments.copy()
    records["route"] = (
        records["origin"].astype(str)
        + "-"
        + records["destination"].astype(str)
    )
    records["shipment_date"] = pd.to_datetime(records["shipment_date"])
    records["week_of"] = (
        records["shipment_date"]
        - pd.to_timedelta(records["shipment_date"].dt.weekday, unit="D")
    )
    records["tonne_km"] = (
        records["quantity_tonnes"] * records["distance_km"]
    )

    weekly = (
        records.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            freight_total=("freight_cost_inr", "sum"),
            tonne_km_total=("tonne_km", "sum"),
        )
    )
    weekly["rate"] = weekly["freight_total"] / weekly["tonne_km_total"]
    weekly = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)

    own_baselines = []
    peer_baselines = []
    for _, current in weekly.iterrows():
        previous = weekly.loc[
            (weekly["route"] == current["route"])
            & (weekly["week_of"] < current["week_of"]),
            "rate",
        ].tail(8)
        own_baselines.append(previous.mean() if not previous.empty else math.nan)

        peers = weekly.loc[
            (weekly["week_of"] == current["week_of"])
            & (weekly["route_type"] == current["route_type"])
            & (weekly["route"] != current["route"]),
            "rate",
        ]
        peer_baselines.append(peers.mean() if not peers.empty else math.nan)

    weekly["own_avg"] = own_baselines
    weekly["peer_avg"] = peer_baselines
    weekly["own_pct"] = (weekly["rate"] - weekly["own_avg"]) / weekly["own_avg"] * 100
    weekly["peer_pct"] = (weekly["rate"] - weekly["peer_avg"]) / weekly["peer_avg"] * 100
    return weekly.set_index(["route", "week_of"])


def main():
    shipments = pd.read_csv(DATA_PATH)
    reference = independent_weekly_metrics(shipments)
    output = pd.read_csv(OUTPUT_PATH, keep_default_na=False)

    assert len(output) == 17, f"Expected 17 output rows, found {len(output)}"
    for _, row in output.iterrows():
        key = (row["route"], pd.Timestamp(row["week_of"]))
        metrics = reference.loc[key]
        own_pct = float(row["vs_own_history"].split("%", maxsplit=1)[0])
        peer_pct = float(row["vs_similar_routes"].split("%", maxsplit=1)[0])
        assert math.isclose(
            row["cost_per_tonne_km"], round(metrics["rate"], 2), abs_tol=1e-9
        ), f"Cost mismatch for {key}"
        assert math.isclose(own_pct, round(metrics["own_pct"], 1), abs_tol=1e-9), (
            f"Own-history percentage mismatch for {key}"
        )
        assert math.isclose(peer_pct, round(metrics["peer_pct"], 1), abs_tol=1e-9), (
            f"Peer percentage mismatch for {key}"
        )

    print(f"Independently recomputed costs and percentages for {len(output)} rows.")


if __name__ == "__main__":
    main()
