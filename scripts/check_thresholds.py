
import pandas as pd
from src.metrics import compute_weekly_metrics

shipments = pd.read_csv("data/shipment_records.csv")
weekly = compute_weekly_metrics(shipments)

# Only rows where both comparisons can be calculated
valid = weekly.dropna(
    subset=[
        "pct_vs_own_history",
        "pct_vs_similar_routes"
    ]
)

print("Total route-weeks:", len(weekly))
print("Route-weeks with both baselines:", len(valid))

print("\nOWN-HISTORY PERCENTILES:")
print(
    weekly["pct_vs_own_history"]
    .dropna()
    .describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
    .to_string()
)

print("\nPEER-COMPARISON PERCENTILES:")
print(
    weekly["pct_vs_similar_routes"]
    .dropna()
    .describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
    .to_string()
)

print("\nCANDIDATE COUNTS FOR POSSIBLE THRESHOLDS:")

for x in [5, 10, 15, 20]:
    for y in [5, 10, 15, 20]:
        candidates = valid[
            (valid["pct_vs_own_history"] > x)
            & (valid["pct_vs_similar_routes"] > y)
        ]

        print(
            f"Own history > +{x}% AND peers > +{y}%: "
            f"{len(candidates)} candidates"
        )
