
import pandas as pd

from src.metrics import compute_weekly_metrics
from src.flagging import flag_candidates


# Load data
shipments = pd.read_csv("data/shipment_records.csv")

# Calculate weekly metrics and baselines
weekly = compute_weekly_metrics(shipments)

# Identify candidates
results = flag_candidates(weekly)

# Display candidate count
candidates = results[results["candidate"]].copy()

print("Total route-weeks:", len(results))
print("Candidate route-weeks:", len(candidates))

print("\nCANDIDATES FOR HUMAN REVIEW:\n")

columns = [
    "route",
    "route_type",
    "week_of",
    "cost_per_tonne_km",
    "own_history_avg",
    "similar_routes_avg",
    "pct_vs_own_history",
    "pct_vs_similar_routes"
]

if candidates.empty:
    print("No candidates found.")
else:
    print(
        candidates[columns]
        .sort_values(
            ["week_of", "route"]
        )
        .round(2)
        .to_string(index=False)
    )

# Save all rows and the candidate subset
results.to_csv(
    "outputs/flagging_results.csv",
    index=False
)

candidates.to_csv(
    "outputs/candidates.csv",
    index=False
)

print("\nSaved outputs/flagging_results.csv")
print("Saved outputs/candidates.csv")
