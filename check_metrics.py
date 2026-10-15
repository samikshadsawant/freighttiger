
import pandas as pd
from src.metrics import compute_weekly_metrics


# 1. Load shipment data
shipments = pd.read_csv("data/shipment_records.csv")

# 2. Compute weekly metrics and baselines
weekly = compute_weekly_metrics(shipments)

# 3. Print basic information
print("Original shipments:", len(shipments))
print("Weekly route groups:", len(weekly))

# 4. Display Step 1 and Step 2 results
print("\nWEEKLY METRICS WITH BASELINES:\n")

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

print(weekly[columns].head(15).to_string(index=False))

# 5. Inspect one route in detail
print("\nAHMEDABAD-MUMBAI ROUTE:\n")

route_data = weekly[
    weekly["route"] == "Ahmedabad-Mumbai"
]

print(route_data[columns].head(12).to_string(index=False))

# 6. Format percentage comparisons for final output
def format_comparison(value, description):
    if pd.isna(value):
        return ""

    return f"{value:+.1f}% vs {description}"


weekly["vs_own_history"] = weekly[
    "pct_vs_own_history"
].apply(
    lambda x: format_comparison(
        x, "this route's past average"
    )
)

weekly["vs_similar_routes"] = weekly[
    "pct_vs_similar_routes"
].apply(
    lambda x: format_comparison(
        x, "similar-length routes this week"
    )
)

# 7. Save results
weekly.to_csv(
    "outputs/weekly_metrics.csv",
    index=False
)

print("\nSaved outputs/weekly_metrics.csv")

# 8. Verify missing history
print(
    "\nRows without own-history baseline:",
    weekly["own_history_avg"].isna().sum()
)

print(
    "Rows without peer baseline:",
    weekly["similar_routes_avg"].isna().sum()
)
