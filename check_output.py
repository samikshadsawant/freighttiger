
import csv
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("outputs/final_output.csv")

EXPECTED_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


def main():
    assert OUTPUT_PATH.exists(), "Output CSV does not exist."

    df = pd.read_csv(OUTPUT_PATH, keep_default_na=False)

    assert df.columns.tolist() == EXPECTED_COLUMNS, (
        f"Unexpected columns: {df.columns.tolist()}"
    )

    assert len(df) == 17, f"Expected 17 rows, found {len(df)}"

    for column in ["route", "week_of", "flagged", "reason"]:
        assert df[column].ne("").all(), f"Blank values found in {column}"

    assert df["flagged"].isin(["Yes", "No (justified)"]).all()

    justified = df["flagged"].eq("No (justified)")
    assert df.loc[justified, "matched_note_id"].ne("").all()

    unexplained = df["flagged"].eq("Yes")
    assert df.loc[unexplained, "matched_note_id"].eq("").all()

    # Check the actual text stored in the CSV.
    with OUTPUT_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for line_number, row in enumerate(reader, start=2):
            cost = row["cost_per_tonne_km"]
            assert len(cost.split(".")[-1]) == 2 and "." in cost, (
                f"Cost formatting error on line {line_number}: {cost!r}"
            )

    print("All output validation checks passed!")
    print(f"Validated {len(df)} rows and {len(EXPECTED_COLUMNS)} columns.")


if __name__ == "__main__":
    main()
