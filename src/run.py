
from pathlib import Path

import pandas as pd

from src.metrics import compute_weekly_metrics
from src.flagging import flag_candidates
from src.explain import explain_candidates
from src.usage_logger import write_usage_log


def main():
    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load input data
    shipments = pd.read_csv(data_dir / "shipment_records.csv")
    notes = pd.read_csv(data_dir / "context_notes.csv")

    # 2. Calculate weekly route metrics
    weekly = compute_weekly_metrics(shipments)

    # 3. Identify candidate route-weeks
    flagged_weekly = flag_candidates(weekly)

    candidates = flagged_weekly.loc[
        flagged_weekly["candidate"]
    ].copy()

    # 4. Explain candidates using context notes
    final_output = explain_candidates(candidates, notes)

    # 5. Save final output
    output_path = output_dir / "final_output.csv"

    final_output.to_csv(
        output_path,
        index=False,
        float_format="%.2f"
    )

    print(f"Total route-week groups: {len(weekly)}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Output rows: {len(final_output)}")
    print(f"Saved: {output_path}")

    print("\nOutput preview:")
    print(final_output.to_string(index=False))

    # 6. Log LLM token usage and estimated cost
    write_usage_log(
        candidate_rows=len(candidates),
        output_rows=len(final_output),
    )


if __name__ == "__main__":
    main()
