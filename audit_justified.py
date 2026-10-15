
import pandas as pd

from src.metrics import compute_weekly_metrics
from src.retrieval import prepare_notes
from src.verdict import assess_note_support


def main():
    output = pd.read_csv(
        "outputs/final_output.csv",
        keep_default_na=False,
    )
    shipments = pd.read_csv("data/shipment_records.csv")
    notes_raw = pd.read_csv("data/context_notes.csv")
    notes = prepare_notes(notes_raw)

    weekly = compute_weekly_metrics(shipments)
    weekly["week_of"] = pd.to_datetime(weekly["week_of"])
    weekly_by_key = weekly.set_index(["route", "week_of"])

    notes_by_id = {
        str(row["note_id"]): row
        for _, row in notes.iterrows()
    }

    justified = output[output["flagged"] == "No (justified)"]
    failures = []

    for _, row in justified.iterrows():
        route = row["route"]
        week = pd.Timestamp(row["week_of"])
        note_id = row["matched_note_id"]

        if not note_id:
            failures.append(f"{route} {week.date()}: missing note ID")
            continue

        if note_id not in notes_by_id:
            failures.append(f"{route} {week.date()}: unknown note {note_id}")
            continue

        if (route, week) not in weekly_by_key.index:
            failures.append(f"{route} {week.date()}: missing weekly metrics")
            continue

        metric = weekly_by_key.loc[(route, week)]
        note = notes_by_id[note_id]

        verdict = assess_note_support(
            note=note,
            route=route,
            week_of=week,
            observed_increase_pct=float(metric["pct_vs_own_history"]),
            peer_increase_pct=float(metric["pct_vs_similar_routes"]),
        )

        if not verdict["explained"]:
            failures.append(
                f"{route} {week.date()}: {note_id} rejected: "
                f"{verdict['reason']}"
            )
        elif verdict["matched_note_id"] != note_id:
            failures.append(
                f"{route} {week.date()}: matched note ID mismatch"
            )
        else:
            print(
                f"PASS | {route} | {week.date()} | "
                f"{note_id} | {note['note']}"
            )

    print(f"\nJustified rows audited: {len(justified)}")

    if failures:
        print("\nAUDIT FAILURES:")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    print("All justified rows passed the audit.")


if __name__ == "__main__":
    main()
