
import pandas as pd

from src.retrieval import prepare_notes
from src.verdict import assess_note_support


def main():
    raw_notes = pd.read_csv("data/context_notes.csv")
    notes = prepare_notes(raw_notes)

    assert len(notes) == 10, (
        f"Expected 10 context notes, found {len(notes)}"
    )

    assert notes["note_id"].is_unique, "Duplicate note IDs found"
    assert notes["can_justify_rise"].notna().all(), (
        "Some notes lack reviewed justification metadata"
    )

    test_route = {
        "N001": "Chennai-Bangalore",
        "N002": "Ahmedabad-Mumbai",
        "N003": "Mumbai-Pune",
        "N004": "Mumbai-Pune",
        "N005": "Mumbai-Delhi",
        "N006": "Mumbai-Pune",
        "N007": "Delhi-Jaipur",
        "N008": "Kolkata-Bhubaneswar",
        "N009": "Chennai-Bangalore",
        "N010": "Mumbai-Pune",
    }

    test_week = {
        "N001": "2025-03-03",
        "N002": "2025-01-20",
        "N003": "2025-09-15",
        "N004": "2024-03-11",
        "N005": "2024-07-29",
        "N006": "2025-09-22",
        "N007": "2024-05-20",
        "N008": "2025-06-09",
        "N009": "2025-03-17",
        "N010": "2025-10-27",
    }

    expected_explained = {
        "N001": True,
        "N002": True,
        "N003": False,
        "N004": False,
        "N005": False,
        "N006": False,
        "N007": False,
        "N008": False,
        "N009": False,
        "N010": False,
    }

    # Use each note directly to test the verdict layer independently
    # of embedding ranking.
    for _, note in notes.iterrows():
        note_id = note["note_id"]

        result = assess_note_support(
            note=note,
            route=test_route[note_id],
            week_of=test_week[note_id],
            observed_increase_pct=20.0,
            peer_increase_pct=20.0,
        )

        expected = expected_explained[note_id]
        passed = result["explained"] == expected

        print(
            f"{note_id}: "
            f"{'PASS' if passed else 'FAIL'} | "
            f"explained={result['explained']} | "
            f"matched_note_id={result['matched_note_id']} | "
            f"{result['reason']}"
        )

        assert passed, f"Unexpected verdict for {note_id}"

    print("\nAll 10 context-note tests passed!")


if __name__ == "__main__":
    main()
