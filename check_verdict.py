
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval import (
    prepare_notes,
    rank_eligible_notes,
)
from src.verdict import assess_note_support


def main():
    # Load and prepare context notes with reviewed metadata.
    notes_raw = pd.read_csv("data/context_notes.csv")
    notes = prepare_notes(notes_raw)

    # Load the local embedding model once.
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    tests = [
        {
            "route": "Chennai-Bangalore",
            "week": "2025-03-03",
            "query": "Flooding caused detours and higher transportation costs.",
            "increase_pct": 31.46,
            "peer_increase_pct": 38.67,
            "expected_note": "N001",
            "expected_explained": True,
        },
        {
            "route": "Ahmedabad-Mumbai",
            "week": "2025-01-20",
            "query": "Festival surcharge increased transportation costs.",
            "increase_pct": 29.47,
            "peer_increase_pct": 22.49,
            "expected_note": "N002",
            "expected_explained": True,
        },
        {
            "route": "Mumbai-Pune",
            "week": "2025-09-15",
            "query": "Nationwide diesel prices increased transportation costs.",
            "increase_pct": 9.25,
            "peer_increase_pct": 23.63,
            "expected_note": "N003",
            "expected_explained": False,
        },
        {
            "route": "Chennai-Bangalore",
            "week": "2025-03-17",
            "query": "Highway repairs are complete and the route returned to normal.",
            "increase_pct": 13.69,
            "peer_increase_pct": 33.48,
            "expected_note": "N009",
            "expected_explained": False,
        },
    ]

    passed = 0

    for test in tests:
        ranked = rank_eligible_notes(
            route=test["route"],
            week_of=test["week"],
            query=test["query"],
            notes=notes,
            model=model,
        )

        # Retrieve the highest-ranked eligible note.
        note = ranked.iloc[0] if not ranked.empty else None

        # Assess the retrieved note directly.
        result = assess_note_support(
            note=note,
            route=test["route"],
            week_of=test["week"],
            observed_increase_pct=test["increase_pct"],
            peer_increase_pct=test["peer_increase_pct"],
        )

        actual_note = (
            str(note["note_id"]) if note is not None else None
        )

        # Check both the retrieval result and the verdict.
        note_matches = actual_note == test["expected_note"]
        verdict_matches = (
            result["explained"] == test["expected_explained"]
        )

        test_passed = note_matches and verdict_matches

        if test_passed:
            passed += 1

        print("\n" + "-" * 70)
        print(f'Route: {test["route"]}')
        print(f'Week: {test["week"]}')
        print(f"Retrieved note: {actual_note}")
        print(f'Expected note: {test["expected_note"]}')
        print(f'Explained: {result["explained"]}')
        print(f'Matched note ID: {result["matched_note_id"]}')
        print(f'Reason: {result["reason"]}')
        print(f"Test result: {'PASS' if test_passed else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"Tests passed: {passed}/{len(tests)}")

    if passed != len(tests):
        raise SystemExit("Some verdict tests failed.")

    print("All verdict tests passed!")


if __name__ == "__main__":
    main()
