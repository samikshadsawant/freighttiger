
import pandas as pd

from src.retrieval import prepare_notes
from src.verdict import assess_note_support


CASES = [
    # Valid, route-specific supporting notes
    ("N001", "Chennai-Bangalore", "2025-02-24", 31.46, 38.67, True, "N001"),
    ("N001", "Chennai-Bangalore", "2025-03-03", 22.78, 31.98, True, "N001"),
    ("N002", "Ahmedabad-Mumbai", "2025-01-20", 29.47, 22.49, True, "N002"),
    ("N002", "Ahmedabad-Mumbai", "2025-01-13", 10.00, 5.00, False, None),
    ("N002", "Ahmedabad-Mumbai", "2025-01-27", 10.00, 5.00, False, None),

    # Correct route, but outside the note's date window
    ("N001", "Chennai-Bangalore", "2025-03-17", 13.69, 33.48, False, None),
    ("N002", "Ahmedabad-Mumbai", "2025-02-03", 20.00, 18.00, False, None),

    # Wrong route
    ("N001", "Delhi-Jaipur", "2025-03-03", 20.00, 18.00, False, None),
    ("N002", "Mumbai-Pune", "2025-01-20", 20.00, 18.00, False, None),

    # Notes that must not justify a cost increase
    ("N005", "Mumbai-Delhi", "2024-07-29", 20.00, 18.00, False, None),
    ("N009", "Chennai-Bangalore", "2025-03-17", 13.69, 33.48, False, None),
    ("N007", "Delhi-Jaipur", "2024-05-20", 20.00, 18.00, False, None),
    ("N008", "Kolkata-Bhubaneswar", "2025-06-09", 20.00, 18.00, False, None),
    ("N010", "Mumbai-Pune", "2025-10-27", 20.00, 18.00, False, None),
    ("N004", "Mumbai-Pune", "2024-03-11", 20.00, 18.00, False, None),
    ("N006", "Mumbai-Pune", "2025-09-22", 20.00, 18.00, False, None),

    # Nationwide diesel note cannot justify route-specific peer divergence
    ("N003", "Mumbai-Pune", "2025-09-15", 9.25, 23.63, False, None),
]


def main():
    raw_notes = pd.read_csv("data/context_notes.csv")
    notes = prepare_notes(raw_notes)
    notes_by_id = {
        str(row["note_id"]): row
        for _, row in notes.iterrows()
    }

    passed = 0

    for number, case in enumerate(CASES, start=1):
        note_id, route, week, own_pct, peer_pct, expected, expected_id = case
        note = notes_by_id[note_id]

        result = assess_note_support(
            note=note,
            route=route,
            week_of=week,
            observed_increase_pct=own_pct,
            peer_increase_pct=peer_pct,
        )

        actual = result["explained"]
        actual_id = result["matched_note_id"] if actual else None

        case_passed = actual == expected and actual_id == expected_id
        passed += int(case_passed)

        print(
            f"Case {number:02}: {'PASS' if case_passed else 'FAIL'} | "
            f"note={note_id} | route={route} | week={week} | "
            f"expected={expected}/{expected_id} | "
            f"actual={actual}/{actual_id}"
        )

        if not case_passed:
            print(f"         Reason: {result['reason']}")

    accuracy = 100 * passed / len(CASES)
    print(f"\nPassed: {passed}/{len(CASES)}")
    print(f"Verdict accuracy: {accuracy:.1f}%")

    if passed != len(CASES):
        raise SystemExit("Evaluation failed. Investigate failed cases.")


if __name__ == "__main__":
    main()
