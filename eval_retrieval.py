
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval import (
    MODEL_NAME,
    prepare_notes,
    get_eligible_notes,
    rank_eligible_notes,
)
from src.verdict import assess_note_support


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "context_notes.csv"

# Each case specifies the expected retrieval result and verdict.
TEST_CASES = [
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-02-24",
        "expected_note": "N001",
        "own_pct": 31.5,
        "peer_pct": 38.7,
        "expected_justified": True,
    },
    {
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "expected_note": "N002",
        "own_pct": 29.5,
        "peer_pct": 22.5,
        "expected_justified": True,
    },
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-17",
        "expected_note": "N009",
        "own_pct": 13.7,
        "peer_pct": 33.5,
        "expected_justified": False,
    },
]


def build_query(case):
    return (
        f"Route: {case['route']}. "
        f"Week starting {case['week_of']}. "
        f"Observed increase is {case['own_pct']}% "
        f"versus own history and {case['peer_pct']}% "
        "versus similar routes. Find relevant context."
    )


def main():
    notes = prepare_notes(pd.read_csv(DATA_PATH))
    model = SentenceTransformer(MODEL_NAME)

    retrieval_passes = 0
    verdict_passes = 0

    print("\nEND-TO-END RETRIEVAL EVALUATION\n")

    for case in TEST_CASES:
        eligible = get_eligible_notes(
            case["route"],
            case["week_of"],
            notes,
        )

        ranked = rank_eligible_notes(
            route=case["route"],
            week_of=case["week_of"],
            query=build_query(case),
            notes=notes,
            model=model,
        )

        expected_id = case["expected_note"]

        # Check whether the expected note is eligible.
        is_eligible = expected_id in eligible["note_id"].values

        # Check whether it is ranked first overall.
        top_id = (
            ranked.iloc[0]["note_id"]
            if not ranked.empty
            else None
        )
        retrieval_ok = is_eligible and top_id == expected_id

        if retrieval_ok:
            retrieval_passes += 1

        # Evaluate the expected note with the existing verdict rules.
        expected_row = notes.loc[
            notes["note_id"] == expected_id
        ].iloc[0]

        verdict = assess_note_support(
            note=expected_row,
            route=case["route"],
            week_of=case["week_of"],
            observed_increase_pct=case["own_pct"],
            peer_increase_pct=case["peer_pct"],
        )

        actual_justified = bool(verdict["explained"])
        verdict_ok = (
            actual_justified == case["expected_justified"]
        )

        if verdict_ok:
            verdict_passes += 1

        print(
            f"{case['route']} | {case['week_of']} | "
            f"Expected: {expected_id} | Top retrieved: {top_id}"
        )
        print(
            f"  Eligible: {is_eligible} | "
            f"Retrieval: {'PASS' if retrieval_ok else 'FAIL'}"
        )
        print(
            f"  Expected justified: {case['expected_justified']} | "
            f"Actual justified: {actual_justified} | "
            f"Verdict: {'PASS' if verdict_ok else 'FAIL'}"
        )

    total = len(TEST_CASES)

    print("\nSUMMARY")
    print(
        f"Retrieval top-1 accuracy: "
        f"{retrieval_passes}/{total} "
        f"({retrieval_passes / total:.1%})"
    )
    print(
        f"Verdict accuracy: "
        f"{verdict_passes}/{total} "
        f"({verdict_passes / total:.1%})"
    )


if __name__ == "__main__":
    main()
