from pathlib import Path

import pandas as pd

from src.explain import choose_note_and_verdict
from src.output_validation import validate_final_output


OUTPUT_PATH = Path("outputs/final_output.csv")
LABEL_PATH = Path(__file__).parent / "data" / "expected_verdicts.csv"


def test_final_output_matches_independent_labeled_verdicts():
    output = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
    expected = pd.read_csv(LABEL_PATH, keep_default_na=False)

    pd.testing.assert_frame_equal(
        output[expected.columns],
        expected,
        check_dtype=False,
    )


def test_supporting_note_beats_higher_ranked_decoy():
    ranked = pd.DataFrame([
        {
            "note_id": "DECOY",
            "date": pd.Timestamp("2025-04-01"),
            "start_date": pd.Timestamp("2025-04-01"),
            "end_date": pd.Timestamp("2025-04-07"),
            "applies_to": "Test-Route",
            "can_justify_rise": False,
            "evidence_type": "non_supporting",
            "reason": "Conditions were normal.",
            "similarity": 0.99,
        },
        {
            "note_id": "SUPPORT",
            "date": pd.Timestamp("2025-04-01"),
            "start_date": pd.Timestamp("2025-04-01"),
            "end_date": pd.Timestamp("2025-04-07"),
            "applies_to": "Test-Route",
            "can_justify_rise": True,
            "evidence_type": "route_specific",
            "reason": "A temporary surcharge increased trip costs.",
            "similarity": 0.42,
        },
    ])

    selected, verdict = choose_note_and_verdict(
        ranked,
        route="Test-Route",
        week_of="2025-04-07",
        observed_increase_pct=25.0,
        peer_increase_pct=20.0,
    )

    assert selected["note_id"] == "SUPPORT"
    assert verdict["explained"] is True
    assert verdict["matched_note_id"] == "SUPPORT"


def test_final_output_format_matches_sample_header():
    sample_columns = pd.read_csv(
        "data/sample_output_format_v2.csv",
        nrows=0,
    ).columns.tolist()
    output = pd.read_csv(OUTPUT_PATH, keep_default_na=False)

    validate_final_output(output, sample_columns)
