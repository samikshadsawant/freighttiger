from pathlib import Path

import pandas as pd
import pytest

from src.explain import choose_note_and_verdict
from src.output_validation import validate_final_output


OUTPUT_PATH = Path("outputs/final_output.csv")
LABEL_PATH = Path(__file__).parent / "data" / "expected_verdicts.csv"


def test_final_output_matches_independent_labeled_verdicts():
    output = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
    expected = pd.read_csv(LABEL_PATH, keep_default_na=False)

    pd.testing.assert_frame_equal(
        output[["route", "week_of", "flagged", "matched_note_id"]],
        expected[["route", "week_of", "flagged", "matched_note_id"]],
        check_dtype=False,
    )
    reason_label = expected.loc[
        expected["expected_reason"].ne("")
    ].iloc[0]
    actual_reason = output.loc[
        output["route"].eq(reason_label["route"])
        & output["week_of"].eq(reason_label["week_of"]),
        "reason",
    ].iloc[0]
    assert actual_reason == reason_label["expected_reason"]


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


@pytest.mark.parametrize(
    "note_order",
    [["N003", "N010"], ["N010", "N003"]],
)
def test_unexplained_citation_prefers_reviewed_cost_pressure(note_order):
    notes = {
        "N003": {
            "note_id": "N003",
            "date": pd.Timestamp("2025-05-05"),
            "start_date": pd.Timestamp("2025-05-05"),
            "end_date": pd.Timestamp("2025-12-31"),
            "applies_to": "All Routes",
            "can_justify_rise": True,
            "evidence_type": "general_cost_pressure",
            "reason": "Nationwide diesel prices rose, increasing transportation costs by roughly 5.0-7.0%.",
            "similarity": 0.4,
        },
        "N010": {
            "note_id": "N010",
            "date": pd.Timestamp("2025-10-27"),
            "start_date": pd.Timestamp("2025-10-27"),
            "end_date": pd.Timestamp("2025-11-02"),
            "applies_to": "All Routes",
            "can_justify_rise": False,
            "evidence_type": "non_supporting",
            "reason": "Compliance costs were absorbed without a rate change.",
            "similarity": 0.9,
        },
    }
    ranked = pd.DataFrame([notes[note_id] for note_id in note_order])

    selected, verdict = choose_note_and_verdict(
        ranked,
        route="Mumbai-Pune",
        week_of="2025-10-27",
        observed_increase_pct=5.5,
        peer_increase_pct=28.3,
    )

    assert selected["note_id"] == "N003"
    assert verdict["explained"] is False
    expected_reason = pd.read_csv(LABEL_PATH).loc[
        lambda labels: labels["route"].eq("Mumbai-Pune")
        & labels["week_of"].eq("2025-10-27"),
        "expected_reason",
    ].iloc[0]
    assert verdict["reason"] == expected_reason


def test_final_output_format_matches_sample_header():
    sample_columns = pd.read_csv(
        "data/sample_output_format_v2.csv",
        nrows=0,
    ).columns.tolist()
    output = pd.read_csv(OUTPUT_PATH, keep_default_na=False)

    validate_final_output(output, sample_columns)
