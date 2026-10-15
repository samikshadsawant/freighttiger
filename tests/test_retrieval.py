
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

from src.retrieval import (
    MODEL_NAME,
    prepare_notes,
    get_eligible_notes,
    rank_eligible_notes,
)
from src.verdict import assess_note_support
from src.explain import choose_note_and_verdict


@pytest.fixture(scope="module")
def notes():
    raw = pd.read_csv("data/context_notes.csv")
    return prepare_notes(raw)


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer(MODEL_NAME)


def test_n001_is_eligible_during_flooding(notes):
    eligible = get_eligible_notes(
        "Chennai-Bangalore",
        "2025-02-24",
        notes,
    )
    assert "N001" in eligible["note_id"].values


def test_n001_is_not_eligible_after_flooding(notes):
    eligible = get_eligible_notes(
        "Chennai-Bangalore",
        "2025-03-17",
        notes,
    )
    assert "N001" not in eligible["note_id"].values


def test_route_specific_note_is_excluded_for_wrong_route(notes):
    eligible = get_eligible_notes(
        "Delhi-Jaipur",
        "2025-02-24",
        notes,
    )
    assert "N001" not in eligible["note_id"].values


def test_all_routes_note_is_eligible(notes):
    eligible = get_eligible_notes(
        "Mumbai-Pune",
        "2025-10-27",
        notes,
    )
    assert "N010" in eligible["note_id"].values


def test_no_eligible_notes_returns_empty_dataframe(notes):
    eligible = get_eligible_notes(
        "Unknown-Route",
        "2026-06-01",
        notes,
    )
    assert eligible.empty


def test_unreviewed_note_is_not_returned_as_supporting(notes, model):
    raw_notes = pd.read_csv("data/context_notes.csv")
    raw_notes.loc[len(raw_notes)] = {
        "note_id": "N011",
        "date": "2026-06-01",
        "applies_to": "All Routes",
        "note": "A new event may have changed transportation costs.",
    }
    notes_with_unreviewed = prepare_notes(raw_notes)

    unreviewed = notes_with_unreviewed.loc[
        notes_with_unreviewed["note_id"].eq("N011")
    ].iloc[0]
    result = assess_note_support(
        note=unreviewed,
        route="Mumbai-Pune",
        week_of="2026-06-01",
        observed_increase_pct=20.0,
        peer_increase_pct=20.0,
    )

    assert result["explained"] is False
    assert result["matched_note_id"] is None


@pytest.mark.parametrize(
    "route,week,expected_note",
    [
        ("Chennai-Bangalore", "2025-02-24", "N001"),
        ("Ahmedabad-Mumbai", "2025-01-20", "N002"),
        ("Chennai-Bangalore", "2025-03-17", "N009"),
    ],
)
def test_expected_note_ranks_first(
    notes, model, route, week, expected_note
):
    query = (
        f"Route: {route}. Week starting {week}. "
        "Find context explaining freight cost changes."
    )

    ranked = rank_eligible_notes(
        route=route,
        week_of=week,
        query=query,
        notes=notes,
        model=model,
    )

    assert not ranked.empty
    assert ranked.iloc[0]["note_id"] == expected_note


def test_ranked_notes_are_all_eligible(notes, model):
    route = "Chennai-Bangalore"
    week = "2025-02-24"

    ranked = rank_eligible_notes(
        route=route,
        week_of=week,
        query="Flooding caused detours and higher freight costs",
        notes=notes,
        model=model,
    )

    eligible_ids = set(
        get_eligible_notes(route, week, notes)["note_id"]
    )

    assert set(ranked["note_id"]).issubset(eligible_ids)


def test_eligible_support_beats_higher_similarity_decoy(model):
    synthetic = pd.DataFrame([
        {
            "note_id": "SYNTHETIC_DECOY",
            "date": pd.Timestamp("2025-04-01"),
            "start_date": pd.Timestamp("2025-04-01"),
            "end_date": pd.Timestamp("2025-04-07"),
            "applies_to": "Test-Route",
            "note": "Stable demand and no disruptions; conditions remained normal.",
            "can_justify_rise": False,
            "evidence_type": "non_supporting",
            "reason": "Demand remained stable and conditions were normal.",
        },
        {
            "note_id": "SYNTHETIC_SUPPORT",
            "date": pd.Timestamp("2025-04-01"),
            "start_date": pd.Timestamp("2025-04-01"),
            "end_date": pd.Timestamp("2025-04-07"),
            "applies_to": "Test-Route",
            "note": "A temporary toll surcharge increased trip costs.",
            "can_justify_rise": True,
            "evidence_type": "route_specific",
            "reason": "A temporary toll surcharge increased trip costs.",
        },
    ])
    ranked = rank_eligible_notes(
        route="Test-Route",
        week_of="2025-04-07",
        query="Stable demand and no disruptions; conditions remained normal.",
        notes=synthetic,
        model=model,
    )

    assert set(ranked["note_id"]) == {
        "SYNTHETIC_DECOY",
        "SYNTHETIC_SUPPORT",
    }
    assert ranked.iloc[0]["note_id"] == "SYNTHETIC_DECOY"

    selected, verdict = choose_note_and_verdict(
        ranked,
        route="Test-Route",
        week_of="2025-04-07",
        observed_increase_pct=25.0,
        peer_increase_pct=20.0,
    )

    assert selected["note_id"] == "SYNTHETIC_SUPPORT"
    assert verdict["explained"] is True
