
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

from src.retrieval import (
    MODEL_NAME,
    prepare_notes,
    get_eligible_notes,
    rank_eligible_notes,
)


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
