
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.config import NEAR_MISS_MAX_GAP_DAYS
from src.retrieval import (
    MODEL_NAME,
    encode_note_embeddings,
    prepare_notes,
    rank_eligible_notes,
)
from src.verdict import assess_note_support


def _near_miss_note(route: str, week_of, notes: pd.DataFrame):
    week_start = pd.Timestamp(week_of).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    applicable = notes.loc[
        notes["applies_to"].isin([route, "All Routes"])
        & notes["start_date"].notna()
        & notes["end_date"].notna()
    ].copy()

    if applicable.empty:
        return None

    def gap_days(note):
        note_start = pd.Timestamp(note["start_date"]).normalize()
        note_end = pd.Timestamp(note["end_date"]).normalize()
        if note_end < week_start:
            return (week_start - note_end).days
        if note_start > week_end:
            return (note_start - week_end).days
        return 0

    applicable["gap_days"] = applicable.apply(gap_days, axis=1)
    nearby = applicable.loc[
        applicable["gap_days"] <= NEAR_MISS_MAX_GAP_DAYS
    ]
    preceding = nearby.loc[nearby["end_date"] < week_start]
    candidates = preceding if not preceding.empty else nearby
    if candidates.empty:
        return None
    closest = candidates.sort_values(
        ["gap_days", "date", "note_id"]
    ).iloc[0]
    return closest.drop(labels="gap_days")


def explain_candidates(
    candidates: pd.DataFrame,
    notes_raw: pd.DataFrame,
    embedding_usage: dict | None = None,
) -> pd.DataFrame:
    """
    Explain candidate route-week records using eligible context notes.

    A candidate is marked 'No (justified)' only when a retrieved note
    passes the explicit verdict checks. Otherwise it remains flagged.
    """

    notes = prepare_notes(notes_raw)
    model = SentenceTransformer(MODEL_NAME)
    if embedding_usage is None:
        embedding_usage = {"encode_calls": 0, "tokens_encoded": 0}
    note_embeddings = encode_note_embeddings(
        notes,
        model,
        usage=embedding_usage,
    )

    output_rows = []

    for _, row in candidates.iterrows():
        route = str(row["route"])
        week_of = pd.Timestamp(row["week_of"])

        own_pct = row["pct_vs_own_history"]
        peer_pct = row["pct_vs_similar_routes"]

        query = (
            f"Route: {route}. Week starting {week_of.date()}. "
            f"Cost per tonne-km increased by {own_pct:.2f}% "
            f"compared with its own historical baseline and by "
            f"{peer_pct:.2f}% compared with similar routes. "
            "Find a context note that genuinely supports the cost increase."
        )

        ranked_notes = rank_eligible_notes(
            route=route,
            week_of=week_of,
            query=query,
            notes=notes,
            model=model,
            note_embeddings=note_embeddings,
            usage=embedding_usage,
        )

        if ranked_notes.empty:
            note = _near_miss_note(route, week_of, notes)
            verdict = assess_note_support(
                note=note,
                route=route,
                week_of=week_of,
                observed_increase_pct=float(own_pct),
                peer_increase_pct=float(peer_pct),
            )
        else:
            note, verdict = choose_note_and_verdict(
                ranked_notes,
                route=route,
                week_of=week_of,
                observed_increase_pct=float(own_pct),
                peer_increase_pct=float(peer_pct),
            )

        if verdict["explained"]:
            flagged = "No (justified)"
            matched_note_id = verdict["matched_note_id"]
        else:
            flagged = "Yes"
            matched_note_id = ""

        output_rows.append({
            "route": route,
            "week_of": week_of.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(float(row["cost_per_tonne_km"]), 2),
            "vs_own_history": (
                f"{float(own_pct):+.1f}% vs this route's past average"
            ),
            "vs_similar_routes": (
                f"{float(peer_pct):+.1f}% vs similar-length routes this week"
            ),
            "flagged": flagged,
            "matched_note_id": matched_note_id,
            "reason": verdict["reason"],
        })


    columns = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
        "reason",
    ]

    return (
        pd.DataFrame(output_rows, columns=columns)
        .sort_values(["route", "week_of"])
        .reset_index(drop=True)
    )


def choose_note_and_verdict(
    ranked_notes: pd.DataFrame,
    route: str,
    week_of,
    observed_increase_pct: float,
    peer_increase_pct: float,
):
    assessed = [
        (
            note,
            assess_note_support(
                note=note,
                route=route,
                week_of=week_of,
                observed_increase_pct=observed_increase_pct,
                peer_increase_pct=peer_increase_pct,
            ),
        )
        for _, note in ranked_notes.iterrows()
    ]
    supporting = [pair for pair in assessed if pair[1]["explained"]]
    if supporting:
        return supporting[0]

    assessed.sort(
        key=lambda pair: not _can_justify_rise(pair[0])
    )
    return assessed[0] if assessed else (None, None)


def _can_justify_rise(note):
    value = note.get("can_justify_rise")
    return bool(pd.notna(value) and value)
