
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.retrieval import prepare_notes, rank_eligible_notes
from src.verdict import assess_note_support


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def explain_candidates(candidates: pd.DataFrame, notes_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Explain candidate route-week records using eligible context notes.

    A candidate is marked 'No (justified)' only when a retrieved note
    passes the explicit verdict checks. Otherwise it remains flagged.
    """

    notes = prepare_notes(notes_raw)
    model = SentenceTransformer(MODEL_NAME)

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
        )

        note = ranked_notes.iloc[0] if not ranked_notes.empty else None

        verdict = assess_note_support(
            note=note,
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

    return pd.DataFrame(output_rows, columns=columns)
