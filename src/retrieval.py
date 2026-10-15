
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from src.note_metadata import NOTE_METADATA

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def prepare_notes(notes: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame(NOTE_METADATA)

    prepared = notes.merge(
        metadata,
        on="note_id",
        how="left",
        validate="one_to_one"
    )

    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["start_date"] = pd.to_datetime(prepared["start_date"])
    prepared["end_date"] = pd.to_datetime(prepared["end_date"])

    if prepared["can_justify_rise"].isna().any():
        raise ValueError("Some notes are missing reviewed metadata.")

    return prepared


def get_eligible_notes(route: str, week_of, notes: pd.DataFrame):
    week_start = pd.Timestamp(week_of).normalize()
    week_end = week_start + pd.Timedelta(days=6)

    route_matches = (
        notes["applies_to"].eq(route)
        | notes["applies_to"].eq("All Routes")
    )

    date_matches = (
        (notes["start_date"] <= week_end)
        & (notes["end_date"] >= week_start)
    )

    return notes.loc[route_matches & date_matches].copy()


def rank_eligible_notes(
    route: str,
    week_of,
    query: str,
    notes: pd.DataFrame,
    model=None
) -> pd.DataFrame:
    """
    Rank eligible notes by semantic similarity.
    Similarity is for retrieval only, not final justification.
    """
    eligible = get_eligible_notes(route, week_of, notes)

    if eligible.empty:
        eligible["similarity"] = pd.Series(dtype=float)
        return eligible

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    note_texts = (
        eligible["note_id"].astype(str)
        + ". "
        + eligible["note"].astype(str)
    ).tolist()

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    note_embeddings = model.encode(
        note_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(
        query_embedding,
        note_embeddings
    )[0].cpu().tolist()

    eligible["similarity"] = scores

    return eligible.sort_values(
        ["similarity", "note_id"],
        ascending=[False, True]
    ).reset_index(drop=True)


def find_supporting_note(
    route: str,
    week_of,
    notes: pd.DataFrame,
    query: str,
    model=None
):
    ranked = rank_eligible_notes(
        route=route,
        week_of=week_of,
        query=query,
        notes=notes,
        model=model
    )

    # Only explicitly reviewed notes can support a rise.
    supporting = ranked[ranked["can_justify_rise"]]

    if supporting.empty:
        return None

    return supporting.iloc[0]
