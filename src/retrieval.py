
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from src.config import NEAR_MISS_MAX_GAP_DAYS
from src.note_metadata import NOTE_METADATA

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _encode_texts(texts, model, usage=None):
    if usage is not None:
        tokenized = model.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=model.max_seq_length,
        )
        usage["encode_calls"] += 1
        usage["tokens_encoded"] += sum(
            len(input_ids) for input_ids in tokenized["input_ids"]
        )
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def encode_note_embeddings(notes: pd.DataFrame, model, usage=None) -> dict:
    note_texts = (
        notes["note_id"].astype(str)
        + ". "
        + notes["note"].astype(str)
    ).tolist()
    embeddings = _encode_texts(note_texts, model, usage)
    return {
        str(note_id): embeddings[index]
        for index, note_id in enumerate(notes["note_id"])
    }


def prepare_notes(notes: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame(NOTE_METADATA)

    prepared = notes.merge(
        metadata,
        on="note_id",
        how="left",
        validate="one_to_one"
    )

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["start_date"] = pd.to_datetime(prepared["start_date"])
    prepared["end_date"] = pd.to_datetime(prepared["end_date"])

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
    model=None,
    note_embeddings=None,
    usage=None,
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

    query_embedding = _encode_texts([query], model, usage)

    if note_embeddings is None:
        note_texts = (
            eligible["note_id"].astype(str)
            + ". "
            + eligible["note"].astype(str)
        ).tolist()
        selected_note_embeddings = _encode_texts(note_texts, model, usage)
    else:
        selected_note_embeddings = torch.stack([
            note_embeddings[str(note_id)]
            for note_id in eligible["note_id"]
        ])

    scores = util.cos_sim(
        query_embedding,
        selected_note_embeddings
    )[0].cpu().tolist()

    eligible["similarity"] = scores

    return eligible.sort_values(
        ["similarity", "note_id"],
        ascending=[False, True]
    ).reset_index(drop=True)
