
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval import prepare_notes, rank_eligible_notes

notes_raw = pd.read_csv("data/context_notes.csv")
notes = prepare_notes(notes_raw)

# Load the model once and reuse it for all queries.
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

tests = [
    (
        "Chennai-Bangalore",
        "2025-03-03",
        "Flooding caused detours and increased transportation costs."
    ),
    (
        "Ahmedabad-Mumbai",
        "2025-01-20",
        "Festival surcharge due to high demand and limited truck availability."
    ),
    (
        "Mumbai-Pune",
        "2025-09-15",
        "Did diesel prices increase transportation costs nationwide?"
    ),
    (
        "Chennai-Bangalore",
        "2025-03-17",
        "Flooding and repairs have ended and the route returned to normal."
    ),
]

for route, week, query in tests:
    ranked = rank_eligible_notes(
        route=route,
        week_of=week,
        query=query,
        notes=notes,
        model=model
    )

    print(f"\nRoute: {route}, week: {week}")

    if ranked.empty:
        print("No eligible notes.")
        continue

    print(
        ranked[["note_id", "similarity", "note"]].to_string(
            index=False
        )
    )
