
import json
from pathlib import Path
from datetime import datetime, timezone


def write_usage_log(
    candidate_rows: int,
    output_rows: int,
    retrieval_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Record actual LLM usage and pipeline retrieval activity."""

    log = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llm": {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_inr": 0.00,
        },
        "retrieval": {
            "model": retrieval_model,
            "candidate_rows_processed": int(candidate_rows),
        },
        "pipeline": {
            "output_rows": int(output_rows),
            "generative_llm_used": False,
        },
        "cost_note": (
            "LLM cost is ₹0.00 because this pipeline uses "
            "embedding-based retrieval and rule-based verdicts, "
            "with no generative LLM calls."
        ),
    }

    output_path = Path("outputs") / "token_cost_log.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(log, file, indent=4)

    print(f"Usage log saved: {output_path}")
    print("LLM calls: 0 | Input tokens: 0 | Output tokens: 0")
    print("Estimated LLM cost: ₹0.00")
