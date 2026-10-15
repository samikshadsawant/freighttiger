import pandas as pd


def assess_note_support(
    note,
    route: str,
    week_of,
    observed_increase_pct: float,
    peer_increase_pct: float | None = None,
) -> dict:
    """Decide whether reviewed context evidence supports a route-week rise."""

    def unexplained(reason):
        return {
            "explained": False,
            "matched_note_id": None,
            "reason": reason,
        }

    if note is None:
        return unexplained(
            "Unexplained. No reviewed context note applies to this route "
            "and week. Flagged for review."
        )

    note_id = str(note["note_id"])
    date_value = note.get("date")
    date_text = (
        pd.Timestamp(date_value).strftime("%Y-%m-%d")
        if pd.notna(date_value)
        else "date not recorded"
    )
    start_value = note.get("start_date")
    end_value = note.get("end_date")
    has_window = pd.notna(start_value) and pd.notna(end_value)

    if has_window:
        note_start = pd.Timestamp(start_value).normalize()
        note_end = pd.Timestamp(end_value).normalize()
        window_text = (
            f"{note_start.strftime('%Y-%m-%d')} to "
            f"{note_end.strftime('%Y-%m-%d')}"
        )
    else:
        window_text = "applicability dates missing"

    note_reference = f"{note_id} ({date_text}; window {window_text})"
    note_reason = str(note.get("reason", "")).strip()
    if not note_reason or note_reason == "nan":
        note_reason = str(note.get("note", "This note has no reviewed summary.")).strip()
    note_summary = note_reason.rstrip(".")
    if note_summary:
        note_summary = note_summary[0].lower() + note_summary[1:]

    if pd.isna(note.get("can_justify_rise")):
        return unexplained(
            f"Unexplained. Note {note_reference} has not been reviewed for "
            "whether it supports a cost rise. Flagged for review."
        )

    if not has_window:
        return unexplained(
            f"Unexplained. Note {note_reference} cannot be matched to this "
            "week because its applicability dates are missing. Flagged for review."
        )

    applies_to = str(note["applies_to"])
    if applies_to != "All Routes" and applies_to != route:
        return unexplained(
            f"Unexplained. Note {note_reference} applies to {applies_to}, "
            f"not {route}. Flagged for review."
        )

    week_start = pd.Timestamp(week_of).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    if note_start > week_end or note_end < week_start:
        if note_end < week_start:
            window_relation = f"ended {note_end.strftime('%Y-%m-%d')} before this week"
        else:
            window_relation = f"starts {note_start.strftime('%Y-%m-%d')} after this week"
        return unexplained(
            f"Unexplained. Closest note {note_reference} {window_relation}. "
            "Flagged for review."
        )

    evidence_type = note.get("evidence_type")
    if bool(note["can_justify_rise"]) and evidence_type == "route_specific":
        return {
            "explained": True,
            "matched_note_id": note_id,
            "reason": (
                f"Justified. Matches note {note_id} dated {date_text} (applicable "
                f"{window_text}): {note_reason} This week overlaps that "
                "window, so the rise has a clear explanation."
            ),
        }

    if evidence_type == "general_cost_pressure":
        peer_comparison = (
            "the peer comparison is unavailable"
            if peer_increase_pct is None
            else f"this route is {peer_increase_pct:.1f}% above its peers"
        )
        return unexplained(
            f"Unexplained. Note {note_reference} says {note_summary}. "
            "That general pressure would affect similar routes too; "
            f"it does not explain why {peer_comparison}. Flagged for review."
        )

    return unexplained(
        f"Unexplained. Closest note {note_reference} says {note_summary} "
        "and does not describe a cost rise. Flagged for review."
    )
