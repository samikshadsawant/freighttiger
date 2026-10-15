
import pandas as pd


ROUTE_SPECIFIC_NOTES = {
    "N001": "Chennai-Bangalore",
    "N002": "Ahmedabad-Mumbai",
}

NON_SUPPORTING_REASONS = {
    "N004": "The affected routes are not part of this dataset.",
    "N005": "The note explicitly says costs were not significantly affected.",
    "N006": "The note reports stable demand and no major disruptions.",
    "N007": "The note describes improved road conditions.",
    "N008": "The note reports normal freight movement without significant disruptions.",
    "N009": "The route returned to normal conditions after repairs.",
    "N010": "The compliance costs were absorbed without a rate change.",
}


def assess_note_support(
    note,
    route: str,
    week_of,
    observed_increase_pct: float,
    peer_increase_pct: float | None = None,
) -> dict:
    """
    Decide whether a context note adequately supports a route-week anomaly.

    The note must be applicable to the route and overlap the week.
    Semantic similarity alone never determines the verdict.

    observed_increase_pct:
        Increase relative to the route's own-history baseline.

    peer_increase_pct:
        Increase relative to similar routes.
    """

    def unexplained(reason):
        return {
            "explained": False,
            "matched_note_id": None,
            "reason": reason,
        }

    if note is None:
        return unexplained(
            "No eligible context note supports the cost increase."
        )

    note_id = str(note["note_id"])

    # A note must have reviewed metadata.
    if pd.isna(note.get("can_justify_rise")):
        return unexplained(
            f"Note {note_id} has no reviewed justification metadata."
        )

    if not bool(note["can_justify_rise"]):
        return unexplained(
            NON_SUPPORTING_REASONS.get(
                note_id,
                f"Note {note_id} was reviewed and cannot justify a cost rise.",
            )
        )

    # Validate route applicability.
    applies_to = str(note["applies_to"])

    if applies_to != "All Routes" and applies_to != route:
        return unexplained(
            f"Note {note_id} does not apply to route {route}."
        )

    # Validate overlap between the note's applicability window and the week.
    if pd.isna(note.get("start_date")) or pd.isna(note.get("end_date")):
        return unexplained(
            f"Note {note_id} is missing its applicability dates."
        )

    week_start = pd.Timestamp(week_of).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    note_start = pd.Timestamp(note["start_date"]).normalize()
    note_end = pd.Timestamp(note["end_date"]).normalize()

    if note_start > week_end or note_end < week_start:
        return unexplained(
            f"Note {note_id} does not overlap the week beginning "
            f"{week_start.date()}."
        )

    # Route-specific evidence must match the exact route.
    if note_id in ROUTE_SPECIFIC_NOTES:
        expected_route = ROUTE_SPECIFIC_NOTES[note_id]

        if route != expected_route:
            return unexplained(
                f"Note {note_id} describes {expected_route}, not {route}."
            )

        if note_id == "N001":
            reason = (
                "Flooding caused longer detours and higher trip costs "
                "on the Chennai-Bangalore route."
            )
        else:
            reason = (
                "A temporary regional festival surcharge was applied "
                "on the Ahmedabad-Mumbai corridor."
            )

        return {
            "explained": True,
            "matched_note_id": note_id,
            "reason": reason,
        }

    # A nationwide diesel note may explain a general increase, but it
    # cannot alone explain why one route is unusually expensive vs peers.
    if note_id == "N003":
        if peer_increase_pct is None:
            return unexplained(
                "The nationwide diesel note does not establish whether "
                "this route's increase is unusual relative to peers."
            )

        if peer_increase_pct > 15.0:
            return unexplained(
                "The nationwide diesel-price note reports a roughly "
                "5-7% general increase, but does not explain why this "
                f"route is {peer_increase_pct:.2f}% above similar routes."
            )

        if 5.0 <= observed_increase_pct <= 7.0:
            return {
                "explained": True,
                "matched_note_id": "N003",
                "reason": (
                    "The nationwide diesel-price note reports a roughly "
                    "5-7% general increase, consistent with the observed "
                    "own-history increase. This does not establish a "
                    "route-specific cause."
                ),
            }

        return unexplained(
            "The nationwide diesel-price note reports a roughly 5-7% "
            "increase, which does not match the observed own-history "
            f"increase of {observed_increase_pct:.2f}%."
        )

    # Unknown notes must never be accepted automatically.
    return unexplained(
        f"Note {note_id} has no explicit reviewed rule supporting "
        "this route-week anomaly."
    )
