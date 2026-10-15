import re


PERCENTAGE_FORMAT = re.compile(r"\+\d+\.\d% vs .+")


def validate_final_output(output, expected_columns):
    if output.columns.tolist() != expected_columns:
        raise ValueError(
            "Output columns do not match the sample format: "
            f"{output.columns.tolist()}"
        )

    if not output["vs_own_history"].map(
        lambda value: bool(PERCENTAGE_FORMAT.fullmatch(str(value)))
    ).all():
        raise ValueError("Invalid vs_own_history percentage format.")

    if not output["vs_similar_routes"].map(
        lambda value: bool(PERCENTAGE_FORMAT.fullmatch(str(value)))
    ).all():
        raise ValueError("Invalid vs_similar_routes percentage format.")

    if not output["flagged"].isin(["Yes", "No (justified)"]).all():
        raise ValueError("Invalid flagged label.")

    unexplained = output["flagged"].eq("Yes")
    justified = output["flagged"].eq("No (justified)")

    if not output.loc[unexplained, "matched_note_id"].eq("").all():
        raise ValueError("Unexplained rows must have a blank matched_note_id.")
    if not output.loc[justified, "matched_note_id"].ne("").all():
        raise ValueError("Justified rows must have a matched_note_id.")
    if not output.loc[unexplained, "reason"].str.startswith("Unexplained.").all():
        raise ValueError("Unexplained reasons must state the verdict.")
    if not output.loc[unexplained, "reason"].str.contains(
        "Flagged for review.", regex=False
    ).all():
        raise ValueError("Unexplained reasons must say they were flagged for review.")
    if not output.loc[justified, "reason"].str.startswith("Justified. Matches note ").all():
        raise ValueError("Justified reasons must identify the matching note.")
    for _, row in output.loc[justified].iterrows():
        if row["matched_note_id"] not in row["reason"]:
            raise ValueError("Justified reason does not name its matched note ID.")
        if "dated " not in row["reason"] or "applicable " not in row["reason"]:
            raise ValueError("Justified reason must include note dates.")
