
import pandas as pd

OWN_THRESHOLD = 5.0
PEER_THRESHOLD = 15.0


def flag_candidates(
    weekly: pd.DataFrame,
    own_threshold: float = OWN_THRESHOLD,
    peer_threshold: float = PEER_THRESHOLD
) -> pd.DataFrame:

    result = weekly.copy()

    # Both baselines must be available
    has_baselines = result[
        [
            "pct_vs_own_history",
            "pct_vs_similar_routes"
        ]
    ].notna().all(axis=1)

    # Cost must be rising compared with own history
    own_history_rising = (
        result["pct_vs_own_history"] > own_threshold
    )

    # Cost must also be above similar routes
    above_peers = (
        result["pct_vs_similar_routes"] > peer_threshold
    )

    # A candidate must satisfy both conditions
    result["candidate"] = (
        has_baselines
        & own_history_rising
        & above_peers
    )

    return result
