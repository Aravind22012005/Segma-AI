"""Feature Engineering Tool (agent-facing): selects the most definitive
numeric features for clustering from the Unified Customer View, pruning
near-constant and highly redundant (collinear) features so KMeans isn't
dominated by duplicated signal (e.g. TxnCount_Total vs MonthlyTxnFrequency)."""

from __future__ import annotations

import pandas as pd

from backend.features import UNIFIED_VIEW_NUMERIC_FEATURES
from backend.json_utils import to_jsonable


def select_features(
    view: pd.DataFrame,
    candidates: list[str] | None = None,
    variance_floor: float = 1e-6,
    corr_threshold: float = 0.92,
) -> dict:
    candidates = candidates or UNIFIED_VIEW_NUMERIC_FEATURES
    candidates = [c for c in candidates if c in view.columns]
    X = view[candidates].astype(float).fillna(0)

    # drop near-constant features
    variances = X.var()
    low_variance = variances[variances <= variance_floor].index.tolist()
    kept = [c for c in candidates if c not in low_variance]

    # drop one of each highly-correlated pair, keeping the one with higher variance
    corr = X[kept].corr().abs()
    dropped_redundant = []
    for i, col_a in enumerate(kept):
        if col_a in dropped_redundant:
            continue
        for col_b in kept[i + 1:]:
            if col_b in dropped_redundant:
                continue
            if corr.loc[col_a, col_b] >= corr_threshold:
                loser = col_a if variances[col_a] < variances[col_b] else col_b
                dropped_redundant.append(loser)

    selected = [c for c in kept if c not in dropped_redundant]

    return to_jsonable({
        "selected_features": selected,
        "dropped_low_variance": low_variance,
        "dropped_redundant_correlated": dropped_redundant,
        "n_candidates": len(candidates),
        "n_selected": len(selected),
    })
