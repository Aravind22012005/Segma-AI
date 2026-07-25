"""Segmentation Tool: rule-based tiering + ML clustering.

Both approaches operate on the Unified Customer View produced by
backend.features.build_unified_customer_view. Nothing here calls an LLM --
the planner decides *which* of these to invoke based on the user's query.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from backend.json_utils import to_jsonable

DEFAULT_TIER_LABELS = ["Dormant", "Regular", "Priority"]


def rule_based_tiering(
    view: pd.DataFrame,
    balance_col: str = "AvgMonthlyBalance",
    freq_col: str = "MonthlyTxnFrequency",
    priority_quantile: float = 0.75,
    dormant_freq_threshold: float | None = None,
    dormant_recency_days: int = 90,
) -> tuple[pd.Series, dict]:
    """Assigns each customer to Priority / Regular / Dormant using an explicit,
    inspectable rule (not a black-box model), exactly like the example query:
    "segment based on balance being maintained and frequency of transactions".
    """
    balance = view[balance_col].astype(float)
    freq = view[freq_col].astype(float)

    if dormant_freq_threshold is None:
        dormant_freq_threshold = max(1.0, freq.quantile(0.15))

    balance_z = (balance - balance.mean()) / (balance.std() or 1)
    freq_z = (freq - freq.mean()) / (freq.std() or 1)
    score = 0.5 * balance_z + 0.5 * freq_z

    balance_threshold = balance.quantile(priority_quantile)
    score_threshold = score.quantile(priority_quantile)

    is_dormant = (freq <= dormant_freq_threshold) | (view["RecencyDays"] >= dormant_recency_days)
    is_priority = (~is_dormant) & (score >= score_threshold) & (balance >= balance.quantile(0.5))

    tier = pd.Series(np.where(is_dormant, "Dormant", np.where(is_priority, "Priority", "Regular")),
                      index=view.index, name="Tier")

    rules_used = {
        "method": "rule_based_composite_score",
        "inputs": [balance_col, freq_col, "RecencyDays"],
        "dormant_rule": f"{freq_col} <= {dormant_freq_threshold:.2f}  OR  RecencyDays >= {dormant_recency_days}",
        "priority_rule": (
            f"NOT dormant AND composite_score(0.5*zscore({balance_col}) + 0.5*zscore({freq_col})) "
            f">= {score_threshold:.3f} (the {priority_quantile:.0%} percentile) "
            f"AND {balance_col} >= median ({balance.quantile(0.5):,.0f})"
        ),
        "regular_rule": "everything else",
        "thresholds": {
            "dormant_freq_threshold": round(float(dormant_freq_threshold), 2),
            "dormant_recency_days": dormant_recency_days,
            "priority_score_threshold": round(float(score_threshold), 3),
            "priority_balance_floor": round(float(balance.quantile(0.5)), 2),
        },
    }
    return tier, rules_used


def ml_clustering(
    view: pd.DataFrame,
    features: list[str],
    k: int | None = None,
    k_range: range = range(3, 7),
    random_state: int = 42,
) -> dict:
    """StandardScaler -> KMeans, with automatic k selection via silhouette
    score if k is not given, plus a PCA(2D) projection for visualization."""
    X = view[features].astype(float).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k is None:
        best_k, best_score, scores = None, -1, {}
        for candidate in k_range:
            if candidate >= len(view):
                continue
            labels = KMeans(n_clusters=candidate, random_state=random_state, n_init=10).fit_predict(X_scaled)
            s = silhouette_score(X_scaled, labels)
            scores[candidate] = round(float(s), 4)
            if s > best_score:
                best_k, best_score = candidate, s
        k = best_k
    else:
        scores = {}

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids_original, columns=features)
    centroid_df.index.name = "cluster"

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    return {
        "k": int(k),
        "silhouette_score": round(float(sil), 4),
        "k_selection_scores": scores,
        "features_used": features,
        "labels": labels,
        "cluster_sizes": to_jsonable(cluster_sizes.to_dict()),
        "centroids": to_jsonable(centroid_df.round(2).to_dict(orient="index")),
        "pca_coords": coords,
        "pca_explained_variance": to_jsonable(pca.explained_variance_ratio_),
        "_scaler": scaler,
        "_model": model,
        "_X_scaled": X_scaled,
    }
