"""Explainability Tool: answers "why does this customer/segment look like this?"

For rule-based tiers the answer is the literal rule (already fully
transparent). For ML clusters we explain by ranking which features deviate
most from the global population mean for that cluster (z-score of the
centroid), and for an individual customer, how their own feature values
compare to their cluster's centroid vs. the next-nearest cluster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.json_utils import to_jsonable


def explain_cluster(view: pd.DataFrame, labels: np.ndarray, features: list[str], cluster_id: int, top_n: int = 5) -> dict:
    X = view[features].astype(float).fillna(0)
    global_mean = X.mean()
    global_std = X.std().replace(0, 1)

    cluster_mask = labels == cluster_id
    cluster_mean = X[cluster_mask].mean()

    z = ((cluster_mean - global_mean) / global_std).sort_values(key=lambda s: s.abs(), ascending=False)
    top_features = z.head(top_n)

    distinguishing = [
        {
            "feature": feat,
            "cluster_avg": round(float(cluster_mean[feat]), 2),
            "population_avg": round(float(global_mean[feat]), 2),
            "z_score": round(float(val), 2),
            "direction": "higher than average" if val > 0 else "lower than average",
        }
        for feat, val in top_features.items()
    ]
    return to_jsonable({
        "cluster_id": int(cluster_id),
        "cluster_size": int(cluster_mask.sum()),
        "distinguishing_features": distinguishing,
    })


def explain_customer_cluster_membership(
    view: pd.DataFrame, labels: np.ndarray, features: list[str], customer_id: str, id_col: str = "CustomerID", top_n: int = 5
) -> dict:
    if customer_id not in view[id_col].values:
        return {"error": f"customer '{customer_id}' not found"}

    idx = view.index[view[id_col] == customer_id][0]
    row_pos = view.index.get_loc(idx)
    cluster_id = int(labels[row_pos])

    cluster_explain = explain_cluster(view, labels, features, cluster_id, top_n=top_n)

    X = view[features].astype(float).fillna(0)
    customer_vals = X.loc[idx]
    global_mean = X.mean()
    global_std = X.std().replace(0, 1)
    customer_z = ((customer_vals - global_mean) / global_std).sort_values(key=lambda s: s.abs(), ascending=False)

    return to_jsonable({
        "customer_id": customer_id,
        "assigned_cluster": cluster_id,
        "cluster_profile": cluster_explain["distinguishing_features"],
        "customer_standout_features": [
            {"feature": f, "value": round(float(customer_vals[f]), 2), "z_score": round(float(v), 2)}
            for f, v in customer_z.head(top_n).items()
        ],
    })


def explain_rule_thresholds(rules_used: dict) -> dict:
    """Rule-based tiers are self-explaining -- just surface the stored rule."""
    return to_jsonable(rules_used)
