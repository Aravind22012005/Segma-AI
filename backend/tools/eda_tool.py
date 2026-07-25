"""EDA Tool: automated + query-driven exploratory data analysis on the
unified customer view. Deterministic pandas/numpy only -- no LLM calls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.json_utils import to_jsonable


def missing_value_report(df: pd.DataFrame) -> dict:
    total = len(df)
    miss = df.isna().sum()
    report = {
        col: {"missing_count": int(miss[col]), "missing_pct": round(float(miss[col]) / total * 100, 2)}
        for col in df.columns if miss[col] > 0
    }
    return to_jsonable({
        "total_rows": total,
        "columns_with_missing": report,
        "clean": len(report) == 0,
    })


def numeric_summary(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if columns:
        numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]
    desc = numeric_df.describe().T
    desc["skew"] = numeric_df.skew()
    return to_jsonable(desc.round(2).to_dict(orient="index"))


def distribution(df: pd.DataFrame, column: str, bins: int = 10) -> dict:
    if column not in df.columns:
        return {"error": f"column '{column}' not found"}
    series = df[column].dropna()
    if pd.api.types.is_numeric_dtype(series):
        counts, edges = np.histogram(series, bins=bins)
        return to_jsonable({
            "type": "numeric",
            "column": column,
            "bins": [f"{edges[i]:.1f} - {edges[i+1]:.1f}" for i in range(len(edges) - 1)],
            "counts": counts,
            "mean": series.mean(), "median": series.median(), "std": series.std(),
            "min": series.min(), "max": series.max(),
        })
    vc = series.value_counts().head(20)
    return to_jsonable({
        "type": "categorical", "column": column,
        "categories": vc.index.tolist(), "counts": vc.values,
    })


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if columns:
        numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]
    corr = numeric_df.corr().round(3)
    # also surface the strongest pairwise relationships for a quick narrative
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return to_jsonable({
        "matrix": corr.to_dict(),
        "top_correlations": [{"feature_a": a, "feature_b": b, "corr": v} for a, b, v in pairs[:10]],
    })


def value_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> dict:
    if column not in df.columns:
        return {"error": f"column '{column}' not found"}
    vc = df[column].value_counts().head(top_n)
    return to_jsonable({"column": column, "categories": vc.index.tolist(), "counts": vc.values})


def groupby_metric(
    df: pd.DataFrame, group_by: str, metric_col: str, agg: str = "mean", filter_query: str | None = None
) -> dict:
    working = df
    if filter_query:
        try:
            working = working.query(filter_query)
        except Exception as e:
            return {"error": f"invalid filter '{filter_query}': {e}"}
    if group_by not in working.columns or metric_col not in working.columns:
        return {"error": f"columns '{group_by}' or '{metric_col}' not found"}
    result = working.groupby(group_by)[metric_col].agg(agg).round(2)
    counts = working.groupby(group_by).size()
    return to_jsonable({
        "group_by": group_by, "metric": metric_col, "aggregation": agg,
        "result": result.to_dict(), "group_sizes": counts.to_dict(),
    })


def dataset_overview(customers: pd.DataFrame, transactions: pd.DataFrame, products: pd.DataFrame) -> dict:
    return to_jsonable({
        "n_customers": len(customers),
        "n_transactions": len(transactions),
        "date_range": [transactions["Date"].min(), transactions["Date"].max()],
        "avg_income": customers["AnnualIncome"].mean(),
        "avg_balance": customers["AvgMonthlyBalance"].mean(),
        "avg_credit_score": customers["CreditScore"].mean(),
        "city_distribution": customers["City"].value_counts().head(8).to_dict(),
        "occupation_distribution": customers["Occupation"].value_counts().to_dict(),
        "product_ownership_rate": {
            c: round(float(products[c].mean()) * 100, 1)
            for c in ["SavingsAccount", "CreditCard", "FixedDeposit", "Insurance", "Investment", "Loan"]
        },
    })
