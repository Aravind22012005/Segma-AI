"""
Feature Engineering ("Feature Engine" in the architecture diagram).

Turns Customer + Transactions + Products into one "Unified Customer View" --
a single wide dataframe that every downstream tool (EDA, segmentation,
explainability, recommendation) reads from. This is the only place
transaction-level aggregation happens, so every tool sees consistent numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _linear_trend_slope(y: np.ndarray) -> float:
    """Slope of y over its own index, normalised by mean(|y|) so it's
    comparable across customers with very different balance scales."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    scale = np.abs(y).mean() or 1.0
    return float(slope / scale)


def build_monthly_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Per-customer, per-calendar-month net cash flow and volume.

    Fully vectorized: credit/debit amounts are split into two numeric
    columns up front so the groupby-agg never needs a per-group Python
    callback (with ~36k CustomerID x YearMonth groups, a lambda that
    re-indexes the parent frame per group is the difference between this
    finishing in ~1s vs. taking minutes).
    """
    df = transactions.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M")

    is_credit = df["DebitCredit"] == "Credit"
    df["CreditAmt"] = np.where(is_credit, df["Amount"], 0.0)
    df["DebitAmt"] = np.where(is_credit, 0.0, df["Amount"])
    df["SignedAmount"] = df["CreditAmt"] - df["DebitAmt"]

    monthly = (
        df.groupby(["CustomerID", "YearMonth"])
        .agg(
            TxnCount=("TransactionID", "count"),
            Credits=("CreditAmt", "sum"),
            Debits=("DebitAmt", "sum"),
            NetFlow=("SignedAmount", "sum"),
        )
        .reset_index()
        .sort_values(["CustomerID", "YearMonth"])
    )
    return monthly


def build_unified_customer_view(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """The single feature table every tool operates on."""
    df = transactions.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    reference_date = df["Date"].max()

    is_credit = df["DebitCredit"] == "Credit"
    df["CreditAmt"] = np.where(is_credit, df["Amount"], 0.0)
    df["DebitAmt"] = np.where(is_credit, 0.0, df["Amount"])
    df["IsDebit"] = (~is_credit).astype(int)
    df["IsCredit"] = is_credit.astype(int)

    agg = df.groupby("CustomerID").agg(
        TxnCount_Total=("TransactionID", "count"),
        TotalCredit=("CreditAmt", "sum"),
        TotalDebit=("DebitAmt", "sum"),
        AvgTxnValue=("Amount", "mean"),
        MedianTxnValue=("Amount", "median"),
        LastTxnDate=("Date", "max"),
        FirstTxnDate=("Date", "min"),
        DebitCount=("IsDebit", "sum"),
        CreditCount=("IsCredit", "sum"),
    )
    agg[["DebitCount", "CreditCount"]] = agg[["DebitCount", "CreditCount"]].fillna(0)
    agg["DebitCreditRatio"] = agg["DebitCount"] / agg["CreditCount"].replace(0, np.nan)
    agg["DebitCreditRatio"] = agg["DebitCreditRatio"].fillna(agg["DebitCount"])

    span_days = (agg["LastTxnDate"] - agg["FirstTxnDate"]).dt.days.clip(lower=1)
    months_active = (span_days / 30).clip(lower=1)
    agg["MonthlyTxnFrequency"] = agg["TxnCount_Total"] / months_active
    agg["MonthlySpend"] = agg["TotalDebit"] / months_active
    agg["RecencyDays"] = (reference_date - agg["LastTxnDate"]).dt.days

    # top merchant category per customer
    top_cat = (
        df.groupby(["CustomerID", "Category"])["TransactionID"]
        .count()
        .reset_index()
        .sort_values(["CustomerID", "TransactionID"], ascending=[True, False])
        .drop_duplicates("CustomerID")
        .set_index("CustomerID")["Category"]
        .rename("TopMerchantCategory")
    )
    agg = agg.join(top_cat)

    # spend split by category (share of total debit spend), used for persona summaries.
    # NOTE: deliberately NOT done via groupby(...).apply(lambda s: s.to_dict()) -- when the
    # applied function returns a dict, pandas broadcasts/expands it back across the group's
    # sub-index instead of keeping one dict per group, silently exploding the row count
    # (bit us during testing: 1,500 customers -> 23M "unified view" rows). A plain per-customer
    # loop is unambiguous and, at 1,500 customers, plenty fast.
    cat_totals = df[df["IsDebit"] == 1].groupby(["CustomerID", "Category"])["Amount"].sum()
    cat_share_map = {}
    for cust_id, sub in cat_totals.groupby(level=0):
        sub = sub.droplevel(0)
        total = sub.sum()
        cat_share_map[cust_id] = (sub / total).round(3).to_dict() if total > 0 else {}
    agg["CategorySpendShare"] = pd.Series(agg.index.map(cat_share_map.get), index=agg.index)

    # balance trend: slope of cumulative net cash flow over months (normalised).
    # Only ~1500 groups (one per customer) so a Python-level apply here is fine.
    monthly = build_monthly_aggregates(transactions)
    trend = (
        monthly.groupby("CustomerID")["NetFlow"]
        .apply(lambda s: _linear_trend_slope(np.cumsum(s.values)))
        .rename("BalanceTrendSlope")
    )
    agg = agg.join(trend)

    # last-6-months vs prior-6-months spend, for "compare spending over the last 6 months".
    # Vectorized via rank-from-end instead of a per-customer apply (36k monthly rows).
    monthly_sorted = monthly.sort_values(["CustomerID", "YearMonth"]).copy()
    rank_from_end = monthly_sorted.groupby("CustomerID").cumcount(ascending=False)
    spend_last6 = (
        monthly_sorted.loc[rank_from_end < 6].groupby("CustomerID")["Debits"].sum().rename("Spend_Last6M")
    )
    spend_prior6 = (
        monthly_sorted.loc[(rank_from_end >= 6) & (rank_from_end < 12)]
        .groupby("CustomerID")["Debits"].sum().rename("Spend_Prior6M")
    )
    windowed = pd.concat([spend_last6, spend_prior6], axis=1)
    agg = agg.join(windowed)
    agg["SpendChangePct_6M"] = np.where(
        agg["Spend_Prior6M"] > 0,
        (agg["Spend_Last6M"] - agg["Spend_Prior6M"]) / agg["Spend_Prior6M"] * 100,
        0.0,
    )

    agg = agg.reset_index()

    view = customers.merge(agg, on="CustomerID", how="left").merge(products, on="CustomerID", how="left")

    # customers with zero transactions in the window (fully dormant) -> fill sensible zeros
    numeric_fill_cols = [
        "TxnCount_Total", "TotalCredit", "TotalDebit", "AvgTxnValue", "MedianTxnValue",
        "DebitCount", "CreditCount", "DebitCreditRatio", "MonthlyTxnFrequency",
        "MonthlySpend", "BalanceTrendSlope", "Spend_Last6M", "Spend_Prior6M", "SpendChangePct_6M",
    ]
    for c in numeric_fill_cols:
        if c in view.columns:
            view[c] = view[c].fillna(0)
    view["RecencyDays"] = view["RecencyDays"].fillna(9999)
    view["TopMerchantCategory"] = view["TopMerchantCategory"].fillna("None")

    return view


UNIFIED_VIEW_NUMERIC_FEATURES = [
    "Age", "AnnualIncome", "AvgMonthlyBalance", "AccountAgeMonths", "CreditScore",
    "EMI", "CardUtilizationPct", "NumProductsOwned",
    "TxnCount_Total", "AvgTxnValue", "MedianTxnValue", "DebitCreditRatio",
    "MonthlyTxnFrequency", "MonthlySpend", "RecencyDays", "BalanceTrendSlope",
    "SpendChangePct_6M",
]
