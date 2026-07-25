"""Recommendation Tool: rule-based cross-sell/up-sell + retention logic +
Regular -> Priority conversion-candidate identification.

Deliberately rule-based (not ML) per the problem statement's "basic
rule-based or ML-based recommendation engine" option -- rules are easy to
explain to a judge/relationship-manager, which matters more than marginal
lift for this use case.
"""

from __future__ import annotations

import pandas as pd

from backend.json_utils import to_jsonable

PRODUCT_COLS = ["SavingsAccount", "CreditCard", "FixedDeposit", "Insurance", "Investment", "Loan"]


def _customer_recommendations(row: pd.Series, tier: str) -> list[str]:
    recs = []
    if tier == "Priority":
        if row.get("CreditCard", 0) == 0:
            recs.append("Offer a premium rewards credit card (high balance, no card on file)")
        if row.get("Investment", 0) == 0:
            recs.append("Introduce a wealth management / mutual fund portfolio")
        if row.get("Insurance", 0) == 0:
            recs.append("Cross-sell a premium life/health insurance plan")
        if row.get("FixedDeposit", 0) == 0 and row.get("AvgMonthlyBalance", 0) > 200_000:
            recs.append("Recommend a Fixed Deposit to lock in idle surplus balance")
    elif tier == "Regular":
        if row.get("Age", 99) < 32 and row.get("Investment", 0) == 0:
            recs.append("Offer a small-ticket SIP / micro-investment plan (young, high txn frequency profile)")
        if row.get("FixedDeposit", 0) == 0 and row.get("AvgMonthlyBalance", 0) > 75_000:
            recs.append("Recommend a Fixed Deposit for surplus balance")
        if row.get("Insurance", 0) == 0:
            recs.append("Cross-sell a term insurance plan")
    elif tier == "Dormant":
        recs.append("Reactivation campaign: cashback/fee waiver on next 3 transactions")
        recs.append("Personal outreach call to understand disengagement reason")

    if row.get("CreditCard", 0) == 1 and row.get("CardUtilizationPct", 0) >= 70:
        recs.append("Risk flag: high card utilization -- proactive credit limit review or balance-transfer offer")
    if row.get("LoanStatus") == "Active" and row.get("CreditScore", 900) < 620 and row.get("EMI", 0) > 0:
        recs.append("Risk flag: low credit score with active EMI -- consider debt consolidation outreach")
    if not recs:
        recs.append("No immediate action -- well-served by current product mix")
    return recs


def generate_recommendations(view: pd.DataFrame, tier_col: str = "Tier", id_col: str = "CustomerID") -> dict:
    out = []
    for _, row in view.iterrows():
        out.append({
            "customer_id": row[id_col],
            "tier": row.get(tier_col, "Unknown"),
            "recommendations": _customer_recommendations(row, row.get(tier_col, "Unknown")),
        })
    return to_jsonable({"n_customers": len(out), "recommendations": out})


def conversion_candidates(
    view: pd.DataFrame,
    thresholds: dict,
    tier_col: str = "Tier",
    balance_col: str = "AvgMonthlyBalance",
    freq_col: str = "MonthlyTxnFrequency",
    top_n: int = 25,
) -> dict:
    """Regular customers closest to crossing into Priority, ranked by gap-to-threshold."""
    regular = view[view[tier_col] == "Regular"].copy()
    if regular.empty:
        return to_jsonable({"n_candidates": 0, "candidates": []})

    balance = view[balance_col].astype(float)
    freq = view[freq_col].astype(float)
    balance_mean, balance_std = balance.mean(), (balance.std() or 1)
    freq_mean, freq_std = freq.mean(), (freq.std() or 1)

    regular["_balance_z"] = (regular[balance_col] - balance_mean) / balance_std
    regular["_freq_z"] = (regular[freq_col] - freq_mean) / freq_std
    regular["_score"] = 0.5 * regular["_balance_z"] + 0.5 * regular["_freq_z"]

    score_threshold = thresholds.get("thresholds", {}).get("priority_score_threshold", regular["_score"].quantile(0.9))
    regular["_gap"] = score_threshold - regular["_score"]
    candidates = regular.sort_values("_gap").head(top_n)

    results = []
    for _, row in candidates.iterrows():
        actions = []
        if row[balance_col] < balance.quantile(0.5):
            actions.append("Encourage higher average balance maintenance (auto-sweep from linked accounts, salary account migration)")
        if row[freq_col] < freq.median():
            actions.append("Drive transaction frequency via UPI cashback / debit card usage incentives")
        if row.get("Investment", 0) == 0:
            actions.append("Cross-sell an investment product to deepen the relationship")
        if not actions:
            actions.append("Very close to threshold -- relationship manager outreach with a Priority-tier preview offer")
        results.append({
            "customer_id": row["CustomerID"],
            "current_balance": round(float(row[balance_col]), 2),
            "current_txn_frequency": round(float(row[freq_col]), 2),
            "gap_to_priority_score": round(float(row["_gap"]), 3),
            "suggested_actions": actions,
        })

    return to_jsonable({"n_candidates": len(results), "score_threshold_used": round(float(score_threshold), 3), "candidates": results})
