"""Explainer: turns raw tool JSON output into a human-readable answer.

LLM mode (if configured) phrases the same numbers more fluently. Offline
mode uses deterministic templates -- still fully readable, no LLM required.
This is the *only* place free-form language is generated; the numbers
themselves were never touched by an LLM.
"""

from __future__ import annotations

import json

from backend.agent.llm_client import NoLLMConfigured, call_llm_text, llm_available

EXPLAINER_SYSTEM_PROMPT = """You are the explanation module of a retail-banking segmentation agent.
You will be given the user's original question and a JSON payload of tool results that were computed
deterministically (pandas/scikit-learn) -- NOT by you. Write a clear, concise, business-friendly answer
(3-8 sentences, use bullet points for lists) that reports the numbers faithfully. Never invent numbers
that aren't in the JSON. If the JSON contains a "question" key (a clarification request), just ask that
question back to the user plainly. Do not mention JSON or tools in your answer."""


def _offline_explain(query: str, plan: dict, exec_result: dict) -> str:
    intent = exec_result.get("intent")
    result = exec_result.get("result", {})

    if intent == "clarify":
        return result.get("question", "Could you clarify what you'd like me to do?")

    if intent == "segment_customers":
        if result.get("method") == "rule_based":
            counts = result["tier_counts"]
            lines = [f"Segmented the customer base using the rule you specified (balance maintained + "
                     f"transaction frequency):"]
            for tier in ["Priority", "Regular", "Dormant"]:
                if tier in counts:
                    lines.append(f"- **{tier}**: {counts[tier]:,} customers")
            lines.append("A CSV of customer -> segment assignments is ready.")
            lines.append(f"Rule used -- Dormant: {result['rules_used']['dormant_rule']}; "
                         f"Priority: {result['rules_used']['priority_rule']}.")
            return "\n".join(lines)
        else:
            sizes = result["cluster_sizes"]
            lines = [f"Ran K-Means clustering (k={result['k']}, silhouette score={result['silhouette_score']}) "
                     f"on {len(result['features_used'])} selected features: {', '.join(result['features_used'])}."]
            for cid, label in result["cluster_label_map"].items():
                lines.append(f"- Cluster {cid} (**{label}**): {sizes.get(str(cid), sizes.get(cid, '?')):,} customers")
            lines.append("A CSV of customer -> cluster assignments is ready.")
            return "\n".join(lines)

    if intent == "explain_segment_basis":
        if "dormant_rule" in result:
            return (f"Segments were assigned with an explicit rule, not a black-box model:\n"
                     f"- **Dormant**: {result['dormant_rule']}\n"
                     f"- **Priority**: {result['priority_rule']}\n"
                     f"- **Regular**: {result['regular_rule']}")
        if "clusters" in result:
            lines = ["Here's what distinguishes each discovered cluster:"]
            for c in result["clusters"]:
                feats = ", ".join(f"{f['feature']} ({f['direction']}, z={f['z_score']})" for f in c["distinguishing_features"][:3])
                lines.append(f"- Cluster {c['cluster_id']} ({c['cluster_size']} customers): {feats}")
            return "\n".join(lines)
        feats = ", ".join(f"{f['feature']} ({f['direction']}, z={f['z_score']})" for f in result.get("distinguishing_features", [])[:5])
        return f"Cluster {result.get('cluster_id')} ({result.get('cluster_size')} customers) is distinguished by: {feats}."

    if intent == "aggregate_metric":
        lines = [f"**{result['aggregation']}({result['metric']})** grouped by {result['group_by']}:"]
        for group, val in result["result"].items():
            n = result["group_sizes"].get(group, "?")
            lines.append(f"- {group}: {val:,.2f}  (n={n})")
        return "\n".join(lines)

    if intent == "conversion_candidates":
        lines = [f"Found {result['n_candidates']} Regular customers closest to crossing into Priority "
                 f"(score threshold used: {result['score_threshold_used']}):"]
        for c in result["candidates"][:10]:
            actions = "; ".join(c["suggested_actions"])
            lines.append(f"- {c['customer_id']}: balance={c['current_balance']:,.0f}, "
                         f"freq={c['current_txn_frequency']:.1f}/mo, gap={c['gap_to_priority_score']:.3f} -> {actions}")
        if result['n_candidates'] > 10:
            lines.append(f"...and {result['n_candidates'] - 10} more in the full CSV.")
        return "\n".join(lines)

    if intent == "customer_lookup":
        profile = result["profile"]
        lines = [f"**{profile.get('CustomerID')}** -- {profile.get('Age')}y {profile.get('Gender')}, "
                 f"{profile.get('Occupation')} in {profile.get('City')}. "
                 f"Tier: {profile.get('Tier', 'not segmented yet')}. "
                 f"Balance: {profile.get('AvgMonthlyBalance', 0):,.0f}, "
                 f"Income: {profile.get('AnnualIncome', 0):,.0f}, "
                 f"Credit score: {profile.get('CreditScore')}, "
                 f"Txn frequency: {profile.get('MonthlyTxnFrequency', 0):.1f}/mo."]
        if "explanation" in result:
            standout = ", ".join(f"{f['feature']}={f['value']}" for f in result["explanation"]["customer_standout_features"][:3])
            lines.append(f"Belongs to cluster {result['explanation']['assigned_cluster']}, standing out on: {standout}.")
        return "\n".join(lines)

    if intent == "recommendation":
        recs = result["recommendations"]
        lines = [f"Recommendations for {result['n_customers']} customer(s):"]
        for r in recs[:10]:
            lines.append(f"- {r['customer_id']} ({r['tier']}): {'; '.join(r['recommendations'])}")
        if result['n_customers'] > 10:
            lines.append(f"...and {result['n_customers'] - 10} more in the full CSV.")
        return "\n".join(lines)

    if intent == "eda_missing_values":
        if result["clean"]:
            return f"No missing values found across {result['total_rows']:,} rows -- the dataset is clean."
        lines = [f"Missing values found (of {result['total_rows']:,} rows):"]
        for col, info in result["columns_with_missing"].items():
            lines.append(f"- {col}: {info['missing_count']:,} missing ({info['missing_pct']}%)")
        return "\n".join(lines)

    if intent == "eda_distribution":
        if result.get("type") == "numeric":
            return (f"**{result['column']}** distribution -- mean={result['mean']:.1f}, "
                    f"median={result['median']:.1f}, std={result['std']:.1f}, "
                    f"range=[{result['min']:.1f}, {result['max']:.1f}].")
        return f"**{result['column']}** top categories: " + ", ".join(
            f"{c} ({n})" for c, n in zip(result["categories"][:8], result["counts"][:8]))

    if intent == "eda_correlation":
        lines = ["Strongest correlations in the data:"]
        for p in result["top_correlations"][:6]:
            lines.append(f"- {p['feature_a']} <-> {p['feature_b']}: {p['corr']:+.2f}")
        return "\n".join(lines)

    if intent == "eda_overview":
        lines = ["Numeric feature summary (mean / std / min / max):"]
        for col, stats in list(result.items())[:8]:
            lines.append(f"- {col}: mean={stats.get('mean')}, std={stats.get('std')}, "
                         f"min={stats.get('min')}, max={stats.get('max')}")
        return "\n".join(lines)

    if intent == "trend_query":
        lines = ["**Customers with steadily increasing balance:** " +
                 ", ".join(r["CustomerID"] for r in result["steadily_increasing_balance"][:8])]
        lines.append("**Customers with steadily declining balance:** " +
                     ", ".join(r["CustomerID"] for r in result["steadily_declining_balance"][:8]))
        lines.append("**Spend up most vs. prior 6 months:** " +
                     ", ".join(r["CustomerID"] for r in result["spend_increased_last_6m"][:8]))
        return "\n".join(lines)

    return "Done."


def explain(query: str, plan: dict, exec_result: dict) -> str:
    if llm_available():
        try:
            payload = json.dumps({"question": query, "plan": plan, "tool_result": exec_result}, default=str)[:12000]
            return call_llm_text(EXPLAINER_SYSTEM_PROMPT, payload)
        except NoLLMConfigured:
            pass
        except Exception as e:
            print(f"[explainer] LLM call failed ({type(e).__name__}: {e}) -- falling back to offline explainer.")
    return _offline_explain(query, plan, exec_result)
