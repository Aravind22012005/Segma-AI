"""Planner intent-coverage battery.

Exercises `parse_query_offline()` (the deterministic default path, used
whenever no LLM API key is configured -- this is what CI and most local
dev runs actually hit) across a spread of realistic natural-language
phrasings for every intent in `INTENTS` except the terminal `clarify`
fallback (which is covered separately, since it's the "none of the above"
bucket rather than something to route *to*).

Each case asserts both:
  - the returned `intent` matches what a human would expect from the
    matcher logic in `parse_query_offline()` (verified by reading the
    source and running the function directly before writing assertions
    here -- see the task report for the full list), and
  - `needs_clarification is False`, i.e. the query is unambiguous enough
    that the offline planner commits to an answer rather than punting
    back to the user.

A small subset also exercises the real LLM path (`parse_query()`), guarded
by `pytest.mark.skipif` on `llm_available()` so these skip cleanly (not
fail/error) in CI or any local environment with no LLM API key configured
-- which is the norm for this repo (see backend/agent/llm_client.py).

Two previously-documented planner gaps -- "who are the priority customers"
and "show me the dormant customers" falling through to the generic clarify
fallback because `list_customers_by_tier` matching required an exact,
article-less substring -- are now fixed (see the `list_tier_regex` in
`parse_query_offline()`) and locked in as normal passing cases in
`OFFLINE_CASES` below rather than `xfail`.
"""

from __future__ import annotations

import pytest

from backend.agent.llm_client import llm_available
from backend.agent.planner import parse_query, parse_query_offline

# ---------------------------------------------------------------------------
# Main battery: (query, expected_intent, expected_params_subset)
#
# expected_params_subset is a dict of key/value pairs that must be present
# (as a subset) in the returned params -- keeps the table readable without
# requiring every case to spell out the full params dict.
# ---------------------------------------------------------------------------
OFFLINE_CASES = [
    # -- segment_customers --
    (
        "segment customers by balance and transaction frequency",
        "segment_customers",
        {"method": "rule"},
    ),
    (
        "cluster customers using kmeans",
        "segment_customers",
        {"method": "ml"},
    ),
    # -- explain_segment_basis --
    (
        "on what basis were priority customers selected",
        "explain_segment_basis",
        {"tier": "Priority"},
    ),
    (
        "how were dormant customers classified",
        "explain_segment_basis",
        {"tier": "Dormant"},
    ),
    # -- conversion_candidates --
    (
        "which regular customers can convert to priority",
        "conversion_candidates",
        {},
    ),
    (
        "who is eligible for conversion",
        "conversion_candidates",
        {},
    ),
    # -- customer_lookup --
    (
        "why does customer CUST00001 belong to their segment",
        "customer_lookup",
        {"customer_id": "CUST00001", "mode": "explain"},
    ),
    (
        "show me the profile for CUST00042",
        "customer_lookup",
        {"customer_id": "CUST00042", "mode": "profile"},
    ),
    # -- recommendation --
    (
        "recommend products for priority customers",
        "recommendation",
        {"tier": "Priority"},
    ),
    (
        "what should be done to increase engagement for dormant customers",
        "recommendation",
        {"tier": "Dormant"},
    ),
    # -- trend_query --
    (
        "has the average balance been increasing over the last 6 months",
        "trend_query",
        {},
    ),
    # -- list_customers_by_tier (the newer intent -- priority/dormant phrasings) --
    (
        "which customers are marked priority",
        "list_customers_by_tier",
        {"tier": "Priority"},
    ),
    (
        "show priority customers",
        "list_customers_by_tier",
        {"tier": "Priority"},
    ),
    (
        "show dormant customers",
        "list_customers_by_tier",
        {"tier": "Dormant"},
    ),
    (
        "list dormant customers",
        "list_customers_by_tier",
        {"tier": "Dormant"},
    ),
    (
        "who are the priority customers",
        "list_customers_by_tier",
        {"tier": "Priority"},
    ),
    (
        "show me the dormant customers",
        "list_customers_by_tier",
        {"tier": "Dormant"},
    ),
    # -- eda_missing_values --
    (
        "how many customers have missing income data",
        "eda_missing_values",
        {},
    ),
    # -- eda_correlation --
    (
        "is there a correlation between income and credit score",
        "eda_correlation",
        {},
    ),
    # -- eda_distribution --
    (
        "what is the distribution of age",
        "eda_distribution",
        {"column": "Age"},
    ),
    # -- aggregate_metric --
    (
        "what is the average monthly spend of customers",
        "aggregate_metric",
        {"metric": "MonthlySpend", "agg": "mean"},
    ),
    (
        "median transaction size",
        "aggregate_metric",
        {"metric": "AvgTxnValue", "agg": "median"},
    ),
    # -- eda_overview --
    (
        "give me an overview of the dataset",
        "eda_overview",
        {},
    ),
    (
        "how many customers do we have",
        "eda_overview",
        {},
    ),
]


@pytest.mark.parametrize(
    "query, expected_intent, expected_params_subset",
    OFFLINE_CASES,
    ids=[c[0] for c in OFFLINE_CASES],
)
def test_offline_intent_classification(query, expected_intent, expected_params_subset):
    result = parse_query_offline(query)
    assert result["intent"] == expected_intent, (
        f"query {query!r} -> intent {result['intent']!r}, expected {expected_intent!r} "
        f"(full result: {result})"
    )
    assert result["needs_clarification"] is False, (
        f"query {query!r} unexpectedly needed clarification: {result}"
    )
    for key, value in expected_params_subset.items():
        assert result["params"].get(key) == value, (
            f"query {query!r} -> params {result['params']}, expected {key}={value!r}"
        )


# ---------------------------------------------------------------------------
# Genuinely ambiguous / out-of-scope queries: the offline planner is
# *supposed* to punt back to the user here, so needs_clarification True is
# the correct assertion (not a gap).
# ---------------------------------------------------------------------------
CLARIFY_CASES = [
    "segment customers",  # no attributes given -- matcher explicitly clarifies
    "what is the weather today",  # not covered by any intent at all
]


@pytest.mark.parametrize("query", CLARIFY_CASES, ids=CLARIFY_CASES)
def test_offline_correctly_asks_for_clarification(query):
    result = parse_query_offline(query)
    assert result["intent"] == "clarify"
    assert result["needs_clarification"] is True
    assert result["clarification_question"]


# ---------------------------------------------------------------------------
# LLM path: a small subset that calls the real `parse_query()` entry point
# (not `parse_query_offline()`), so it only actually exercises the LLM when
# a provider key is configured. Skips cleanly otherwise -- reusing
# `llm_available()` (the same detection `parse_query()` itself uses) rather
# than re-deriving key-presence logic here.
# ---------------------------------------------------------------------------
LLM_CASES = [
    (
        "segment customers by balance and transaction frequency",
        "segment_customers",
    ),
    (
        "which customers are marked priority",
        "list_customers_by_tier",
    ),
    (
        "on what basis were priority customers selected",
        "explain_segment_basis",
    ),
]


@pytest.mark.skipif(
    not llm_available(),
    reason="No GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY configured in this environment.",
)
@pytest.mark.parametrize(
    "query, expected_intent", LLM_CASES, ids=[c[0] for c in LLM_CASES]
)
def test_llm_path_intent_classification(query, expected_intent):
    result = parse_query(query)
    assert result["intent"] == expected_intent, (
        f"LLM planner: query {query!r} -> intent {result['intent']!r}, "
        f"expected {expected_intent!r} (full result: {result})"
    )
    assert result["needs_clarification"] is False
