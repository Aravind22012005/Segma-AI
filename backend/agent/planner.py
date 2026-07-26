"""Planner: turns a natural-language query into a structured intent + params.

Two modes:
  - LLM mode (if an API key is configured): sends the query + a description
    of available intents/columns to the configured LLM and asks for strict
    JSON back.
  - Offline mode (default, always available): deterministic keyword/regex
    matching. Covers every example query in the problem statement plus
    common rephrasings, and is what keeps the demo 100% reliable with zero
    API keys configured.

Either path can return `needs_clarification: true` -- this is the
human-in-the-loop behaviour the spec calls out explicitly ("at any point,
if more input is needed from the user, the agent should ask").
"""

from __future__ import annotations

import re

from backend.agent.llm_client import NoLLMConfigured, call_llm_json, llm_available

INTENTS = [
    "segment_customers", "explain_segment_basis", "aggregate_metric",
    "conversion_candidates", "customer_lookup", "recommendation",
    "eda_missing_values", "eda_distribution", "eda_correlation",
    "eda_overview", "trend_query", "list_customers_by_tier", "clarify",
]

COLUMN_SYNONYMS = {
    "AvgMonthlyBalance": ["balance", "avg balance", "average balance", "balance maintained"],
    "MonthlyTxnFrequency": ["frequency", "transaction frequency", "how often", "txn frequency", "number of transactions"],
    "AvgTxnValue": ["transaction size", "transaction value", "transaction amount", "average transaction", "avg transaction", "size of transactions"],
    "MonthlySpend": ["monthly spend", "spending", "spend"],
    "AnnualIncome": ["income", "salary", "earnings"],
    "CreditScore": ["credit score", "cibil"],
    "Age": ["age"],
    "CardUtilizationPct": ["card utilization", "credit utilization", "utilization"],
    "EMI": ["emi"],
    "DebitCreditRatio": ["debit credit ratio", "debit/credit ratio", "debit to credit"],
    "NumProductsOwned": ["number of products", "products owned"],
    "AccountAgeMonths": ["account age"],
    "RecencyDays": ["recency", "days since last transaction", "last active"],
    "MaritalStatus": ["marital status", "married", "marital"],
    "EducationLevel": ["education level", "education"],
    "PriorCampaignContacts": ["campaign contacts", "prior contacts", "number of contacts"],
    "PriorCampaignOutcome": ["campaign outcome", "campaign response", "prior campaign"],
}

TIER_WORDS = ["priority", "regular", "dormant"]
AGG_WORDS = {"average": "mean", "avg": "mean", "mean": "mean", "median": "median",
             "sum": "sum", "total": "sum", "count": "count", "max": "max", "min": "min"}


def _clarify(question: str) -> dict:
    return {"intent": "clarify", "params": {}, "needs_clarification": True, "clarification_question": question}


def _extract_customer_id(q: str) -> str | None:
    m = re.search(r"cust\s*0*\d{1,6}", q, re.IGNORECASE)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(0))
    return f"CUST{int(digits):05d}"


def _extract_tier(q: str) -> str | None:
    for t in TIER_WORDS:
        if t in q:
            return t.capitalize()
    return None


def _extract_column(q: str) -> str | None:
    best, best_len = None, 0
    for col, synonyms in COLUMN_SYNONYMS.items():
        for syn in synonyms:
            if syn in q and len(syn) > best_len:
                best, best_len = col, len(syn)
    return best


def _extract_agg(q: str) -> str:
    for word, agg in AGG_WORDS.items():
        if word in q:
            return agg
    return "mean"


def parse_query_offline(query: str) -> dict:
    q = query.lower().strip()

    customer_id = _extract_customer_id(q)

    # 0. A specific customer ID + "why/belong/explain" always means customer_lookup, even
    # though such questions often also contain the words "customer" and "segment"
    # (e.g. "why does customer CUST00001 belong to their segment?") which would otherwise
    # match the segmentation trigger below.
    if customer_id and any(k in q for k in ["why", "belong", "explain"]):
        return {"intent": "customer_lookup", "params": {"customer_id": customer_id, "mode": "explain"},
                "needs_clarification": False}

    # 1. Segmentation
    segment_triggers = ["segment customer", "segment the customer", "cluster customer", "cluster the customer",
                         "divide customer", "group customer", "classify customer", "segment them",
                         "segment the customer base", "create segments", "create customer segments"]
    if any(t in q for t in segment_triggers) or (("segment" in q or "cluster" in q) and "customer" in q):
        wants_ml = any(k in q for k in ["cluster", "clustering", "kmeans", "k-means", "machine learning",
                                          "unsupervised", " ml "])
        wants_rule = ("priority" in q and "dormant" in q) or "regular" in q
        method = "rule" if (wants_rule or not wants_ml) else "ml"

        balance_col = "AvgMonthlyBalance" if any(k in q for k in COLUMN_SYNONYMS["AvgMonthlyBalance"]) else None
        freq_col = "MonthlyTxnFrequency" if any(k in q for k in COLUMN_SYNONYMS["MonthlyTxnFrequency"]) else None

        if method == "rule" and not balance_col and not freq_col:
            return _clarify(
                "Which attributes should I use to segment customers? For example: "
                "\"balance maintained and frequency of transactions\", or I can run an ML "
                "clustering across income, balance, spend and credit behaviour instead."
            )
        return {
            "intent": "segment_customers",
            "params": {"method": method, "balance_col": balance_col or "AvgMonthlyBalance",
                       "freq_col": freq_col or "MonthlyTxnFrequency"},
            "needs_clarification": False,
        }

    # 2. Explain basis of a segment
    basis_triggers = ["on what basis", "why were", "how were", "basis for", "how did you decide",
                       "what criteria", "what basis", "how are", "how is"]
    if any(t in q for t in basis_triggers) and (_extract_tier(q) or "segment" in q or "cluster" in q):
        return {"intent": "explain_segment_basis", "params": {"tier": _extract_tier(q)}, "needs_clarification": False}

    # 3. Conversion candidates
    if "convert" in q or "conversion" in q or (
        "regular" in q and "priority" in q and any(k in q for k in ["can be", "upgrade", "become", "move to", "potential"])
    ):
        return {"intent": "conversion_candidates", "params": {}, "needs_clarification": False}

    # 4. Customer-specific
    if customer_id:
        mode = "explain" if any(k in q for k in ["why", "belong", "explain", "basis"]) else "profile"
        return {"intent": "customer_lookup", "params": {"customer_id": customer_id, "mode": mode},
                "needs_clarification": False}

    # 5. Recommendations
    if any(k in q for k in ["recommend", "recommendation", "cross-sell", "cross sell", "upsell", "up-sell",
                              "what should be done", "what product", "which product", "next best action"]):
        return {"intent": "recommendation", "params": {"tier": _extract_tier(q)}, "needs_clarification": False}

    # 6. Trend / time-based
    if any(k in q for k in ["steadily increased", "steadily increasing", "trend", "increasing balance",
                              "declining balance", "decreasing balance", "last six month", "last 6 month",
                              "compare spending", "over time", "past year"]):
        return {"intent": "trend_query", "params": {}, "needs_clarification": False}

    # 6.5 List customers by tier -- standalone "who/which/show/list ... priority/dormant/regular
    # customers" requests (e.g. "which customers are marked priority", "show priority customers").
    # Must come after the more specific triggers above (segmentation, explain-basis,
    # conversion-candidates, customer-lookup, recommendation, trend) so it doesn't steal their
    # queries, and before the AGG_WORDS/overview fallthrough below so these phrases don't fall
    # through to the generic clarify at the end of this function.
    # Literal "marked"/"in" phrasings don't need the tier word directly
    # adjacent (it's picked up anywhere in `q` by `_extract_tier` below).
    list_tier_triggers = [
        "which customers are marked", "which customers are in",
        "who is marked", "who are marked",
    ]
    # "show/show me/list/who is/who are (the|a|an) <tier> customer(s)" -- tolerate
    # an optional article and an optional "me" between the trigger verb and the
    # tier word, since real phrasings like "show me the dormant customers" or
    # "who are the priority customers" are just as unambiguous as the
    # article-less forms.
    list_tier_regex = re.search(
        r"\b(?:(?:show|list)(?:\s+me)?|who\s+(?:is|are))\s+"
        r"(?:the|a|an)?\s*(?:priority|regular|dormant)(?:\s+customers?)?\b",
        q,
    )
    standalone_tier_list = re.fullmatch(r"(priority|regular|dormant)\s+customers?", q.strip())
    if (
        any(t in q for t in list_tier_triggers) or list_tier_regex or standalone_tier_list
    ) and not any(k in q for k in AGG_WORDS):
        tier = _extract_tier(q)
        if not tier:
            return _clarify("Which tier would you like to list -- Priority, Regular, or Dormant?")
        return {"intent": "list_customers_by_tier", "params": {"tier": tier}, "needs_clarification": False}

    # 7. EDA
    if "missing" in q or "null" in q or "nan" in q:
        return {"intent": "eda_missing_values", "params": {}, "needs_clarification": False}

    if "correlat" in q:
        return {"intent": "eda_correlation", "params": {}, "needs_clarification": False}

    if "distribut" in q or "histogram" in q or "spread of" in q:
        col = _extract_column(q)
        if not col:
            return _clarify("Which column's distribution would you like to see? (e.g. Age, AnnualIncome, AvgMonthlyBalance)")
        return {"intent": "eda_distribution", "params": {"column": col}, "needs_clarification": False}

    if any(k in q for k in AGG_WORDS) or "how much" in q or "what is the" in q:
        metric = _extract_column(q)
        group_by = "Tier" if (_extract_tier(q) or "segment" in q or "tier" in q) else None
        if not metric:
            return _clarify(
                "Which metric would you like aggregated? For example: transaction size, "
                "monthly spend, balance, income, or credit score."
            )
        return {
            "intent": "aggregate_metric",
            "params": {"metric": metric, "group_by": group_by or "Tier", "agg": _extract_agg(q)},
            "needs_clarification": False,
        }

    if any(k in q for k in ["overview", "summary", "describe the data", "tell me about the dataset",
                              "how many customers", "what data"]):
        return {"intent": "eda_overview", "params": {}, "needs_clarification": False}

    return _clarify(
        "I'm not sure what you're asking. Try things like: \"segment customers by balance and "
        "transaction frequency\", \"on what basis were priority customers selected\", \"average "
        "transaction size for priority customers\", or \"which regular customers can convert to priority\"."
    )


LLM_SYSTEM_PROMPT = """You are the planning module of a retail-banking customer segmentation agent.
Given a user's natural-language question, output STRICT JSON (no prose, no markdown fences) with this shape:

{
  "intent": one of ["segment_customers","explain_segment_basis","aggregate_metric","conversion_candidates",
                     "customer_lookup","recommendation","eda_missing_values","eda_distribution",
                     "eda_correlation","eda_overview","trend_query","list_customers_by_tier","clarify"],
  "params": { ...intent-specific... },
  "needs_clarification": boolean,
  "clarification_question": string (only if needs_clarification is true)
}

Intent param shapes:
- segment_customers: {"method": "rule"|"ml", "balance_col": "AvgMonthlyBalance", "freq_col": "MonthlyTxnFrequency"}
- explain_segment_basis: {"tier": "Priority"|"Regular"|"Dormant"|null}
- aggregate_metric: {"metric": <column name>, "group_by": "Tier", "agg": "mean"|"median"|"sum"|"count"|"max"|"min"}
- conversion_candidates: {}
- customer_lookup: {"customer_id": "CUSTxxxxx", "mode": "profile"|"explain"}
- recommendation: {"tier": "Priority"|"Regular"|"Dormant"|null}
- eda_distribution: {"column": <column name>}
- trend_query: {}
- list_customers_by_tier: {"tier": "Priority"|"Regular"|"Dormant"}
  Example phrasings: "which customers are marked priority", "show me the dormant customers",
  "list the priority customers".

Available unified-view columns: Age, Gender, Occupation, AnnualIncome, MaritalStatus, EducationLevel,
City, CityTier, AccountType, AccountAgeMonths, AvgMonthlyBalance, CreditScore, LoanType, LoanStatus, EMI,
HasCreditCard, CardUtilizationPct, PriorCampaignContacts, DaysSinceLastCampaignContact,
PriorCampaignOutcome (Success/Failure/Other/Never Contacted), TxnCount_Total, AvgTxnValue, MedianTxnValue,
DebitCreditRatio, MonthlyTxnFrequency, MonthlySpend, RecencyDays, BalanceTrendSlope, SpendChangePct_6M,
TopMerchantCategory, SavingsAccount, CreditCard, FixedDeposit, Insurance, Investment, Loan,
NumProductsOwned, Tier (Priority/Regular/Dormant, only exists after segmentation has been run).
MaritalStatus, EducationLevel, and the PriorCampaign* fields are real data from the UCI Bank Marketing
dataset, bootstrapped per customer -- treat them as first-class queryable/groupable columns.

If the question is genuinely ambiguous (e.g. "segment customers" with no criteria given), set
needs_clarification true and ask a short, specific clarifying question instead of guessing.
Output ONLY the JSON object.
"""


def parse_query(query: str) -> dict:
    """Public entry point. Tries the LLM planner if configured, always falls
    back to the deterministic offline planner on any failure."""
    if llm_available():
        try:
            result = call_llm_json(LLM_SYSTEM_PROMPT, f"User question: {query}")
            if "intent" in result and result["intent"] in INTENTS:
                result.setdefault("params", {})
                result.setdefault("needs_clarification", False)
                return result
            print(f"[planner] LLM returned an unrecognised intent {result.get('intent')!r} -- falling back to offline planner.")
        except NoLLMConfigured:
            pass
        except Exception as e:
            # Falling back silently is the right behaviour for the demo, but a bare
            # `except: pass` here once hid a real bug (wrong model name) for a while --
            # so at least surface it in the server console.
            print(f"[planner] LLM call failed ({type(e).__name__}: {e}) -- falling back to offline planner.")
    return parse_query_offline(query)
