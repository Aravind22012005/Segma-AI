"""End-to-end tests for the executor-autonomy bug fix.

Prior to this fix, several executor functions (`_explain_basis`,
`_aggregate_metric`, `_conversion_candidates`, `_recommendation`) refused to
answer segmentation-dependent questions on a session that hadn't been
segmented yet, returning a "run segmentation first" / "click a button"
clarify response instead of just auto-running rule-based tiering and
proceeding. A related gap meant phrasings like "which customers are marked
priority" fell through the planner to a generic clarify with no intent at
all -- this was closed by adding a new `list_customers_by_tier` intent
end-to-end (planner + executor).

These tests exercise the FULL pipeline through the real FastAPI app --
Planner -> Executor -> response -- via `fastapi.testclient.TestClient`
against `backend.main.app`'s `/api/chat` endpoint, each on a freshly
generated session_id with no prior segmentation. This proves the fix works
through the real HTTP-shaped code path (not just that the underlying
tool functions are individually correct -- see test_segmentation_tool.py
for that level of coverage).

Refusal/clarification text to guard against, as seen in the pre-fix
behaviour: `"intent": "clarify"`, `needs_clarification: true`, and phrases
like "run segmentation first", "click", "ask me to segment".
"""

from __future__ import annotations


def _assert_not_refused(data: dict) -> None:
    """Shared assertion: the response must be a real answer, not a punt back
    to the user to segment manually first."""
    assert data["intent"] != "clarify", f"unexpected clarify response: {data}"
    assert data["needs_clarification"] is False, f"unexpected needs_clarification: {data}"
    answer_lower = data["answer"].lower()
    for phrase in ["run segmentation first", "click a button", "ask me to segment",
                   "segment the customers first", "please segment"]:
        assert phrase not in answer_lower, f"found refusal phrase {phrase!r} in answer: {data['answer']!r}"


class TestExplainBasisAutonomy:
    """_explain_basis must auto-run rule-based tiering on a fresh session
    instead of refusing to explain a basis that doesn't exist yet."""

    def test_explain_basis_on_fresh_session_auto_segments(self, app_client, session_id):
        resp = app_client.post(
            "/api/chat",
            json={"session_id": session_id, "query": "on what basis were priority customers selected"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "explain_segment_basis"
        _assert_not_refused(data)

        result = data["result"]
        assert result["method"] == "rule_based_composite_score"
        for key in ["inputs", "dormant_rule", "priority_rule", "regular_rule", "thresholds"]:
            assert key in result
        assert "priority_score_threshold" in result["thresholds"]
        assert "priority_balance_floor" in result["thresholds"]


class TestAggregateMetricAutonomy:
    """_aggregate_metric grouped by Tier must auto-run rule-based tiering on
    a fresh session instead of refusing because Tier doesn't exist yet."""

    def test_aggregate_metric_by_tier_on_fresh_session_auto_segments(self, app_client, session_id):
        resp = app_client.post(
            "/api/chat",
            json={"session_id": session_id, "query": "average balance by tier"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "aggregate_metric"
        _assert_not_refused(data)

        result = data["result"]
        assert result["group_by"] == "Tier"
        assert result["metric"] == "AvgMonthlyBalance"
        assert result["aggregation"] == "mean"
        assert set(result["result"].keys()) == {"Priority", "Regular", "Dormant"}
        assert set(result["group_sizes"].keys()) == {"Priority", "Regular", "Dormant"}
        assert all(v > 0 for v in result["group_sizes"].values())


class TestConversionCandidatesAutonomy:
    """_conversion_candidates must auto-run rule-based tiering on a fresh
    session instead of refusing because there's no Tier column yet."""

    def test_conversion_candidates_on_fresh_session_auto_segments(self, app_client, session_id):
        resp = app_client.post(
            "/api/chat",
            json={"session_id": session_id, "query": "which regular customers can convert to priority"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "conversion_candidates"
        _assert_not_refused(data)

        result = data["result"]
        for key in ["n_candidates", "score_threshold_used", "candidates"]:
            assert key in result
        assert isinstance(result["candidates"], list)


class TestRecommendationAutonomy:
    """_recommendation must auto-run rule-based tiering on a fresh session
    instead of refusing because there's no Tier column yet."""

    def test_recommendation_on_fresh_session_auto_segments(self, app_client, session_id):
        resp = app_client.post(
            "/api/chat",
            json={"session_id": session_id, "query": "what should be recommended for priority customers"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "recommendation"
        _assert_not_refused(data)

        result = data["result"]
        assert "n_customers" in result
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
        assert result["n_customers"] > 0


class TestListCustomersByTierIntent:
    """The new list_customers_by_tier intent must resolve end-to-end for the
    original bug-report phrasing ("which customers are marked priority"),
    auto-running rule-based tiering on a fresh session rather than falling
    through to the generic clarify."""

    def test_which_customers_are_marked_priority_on_fresh_session(self, app_client, session_id):
        resp = app_client.post(
            "/api/chat",
            json={"session_id": session_id, "query": "which customers are marked priority"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "list_customers_by_tier"
        assert data["plan"]["intent"] == "list_customers_by_tier"
        _assert_not_refused(data)

        result = data["result"]
        assert result["tier"] == "Priority"
        assert result["count"] > 0
        assert isinstance(result["sample"], list)
        assert len(result["sample"]) > 0
        for row in result["sample"]:
            assert row["Tier"] == "Priority"
            assert "CustomerID" in row
