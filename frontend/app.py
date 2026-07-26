"""Streamlit frontend -- chat + dashboards on top of the FastAPI agent backend.

Run from the `bank-agent/` directory (with the backend already running):
    streamlit run frontend/app.py
"""

from __future__ import annotations

import os
import sys
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_client as api
from theme import CATEGORICAL, CUSTOM_CSS, PLOTLY_TEMPLATE, SURFACE, TEXT_PRIMARY, TIER_COLORS, tier_pill

st.set_page_config(page_title="SegmaAI · Retail Banking Agent", page_icon="🏦", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
import plotly.io as pio
pio.templates["segmaai"] = PLOTLY_TEMPLATE
pio.templates.default = "segmaai"

QUICK_QUERIES = [
    ("🧩 Segment by balance & frequency", "Segment customers into priority, regular and dormant customers based on balance being maintained and frequency of transactions"),
    ("❓ Explain priority basis", "On what basis were priority customers selected?"),
    ("💳 Avg transaction size by tier", "What is the average size of transactions for priority and regular customers?"),
    ("📈 Conversion candidates", "Which regular customers can be converted to priority? What should be done for them?"),
]

# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"


def handle_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        resp = api.chat(st.session_state.session_id, query)
    except api.APIError as e:
        resp = {"answer": f"Backend unreachable at {st.session_state.backend_url} ({e}). "
                           f"Is `uvicorn backend.main:app` running?", "intent": "error", "result": {}}
    st.session_state.messages.append({"role": "agent", "content": resp.get("answer", ""), "response": resp})


# ---------------------------------------------------------------------------
# Masthead: centered wordmark + pill navigation (replaces the sidebar)
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("chat", "💬", "Chat"),
    ("eda", "📊", "EDA"),
    ("segments", "🧩", "Segments"),
    ("lookup", "👤", "Lookup"),
    ("personas", "🎯", "Personas"),
]

if "page" not in st.session_state:
    st.session_state.page = "chat"


def render_masthead():
    st.markdown(
        '<div class="masthead">'
        '<div class="masthead-mark">Segma<span class="accent">AI</span></div>'
        '<div class="masthead-rule"></div>'
        '<div class="masthead-tagline">Ask a question, get a segmented answer.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(key="nav_row"):
        cols = st.columns(len(NAV_ITEMS))
        for c, (key, icon, label) in zip(cols, NAV_ITEMS):
            if st.session_state.page == key:
                c.markdown(f'<div class="nav-pill-active">{icon} {label}</div>', unsafe_allow_html=True)
            elif c.button(f"{icon} {label}", key=f"nav_{key}", width="stretch"):
                st.session_state.page = key
                st.rerun()

    try:
        health = api.health()
        llm_on = health.get("llm_active")
        dot = "●" if llm_on else "○"
        status = f"{dot} {health.get('planner', 'offline').title()} · {health.get('n_customers', '?'):,} customers"
    except api.APIError:
        status = "○ Backend unreachable"

    _, status_col, settings_col, _ = st.columns([3, 2, 0.6, 3])
    with status_col:
        st.markdown(f'<div class="status-strip">{status}</div>', unsafe_allow_html=True)
    with settings_col:
        with st.container(key="settings_pop"):
            with st.popover("⚙"):
                st.session_state.backend_url = st.text_input("Backend URL", st.session_state.backend_url)
                if st.button("Reset conversation & segmentation", width="stretch"):
                    api.reset_session(st.session_state.session_id)
                    st.session_state.messages = []
                    st.rerun()


render_masthead()


def kpi_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Rich result rendering for chat messages
# ---------------------------------------------------------------------------
def render_result_widget(resp: dict):
    intent = resp.get("intent")
    result = resp.get("result", {})
    if not result:
        return

    if intent == "segment_customers":
        counts = result.get("tier_counts") or {
            v: k for k, v in {}.items()
        }
        if "cluster_sizes" in result:
            counts = {result["cluster_label_map"].get(str(k), k): v for k, v in result["cluster_sizes"].items()}
        if counts:
            fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                         labels={"x": "Segment", "y": "Customers"}, title="Segment sizes")
            fig.update_traces(marker_color=[TIER_COLORS.get(str(k).split(" ")[0], CATEGORICAL[i % 8])
                                             for i, k in enumerate(counts.keys())])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.link_button("⬇ Download segment assignments (CSV)", api.download_segments_url(st.session_state.session_id))

    elif intent == "aggregate_metric":
        r = result.get("result", {})
        if r:
            fig = px.bar(x=list(r.keys()), y=list(r.values()),
                         labels={"x": result.get("group_by", "group"), "y": result.get("metric", "value")},
                         title=f"{result.get('aggregation')}({result.get('metric')}) by {result.get('group_by')}")
            fig.update_traces(marker_color=[TIER_COLORS.get(str(k).split(" ")[0], CATEGORICAL[i % 8])
                                             for i, k in enumerate(r.keys())])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    elif intent == "conversion_candidates":
        cands = result.get("candidates", [])
        if cands:
            df = pd.DataFrame(cands)
            st.dataframe(df, width="stretch", hide_index=True)

    elif intent == "recommendation":
        recs = result.get("recommendations", [])
        if recs:
            df = pd.DataFrame(recs)
            df["recommendations"] = df["recommendations"].apply(lambda x: " • ".join(x))
            st.dataframe(df, width="stretch", hide_index=True)

    elif intent == "eda_distribution" and result.get("type") == "numeric":
        fig = px.bar(x=result["bins"], y=result["counts"], labels={"x": result["column"], "y": "count"},
                     title=f"Distribution of {result['column']}")
        fig.update_traces(marker_color=CATEGORICAL[0])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    elif intent == "eda_correlation":
        pairs = result.get("top_correlations", [])
        if pairs:
            df = pd.DataFrame(pairs)
            st.dataframe(df, width="stretch", hide_index=True)

    elif intent == "explain_segment_basis" and "clusters" not in result and "distinguishing_features" in result:
        df = pd.DataFrame(result["distinguishing_features"])
        if not df.empty:
            st.dataframe(df, width="stretch", hide_index=True)


def render_chat_page():
    st.markdown('<div class="hero-title">Ask your Segmentation Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Natural language in → deterministic pandas/scikit-learn analysis out. '
                'Try one of the example judge queries below, or type your own.</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    clicked_query = None
    for c, (label, query) in zip(cols, QUICK_QUERIES):
        if c.button(label, width="stretch"):
            clicked_query = query

    chat_container = st.container()

    typed = st.chat_input("Ask about segments, EDA, recommendations, or a specific customer…")
    final_query = clicked_query or typed
    if final_query:
        handle_query(final_query)

    with chat_container:
        if not st.session_state.messages:
            st.info("No messages yet -- try a quick query above, or ask something like "
                     "\"Which regular customers can be converted to priority?\"")
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                resp = msg.get("response", {})
                needs_clarify = resp.get("needs_clarification")
                icon = "🤔" if needs_clarify else "🤖"
                st.markdown(f'<div class="chat-bubble-agent">{icon} {msg["content"]}</div>', unsafe_allow_html=True)
                if resp:
                    render_result_widget(resp)


def render_eda_page():
    st.markdown('<div class="hero-title">EDA Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Automated exploratory analysis over the unified customer view.</div>', unsafe_allow_html=True)
    try:
        ov = api.overview()
    except api.APIError:
        st.error("Backend not reachable.")
        return

    kpi_row([
        ("Customers", f"{ov['n_customers']:,}"),
        ("Transactions", f"{ov['n_transactions']:,}"),
        ("Avg. Income", f"₹{ov['avg_income']:,.0f}"),
        ("Avg. Balance", f"₹{ov['avg_balance']:,.0f}"),
        ("Avg. Credit Score", f"{ov['avg_credit_score']:.0f}"),
    ])
    st.caption(f"Transaction window: {ov['date_range'][0]} → {ov['date_range'][1]}")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        city = ov["city_distribution"]
        fig = px.bar(x=list(city.values()), y=list(city.keys()), orientation="h",
                     labels={"x": "Customers", "y": ""}, title="Top cities")
        fig.update_traces(marker_color=CATEGORICAL[0])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        occ = ov["occupation_distribution"]
        fig = px.pie(names=list(occ.keys()), values=list(occ.values()), hole=0.55, title="Occupation mix")
        fig.update_traces(marker=dict(colors=CATEGORICAL), textfont_color=TEXT_PRIMARY)
        fig.update_layout(paper_bgcolor=SURFACE)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    prod = ov["product_ownership_rate"]
    fig = px.bar(x=list(prod.keys()), y=list(prod.values()), labels={"x": "Product", "y": "Ownership %"},
                 title="Product ownership rate")
    fig.update_traces(marker_color=CATEGORICAL[2])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Ask the agent")
    c1, c2 = st.columns(2)
    with c1:
        col = st.selectbox("Distribution of…", ["Age", "AnnualIncome", "AvgMonthlyBalance", "CreditScore",
                                                   "MonthlyTxnFrequency", "MonthlySpend", "CardUtilizationPct"])
        if st.button("Show distribution"):
            handle_query(f"Show the distribution of {col}")
    with c2:
        if st.button("Show correlations"):
            handle_query("Show correlations between customer features")
        if st.button("Check for missing values"):
            handle_query("Are there any missing values in the data?")

    last = next((m for m in reversed(st.session_state.messages) if m["role"] == "agent"), None)
    if last and last.get("response", {}).get("intent") in ("eda_distribution", "eda_correlation", "eda_missing_values"):
        st.markdown(f'<div class="glass-card">{last["content"]}</div>', unsafe_allow_html=True)
        render_result_widget(last["response"])


def render_segments_page():
    st.markdown('<div class="hero-title">Segments Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Run rule-based tiering or unsupervised clustering, then explore the result.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    if c1.button("🧩 Run rule-based tiering (balance + frequency)", width="stretch"):
        handle_query("Segment customers into priority, regular and dormant customers based on balance being maintained and frequency of transactions")
        st.rerun()
    if c2.button("🤖 Run ML clustering (KMeans, auto-k)", width="stretch"):
        handle_query("Segment customers using ML clustering across income, balance, spend and credit behaviour")
        st.rerun()
    if c3.button("❓ Explain current segmentation basis", width="stretch"):
        handle_query("On what basis were the current segments selected?")

    try:
        seg = api.segments(st.session_state.session_id)
    except api.APIError:
        st.error("Backend not reachable.")
        return

    if not seg.get("segmented"):
        st.info("No segmentation run yet this session -- click a button above.")
        return

    counts = seg["tier_counts"]
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = px.pie(names=list(counts.keys()), values=list(counts.values()), hole=0.55,
                     title=f"Tier distribution ({seg['method']})")
        fig.update_traces(marker=dict(colors=[TIER_COLORS.get(str(k).split(" ")[0], CATEGORICAL[i % 8])
                                               for i, k in enumerate(counts.keys())]), textfont_color=TEXT_PRIMARY)
        fig.update_layout(paper_bgcolor=SURFACE)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        if "scatter" in seg:
            sc = seg["scatter"]
            df = pd.DataFrame(sc)
            fig = px.scatter(df, x="x", y="y", color="tier",
                             color_discrete_map={t: TIER_COLORS.get(str(t).split(" ")[0], CATEGORICAL[i])
                                                  for i, t in enumerate(df["tier"].unique())},
                             hover_data=["customer_id"], title="Cluster projection (PCA, 2D)")
            fig.update_traces(marker=dict(size=6, opacity=0.75))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            balance_freq_scatter()

    st.markdown("#### Customers in segment")
    tier_filter = st.selectbox("Filter by tier", ["All"] + list(counts.keys()))
    data = api.list_customers(st.session_state.session_id, None if tier_filter == "All" else tier_filter, limit=300)
    df = pd.DataFrame(data["customers"])
    st.caption(f"Showing {len(df)} of {data['total']} customers")
    st.dataframe(df, width="stretch", hide_index=True)
    st.link_button("⬇ Download full segment assignments (CSV)", api.download_segments_url(st.session_state.session_id))


def balance_freq_scatter():
    data = api.list_customers(st.session_state.session_id, limit=1500)
    df = pd.DataFrame(data["customers"])
    if df.empty or "Tier" not in df.columns:
        return
    fig = px.scatter(df, x="AvgMonthlyBalance", y="MonthlyTxnFrequency", color="Tier",
                     color_discrete_map={t: TIER_COLORS.get(str(t).split(" ")[0], CATEGORICAL[0]) for t in df["Tier"].unique()},
                     hover_data=["CustomerID"], title="Balance vs. Transaction Frequency")
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_customer_page():
    st.markdown('<div class="hero-title">Customer Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Look up any customer\'s profile and, if segmented, why they landed in their segment.</div>', unsafe_allow_html=True)

    cid = st.text_input("Customer ID", placeholder="CUST00001")
    if st.button("Look up") and cid:
        try:
            profile = api.customer_detail(cid.strip().upper(), st.session_state.session_id)
        except Exception:
            st.error(f"Customer '{cid}' not found.")
            return

        tier = profile.get("Tier")
        header = f"### {profile['CustomerID']}"
        if tier:
            header += "  " + tier_pill(tier)
        st.markdown(header, unsafe_allow_html=True)

        kpi_row([
            ("Age / Gender", f"{profile['Age']} / {profile['Gender']}"),
            ("Occupation", profile["Occupation"]),
            ("City", profile["City"]),
            ("Account Type", profile["AccountType"]),
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        kpi_row([
            ("Avg. Balance", f"₹{profile['AvgMonthlyBalance']:,.0f}"),
            ("Annual Income", f"₹{profile['AnnualIncome']:,.0f}"),
            ("Credit Score", f"{profile['CreditScore']:.0f}"),
            ("Txn Frequency", f"{profile.get('MonthlyTxnFrequency', 0):.1f}/mo"),
        ])
        st.markdown("<br>", unsafe_allow_html=True)

        products = [p for p in ["SavingsAccount", "CreditCard", "FixedDeposit", "Insurance", "Investment", "Loan"]
                    if profile.get(p) == 1]
        st.markdown("**Products owned:** " + (", ".join(products) if products else "None"))
        st.markdown(f"**Top spend category:** {profile.get('TopMerchantCategory', 'N/A')} · "
                    f"**Card utilization:** {profile.get('CardUtilizationPct', 0):.0f}% · "
                    f"**EMI:** ₹{profile.get('EMI', 0):,.0f}/mo")

        if st.button(f"Why does {profile['CustomerID']} belong to this segment?"):
            handle_query(f"Why does customer {profile['CustomerID']} belong to their segment?")

        last = next((m for m in reversed(st.session_state.messages) if m["role"] == "agent"), None)
        if last and last.get("response", {}).get("intent") == "customer_lookup":
            st.markdown(f'<div class="glass-card">{last["content"]}</div>', unsafe_allow_html=True)


PERSONA_FLAIR = {
    "Priority": ("🏆", "The Affluent Loyalists", "High balance, high engagement -- your most valuable relationship."),
    "Regular": ("💼", "The Core Base", "Steady, dependable -- the bulk of the customer book."),
    "Dormant": ("💤", "The Disengaged", "Low activity -- reactivation candidates before they churn."),
}


def render_personas_page():
    st.markdown('<div class="hero-title">Personas & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Auto-generated segment personas with rule-based cross-sell/retention actions.</div>', unsafe_allow_html=True)

    try:
        seg = api.segments(st.session_state.session_id)
    except api.APIError:
        st.error("Backend not reachable.")
        return
    if not seg.get("segmented"):
        st.info("Run a segmentation first (see 'Segments Explorer' or ask the chat agent).")
        if st.button("🧩 Run rule-based tiering now"):
            handle_query("Segment customers into priority, regular and dormant customers based on balance being maintained and frequency of transactions")
            st.rerun()
        return

    for tier_full in seg["tier_counts"]:
        base_tier = str(tier_full).split(" ")[0]
        icon, title, blurb = PERSONA_FLAIR.get(base_tier, ("👥", tier_full, ""))
        data = api.list_customers(st.session_state.session_id, tier_full, limit=2000)
        df = pd.DataFrame(data["customers"])

        with st.container():
            st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"#### {icon} {tier_full} — {title}")
            st.caption(blurb)
            if not df.empty:
                kpi_row([
                    ("Customers", f"{len(df):,}"),
                    ("Avg. Balance", f"₹{df['AvgMonthlyBalance'].mean():,.0f}"),
                    ("Avg. Txn Frequency", f"{df['MonthlyTxnFrequency'].mean():.1f}/mo"),
                    ("Avg. Credit Score", f"{df['CreditScore'].mean():.0f}"),
                ])
            if st.button(f"Get recommendations for {tier_full}", key=f"rec_{tier_full}"):
                handle_query(f"Recommend products for {tier_full} customers")
            st.markdown("</div>", unsafe_allow_html=True)

        last = next((m for m in reversed(st.session_state.messages)
                     if m["role"] == "agent" and m.get("response", {}).get("intent") == "recommendation"), None)
        if last:
            recs = last["response"]["result"].get("recommendations", [])
            if recs and recs[0]["tier"] == tier_full:
                sample = pd.DataFrame(recs[:8])
                sample["recommendations"] = sample["recommendations"].apply(lambda x: " • ".join(x))
                st.dataframe(sample, width="stretch", hide_index=True)


PAGES = {
    "chat": render_chat_page,
    "eda": render_eda_page,
    "segments": render_segments_page,
    "lookup": render_customer_page,
    "personas": render_personas_page,
}
PAGES[st.session_state.page]()
