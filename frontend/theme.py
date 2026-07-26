"""Visual theme: CSS injected into the Streamlit app + a matching Plotly
template.

Direction: "ledger paper" -- a retail-banking statement, not a SaaS dashboard.
Near-white ruled paper plane, true-white cards held by hairlines instead of
shadows, a deep ledger green for value and a warm terracotta red for risk.
Signature element is the ledger rule: a short green bar that draws itself in
under every section title, echoing the ruled line on a bank statement. The
same rule literalizes as an interactive sweep-underline on the centered nav
pills -- the one signature motif carried from static hero into live control.

Type roles: Bricolage Grotesque (display), Instrument Sans (body),
IBM Plex Mono (labels and figures -- the statement voice).

The categorical palette is not hand-picked: it was run through the dataviz
validator against a #ffffff surface. The 8-slot order passes lightness band,
chroma floor, adjacent-pair CVD separation (worst 12.6 deutan) and the
normal-vision floor (worst 15.1). The 3 tier colors additionally pass on the
*all-pairs* list (worst CVD 9.6, normal-vision 17.0, all >= 3:1 contrast),
which is what scatter and pie need.
"""

# --- Surfaces & ink -------------------------------------------------------
PAGE = "#f4f4f0"          # ruled paper plane
SURFACE = "#ffffff"       # card + chart surface
CARD = "#ffffff"
CARD_BORDER = "rgba(16,26,21,0.11)"
CARD_BORDER_SOLID = "#e2e4de"
TEXT_PRIMARY = "#101a15"
TEXT_SECONDARY = "#4b5750"
TEXT_MUTED = "#66706a"    # 5.13:1 on white, 4.66:1 on PAGE -- clears AA for small text
GRID = "#e7e8e2"
AXIS = "#c9cdc6"

# --- Brand ----------------------------------------------------------------
BRAND = "#0a7147"         # ledger green -- value, growth, "in the black"
BRAND_DEEP = "#075436"
BRAND_TINT = "rgba(10,113,71,0.09)"
ALERT = "#b8352b"         # reserved for risk / offline, never a series color

# --- Data colors (validated, see module docstring) ------------------------
CATEGORICAL = ["#0a7147", "#45619e", "#dd7264", "#eda100", "#7a4fa8", "#1baf7a", "#96601f", "#e87ba4"]

# Role mapping, deliberately not a traffic light: value reads as the deepest,
# most saturated color; the core base is an institutional indigo (neutral, not
# "warning yellow"); at-risk is a warm terracotta -- red in family, but muted
# and light enough to separate from the green under deuteranopia.
TIER_COLORS = {"Priority": "#0a7147", "Regular": "#45619e", "Dormant": "#dd7264"}

ACCENT_GRADIENT = f"linear-gradient(90deg, {BRAND_DEEP} 0%, {BRAND} 60%, #16a06a 100%)"

FONT_BODY = "Instrument Sans, Inter, -apple-system, Segoe UI, sans-serif"
FONT_DISPLAY = "Bricolage Grotesque, Instrument Sans, Segoe UI, sans-serif"
FONT_MONO = "IBM Plex Mono, JetBrains Mono, Consolas, monospace"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_SECONDARY, family=FONT_BODY, size=13),
        title=dict(font=dict(color=TEXT_PRIMARY, family=FONT_DISPLAY, size=16),
                   x=0.0, xanchor="left", y=0.96, yanchor="top"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, color=TEXT_MUTED,
                   showgrid=False, ticks="outside", ticklen=4, tickcolor=AXIS,
                   tickfont=dict(family=FONT_MONO, size=11), automargin=True,
                   title=dict(font=dict(family=FONT_BODY, size=12, color=TEXT_MUTED))),
        yaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=GRID, color=TEXT_MUTED,
                   showgrid=True, showline=False, ticks="",
                   tickfont=dict(family=FONT_MONO, size=11), automargin=True,
                   title=dict(font=dict(family=FONT_BODY, size=12, color=TEXT_MUTED))),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=CARD_BORDER_SOLID, borderwidth=0,
                    font=dict(color=TEXT_SECONDARY, size=12), itemsizing="constant",
                    title=dict(font=dict(family=FONT_MONO, size=10, color=TEXT_MUTED))),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=CARD_BORDER_SOLID, align="left",
                        font=dict(color=TEXT_PRIMARY, family=FONT_BODY, size=12)),
        hovermode="closest",
        colorway=CATEGORICAL,
        margin=dict(l=8, r=8, t=56, b=8),
        height=380,
        barcornerradius=4,
        bargap=0.34,
        separators=".,",
    ),
    # 2px surface gap between adjacent fills, and a hover layer by default.
    data=dict(
        bar=[dict(marker=dict(line=dict(color=SURFACE, width=2)),
                  hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>")],
        pie=[dict(marker=dict(line=dict(color=SURFACE, width=2)),
                  textposition="outside", textinfo="label+percent",
                  hovertemplate="<b>%{label}</b><br>%{value:,} · %{percent}<extra></extra>")],
        scattergl=[dict(marker=dict(line=dict(color=SURFACE, width=1)))],
    ),
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* --- Motion: one orchestrated page-load sequence, nothing scattered --- */
@keyframes ledger-rise {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes rule-draw {{
    from {{ transform: scaleX(0); }}
    to   {{ transform: scaleX(1); }}
}}
@keyframes slide-in-right {{
    from {{ opacity: 0; transform: translateX(16px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes slide-in-left {{
    from {{ opacity: 0; transform: translateX(-16px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}

html {{
    scroll-behavior: smooth;
}}

html, body, [class*="css"] {{
    font-family: {FONT_BODY};
}}

.stApp {{
    /* ruled paper: hairlines every 34px, barely there */
    background:
        repeating-linear-gradient(
            to bottom,
            rgba(10,113,71,0.045) 0px,
            rgba(10,113,71,0.045) 1px,
            transparent 1px,
            transparent 34px
        ),
        {PAGE};
    background-attachment: fixed;
    color: {TEXT_PRIMARY};
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1180px;
    margin-left: auto;
    margin-right: auto;
}}

#MainMenu, footer, header {{visibility: hidden;}}

/* --- Masthead: centered wordmark replacing the sidebar identity block --- */
.masthead {{
    text-align: center;
    padding: 0.8rem 0 0.3rem;
}}
.masthead-mark {{
    font-family: {FONT_DISPLAY};
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: {TEXT_PRIMARY};
    line-height: 1;
    animation: ledger-rise 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
.masthead-mark .accent {{ color: {BRAND}; }}
.masthead-rule {{
    width: 84px;
    height: 3px;
    margin: 0.65rem auto 0;
    background: {ACCENT_GRADIENT};
    border-radius: 2px;
    animation: rule-draw 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.2s both;
}}
.masthead-tagline {{
    color: {TEXT_MUTED};
    font-size: 0.92rem;
    margin-top: 0.6rem;
    animation: ledger-rise 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.12s both;
}}

/* --- Centered pill navigation: the signature ledger-rule, reincarnated as
   a hover sweep-underline under each destination --- */
.st-key-nav_row {{
    max-width: 780px;
    margin: 1.1rem auto 0;
}}
.st-key-nav_row div[data-testid="stHorizontalBlock"] {{
    gap: 0.5rem;
}}
.st-key-nav_row div.stButton > button {{
    border-radius: 999px;
    border: 1px solid {CARD_BORDER_SOLID};
    background: {SURFACE};
    color: {TEXT_SECONDARY};
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.55rem 0.9rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(16,26,21,0.03);
    transition: color 0.2s ease, border-color 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
}}
.st-key-nav_row div.stButton > button::after {{
    content: "";
    position: absolute;
    left: 50%; right: 50%; bottom: 7px;
    height: 2px;
    background: {BRAND};
    border-radius: 2px;
    transition: left 0.25s cubic-bezier(0.22, 1, 0.36, 1), right 0.25s cubic-bezier(0.22, 1, 0.36, 1);
}}
.st-key-nav_row div.stButton > button:hover {{
    color: {BRAND_DEEP};
    border-color: {BRAND};
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(10,113,71,0.16);
}}
.st-key-nav_row div.stButton > button:hover::after {{ left: 16px; right: 16px; }}
.st-key-nav_row div.stButton > button:active {{ transform: translateY(0) scale(0.97); }}
.nav-pill-active {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    min-height: 2.5rem;
    border-radius: 999px;
    background: {ACCENT_GRADIENT};
    color: #ffffff;
    font-weight: 700;
    font-size: 0.86rem;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 20px rgba(10,113,71,0.28);
}}

/* --- Status strip + settings popover trigger --- */
.status-strip {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    font-family: {FONT_MONO};
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    height: 2.1rem;
}}
.st-key-settings_pop button {{
    border-radius: 999px !important;
    width: 2.1rem;
    padding: 0 !important;
}}

/* --- Hero + the signature ledger rule --- */
.hero-title {{
    font-family: {FONT_DISPLAY};
    font-size: 2.2rem;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin-bottom: 0.55rem;
    animation: ledger-rise 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
.hero-title::after {{
    content: "";
    display: block;
    width: 62px;
    height: 3px;
    margin-top: 0.5rem;
    background: {ACCENT_GRADIENT};
    border-radius: 2px;
    transform-origin: left center;
    animation: rule-draw 0.55s cubic-bezier(0.22, 1, 0.36, 1) 0.22s both;
}}
.hero-sub {{
    color: {TEXT_MUTED};
    font-size: 0.94rem;
    line-height: 1.5;
    margin-top: 0.35rem;
    margin-bottom: 1.5rem;
    max-width: 62ch;
    animation: ledger-rise 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.1s both;
}}

/* --- Cards: hairline + ruled edge, no shadow, no blur --- */
.glass-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-left: 3px solid {BRAND};
    border-radius: 10px;
    padding: 1.05rem 1.25rem;
    margin-bottom: 0.9rem;
    color: {TEXT_SECONDARY};
    animation: ledger-rise 0.45s cubic-bezier(0.22, 1, 0.36, 1) 0.12s both;
}}

.kpi-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 0.85rem 1rem 0.9rem;
    text-align: left;
    animation: ledger-rise 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
.kpi-label {{
    font-family: {FONT_MONO};
    color: {TEXT_MUTED};
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    font-weight: 500;
}}
.kpi-value {{
    font-family: {FONT_DISPLAY};
    color: {TEXT_PRIMARY};
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-top: 0.2rem;
    font-variant-numeric: tabular-nums;
}}

/* Staggered entrance across a row of columns (KPI rows, quick-query buttons) */
div[data-testid="stColumn"]:nth-of-type(1) .kpi-card {{ animation-delay: 0.06s; }}
div[data-testid="stColumn"]:nth-of-type(2) .kpi-card {{ animation-delay: 0.12s; }}
div[data-testid="stColumn"]:nth-of-type(3) .kpi-card {{ animation-delay: 0.18s; }}
div[data-testid="stColumn"]:nth-of-type(4) .kpi-card {{ animation-delay: 0.24s; }}
div[data-testid="stColumn"]:nth-of-type(5) .kpi-card {{ animation-delay: 0.30s; }}

/* Charts and tables arrive after the copy they belong to */
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {{
    background: {SURFACE};
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 0.35rem;
    animation: ledger-rise 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.18s both;
}}

/* --- Tier pills --- */
.pill {{
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 3px;
    font-family: {FONT_MONO};
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.pill-priority {{ background: rgba(10,113,71,0.10);  color: {BRAND_DEEP}; border: 1px solid rgba(10,113,71,0.30); }}
.pill-regular  {{ background: rgba(69,97,158,0.10);  color: #34497a;      border: 1px solid rgba(69,97,158,0.30); }}
.pill-dormant  {{ background: rgba(221,114,100,0.14); color: #9d3a2b;     border: 1px solid rgba(221,114,100,0.38); }}
.pill-llm      {{ background: rgba(10,113,71,0.10);  color: {BRAND_DEEP}; border: 1px solid rgba(10,113,71,0.30); }}
.pill-offline  {{ background: rgba(184,53,43,0.09);  color: {ALERT};      border: 1px solid rgba(184,53,43,0.30); }}

/* --- Chat --- */
.chat-bubble-user {{
    background: {BRAND};
    color: #ffffff;
    padding: 0.65rem 0.95rem;
    border-radius: 10px 10px 2px 10px;
    margin: 0.4rem 0;
    max-width: 78%;
    margin-left: auto;
    font-size: 0.92rem;
    line-height: 1.5;
    animation: slide-in-right 0.32s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
.chat-bubble-agent {{
    background: {SURFACE};
    border: 1px solid {CARD_BORDER};
    border-left: 3px solid {CARD_BORDER_SOLID};
    color: {TEXT_PRIMARY};
    padding: 0.7rem 1rem;
    border-radius: 2px 10px 10px 2px;
    margin: 0.4rem 0;
    max-width: 84%;
    font-size: 0.92rem;
    line-height: 1.55;
    white-space: pre-wrap;
    animation: slide-in-left 0.32s cubic-bezier(0.22, 1, 0.36, 1) both;
}}
.chat-meta {{
    font-family: {FONT_MONO};
    color: {TEXT_MUTED};
    font-size: 0.66rem;
    margin-top: 0.25rem;
}}

/* --- Controls: soft, rounded, tactile --- */
div.stButton > button {{
    border-radius: 12px;
    border: 1px solid {CARD_BORDER_SOLID};
    background: {SURFACE};
    color: {TEXT_PRIMARY};
    font-weight: 500;
    box-shadow: 0 1px 2px rgba(16,26,21,0.04);
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease,
                transform 0.15s ease, box-shadow 0.2s ease;
}}
div.stButton > button:hover {{
    border-color: {BRAND};
    color: {BRAND_DEEP};
    background: {BRAND_TINT};
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(10,113,71,0.14);
}}
div.stButton > button:active {{
    transform: translateY(0) scale(0.98);
    box-shadow: 0 1px 2px rgba(16,26,21,0.06);
}}
div.stButton > button:focus-visible,
a:focus-visible, input:focus-visible, textarea:focus-visible {{
    outline: 2px solid {BRAND};
    outline-offset: 2px;
}}

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] > div {{
    background: {SURFACE} !important;
    border-color: {CARD_BORDER_SOLID} !important;
    border-radius: 10px !important;
}}
/* --- Chat input: force the "ledger paper" surface regardless of the
   visitor's OS/browser dark-mode preference. config.toml pins Streamlit's
   own theme to light so native chrome (this widget's shadow-lite internals)
   agrees with our injected CSS instead of fighting it -- this block is the
   belt-and-suspenders layer on top of that. --- */
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {{
    background: {PAGE} !important;
}}
[data-testid="stChatInput"] {{
    background: {SURFACE} !important;
    border: 1px solid {CARD_BORDER_SOLID} !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(16,26,21,0.06);
}}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {{
    background: {SURFACE} !important;
    color: {TEXT_PRIMARY} !important;
    caret-color: {BRAND} !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {TEXT_MUTED} !important;
    opacity: 1 !important;
}}
[data-testid="stChatInput"] button {{
    background: {BRAND} !important;
    border-radius: 999px !important;
    transition: background 0.15s ease, transform 0.15s ease;
}}
[data-testid="stChatInput"] button:hover {{ background: {BRAND_DEEP} !important; transform: scale(1.06); }}
[data-testid="stChatInput"] button svg {{ fill: #ffffff !important; }}

hr {{ border-color: {CARD_BORDER_SOLID}; }}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: rgba(10,113,71,0.30); border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation: none !important;
        transition: none !important;
    }}
}}
</style>
"""


def tier_pill(tier: str) -> str:
    key = (tier or "").split(" ")[0].lower()
    cls = {"priority": "pill-priority", "regular": "pill-regular", "dormant": "pill-dormant"}.get(key, "pill-regular")
    return f'<span class="pill {cls}">{tier}</span>'
