"""Visual theme: CSS injected into the Streamlit app + a matching Plotly
template. Dark surface, glassmorphic cards, animated gradient backdrop.
Palette follows a validated categorical set (dark-mode steps) so charts stay
colorblind-safe instead of default Plotly rainbow."""

SURFACE = "#14141a"
PAGE = "#0b0b10"
CARD = "rgba(255,255,255,0.045)"
CARD_BORDER = "rgba(255,255,255,0.09)"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#898781"
GRID = "#2c2c2a"

CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#59d35a", "#9085e9", "#e66767"]
TIER_COLORS = {"Priority": "#c98500", "Regular": "#3987e5", "Dormant": "#5a5a66"}
ACCENT_GRADIENT = "linear-gradient(135deg, #3987e5 0%, #9085e9 45%, #d55181 100%)"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_SECONDARY, family="Inter, -apple-system, Segoe UI, sans-serif", size=13),
        title=dict(font=dict(color=TEXT_PRIMARY, size=16)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_MUTED),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY)),
        colorway=CATEGORICAL,
        margin=dict(l=10, r=10, t=40, b=10),
    )
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at 15% 10%, rgba(57,135,229,0.16), transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(213,81,129,0.14), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(144,133,233,0.12), transparent 50%),
        {PAGE};
    background-attachment: fixed;
}}

section[data-testid="stSidebar"] {{
    background: rgba(10,10,14,0.85);
    backdrop-filter: blur(18px);
    border-right: 1px solid {CARD_BORDER};
}}

#MainMenu, footer, header {{visibility: hidden;}}

.hero-title {{
    font-size: 2.3rem;
    font-weight: 800;
    background: {ACCENT_GRADIENT};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -0.02em;
}}
.hero-sub {{
    color: {TEXT_MUTED};
    font-size: 0.95rem;
    margin-top: 0.1rem;
    margin-bottom: 1.4rem;
}}

.glass-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 0.9rem;
}}

.kpi-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 1rem 1.1rem;
    backdrop-filter: blur(14px);
    text-align: left;
}}
.kpi-label {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}}
.kpi-value {{
    color: {TEXT_PRIMARY};
    font-size: 1.65rem;
    font-weight: 700;
    margin-top: 0.15rem;
    font-variant-numeric: tabular-nums;
}}

.pill {{
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.pill-priority {{ background: rgba(201,133,0,0.18); color: #f0b429; border: 1px solid rgba(240,180,41,0.35); }}
.pill-regular  {{ background: rgba(57,135,229,0.18); color: #6ea8f0; border: 1px solid rgba(110,168,240,0.35); }}
.pill-dormant  {{ background: rgba(120,120,130,0.18); color: #9d9dae; border: 1px solid rgba(157,157,174,0.35); }}
.pill-llm      {{ background: rgba(25,158,112,0.18); color: #3fd39a; border: 1px solid rgba(63,211,154,0.35); }}
.pill-offline  {{ background: rgba(214,88,88,0.14); color: #e88; border: 1px solid rgba(232,136,136,0.35); }}

.chat-bubble-user {{
    background: {ACCENT_GRADIENT};
    color: white;
    padding: 0.7rem 1rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.35rem 0;
    max-width: 78%;
    margin-left: auto;
    font-size: 0.92rem;
    box-shadow: 0 4px 18px rgba(144,133,233,0.25);
}}
.chat-bubble-agent {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    color: {TEXT_PRIMARY};
    padding: 0.7rem 1rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.35rem 0;
    max-width: 82%;
    font-size: 0.92rem;
    backdrop-filter: blur(14px);
    white-space: pre-wrap;
}}
.chat-meta {{
    color: {TEXT_MUTED};
    font-size: 0.68rem;
    margin-top: 0.25rem;
}}

div.stButton > button {{
    border-radius: 10px;
    border: 1px solid {CARD_BORDER};
    background: rgba(255,255,255,0.04);
    color: {TEXT_PRIMARY};
    font-weight: 600;
    transition: all 0.15s ease;
}}
div.stButton > button:hover {{
    border-color: #9085e9;
    color: #b7aef2;
    background: rgba(144,133,233,0.10);
}}

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] > div {{
    background: rgba(255,255,255,0.045) !important;
    border-color: {CARD_BORDER} !important;
    border-radius: 10px !important;
}}

hr {{ border-color: {CARD_BORDER}; }}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: rgba(144,133,233,0.35); border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
</style>
"""


def tier_pill(tier: str) -> str:
    key = (tier or "").split(" ")[0].lower()
    cls = {"priority": "pill-priority", "regular": "pill-regular", "dormant": "pill-dormant"}.get(key, "pill-regular")
    return f'<span class="pill {cls}">{tier}</span>'
