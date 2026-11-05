"""Visual theme: colour tokens, CSS injection, hero banner and Plotly layout defaults."""
from __future__ import annotations

import streamlit as st

COLORS = {
    "bg": "#050816", "panel": "#0b1226", "panel2": "#101a36", "border": "#1e2a4a",
    "text": "#e6edf7", "muted": "#8a9ac2",
    "cyan": "#22d3ee", "blue": "#3b82f6", "purple": "#a855f7", "amber": "#f59e0b",
    "green": "#22c55e", "red": "#ef4444", "yellow": "#facc15", "grey": "#6b7280",
}

# NOTE: no blank lines inside the <style> block (Markdown would end the HTML block there).
_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Poppins:wght@500;600;700&family=Fira+Code:wght@400;500&display=swap');
:root { --font-body: 'Nunito', 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif; --font-head: 'Poppins', 'Nunito', 'Segoe UI', system-ui, sans-serif; --font-code: 'Fira Code', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
html, body, [class*="css"], .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp select, .stApp table, .stApp th, .stApp td, .stApp summary, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stWidgetLabel"] p, [data-testid="stTickBarMin"], [data-testid="stTickBarMax"], [data-baseweb="tab"] p, [data-baseweb="select"] div { font-family: var(--font-body); }
.stApp { font-size: clamp(15px, 0.35vw + 14px, 17px); line-height: 1.65; overflow-wrap: anywhere; background: radial-gradient(1100px 560px at 8% -10%, rgba(59,130,246,.20), transparent 60%), radial-gradient(900px 480px at 100% 0%, rgba(168,85,247,.17), transparent 55%), #050816; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #070d20, #0a1330); border-right: 1px solid #1e2a4a; }
h1, h2, h3, h4, h5, h6, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--font-head); letter-spacing: -0.005em; line-height: 1.25; }
.stApp h2 { font-size: clamp(1.35rem, 1.1rem + 1.2vw, 1.9rem); }
.stApp h3 { font-size: clamp(1.15rem, 1rem + .9vw, 1.5rem); }
.stApp h4 { font-size: clamp(1.02rem, .95rem + .5vw, 1.2rem); }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
.hero { padding: 4px 0 14px 0; border-bottom: 1px solid rgba(34,211,238,.18); margin-bottom: 14px; }
.hero-kicker { font: 700 clamp(10.5px, 1.4vw, 12.5px) var(--font-body); letter-spacing: .2em; text-transform: uppercase; color: #22d3ee; }
.hero-kicker::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22d3ee; margin-right: 10px; box-shadow: 0 0 12px #22d3ee; animation: bsPulse 2.4s ease-in-out infinite; }
@keyframes bsPulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
@media (prefers-reduced-motion: reduce) { .hero-kicker::before { animation: none; } }
.hero h1 { margin: 4px 0 0 0; padding: 0; font-family: var(--font-head); font-size: clamp(30px, 6vw, 58px); font-weight: 700; line-height: 1.1; background: linear-gradient(90deg, #22d3ee, #3b82f6 45%, #a855f7); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-tag { font: 600 clamp(16px, 2.6vw, 22px) var(--font-head); color: #e6edf7; margin-top: 2px; }
.hero-sub { color: #8a9ac2; font-size: clamp(13.5px, 1.6vw, 15.5px); margin-top: 2px; }
.card { background: linear-gradient(180deg, rgba(16,26,54,.88), rgba(11,18,38,.88)); border: 1px solid #1e2a4a; border-radius: 16px; padding: 14px 18px; margin: 10px 0; box-shadow: 0 8px 30px rgba(0,0,0,.35); transition: border-color .25s; }
.card:hover { border-color: rgba(34,211,238,.45); }
.card-title { font: 700 clamp(13px, 1.5vw, 14.5px) var(--font-body); letter-spacing: .03em; color: #22d3ee; margin-bottom: 8px; }
.card p { margin: 4px 0; line-height: 1.65; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 112px), 1fr)); gap: 10px; margin: 6px 0 12px; }
.stat { background: #0b1226; border: 1px solid #1e2a4a; border-radius: 14px; padding: 10px 12px; min-width: 0; }
.stat .v { font: 700 clamp(18px, 2.4vw, 23px) var(--font-head); color: #e6edf7; }
.stat .l { font: 600 clamp(11.5px, 1.3vw, 12.5px) var(--font-body); color: #8a9ac2; letter-spacing: .02em; }
.stat .h { font-size: clamp(11px, 1.2vw, 12px); color: #7c8aa5; }
.stat.good .v { color: #22c55e; } .stat.bad .v { color: #ef4444; } .stat.accent .v { color: #22d3ee; } .stat.warn .v { color: #f59e0b; }
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: clamp(12px, 1.4vw, 13px); color: #cbd5e1; background: #0b1226; border: 1px solid #1e2a4a; border-radius: 999px; padding: 4px 12px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
.callout { border-left: 3px solid; border-radius: 10px; padding: 12px 16px; margin: 10px 0; background: rgba(16,26,54,.7); font-size: clamp(14.5px, 1.5vw, 16px); line-height: 1.65; }
.callout.info { border-color: #22d3ee; } .callout.ok { border-color: #22c55e; } .callout.warn { border-color: #f59e0b; } .callout.bad { border-color: #ef4444; }
.pseudo { background: #050a1a; border: 1px solid #1e2a4a; border-radius: 14px; padding: 8px 0; font: 400 clamp(11.5px, 1.4vw, 13px)/1.75 var(--font-code); overflow-x: auto; -webkit-overflow-scrolling: touch; }
.pseudo .ln { display: block; padding: 0 12px; white-space: pre; color: #94a3b8; border-left: 3px solid transparent; transition: background .25s; }
.pseudo .ln b { color: #5b6b86; font-weight: 400; margin-right: 10px; }
.pseudo .ln.hl { background: rgba(168,85,247,.24); color: #fff; border-left-color: #a855f7; }
.step-list { margin: 0; padding-left: 20px; font-size: clamp(14px, 1.5vw, 15.5px); line-height: 1.6; } .step-list li { margin: 6px 0; } .step-list li.todo { opacity: .32; }
.step-list ul { margin: 4px 0 2px 0; padding-left: 18px; color: #b6c2df; font-size: .93em; }
.kv { color: #a5f3fc; font-family: var(--font-code); font-size: .92em; }
.stButton > button { border-radius: 12px; border: 1px solid #1e2a4a; background: #0f1a38; color: #e6edf7; font-weight: 600; transition: border-color .2s, box-shadow .2s; }
.stButton > button:hover:not(:disabled) { border-color: #22d3ee; box-shadow: 0 0 14px rgba(34,211,238,.35); color: #fff; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #0891b2, #6d28d9); border: none; }
[data-testid="stExpander"] { border: 1px solid #1e2a4a; border-radius: 14px; background: rgba(11,18,38,.6); }
[data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stPlotlyChart"] { max-width: 100%; }
[data-testid="stTable"], [data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
@media (max-width: 992px) {
.block-container { padding-left: 1.4rem; padding-right: 1.4rem; }
[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; row-gap: .6rem; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"], [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4)) > [data-testid="stColumn"], [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(4)) > [data-testid="column"] { flex: 1 1 calc(50% - .5rem) !important; min-width: calc(50% - .5rem) !important; width: calc(50% - .5rem) !important; }
}
@media (max-width: 640px) {
.block-container { padding: 1rem .85rem 3rem .85rem; }
.hero { padding-bottom: 10px; margin-bottom: 10px; }
.card { padding: 12px 14px; border-radius: 14px; }
.callout { padding: 10px 12px; }
.stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.stButton > button { min-height: 46px; }
.stApp input, .stApp textarea { font-size: 16px; }
[data-testid="stSidebar"] { max-width: 88vw; }
}
@media (min-width: 1600px) { .block-container { max-width: 1400px; } }
</style>"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">AI SEARCH LABORATORY</div>'
        '<h1>BEAM SEARCH</h1>'
        '<div class="hero-tag">The Intelligence Behind Limited Search</div>'
        '<div class="hero-sub">An Interactive AI Search and Decision-Making Explorer.</div></div>',
        unsafe_allow_html=True)


def plotly_layout(height: int = 460, margin: dict | None = None) -> dict:
    """Common Plotly layout: transparent background, light text, dark hover labels."""
    return dict(
        height=height,
        margin=margin or dict(l=8, r=8, t=30, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Nunito, Segoe UI, system-ui, sans-serif", size=12),
        hoverlabel=dict(bgcolor="#0b1226", bordercolor="#22d3ee", font=dict(color="#e6edf7", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
