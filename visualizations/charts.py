"""Generic Plotly charts for comparisons and complexity."""
from __future__ import annotations

from typing import Dict, Sequence

import plotly.graph_objects as go

from components.theme import COLORS, plotly_layout


def bars(x: Sequence, y: Sequence[float], title: str, color: str = COLORS["cyan"], height: int = 240,
         highlight_x=None, text_fmt: str = "{:g}") -> go.Figure:
    """Single bar chart; the bar whose x equals ``highlight_x`` is drawn in purple."""
    colors = [COLORS["purple"] if (highlight_x is not None and xi == highlight_x) else color for xi in x]
    fig = go.Figure(go.Bar(x=[str(v) for v in x], y=list(y), marker=dict(color=colors, line=dict(width=0)),
                           text=[text_fmt.format(v) if v is not None else "" for v in y], textposition="outside",
                           cliponaxis=False, hoverinfo="skip"))
    fig.update_layout(**plotly_layout(height, margin=dict(l=8, r=8, t=34, b=8)),
                      title=dict(text=title, x=0.01, font=dict(size=13, color="#8a9ac2")), bargap=0.35)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(30,42,74,.7)", zeroline=False, fixedrange=True)
    fig.update_xaxes(fixedrange=True, title=None)
    return fig


def grouped_bars(categories: Sequence[str], series: Dict[str, Sequence[float]], title: str, height: int = 280,
                 colors: Sequence[str] = (COLORS["cyan"], COLORS["purple"], COLORS["amber"], COLORS["green"])) -> go.Figure:
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        fig.add_trace(go.Bar(name=name, x=list(categories), y=list(ys), marker=dict(color=colors[i % len(colors)]),
                             text=[f"{v:g}" for v in ys], textposition="outside", cliponaxis=False))
    fig.update_layout(**plotly_layout(height), barmode="group", bargap=0.3,
                      title=dict(text=title, x=0.01, font=dict(size=13, color="#8a9ac2")))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(30,42,74,.7)", zeroline=False, fixedrange=True)
    fig.update_xaxes(fixedrange=True)
    return fig


def growth_chart(depths: Sequence[int], bfs: Sequence[float], beam: Sequence[float], k: int, height: int = 340) -> go.Figure:
    """Log-scale comparison of nodes generated per level: exponential BFS vs bounded beam."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(depths), y=list(bfs), mode="lines+markers", name="BFS: b^d nodes at depth d",
                             line=dict(color=COLORS["blue"], width=3)))
    fig.add_trace(go.Scatter(x=list(depths), y=list(beam), mode="lines+markers",
                             name=f"Beam (k={k}): candidates per level", line=dict(color=COLORS["purple"], width=3)))
    fig.update_layout(**plotly_layout(height), xaxis_title="depth", yaxis_title="nodes (log scale)")
    fig.update_yaxes(type="log", showgrid=True, gridcolor="rgba(30,42,74,.7)", fixedrange=True)
    fig.update_xaxes(dtick=1, fixedrange=True)
    return fig


def score_bars(labels: Sequence[str], scores: Sequence[float], kept: Sequence[bool], title: str, height: int = 300) -> go.Figure:
    colors = [COLORS["purple"] if k else COLORS["grey"] for k in kept]
    fig = go.Figure(go.Bar(x=list(scores), y=list(labels), orientation="h", marker=dict(color=colors),
                           text=[f"{s:.2f}" for s in scores], textposition="outside", cliponaxis=False, hoverinfo="skip"))
    fig.update_layout(**plotly_layout(height, margin=dict(l=8, r=40, t=34, b=8)),
                      title=dict(text=title, x=0.01, font=dict(size=13, color="#8a9ac2")))
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(30,42,74,.7)", fixedrange=True, title="score (higher = better)")
    return fig
