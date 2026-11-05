"""Plotly rendering of search trees (explicit trees and the text-generation hypothesis tree).

Every node gets a *halo* (large translucent marker) under a solid *core* marker, which
produces the glowing look. Traces are always created in the same fixed order, even
when empty, so animation frames stay consistent.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go

from algorithms.beam_search import (BEAM, EXPANDING, EXPLORED, PATH, PENDING, PRUNED, UNSEEN, BeamStep)
from algorithms.tree import SearchTree
from components.theme import plotly_layout

STATUS_ORDER = [UNSEEN, PENDING, EXPANDING, BEAM, EXPLORED, PRUNED, PATH]

STATUS_STYLE: Dict[str, dict] = {
    UNSEEN:    dict(fill="rgba(30,41,59,0.30)", line="#334155", glow="rgba(0,0,0,0)", text="#64748b", name="Not generated yet"),
    PENDING:   dict(fill="rgba(34,211,238,0.16)", line="#22d3ee", glow="rgba(34,211,238,0.22)", text="#a5f3fc", name="Candidate (pending)"),
    EXPANDING: dict(fill="#f59e0b", line="#fde68a", glow="rgba(245,158,11,0.34)", text="#1f1300", name="Being expanded"),
    BEAM:      dict(fill="#a855f7", line="#e9d5ff", glow="rgba(168,85,247,0.36)", text="#ffffff", name="Selected (in beam)"),
    EXPLORED:  dict(fill="#2563eb", line="#93c5fd", glow="rgba(59,130,246,0.22)", text="#ffffff", name="Explored"),
    PRUNED:    dict(fill="#475569", line="#94a3b8", glow="rgba(0,0,0,0)", text="#e2e8f0", name="Pruned (discarded)"),
    PATH:      dict(fill="#facc15", line="#fef9c3", glow="rgba(250,204,21,0.42)", text="#1c1400", name="Final path"),
}

EDGE_STYLE: Dict[str, dict] = {
    UNSEEN:    dict(color="rgba(71,85,105,0.30)", width=1, dash="dot"),
    PENDING:   dict(color="rgba(34,211,238,0.55)", width=1.5, dash="dash"),
    EXPANDING: dict(color="rgba(245,158,11,0.85)", width=2.5, dash="solid"),
    BEAM:      dict(color="rgba(168,85,247,0.90)", width=2.5, dash="solid"),
    EXPLORED:  dict(color="rgba(59,130,246,0.65)", width=2, dash="solid"),
    PRUNED:    dict(color="rgba(107,114,128,0.55)", width=1.2, dash="dash"),
    PATH:      dict(color="rgba(250,204,21,0.95)", width=5, dash="solid"),
}


def legend_entries(statuses: Sequence[str] = STATUS_ORDER) -> List[Tuple[str, str]]:
    return [(STATUS_STYLE[s]["fill"] if s != UNSEEN else "#334155", STATUS_STYLE[s]["name"]) for s in statuses]


@dataclass
class NodeView:
    """Everything needed to draw one node, independent of where it came from."""
    id: int
    parent: Optional[int]
    label: str
    value: str                 # short text drawn inside the node (e.g. the h-value)
    status: str
    depth: int
    is_goal: bool = False
    hover: str = ""
    edge_text: str = ""        # text on the edge parent -> node (e.g. a probability)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
def layout_positions(nodes: Sequence[NodeView]) -> Dict[int, Tuple[float, float]]:
    """Tidy top-down layout: leaves get consecutive x slots, parents sit above the mean of their children."""
    children: Dict[int, List[int]] = {n.id: [] for n in nodes}
    root = None
    for n in nodes:
        if n.parent is None:
            root = n.id
        else:
            children[n.parent].append(n.id)
    pos: Dict[int, Tuple[float, float]] = {}
    depth = {n.id: n.depth for n in nodes}
    counter = [0.0]

    def place(nid: int) -> float:
        kids = children[nid]
        if not kids:
            x = counter[0]
            counter[0] += 1.0
        else:
            xs = [place(c) for c in kids]
            x = sum(xs) / len(xs)
        pos[nid] = (x, -float(depth[nid]))
        return x

    if root is not None:
        place(root)
    return pos


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def _marker_size(width_units: float) -> float:
    return max(9.0, min(40.0, 640.0 / max(width_units, 1.0) * 0.82))


def build_traces(nodes: Sequence[NodeView], pos: Dict[int, Tuple[float, float]], show_unseen: bool = True,
                 size: float = 30.0, show_edge_text: bool = False) -> list:
    """Return the fixed list of traces: 7 edge traces, 7x(halo, core) node traces, 1 label trace, 1 edge-text trace."""
    traces: list = []
    by_status: Dict[str, List[NodeView]] = {s: [] for s in STATUS_ORDER}
    for n in nodes:
        by_status[n.status].append(n)

    for s in STATUS_ORDER:                                     # edges, coloured by the status of the child
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        if not (s == UNSEEN and not show_unseen):
            for n in by_status[s]:
                if n.parent is None:
                    continue
                (x0, y0), (x1, y1) = pos[n.parent], pos[n.id]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
        e = EDGE_STYLE[s]
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=e["color"], width=e["width"], dash=e["dash"]),
                                 hoverinfo="skip", showlegend=False))

    show_value = size >= 22
    font_size = 11 if size >= 30 else 9
    for s in STATUS_ORDER:                                     # halo + core per status
        group = [] if (s == UNSEEN and not show_unseen) else by_status[s]
        st_ = STATUS_STYLE[s]
        x = [pos[n.id][0] for n in group]
        y = [pos[n.id][1] for n in group]
        traces.append(go.Scatter(x=x, y=y, mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=size * 1.75, color=st_["glow"], line=dict(width=0))))
        traces.append(go.Scatter(
            x=x, y=y, mode="markers+text" if show_value else "markers",
            text=[n.value for n in group] if show_value else None, textposition="middle center",
            textfont=dict(color=st_["text"], size=font_size),
            marker=dict(size=[size * (1.22 if n.is_goal else 1.0) for n in group], color=st_["fill"],
                        symbol=["star" if n.is_goal else "circle" for n in group],
                        line=dict(color=st_["line"], width=1.6)),
            hovertext=[n.hover for n in group], hoverinfo="text", showlegend=False))

    visible = [n for n in nodes if show_unseen or n.status != UNSEEN]
    show_labels = size >= 16
    # An invisible marker the size of the node makes Plotly place the text *above* the circle, not on it.
    traces.append(go.Scatter(
        x=[pos[n.id][0] for n in visible], y=[pos[n.id][1] for n in visible],
        mode="markers+text" if show_labels else "markers", text=[n.label for n in visible] if show_labels else None,
        textposition="top center",
        textfont=dict(size=10, color=["#64748b" if n.status == UNSEEN else "#cbd5e1" for n in visible]),
        marker=dict(size=size, color="rgba(0,0,0,0)", line=dict(width=0)), hoverinfo="skip", showlegend=False))

    et = [n for n in nodes if n.parent is not None and n.edge_text and (show_unseen or n.status != UNSEEN)]
    traces.append(go.Scatter(
        x=[(pos[n.id][0] + pos[n.parent][0]) / 2 for n in et], y=[(pos[n.id][1] + pos[n.parent][1]) / 2 for n in et],
        mode="text" if show_edge_text else "markers", text=[n.edge_text for n in et] if show_edge_text else None,
        textfont=dict(size=9, color="#94a3b8"), marker=dict(size=0.1, color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False))
    return traces


def _apply_layout(fig: go.Figure, pos: Dict[int, Tuple[float, float]], height: int, title: Optional[str]) -> None:
    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    fig.update_layout(**plotly_layout(height), dragmode=False)
    fig.update_xaxes(visible=False, range=[min(xs) - 0.7, max(xs) + 0.7], fixedrange=True)
    fig.update_yaxes(visible=False, range=[min(ys) - 0.55, max(ys) + 0.75], fixedrange=True)
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=13, color="#8a9ac2")))


def _size_for(pos: Dict[int, Tuple[float, float]]) -> float:
    xs = [p[0] for p in pos.values()] or [0.0]
    return _marker_size(max(xs) - min(xs) + 1.4)


def draw_tree(nodes: Sequence[NodeView], show_unseen: bool = True, height: int = 460,
              title: Optional[str] = None, show_edge_text: bool = False) -> go.Figure:
    """Static figure for one snapshot."""
    pos = layout_positions(nodes)
    fig = go.Figure(build_traces(nodes, pos, show_unseen, _size_for(pos), show_edge_text))
    _apply_layout(fig, pos, height, title)
    return fig


def animated_tree(frames: Sequence[Sequence[NodeView]], captions: Sequence[str], height: int = 460,
                  duration_ms: int = 1400) -> go.Figure:
    """Figure with Play / Pause buttons and a slider that steps through ``frames``."""
    pos = layout_positions(frames[0])
    size = _size_for(pos)
    fig = go.Figure(build_traces(frames[0], pos, True, size))
    _apply_layout(fig, pos, height, None)
    cap = lambda t: [dict(text=t, x=0.5, y=-0.02, xref="paper", yref="paper", showarrow=False,
                          font=dict(size=13, color="#e6edf7"), align="center")]
    fig.frames = [go.Frame(data=build_traces(f, pos, True, size), name=str(i), layout=go.Layout(annotations=cap(captions[i])))
                  for i, f in enumerate(frames)]
    fig.update_layout(
        annotations=cap(captions[0]),
        margin=dict(l=8, r=8, t=10, b=96),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", yanchor="top",
                          bgcolor="#0f1a38", bordercolor="#22d3ee", font=dict(color="#e6edf7"),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=duration_ms, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.22, len=0.78, y=-0.1, yanchor="top", currentvalue=dict(visible=False),
                      font=dict(color="#8a9ac2"), bgcolor="#1e2a4a", activebgcolor="#a855f7",
                      steps=[dict(method="animate", label=str(i),
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))]) for i in range(len(frames))])])
    return fig


# --------------------------------------------------------------------------
# conversions from algorithm output to NodeViews
# --------------------------------------------------------------------------
def _g_cost(tree: SearchTree, nid: int) -> float:
    return tree.path_cost(nid)


def nodes_from_status(tree: SearchTree, status: Dict[int, str]) -> List[NodeView]:
    """NodeViews for an explicit tree given a node-id -> status mapping."""
    out = []
    for nid, n in tree.nodes.items():
        s = status.get(nid, UNSEEN)
        hover = (f"<b>{html.escape(n.label)}</b><br>h = {n.h:g}<br>depth = {n.depth}"
                 f"<br>path cost g = {_g_cost(tree, nid):g}<br>{STATUS_STYLE[s]['name']}"
                 + ("<br><b>GOAL</b>" if n.is_goal else ""))
        out.append(NodeView(nid, n.parent, n.label, f"{n.h:g}", s, n.depth, n.is_goal, hover))
    return out


def nodes_from_step(tree: SearchTree, step: BeamStep) -> List[NodeView]:
    return nodes_from_status(tree, step.status)


def status_from_summary(tree: SearchTree, summary) -> Dict[int, str]:
    """Status map for any algorithm summary (used by the comparison lab)."""
    status: Dict[int, str] = {}
    expanded = set(summary.expanded_order)
    on_path = set(summary.path)
    for nid in tree.nodes:
        if nid in on_path:
            status[nid] = PATH
        elif nid in summary.pruned_ids:
            status[nid] = PRUNED
        elif nid in expanded:
            status[nid] = EXPLORED
        elif nid in summary.generated_ids:
            status[nid] = PENDING
        else:
            status[nid] = UNSEEN
    return status
