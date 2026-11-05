"""Plotly rendering of the maze. A transparent scatter layer on top of the heatmap makes cells clickable."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

import plotly.graph_objects as go

from algorithms.grid import Cell, Grid
from components.theme import plotly_layout

# cell codes
EMPTY, OBSTACLE, EXPLORED, BEAM, PRUNED, PATH, START, GOAL = range(8)

CELL_COLORS = {
    EMPTY: "#15234a", OBSTACLE: "#010208", EXPLORED: "#3b82f6", BEAM: "#a855f7",
    PRUNED: "#6b7280", PATH: "#facc15", START: "#22c55e", GOAL: "#ef4444",
}
CELL_NAMES = {
    START: "Start", GOAL: "Goal", OBSTACLE: "Obstacle", EXPLORED: "Explored", BEAM: "Selected beam",
    PRUNED: "Pruned", PATH: "Final path",
}


def maze_legend() -> List[tuple]:
    return [(CELL_COLORS[c], CELL_NAMES[c]) for c in (START, GOAL, OBSTACLE, EXPLORED, BEAM, PRUNED, PATH)]


def _colorscale() -> list:
    n = len(CELL_COLORS)
    scale = []
    for code in range(n):
        scale += [[code / n, CELL_COLORS[code]], [(code + 1) / n, CELL_COLORS[code]]]
    return scale


def maze_figure(grid: Grid, explored: Iterable[Cell] = (), beam: Iterable[Cell] = (), pruned: Iterable[Cell] = (),
                path: Optional[Sequence[Cell]] = None, height: int = 460, clickable: bool = True) -> go.Figure:
    """Heatmap of the grid. Colour precedence: start/goal > path > beam > pruned > explored > obstacle > empty."""
    explored_s: Set[Cell] = set(explored)
    beam_s: Set[Cell] = set(beam)
    pruned_s: Set[Cell] = set(pruned)
    path_s: Set[Cell] = set(path or [])
    z: List[List[int]] = []
    hover: List[List[str]] = []
    for r in range(grid.rows):
        zr, hr = [], []
        for c in range(grid.cols):
            cell = (r, c)
            if cell == grid.start:
                code = START
            elif cell == grid.goal:
                code = GOAL
            elif cell in grid.obstacles:
                code = OBSTACLE
            elif cell in path_s:
                code = PATH
            elif cell in beam_s:
                code = BEAM
            elif cell in pruned_s:
                code = PRUNED
            elif cell in explored_s:
                code = EXPLORED
            else:
                code = EMPTY
            zr.append(code)
            name = CELL_NAMES.get(code, "Free")
            hr.append(f"cell (row {r}, col {c})<br>{name}<br>Manhattan h = {grid.manhattan(cell, grid.goal)}")
        z.append(zr)
        hover.append(hr)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, x=list(range(grid.cols)), y=list(range(grid.rows)), colorscale=_colorscale(),
                             zmin=-0.5, zmax=len(CELL_COLORS) - 0.5, showscale=False, xgap=2, ygap=2,
                             hoverinfo="skip"))
    xs = [c for r in range(grid.rows) for c in range(grid.cols)]
    ys = [r for r in range(grid.rows) for c in range(grid.cols)]
    texts = ["S" if (r, c) == grid.start else "G" if (r, c) == grid.goal else ""
             for r in range(grid.rows) for c in range(grid.cols)]
    cell_px = max(10, min(34, 440 // max(grid.rows, grid.cols)))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textfont=dict(color="#05110a", size=max(10, cell_px // 2)),
        marker=dict(size=cell_px, color="rgba(0,0,0,0.01)"),
        hovertext=[hover[r][c] for r in range(grid.rows) for c in range(grid.cols)], hoverinfo="text",
        showlegend=False, selected=dict(marker=dict(opacity=0.35, color="#22d3ee")),
        unselected=dict(marker=dict(opacity=1))))
    fig.update_layout(**plotly_layout(height, margin=dict(l=4, r=4, t=4, b=4)),
                      dragmode="select" if clickable else False, clickmode="event+select")
    fig.update_xaxes(visible=False, range=[-0.5, grid.cols - 0.5], fixedrange=True, constrain="domain")
    fig.update_yaxes(visible=False, range=[grid.rows - 0.5, -0.5], fixedrange=True, scaleanchor="x", scaleratio=1)
    return fig
