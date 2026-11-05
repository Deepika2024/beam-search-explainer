"""3D view for the four-directional Beam Search grid simulation."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import plotly.graph_objects as go

from algorithms.grid import Cell, Grid


PALETTE = {
    "free": "#25334b", "start": "#22c55e", "goal": "#f43f5e",
    "obstacle": "#64748b", "explored": "#38bdf8", "beam": "#a78bfa",
    "pruned": "#f59e0b", "path": "#facc15",
}


def _cube(x: float, y: float, z: float, size: float = 0.78, height: float = 1.25):
    """Return vertices/faces for one raised cube centered at (x,y)."""
    h = size / 2
    vertices = [
        (x-h, y-h, z), (x+h, y-h, z), (x+h, y+h, z), (x-h, y+h, z),
        (x-h, y-h, z+height), (x+h, y-h, z+height),
        (x+h, y+h, z+height), (x-h, y+h, z+height),
    ]
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return vertices, faces


def maze_3d_figure(
    grid: Grid,
    explored: Iterable[Cell] = (),
    beam: Iterable[Cell] = (),
    pruned: Iterable[Cell] = (),
    path: Optional[Sequence[Cell]] = None,
    height: int = 680,
) -> go.Figure:
    """Render the grid as a raised 3D maze; legal moves remain N/E/S/W in its floor plane."""
    explored_s, beam_s = set(explored), set(beam)
    pruned_s, path_s = set(pruned), set(path or [])
    fig = go.Figure()

    # Raised obstacle blocks make the environment visibly three-dimensional.
    for r, c in sorted(grid.obstacles):
        verts, faces = _cube(c, -r, 0.02, height=1.15)
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in verts], y=[v[1] for v in verts], z=[v[2] for v in verts],
            i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
            color=PALETTE["obstacle"], opacity=0.95, flatshading=True,
            hovertext=f"Obstacle at row {r}, column {c}", hoverinfo="text", showscale=False,
            name="Obstacle", showlegend=False,
        ))

    # One square tile per traversable cell. The z offsets distinguish live states.
    groups = {name: [] for name in ("free", "explored", "beam", "pruned", "path", "start", "goal")}
    for r in range(grid.rows):
        for c in range(grid.cols):
            cell = (r, c)
            if cell in grid.obstacles:
                continue
            if cell == grid.start:
                group = "start"
            elif cell == grid.goal:
                group = "goal"
            elif cell in path_s:
                group = "path"
            elif cell in beam_s:
                group = "beam"
            elif cell in pruned_s:
                group = "pruned"
            elif cell in explored_s:
                group = "explored"
            else:
                group = "free"
            groups[group].append((c, -r, 0.13 if group in ("beam", "path", "start", "goal") else 0.045,
                                  f"{group.title()}<br>Row: {r}, Column: {c}<br>h(n): {grid.manhattan(cell, grid.goal)}"))

    for group, pts in groups.items():
        if not pts:
            continue
        marker_size = 15 if group in ("start", "goal") else 11
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in pts], y=[p[1] for p in pts], z=[p[2] for p in pts],
            mode="markers+text" if group in ("start", "goal") else "markers",
            text=["S" if group == "start" else "G" for _ in pts] if group in ("start", "goal") else None,
            textposition="top center", textfont=dict(color="white", size=13),
            marker=dict(size=marker_size, color=PALETTE[group], symbol="square",
                        line=dict(color="#dbeafe", width=0.6)),
            hovertext=[p[3] for p in pts], hoverinfo="text", name=group.title(),
        ))

    # Draw the current best path as a bright elevated line.
    if path and len(path) > 1:
        fig.add_trace(go.Scatter3d(
            x=[c for r, c in path], y=[-r for r, c in path], z=[0.32] * len(path),
            mode="lines+markers", line=dict(color=PALETTE["path"], width=9),
            marker=dict(size=4, color=PALETTE["path"]), name="Final path",
            hovertemplate="Path cell<extra></extra>",
        ))

    # Four compass arrows communicate the permitted movement directions.
    sr, sc = grid.start
    arrow_specs = [
        (sc, -sr, 0.55, 0, 0.8, 0, "North / Up", "#22d3ee"),
        (sc, -sr, 0.55, 0, -0.8, 0, "South / Down", "#22d3ee"),
        (sc, -sr, 0.55, 0.8, 0, 0, "East / Right", "#c084fc"),
        (sc, -sr, 0.55, -0.8, 0, 0, "West / Left", "#c084fc"),
    ]
    for x, y, z, u, v, w, label, color in arrow_specs:
        fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[u], v=[v], w=[w],
                              anchor="tail", sizemode="absolute", sizeref=0.34,
                              colorscale=[[0, color], [1, color]], showscale=False,
                              name=label, showlegend=False, hovertext=label, hoverinfo="text"))

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=25, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe", family="Nunito, Segoe UI, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(title="East  ←  Column  →  West", showbackground=False, gridcolor="#334155",
                       zerolinecolor="#475569", dtick=1),
            yaxis=dict(title="South  ←  Row  →  North", showbackground=False, gridcolor="#334155",
                       zerolinecolor="#475569", dtick=1),
            zaxis=dict(title="Elevation", range=[-0.1, 1.8], showbackground=False,
                       gridcolor="#334155", zerolinecolor="#475569", showticklabels=False),
            aspectmode="manual",
            aspectratio=dict(x=max(1, grid.cols), y=max(1, grid.rows), z=max(4, min(grid.rows, grid.cols) * 0.42)),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=1.35)),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", y=1.02, x=0),
        title=dict(text="3D Four-Direction Beam Search Environment", x=0.02, y=0.98,
                   font=dict(size=17, color="#e2e8f0")),
    )
    return fig
