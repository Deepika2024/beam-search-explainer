"""Module 1 - concepts, the selection rule, and an animated walk-through driven by the real algorithm."""
from __future__ import annotations

import textwrap

import pandas as pd
import streamlit as st

from algorithms.beam_search import SELECT, beam_search
from components.explanations import (APPLICATIONS, COMPARISON_TABLE, CONCEPT_QA, SCORE_DIRECTION)
from components.ui import callout, html_block, legend, show_df, show_plot
from data.sample_search_trees import demo_tree
from visualizations.search_tree import animated_tree, legend_entries, nodes_from_step


def _walkthrough_table(tree, result) -> pd.DataFrame:
    """One row per level, read straight from the recorded run."""
    rows = []
    for s in result.steps:
        if s.phase != SELECT:
            continue
        rows.append({
            "Depth": s.depth,
            "Candidates C(t+1) with h": ", ".join(f"{tree.label(c)}={tree.nodes[c].h:g}" for c in s.candidates),
            f"Kept B(t+1) (k={result.k})": ", ".join(tree.label(c) for c in s.selected),
            "Pruned": ", ".join(tree.label(c) for c in s.pruned) or "-",
        })
    return pd.DataFrame(rows)


def render() -> None:
    st.markdown("## Understand Beam Search")
    callout("info", "<b>In one sentence:</b> Beam Search explores a tree level by level, but at every level it keeps "
                    "only the <b>k most promising nodes</b> and permanently forgets the rest. It is fast and "
                    "memory-light - and it can lose the answer.")

    # ---- animated concept ---------------------------------------------------
    st.markdown("### Watch it work")
    st.caption("Small tree, real run. Press Play, or drag the slider. Lower heuristic value h = more promising.")
    k = st.select_slider("Beam width k for the animation", options=[1, 2, 3], value=2, key="m1_k")
    tree = demo_tree()
    result = beam_search(tree, k)
    frames = [nodes_from_step(tree, s) for s in result.steps]
    captions = ["<br>".join(textwrap.wrap(f"Step {s.index}: {s.message}", 92)) for s in result.steps]
    show_plot(animated_tree(frames, captions, height=470), key="m1_anim")
    legend(legend_entries())

    st.markdown("#### The same run, level by level")
    show_df(_walkthrough_table(tree, result))
    if result.found:
        st.success("Goal found: " + " → ".join(tree.label(n) for n in result.path) + f"  (path cost {result.cost:g})")

    # ---- the rule -----------------------------------------------------------
    st.markdown("### The selection rule")
    st.latex(r"B_{t+1} \;=\; \operatorname{TopK}\!\big(C_{t+1},\,k\big)")
    st.latex(r"C_{t+1} \;=\; \bigcup_{n \in B_t} \operatorname{children}(n)"
             r"\qquad\qquad f(n) = h(n)")
    st.markdown("""
| Symbol | Meaning |
|---|---|
| **B(t)** | the **beam** at depth *t*: the set of nodes that survived pruning. B(0) = {start}. |
| **C(t+1)** | the **candidate set**: every child of every node in B(t). |
| **k** | the **beam width** (a positive integer). |
| **TopK(S, k)** | the *k* elements of *S* with the best score. If |S| < k, all of them. Ties are broken by generation order. |
| **h(n)** | the **heuristic** score of node *n*, e.g. estimated distance to the goal. |
| **f(n) = h(n)** | beam search ranks candidates by the heuristic alone (no path cost, unlike A*). |
""")
    st.markdown("**Which direction is better?** It depends on what the score means - this lab always says which:")
    show_df(SCORE_DIRECTION)

    # ---- concepts -----------------------------------------------------------
    st.markdown("### The ideas, one question at a time")
    for i, (q, a) in enumerate(CONCEPT_QA):
        with st.expander(q, expanded=(i == 2)):
            st.markdown(a)

    st.markdown("### Beam Search vs BFS vs Greedy Best-First")
    show_df(COMPARISON_TABLE)
    st.caption("Beam Search is a trade-off, not an upgrade: it is not better than BFS or Greedy in every respect. "
               "Module 6 runs all three on the same problem.")

    st.markdown("### Where it is used")
    for title, text in APPLICATIONS:
        html_block(f'<div class="card"><div class="card-title">{title}</div><p>{text}</p></div>')
