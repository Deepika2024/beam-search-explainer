"""Module 7 - academic analysis: pseudocode, formulation, complexity (with a calculator), completeness, optimality."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from algorithms.analysis import level_table, totals
from algorithms.beam_search import PSEUDOCODE, beam_search
from components.explanations import APPLICATIONS
from components.ui import callout, html_block, show_df, show_plot, stat_cards
from data.sample_search_trees import cost_trap_tree, failure_tree, uniform_tree
from visualizations.charts import growth_chart

ss = st.session_state


def _complexity() -> None:
    st.markdown("#### Assumptions (a bound means nothing without them)")
    st.markdown(
        "- Uniform branching factor **b** (every node has b children), search depth **d**, beam width **k**.\n"
        "- Worst case for time: the search runs all d levels without meeting a goal.\n"
        "- Evaluating h(n) and generating a child each cost O(1).\n"
        "- *Generated* nodes are all candidates created; *retained* nodes are the at most k that survive each level. "
        "They are different quantities and must not be confused.")
    st.markdown("**Per level i = 1…d**")
    st.latex(r"\text{expanded}_i=\min(b^{\,i-1},k),\qquad \text{generated}_i=b\cdot\text{expanded}_i=\min(b^{\,i},\,kb),"
             r"\qquad \text{retained}_i=\min(\text{generated}_i,\,k)")
    st.markdown("**Totals**")
    st.latex(r"N_{\text{generated}}=\sum_{i=1}^{d}\min(b^{\,i},\,kb)\;\le\; d\,k\,b")
    st.latex(r"T_{\text{beam}}=O\big(d\cdot kb\cdot\log(kb)\big)\ \text{(sorting each level's candidates)}\qquad"
             r"T_{\text{BFS}}=O(b^{d})")
    st.latex(r"S_{\text{beam}}=O(kb)\ \text{candidates}\ +\ O(k)\ \text{retained}\ (+\,O(kd)\ \text{for stored paths})\qquad"
             r"S_{\text{BFS}}=O(b^{d})")
    callout("info", "Once <b>b^(i-1) ≥ k</b> the beam is full and every further level costs the same, about k·b candidates. "
                    "So beam search grows <b>linearly</b> in depth, while BFS grows <b>exponentially</b>. "
                    "A selection algorithm can pick the top k in O(kb) rather than O(kb log kb); the log factor "
                    "depends on how the top-k is implemented.")

    st.markdown("#### Try it: closed form vs a real run")
    c1, c2, c3 = st.columns(3)
    b = c1.slider("Branching factor b", 2, 10, 3, key="m7_b")
    d = c2.slider("Depth d", 1, 12, 5, key="m7_d")
    k = c3.slider("Beam width k", 1, 50, 4, key="m7_k")
    rows = level_table(b, d, k)
    tot = totals(b, d, k)
    stat_cards([
        ("BFS nodes generated", f"{tot['bfs_generated']:,}", "sum of b^i"),
        ("Beam nodes generated", f"{tot['beam_generated']:,}", f"≤ d·k·b = {d * k * b:,}"),
        ("Beam nodes expanded", f"{tot['beam_expanded']:,}", "sum of min(b^(i-1), k)"),
        ("Beam peak retained", f"{tot['beam_peak_retained']:,}", "≤ k"),
        ("BFS peak frontier", f"{tot['bfs_peak_frontier']:,}", "b^d"),
    ], ["", "accent", "", "accent", "bad"])
    show_df(pd.DataFrame(rows).rename(columns={
        "depth": "Depth i", "bfs_frontier": "BFS nodes at depth i (b^i)", "beam_expanded": "Beam expanded",
        "beam_generated": "Beam generated (candidates)", "beam_retained": "Beam retained"}))
    show_plot(growth_chart([r["depth"] for r in rows], [r["bfs_frontier"] for r in rows],
                           [r["beam_generated"] for r in rows], k), key="m7_growth")

    # ---- validate the formulas against the actual algorithm ----------------------
    if sum(b ** i for i in range(d + 1)) <= 2000:
        tree = uniform_tree(b, d, seed=1)
        res = beam_search(tree, k, max_depth=d, record=False)
        ok = (int(res.stats["generated"]) == tot["beam_generated"] and int(res.stats["expanded"]) == tot["beam_expanded"])
        callout("ok" if ok else "bad",
                f"<b>Check against the real algorithm</b> on a complete {b}-ary tree of depth {d} (no goal): "
                f"measured generated = {int(res.stats['generated']):,}, expanded = {int(res.stats['expanded']):,}; "
                f"formula = {tot['beam_generated']:,} and {tot['beam_expanded']:,}. "
                + ("They match." if ok else "They differ - please report this."))
    else:
        st.caption("This tree is too large to build here, so only the closed form is shown.")


def _completeness_optimality() -> None:
    st.markdown("#### Completeness - generally **not** complete")
    st.markdown("An algorithm is *complete* if it finds a solution whenever one exists. Beam search throws candidates away "
                "for good, so the only route to the goal may be pruned before it is reached. "
                "*Exception:* if k is at least the size of the widest level, nothing is ever pruned and beam search "
                "explores like BFS.")
    tree = failure_tree()
    rows = []
    for kk in (1, 2, 3, 5):
        r = beam_search(tree, kk, record=False)
        rows.append({"k": kk, "Goal found": "✅ yes" if r.found else "❌ no",
                     "Why": "-" if r.found else r.failure_reason})
    st.caption("Live evidence - the failure-demo tree contains a goal, but:")
    show_df(pd.DataFrame(rows))

    st.markdown("#### Optimality - generally **not** optimal")
    st.markdown("Beam search ranks by h(n) only, ignores the cost already paid g(n), and stops at the first goal that "
                "survives pruning. A wider beam changes *which* goal is reached first - not necessarily to a better one.")
    tree = cost_trap_tree()
    rows = []
    for kk in (1, 2, 3):
        r = beam_search(tree, kk, record=False)
        rows.append({"k": kk, "Path": " → ".join(tree.label(n) for n in r.path), "Cost": r.cost})
    st.caption(f"Live evidence - cost-trap tree (optimal cost = {tree.optimal_cost():g}):")
    show_df(pd.DataFrame(rows))


def render() -> None:
    st.markdown("## Mathematical & Theoretical Analysis")
    tabs = st.tabs(["Pseudocode", "Formulation", "Complexity", "Completeness & optimality", "Pros, cons, uses"])

    with tabs[0]:
        st.code("\n".join(PSEUDOCODE), language="text")
        st.markdown("The goal test happens **after pruning**: a goal that is pruned is never seen. Ties between equal "
                    "scores are broken by generation order (stable sort), so results are deterministic.")

    with tabs[1]:
        st.markdown("**Search problem.** States S, start s₀, successor function children(n), goal test is_goal(n), "
                    "heuristic h(n) ≥ 0 (estimated remaining cost; *lower is better*).")
        st.latex(r"B_0=\{s_0\},\qquad C_{t+1}=\bigcup_{n\in B_t}\operatorname{children}(n),"
                 r"\qquad B_{t+1}=\operatorname{TopK}(C_{t+1},k)")
        st.latex(r"\operatorname{TopK}(C,k)=\text{the } \min(k,|C|) \text{ elements of } C \text{ with smallest } f(n)=h(n)")
        st.latex(r"\text{pruned}_{t+1}=C_{t+1}\setminus B_{t+1}\quad(\text{never revisited})")
        st.latex(r"\text{terminate when } B_t\cap \text{Goals}\neq\emptyset\ \ (\text{success})"
                 r"\ \text{ or } C_{t+1}=\emptyset\ \ (\text{failure})")
        st.markdown("- **Heuristic function** h(n): any estimate of promise. The better it correlates with true remaining cost, the "
                    "smaller a beam you need.\n"
                    "- **Beam width** k: the only structural parameter; k = 1 is a greedy walk without backtracking.\n"
                    "- **Pruning strategy**: rank-based (keep the best k), as opposed to threshold-based beams that keep "
                    "everything within some margin of the best score.\n"
                    "- For sequence models the score is the cumulative log-probability (higher is better); see Module 5.")

    with tabs[2]:
        _complexity()

    with tabs[3]:
        _completeness_optimality()

    with tabs[4]:
        a, b = st.columns(2)
        with a:
            html_block('<div class="card"><div class="card-title">Advantages</div><ul>'
                       "<li>Memory bounded by k instead of exponential in depth.</li>"
                       "<li>Time per level bounded by about k·b candidates.</li>"
                       "<li>k is a simple dial between speed and thoroughness.</li>"
                       "<li>Works with any scoring function, including probabilities from neural networks.</li>"
                       "<li>Often finds good solutions even when the heuristic is imperfect.</li></ul></div>")
        with b:
            html_block('<div class="card"><div class="card-title">Disadvantages</div><ul>'
                       "<li>Not complete: pruning can remove the only path to the goal.</li>"
                       "<li>Not optimal, and a larger k can still return a worse answer than a smaller k.</li>"
                       "<li>Depends on the quality of h; a misleading heuristic misleads the beam.</li>"
                       "<li>Duplicate states must be handled explicitly in graphs (this lab closes visited cells).</li>"
                       "<li>Choosing k is a tuning problem with no universal answer.</li></ul></div>")
        st.markdown("#### Real-world applications")
        for t, txt in APPLICATIONS:
            html_block(f'<div class="card"><div class="card-title">{t}</div><p>{txt}</p></div>')
