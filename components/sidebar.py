"""Sidebar navigation."""
from __future__ import annotations

import streamlit as st

PAGES = [
    "1 · Understand Beam Search",
    "2 · Search Tree Explorer",
    "3 · Beam Width Playground",
    "4 · Maze Solver",
    "5 · Text Generation",
    "6 · Algorithm Comparison",
    "7 · Theory & Analysis",
    "8 · Quiz Arena",
    "9 · 3D Four-Direction Simulator",
]


def reset_all_state() -> None:
    """Forget the progress of every module (keeps the current page)."""
    for key in list(st.session_state.keys()):
        if key != "nav" and key[:1] == "m" and key[1:2].isdigit() and key[2:3] == "_":
            del st.session_state[key]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🔦 Beam Search Lab")
        st.caption("Pick a module. Every control runs the real algorithm.")
        page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.button("↺ Reset all demos", key="reset_all", on_click=reset_all_state,
                  help="Clears the progress of every module (tree, 2D/3D maze, text, quiz).")
        st.caption("Rule used in tree and maze demos: **lower h(n) is better**. "
                   "In the text demo: **higher log-probability is better**.")
    return page
