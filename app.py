"""BEAM SEARCH - The Intelligence Behind Limited Search.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Beam Search Lab", page_icon="🔦", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import PAGES, render_sidebar          # noqa: E402  (after set_page_config)
from components.theme import hero, inject_css                  # noqa: E402
from modules import (module1_understand, module2_tree_explorer, module3_beam_width, module4_maze,  # noqa: E402
                     module5_text_generation, module6_comparison, module7_theory, module8_quiz,
                     module9_3d_simulation)

MODULES = {
    PAGES[0]: module1_understand, PAGES[1]: module2_tree_explorer, PAGES[2]: module3_beam_width,
    PAGES[3]: module4_maze, PAGES[4]: module5_text_generation, PAGES[5]: module6_comparison,
    PAGES[6]: module7_theory, PAGES[7]: module8_quiz, PAGES[8]: module9_3d_simulation,
}

inject_css()
page = render_sidebar()
hero()
MODULES[page].render()
