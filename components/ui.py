"""Small UI helpers: Streamlit version compatibility and HTML snippets."""
from __future__ import annotations

import html
from typing import Any, Iterable, Optional, Sequence, Tuple

import streamlit as st


def _call_stretched(fn, *args, **kwargs):
    """Call a widget so that it fills its container, whatever the Streamlit version.

    Newer releases use ``width="stretch"``; older ones use ``use_container_width=True``
    (and some old widgets have an integer ``width`` that rejects "stretch").
    Each variant is tried in turn; the final attempt passes no sizing argument at all.
    """
    last_error = None
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return fn(*args, **extra, **kwargs)
        except (TypeError, st.errors.StreamlitAPIException) as exc:   # unsupported keyword in this version
            last_error = exc
    raise last_error


def show_plot(fig, key: Optional[str] = None, **kwargs):
    """Render a Plotly figure (full width by default in every Streamlit version) without the mode bar."""
    return st.plotly_chart(fig, key=key, config={"displayModeBar": False, "responsive": True}, **kwargs)


def show_df(df, **kwargs) -> None:
    _call_stretched(st.dataframe, df, hide_index=True, **kwargs)


def button(label: str, **kwargs) -> bool:
    return _call_stretched(st.button, label, **kwargs)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    html_block(f'<div class="card"><div class="card-title">{html.escape(title)}</div>{body_html}</div>')


def callout(kind: str, body_html: str) -> None:
    """kind: info | ok | warn | bad"""
    html_block(f'<div class="callout {kind}">{body_html}</div>')


def stat_cards(items: Sequence[Tuple[str, Any, str]], classes: Optional[Sequence[str]] = None) -> None:
    """Row of metric cards. ``items`` = (label, value, hint)."""
    cells = []
    for i, (label, value, hint) in enumerate(items):
        cls = classes[i] if classes and i < len(classes) else ""
        cells.append(f'<div class="stat {cls}"><div class="l">{html.escape(str(label))}</div>'
                     f'<div class="v">{html.escape(str(value))}</div>'
                     f'<div class="h">{html.escape(str(hint))}</div></div>')
    html_block('<div class="stat-grid">' + "".join(cells) + "</div>")


def legend(entries: Iterable[Tuple[str, str]]) -> None:
    """entries = (colour, text)."""
    chips = "".join(f'<span class="chip"><span class="dot" style="background:{c}"></span>{html.escape(t)}</span>'
                    for c, t in entries)
    html_block(f'<div class="legend">{chips}</div>')
