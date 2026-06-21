"""Self-contained editorial Plotly theme for the package's own figures.

The package ships its own light theme (no external/silico dependency) so
``report.curve(...)`` etc. look consistent in any notebook. Palette is the
"Editorial 8" colorway.
"""

from __future__ import annotations

import plotly.graph_objects as go

EDITORIAL_8 = [
    "#C4650D",  # ember — primary
    "#4E728A",  # slate
    "#2E6E4E",  # forest — good
    "#988453",  # wheat
    "#B9605B",  # rose — bad
    "#7495AB",  # pale-slate
    "#84713A",  # olive
    "#31362E",  # graphite
]

SEQUENTIAL = [
    [0.00, "#F0E2D0"],
    [0.35, "#EDD8C5"],
    [0.65, "#DE9D50"],
    [1.00, "#C4650D"],
]

_FONT = "'Suisse Intl', -apple-system, BlinkMacSystemFont, Arial, sans-serif"


def apply_theme(fig: go.Figure, *, height: int = 420) -> go.Figure:
    """Apply the editorial cream theme + safe layout defaults to a figure."""
    fig.update_layout(
        height=height,
        margin=dict(t=48, r=24, b=80, l=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1D272A", family=_FONT, size=13),
        colorway=EDITORIAL_8,
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.22,
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1D272A", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#B4B4B4",
            font=dict(family="ui-monospace, Menlo, monospace", size=12, color="#1D272A"),
        ),
    )
    fig.update_xaxes(
        gridcolor="#E2E0D8",
        zerolinecolor="#B4B4B4",
        automargin=True,
        ticks="outside",
        nticks=8,
    )
    fig.update_yaxes(
        gridcolor="#E2E0D8",
        zerolinecolor="#B4B4B4",
        automargin=True,
        ticks="outside",
        nticks=8,
    )
    fig.update_layout(title=None)
    return fig


PLOTLY_CONFIG = {
    "responsive": False,
    "displayModeBar": "hover",
    "displaylogo": False,
}
