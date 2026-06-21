"""Per-check curve figures (energy & force vs the swept coordinate)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlip_smoothness_eval.checks.base import CheckResult
from mlip_smoothness_eval.viz.theme import EDITORIAL_8, apply_theme


def _two_panel(x, y_top, y_bot, *, xlabel, top_label, bot_label, height=520) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=x, y=y_top, mode="lines+markers", name=top_label,
                             line=dict(color=EDITORIAL_8[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y_bot, mode="lines+markers", name=bot_label,
                             line=dict(color=EDITORIAL_8[1])), row=2, col=1)
    fig.update_yaxes(title_text=top_label, row=1, col=1)
    fig.update_yaxes(title_text=bot_label, row=2, col=1)
    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    apply_theme(fig, height=height)
    return fig


def curve(result: CheckResult) -> go.Figure:
    """Build the energy/force curve figure for a check result."""
    name = result.name
    t = result.trace
    if name == "diatomic":
        return _two_panel(
            t["x"], t["energy"], t["force"],
            xlabel=t.get("xlabel", "bond length (Å)"),
            top_label="energy (eV)",
            bot_label="axial force (eV/Å)",
        )
    if name == "displacement_scan":
        return _two_panel(
            t["x"], t["energy"], t["proj_force"],
            xlabel=t.get("xlabel", "displacement s (Å)"),
            top_label="energy (eV)",
            bot_label="-F·d (eV/Å)",
        )
    if name == "cutoff_smoothness":
        return _two_panel(
            t["x"], t["energy"], t["force"],
            xlabel=t.get("xlabel", "drag distance (Å)"),
            top_label="energy (eV)",
            bot_label="|force on atom 0| (eV/Å)",
        )
    if name == "nve_drift":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t["x"], y=t["drift_per_atom"], mode="lines",
                                 name="total-E drift / atom", line=dict(color=EDITORIAL_8[0])))
        fig.add_hline(y=0.0, line=dict(color="#B4B4B4", dash="dot"))
        fig.update_xaxes(title_text=t.get("xlabel", "time (fs)"))
        fig.update_yaxes(title_text="total-energy drift / atom (eV)")
        apply_theme(fig, height=420)
        return fig
    if name == "force_jacobian":
        jac = np.asarray(t["jacobian"])
        asym = jac - jac.T
        fig = go.Figure(go.Heatmap(z=asym, colorscale="RdBu", zmid=0.0,
                                   colorbar=dict(title="J - Jᵀ")))
        fig.update_xaxes(title_text="position dof j")
        fig.update_yaxes(title_text="force dof i", autorange="reversed")
        apply_theme(fig, height=460)
        return fig
    raise ValueError(f"no curve renderer for check {name!r}")
