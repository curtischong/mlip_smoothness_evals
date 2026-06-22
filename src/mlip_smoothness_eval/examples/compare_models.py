"""Aggregate per-model smoothness JSON into one self-contained comparison page.

This is the env-agnostic half of the comparison: it imports no MLIP, only
plotly. It reads every ``*.json`` written by ``compare_worker.py`` and renders
(1) a metrics heatmap across models and (2) overlaid per-check curves, into a
single standalone ``comparison.html``.

    uv run python src/mlip_smoothness_eval/examples/compare_models.py   # reads comparison_results/
    uv run python src/mlip_smoothness_eval/examples/compare_models.py --outdir results --html out.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# This env ships a hollow `orjson` namespace (no real module); plotly's auto-
# detect picks it and crashes serializing figures. Force the stdlib json engine.
pio.json.config.default_engine = "json"

from mlip_smoothness_eval.report import _METRIC_ORDER
from mlip_smoothness_eval.viz.theme import EDITORIAL_8, apply_theme

# (check name, trace x-key, trace y-key, axis labels, whether to baseline-subtract
# the rightmost point so shapes overlay regardless of each model's energy offset).
_CURVES = (
    ("diatomic", "x", "energy", "bond length (Å)", "energy − E(r_max) (eV)", True),
    ("displacement_scan", "x", "energy", "displacement s (Å)", "energy − E(0) (eV)", "first"),
    ("cutoff_smoothness", "x", "energy", "drag distance (Å)", "energy − E(0) (eV)", "first"),
    ("nve_drift", "x", "drift_per_atom", "time (fs)", "total-E drift / atom (eV)", False),
)


def _load(outdir: Path) -> list[dict]:
    reports = [json.loads(p.read_text()) for p in sorted(outdir.glob("*.json"))]
    if not reports:
        raise SystemExit(f"no *.json found in {outdir}")
    return reports


def _heatmap(reports: list[dict]) -> go.Figure:
    """Models × metrics; each metric column normalized independently (greener = lower)."""
    models = [r["model_name"] for r in reports]
    metrics = [m for m in _METRIC_ORDER if any(m in r["metrics"] for r in reports)]

    raw = np.array([[r["metrics"].get(m, math.nan) for m in metrics] for r in reports])
    # Per-metric min-max over log10(|value|) so disparate units/scales are comparable.
    z = np.full_like(raw, math.nan)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        logged = np.log10(np.abs(col) + 1e-30)
        finite = logged[np.isfinite(logged)]
        if finite.size:
            lo, hi = finite.min(), finite.max()
            z[:, j] = 0.0 if hi == lo else (logged - lo) / (hi - lo)

    text = [["—" if not math.isfinite(v) else f"{v:.3g}" for v in row] for row in raw]
    fig = go.Figure(
        go.Heatmap(
            z=z.T,
            x=models,
            y=metrics,
            text=np.array(text).T,
            texttemplate="%{text}",
            colorscale=[[0.0, "#2E6E4E"], [0.5, "#E8D9A0"], [1.0, "#B9605B"]],
            showscale=True,
            colorbar=dict(title="rank<br>(per row)", tickvals=[0, 1], ticktext=["best", "worst"]),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed")
    apply_theme(fig, height=80 + 26 * len(metrics))
    return fig


def _overlay(reports: list[dict], check: str, xk: str, yk: str, xlabel: str, ylabel: str, baseline) -> list[go.Figure]:
    """One figure per (check, symbol); one trace per model. Returns [] if absent."""
    # group by the diatomic symbol so each element gets its own panel
    by_symbol: dict[str | None, list[tuple[str, dict]]] = {}
    for r in reports:
        for c in r["checks"]:
            if c["name"] != check or xk not in c["trace"] or yk not in c["trace"]:
                continue
            by_symbol.setdefault(c["trace"].get("symbol"), []).append((r["model_name"], c["trace"]))

    figs = []
    for symbol, entries in by_symbol.items():
        fig = go.Figure()
        for i, (model, trace) in enumerate(entries):
            x = np.asarray(trace[xk], dtype=float)
            y = np.asarray(trace[yk], dtype=float)
            if baseline == "first" and y.size:
                y = y - y[0]
            elif baseline is True and y.size:
                y = y - y[-1]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=model,
                                     line=dict(color=EDITORIAL_8[i % len(EDITORIAL_8)])))
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=ylabel)
        apply_theme(fig, height=440)
        title = check if symbol is None else f"{check} — {symbol}"
        figs.append((title, fig))
    return figs


def build(outdir: Path, html: Path) -> Path:
    reports = _load(outdir)
    models = ", ".join(r["model_name"] for r in reports)

    sections: list[tuple[str, go.Figure]] = [("metrics heatmap", _heatmap(reports))]
    for check, xk, yk, xlabel, ylabel, baseline in _CURVES:
        sections.extend(_overlay(reports, check, xk, yk, xlabel, ylabel, baseline))

    # Inline plotly.js once (first figure), then embed the rest as div-only fragments.
    blocks = []
    for i, (title, fig) in enumerate(sections):
        blocks.append(f"<h2 style='margin:28px 0 6px;font-weight:600'>{title}</h2>")
        blocks.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))

    page = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>MLIP smoothness comparison</title>"
        "<div style=\"font-family:'Suisse Intl',-apple-system,Arial,sans-serif;"
        "color:#1D272A;max-width:1000px;margin:24px auto;padding:0 16px\">"
        "<h1 style='margin:0 0 4px'>MLIP smoothness comparison</h1>"
        f"<p style='color:#7B7B7B;margin:0 0 8px'>{len(reports)} models: {models}. "
        "Heatmap colors are per-metric ranks (green = best in row, red = worst); "
        "raw values are printed. Curves are baseline-shifted so shapes overlay.</p>"
        + "".join(blocks)
        + "</div>"
    )
    html.write_text(page)
    return html


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="examples/comparison_results")
    p.add_argument("--html", default="examples/comparison.html")
    args = p.parse_args()
    out = build(Path(args.outdir), Path(args.html))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
