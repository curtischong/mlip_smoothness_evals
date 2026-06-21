"""Smoke tests for the report object and the visualization renderers."""

from __future__ import annotations

import os

import plotly.graph_objects as go

from mlip_smoothness_eval import evaluate_smoothness


def test_evaluate_and_scorecard(lj_model, crystal):
    report = evaluate_smoothness(
        lj_model,
        structures=[crystal],
        diatomic_symbols=("O",),
        nve_steps=200,
        model_name="LJ",
    )
    metrics = report.metrics
    assert "nonconservativity_rmse" in metrics
    assert "diatomic_tortuosity" in metrics
    html = report.to_html()
    assert "Smoothness scorecard" in html
    assert "nonconservativity_rmse" in html


def test_curves_return_figures(lj_model, crystal):
    report = evaluate_smoothness(
        lj_model, structures=[crystal], diatomic_symbols=("O",), nve_steps=200
    )
    for name in ("diatomic", "displacement_scan", "cutoff_smoothness", "nve_drift", "force_jacobian"):
        fig = report.curve(name) if name != "diatomic" else report.curve(name, symbol="O")
        assert isinstance(fig, go.Figure)


def test_gif_written(tmp_path, lj_model, crystal):
    report = evaluate_smoothness(
        lj_model, structures=[crystal], diatomic_symbols=("O",), run_nve=False
    )
    path = str(tmp_path / "scan.gif")
    out = report.gif("displacement_scan", path=path, max_frames=8)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_pca_surface(lj_model, crystal):
    report = evaluate_smoothness(
        lj_model, structures=[crystal], diatomic_symbols=("O",), run_nve=False
    )
    fig = report.pca_surface(n_samples=60, grid=20)
    assert isinstance(fig, go.Figure)
