"""Smoke tests for the report object and the visualization renderers."""

from __future__ import annotations

import os

import plotly.graph_objects as go
from ase import Atoms

from mlip_smoothness_eval import evaluate_smoothness
from mlip_smoothness_eval.checks.bsct import TorchSimCalculator


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
    assert "nonconservativity_rmse" in report.to_frame().index
    assert "nonconservativity_rmse" in report._repr_html_()


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
    for name, symbol in (("displacement_scan", None), ("diatomic", "O")):
        out = report.gif(name, path=str(tmp_path / f"{name}.gif"), symbol=symbol, max_frames=8)
        assert os.path.exists(out) and os.path.getsize(out) > 0


def test_bsct_ase_adapter(lj_model):
    # the bridge BSCT drives: torch-sim model -> ASE energy/forces, obeying Newton's 3rd law
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    atoms.calc = TorchSimCalculator(lj_model)
    assert atoms.get_forces().shape == (2, 3)
    assert atoms.get_forces()[0, 0] == -atoms.get_forces()[1, 0]


def test_pca_surface(lj_model, crystal):
    report = evaluate_smoothness(
        lj_model, structures=[crystal], diatomic_symbols=("O",), run_nve=False
    )
    fig = report.pca_surface(n_samples=60, grid=20)
    assert isinstance(fig, go.Figure)
