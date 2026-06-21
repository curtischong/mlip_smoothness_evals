"""``evaluate_smoothness``: run every probe and assemble a ``SmoothnessReport``."""

from __future__ import annotations

import torch
from torch_sim.state import SimState

from mlip_smoothness_eval.checks import (
    cutoff_smoothness,
    diatomic_smoothness,
    displacement_scan,
    force_jacobian_asymmetry,
    nonconservativity,
    nve_energy_drift,
)
from mlip_smoothness_eval.report import SmoothnessReport
from mlip_smoothness_eval.structures import random_crystal


def evaluate_smoothness(
    model: object,
    *,
    structures: list[SimState] | None = None,
    diatomic_symbols: tuple[str, ...] = ("H", "O"),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    run_nve: bool = True,
    nve_steps: int = 1000,
    method: str = "auto",
    model_name: str | None = None,
    notes: list[str] | None = None,
) -> SmoothnessReport:
    """Score the smoothness / energy-conservation of a torch-sim MLIP.

    Parameters
    ----------
    model:
        A torch-sim ``ModelInterface``: ``model(state) -> {"energy", "forces"}``,
        with a differentiable energy (so ``-dE/dr`` can be taken by autograd).
    structures:
        States to run the per-structure probes on. Defaults to a single small
        rattled FCC crystal.
    diatomic_symbols:
        Elements whose homonuclear diatomic PEC is scored.
    run_nve / nve_steps:
        Whether to run the NVE energy-drift gate, and for how many steps.
    """
    if structures is None:
        structures = [random_crystal(device=device, dtype=dtype)]

    results = []
    for state in structures:
        results.append(nonconservativity(model, state, method=method))
        results.append(displacement_scan(model, state))
        results.append(cutoff_smoothness(model, state))
        results.append(force_jacobian_asymmetry(model, state, method=method))
        if run_nve:
            results.append(nve_energy_drift(model, state, steps=nve_steps))

    for symbol in diatomic_symbols:
        results.append(diatomic_smoothness(model, symbol, device=device, dtype=dtype))

    return SmoothnessReport(
        results=results,
        model_name=model_name or type(model).__name__,
        structures=list(structures),
        _model=model,
        notes=notes or [],
    )
