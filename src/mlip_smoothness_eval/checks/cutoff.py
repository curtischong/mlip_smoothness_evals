"""Drag one atom in a straight line and watch for graph discontinuities.

As an atom crosses a neighbor-list cutoff, a non-smooth model gains/loses an
edge and the energy/force jump. Spikes in consecutive ``dE`` / ``dF`` relative
to their median flag those discontinuities.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_sim.state import SimState

from mlip_smoothness_eval.structures import displaced
from mlip_smoothness_eval.checks.base import CheckResult, predict


def cutoff_smoothness(
    model: object,
    state: SimState,
    *,
    step: float = 0.01,
    num_steps: int = 120,
    seed: int = 0,
) -> CheckResult:
    generator = torch.Generator().manual_seed(seed)
    direction = torch.zeros_like(state.positions, dtype=torch.float64)
    d0 = torch.randn(3, generator=generator, dtype=torch.float64)
    direction[0] = d0 / d0.norm()
    direction = direction.to(state.positions)

    energies = torch.empty(num_steps, dtype=torch.float64)
    forces = torch.empty(num_steps, dtype=torch.float64)
    frames = np.empty((num_steps, state.positions.shape[0], 3), dtype=np.float64)
    base_pos = state.positions.detach().cpu().numpy()
    dir_np = direction.cpu().numpy()
    for k in range(num_steps):
        disp = (k * step) * direction
        pred = predict(model, displaced(state, disp))
        energies[k] = float(pred.energy)
        forces[k] = float(pred.forces[0].norm())
        frames[k] = base_pos + (k * step) * dir_np

    dE = energies.diff().abs()
    dF = forces.diff().abs()
    metrics = {
        "cutoff_energy_spike_ratio": float(dE.max() / dE.median().clamp_min(1e-8)),
        "cutoff_force_spike_ratio": float(dF.max() / dF.median().clamp_min(1e-8)),
    }
    trace = {
        "x": (torch.arange(num_steps) * step).numpy(),
        "energy": energies.numpy(),
        "force": forces.numpy(),
        "frames": frames,
        "xlabel": "drag distance (Å)",
    }
    return CheckResult("cutoff_smoothness", metrics, trace, {"seed": seed})
