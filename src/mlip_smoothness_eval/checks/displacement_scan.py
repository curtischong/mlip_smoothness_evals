"""Scan along one random displacement direction.

Two things are checked along the 1D cut ``E(s)``:
- ``-F.d`` should equal ``dE/ds`` for a conservative field (force/energy
  consistency), and
- ``E(s)`` should be smooth — a large third difference flags kinks / sawtooth.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_sim.state import SimState

from ..structures import displaced
from .base import CheckResult, predict


def displacement_scan(
    model: object,
    state: SimState,
    *,
    amplitude: float = 0.1,
    num_points: int = 21,
    seed: int = 0,
) -> CheckResult:
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(state.positions.shape, generator=generator, dtype=torch.float64)
    direction = (direction / direction.norm()).to(state.positions)

    svals = torch.linspace(-amplitude, amplitude, num_points, dtype=torch.float64)
    energies = torch.empty(num_points, dtype=torch.float64)
    proj_force = torch.empty(num_points, dtype=torch.float64)  # -F.d == dE/ds (conservative)
    frames = np.empty((num_points, state.positions.shape[0], 3), dtype=np.float64)
    base_pos = state.positions.detach().cpu().numpy()
    for i, s in enumerate(svals):
        disp = s.item() * direction
        pred = predict(model, displaced(state, disp))
        energies[i] = float(pred.energy)
        proj_force[i] = -float((pred.forces.cpu().double() * direction.cpu().double()).sum())
        frames[i] = base_pos + (s.item() * direction.cpu().numpy())

    ds = float(svals[1] - svals[0])
    dE_ds = (energies[2:] - energies[:-2]) / (2.0 * ds)  # central diff, interior
    inconsistency = (dE_ds - proj_force[1:-1]).abs().max()
    second = energies[2:] - 2.0 * energies[1:-1] + energies[:-2]
    roughness = (second[1:] - second[:-1]).abs().max() / (ds ** 3)

    metrics = {
        "scan_force_energy_inconsistency": float(inconsistency),
        "scan_energy_roughness": float(roughness),
    }
    trace = {
        "x": svals.numpy(),
        "energy": energies.numpy(),
        "proj_force": proj_force.numpy(),
        "frames": frames,
        "xlabel": "displacement s (Å)",
    }
    return CheckResult("displacement_scan", metrics, trace, {"seed": seed})
