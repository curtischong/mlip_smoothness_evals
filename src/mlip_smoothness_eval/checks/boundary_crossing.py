"""Translate one atom across a periodic boundary and watch for an energy jump.

When an atom crosses a cell face, a minimum-image model must keep the energy
continuous *and* periodic: translating an atom by a full lattice vector
reproduces a physically identical configuration, so ``E`` has to return to its
starting value with no jump at the crossing. The model is fed the continuous
(unwrapped) trajectory, so its own PBC handling is on trial:

- a discontinuous wrap shows up as a spike in consecutive ``dE`` / ``dF`` at the
  boundary (``boundary_energy_spike_ratio`` / ``boundary_force_spike_ratio``),
- a model that drops PBC entirely instead ramps the energy as the atom leaves
  the cell, so it fails to come back (``boundary_periodicity_error``).

Frames are wrapped back into the cell so the gif shows the atom leaving one face
and re-entering the opposite one while the energy stays smooth.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_sim.state import SimState

from mlip_smoothness_eval.structures import with_positions
from mlip_smoothness_eval.checks.base import CheckResult, predict


def boundary_crossing(
    model: object,
    state: SimState,
    *,
    atom: int = 0,
    axis: int = 0,
    num_steps: int = 120,
) -> CheckResult:
    """Drag ``atom`` by one full lattice vector along ``axis`` across the boundary.

    The endpoint (full-period translation) is the same configuration as the
    start, so a correct periodic model returns to its initial energy.
    """
    if not bool(torch.as_tensor(state.pbc).any()):
        raise ValueError("boundary_crossing needs a periodic state (pbc=True)")

    cell = state.cell[0].detach()  # (3, 3) column-vector lattice
    lat = cell[:, axis]  # cartesian lattice vector for this axis
    fracs = torch.linspace(0.0, 1.0, num_steps, dtype=torch.float64)

    base = state.positions.detach()
    lat_p = lat.to(base)
    cell_p = cell.to(base)
    inv_cell = torch.linalg.inv(cell_p)  # cartesian -> fractional
    energies = torch.empty(num_steps, dtype=torch.float64)
    forces = torch.empty(num_steps, dtype=torch.float64)
    frames = np.empty((num_steps, base.shape[0], 3), dtype=np.float64)

    for k, f in enumerate(fracs):
        pos = base.clone()
        pos[atom] = base[atom] + f.item() * lat_p
        pred = predict(model, with_positions(state, pos))
        energies[k] = float(pred.energy)
        forces[k] = float(pred.forces[atom].norm())
        # wrap only the dragged atom into the cell so the gif shows the crossing
        wrapped = pos.clone()
        frac = inv_cell @ wrapped[atom]
        wrapped[atom] = cell_p @ (frac - torch.floor(frac))
        frames[k] = wrapped.cpu().numpy()

    dE = energies.diff().abs()
    dF = forces.diff().abs()
    metrics = {
        "boundary_energy_spike_ratio": float(dE.max() / dE.median().clamp_min(1e-8)),
        "boundary_force_spike_ratio": float(dF.max() / dF.median().clamp_min(1e-8)),
        "boundary_periodicity_error": float((energies[-1] - energies[0]).abs()),
    }
    trace = {
        "x": (fracs * float(lat_p.norm())).numpy(),
        "energy": energies.numpy(),
        "force": forces.numpy(),
        "frames": frames,
        "xlabel": "distance along lattice vector (Å)",
    }
    return CheckResult("boundary_crossing", metrics, trace, {"atom": atom, "axis": axis})
