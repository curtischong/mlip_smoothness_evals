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

import math

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
    """Sweep ``atom`` one full turn of the periodic phase along ``axis``.

    The swept coordinate is the phase ``theta`` of the periodic axis, run from
    ``+pi`` to ``-pi`` — one full lattice vector of travel. ``+pi`` and ``-pi``
    are the same point on the circle (a full-period translation apart), so a
    correct periodic model comes back to its initial energy with no jump.
    """
    if not bool(torch.as_tensor(state.pbc).any()):
        raise ValueError("boundary_crossing needs a periodic state (pbc=True)")

    cell = state.cell[0].detach()  # (3, 3) column-vector lattice
    lat = cell[:, axis]  # cartesian lattice vector for this axis
    # one full turn of the periodic phase: +pi -> -pi. The fractional
    # displacement along the lattice vector is theta / 2pi (+1/2 .. -1/2), so the
    # endpoints sit one full lattice vector apart — the same periodic image.
    thetas = torch.linspace(math.pi, -math.pi, num_steps, dtype=torch.float64)
    fracs = thetas / (2.0 * math.pi)

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
        "x": thetas.numpy(),
        "energy": energies.numpy(),
        "force": forces.numpy(),
        "frames": frames,
        "cell": cell_p.cpu().numpy(),  # (3,3) column-vector lattice, for the gif box
        "xlabel": "periodic phase θ (rad)",
    }
    return CheckResult("boundary_crossing", metrics, trace, {"atom": atom, "axis": axis})
