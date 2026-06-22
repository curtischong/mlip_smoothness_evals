"""Homonuclear diatomic potential-energy curves (arXiv 2509.20630).

Sweep a single bond length and score the 1D PEC by:
- tortuosity: total energy variation over the ideal r_min -> r_eq -> r_max path
  (a clean single-well curve like Lennard-Jones gives ~1; wiggles raise it),
- energy jump: magnitude-weighted sign changes in the discrete energy gradient,
- force flips: a clean single-well curve flips axial-force direction ~once.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from duecredit import Doi, due
from torch_sim.state import SimState

from mlip_smoothness_eval.structures import diatomic as _diatomic_state
from mlip_smoothness_eval.checks.base import CheckResult, predict

_NAN_METRICS = {
    "diatomic_tortuosity": float("nan"),
    "diatomic_energy_jump": float("nan"),
    "diatomic_force_flips": float("nan"),
}


@due.dcite(
    Doi("10.48550/arXiv.2509.20630"),
    description="Homonuclear diatomic potential-energy-curve smoothness diagnostics",
    path="mlip_smoothness_eval.checks.diatomic",
)
def diatomic_smoothness(
    model: object,
    symbol: str,
    *,
    step: float = 0.01,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> CheckResult:
    """Sweep the bond length of a homonuclear ``symbol`` dimer and score the PEC.

    Bond goes from 0.9x the covalent radius to 3.1x the van der Waals radius (or
    6 Å when no vdW radius exists), every ``step`` Å.
    """
    z = atomic_numbers[symbol]
    r_min = 0.9 * float(covalent_radii[z])
    vdw = float(vdw_radii[z])
    r_max = 6.0 if math.isnan(vdw) else 3.1 * vdw
    num = int(round((r_max - r_min) / step)) + 1
    radii = torch.linspace(r_min, r_max, num, dtype=torch.float64)
    box = r_max + 20.0

    energies = torch.empty(num, dtype=torch.float64)
    axial_force = torch.empty(num, dtype=torch.float64)  # force on atom 1 along +x
    i = num
    for j, r in enumerate(radii):
        # Past the model's neighbor cutoff the pair may have no edges; some
        # backbones raise on the empty graph. Stop and score the bonded portion.
        try:
            state = _diatomic_state(symbol, float(r), box=box, device=device, dtype=dtype)
            pred = predict(model, state)
        except (RuntimeError, IndexError):
            i = j
            break
        if not torch.isfinite(pred.energy):
            i = j
            break
        energies[j] = float(pred.energy)
        axial_force[j] = float(pred.forces[1, 0])

    if i < 4:
        return CheckResult("diatomic", dict(_NAN_METRICS), {"symbol": symbol}, {"symbol": symbol})
    energies = energies[:i]
    axial_force = axial_force[:i]
    radii = radii[:i]

    eq = int(energies.argmin())
    coarse = (energies[0] - energies[eq]).abs() + (energies[eq] - energies[-1]).abs()
    tortuosity = energies.diff().abs().sum() / coarse.clamp_min(1e-8)

    dE = energies.diff()
    sign_change = (dE[1:].sign() - dE[:-1].sign()).abs()
    energy_jump = (sign_change * (dE[1:].abs() + dE[:-1].abs())).sum()

    force_flips = (axial_force[1:].sign() != axial_force[:-1].sign()).sum()

    metrics = {
        "diatomic_tortuosity": float(tortuosity),
        "diatomic_energy_jump": float(energy_jump),
        "diatomic_force_flips": float(force_flips),
    }
    trace = {
        "x": radii.numpy(),
        "energy": energies.numpy(),
        "force": axial_force.numpy(),
        "symbol": symbol,
        "r_eq": float(radii[eq]),
        "xlabel": "bond length (Å)",
    }
    return CheckResult("diatomic", metrics, trace, {"symbol": symbol})


def diatomic_frames(symbol: str, radii: np.ndarray) -> np.ndarray:
    """``(n, 2, 3)`` positions for each bond length, for gif rendering."""
    frames = np.zeros((len(radii), 2, 3), dtype=np.float64)
    frames[:, 1, 0] = radii
    return frames
