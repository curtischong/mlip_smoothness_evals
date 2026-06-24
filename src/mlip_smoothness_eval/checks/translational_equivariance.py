"""Rigidly translate a whole cell and watch for an energy that should not move.

A potential is translationally invariant by construction only if its neighbour
search is exactly periodic: sliding *every* atom by the same vector relabels
which periodic image sits in the cell but leaves the physical configuration
untouched, so ``E`` must stay flat. Models with a hidden grid (mesh-based
long-range terms, fixed FFT boxes) instead ripple as the lattice slides under
the grid — the classic "egg-box" effect — and forces, which should be invariant,
wobble with it.

Each structure is swept one full lattice vector along ``axis`` (phase ``theta``
from ``0`` to ``2*pi``; the fractional shift is ``theta / 2pi``), so the endpoint
is the same periodic image as the start and a correct model returns a perfectly
constant curve. The probe is batched: every (structure, step) pair is one system
in a torch-sim ``SimState``, packed up to ``max_atoms_per_batch`` atoms and scored
in a handful of forward passes instead of one per frame.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch_sim.state import SimState

from mlip_smoothness_eval.checks.base import CheckResult


def _batched_simstate(
    frames: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SimState:
    """Pack per-frame ``(positions, atomic numbers, cell)`` into one batched periodic state."""
    from ase.data import atomic_masses

    counts = [f[2].shape[0] for f in frames]
    pos = np.concatenate([f[2] for f in frames])
    zs = np.concatenate([f[3] for f in frames])
    cells = np.stack([f[4] for f in frames])  # (n_systems, 3, 3) column vectors
    system_idx = np.repeat(np.arange(len(frames)), counts)
    return SimState(
        positions=torch.tensor(pos, dtype=dtype, device=device),
        masses=torch.tensor(atomic_masses[zs], dtype=dtype, device=device),
        cell=torch.tensor(cells, dtype=dtype, device=device),
        pbc=True,
        atomic_numbers=torch.tensor(zs, dtype=torch.long, device=device),
        system_idx=torch.tensor(system_idx, dtype=torch.long, device=device),
    )


def _batch_predict(model: object, state: SimState) -> tuple[Tensor, Tensor]:
    """Per-system energy ``(n_systems,)`` and per-atom forces ``(n_atoms, 3)``."""
    work = state.clone()
    work.positions = state.positions.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = model(work)
    return out["energy"].detach().reshape(-1), out["forces"].detach().reshape(-1, 3)


def translational_equivariance(
    model: object,
    states: SimState | list[SimState],
    *,
    axis: int = 0,
    num_steps: int = 120,
    max_atoms_per_batch: int = 4096,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> CheckResult:
    """Score how constant the energy stays as each cell is rigidly translated.

    Translates every atom of each periodic state by ``frac`` of lattice vector
    ``axis`` (``frac = theta / 2pi`` over ``theta in [0, 2pi]``) and measures the
    spurious energy / force variation a translationally invariant model must not
    have. The headline consistency metric is the standard deviation of the energy
    across the ``num_steps - 1`` *distinct* translated images (the ``theta = 2pi``
    sample is the same periodic image as ``theta = 0`` and is dropped so the start
    is not double-counted), normalised per atom. Metrics aggregate across the
    structures: ``*_range_max`` / ``*_per_atom_max`` are the worst structure,
    ``*_std_per_atom_{mean,max}`` the average / worst per-atom energy wobble,
    ``periodicity_error_max`` the residual after a full-lattice-vector shift.
    Smaller is better on every metric; a perfect model scores zero.
    """
    states = [states] if isinstance(states, SimState) else list(states)
    for s in states:
        if not bool(torch.as_tensor(s.pbc).any()):
            raise ValueError("translational_equivariance needs periodic states (pbc=True)")

    device = torch.device(device)
    thetas = torch.linspace(0.0, 2.0 * math.pi, num_steps, dtype=torch.float64)
    fracs = (thetas / (2.0 * math.pi)).tolist()

    # one (structure, step) pair per batched system
    frames: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    cells: list[np.ndarray] = []
    n_atoms: list[int] = []
    for si, state in enumerate(states):
        cell = state.cell[0].detach()  # (3, 3) column-vector lattice
        lat = cell[:, axis]
        base = state.positions.detach()
        zs = state.atomic_numbers.detach().cpu().numpy()
        cell_np = cell.cpu().numpy()
        cells.append(cell_np)
        n_atoms.append(base.shape[0])
        for ki, f in enumerate(fracs):
            pos = (base + f * lat).cpu().numpy()
            frames.append((si, ki, pos, zs, cell_np))

    calc_energy = [np.empty(num_steps) for _ in states]
    calc_forces: list[list[np.ndarray]] = [[None] * num_steps for _ in states]

    i = 0
    while i < len(frames):
        counts, j = [], i
        while j < len(frames) and (not counts or sum(counts) + frames[j][2].shape[0] <= max_atoms_per_batch):
            counts.append(frames[j][2].shape[0])
            j += 1
        batch = _batched_simstate(frames[i:j], device=device, dtype=dtype)
        energy, forces = _batch_predict(model, batch)
        for k, fk in enumerate(torch.split(forces, counts)):
            si, ki, _, _, _ = frames[i + k]
            calc_energy[si][ki] = float(energy[k])
            calc_forces[si][ki] = fk.cpu().numpy()
        i = j

    e_range, e_std_pa, e_per_atom, f_dev, periodicity = [], [], [], [], []
    for si in range(len(states)):
        e = calc_energy[si]
        f = np.stack(calc_forces[si])  # (num_steps, N, 3)
        # θ=2π is the same periodic image as θ=0; drop it so the std is over
        # distinct translations only and the start image isn't double-counted.
        e_unique = e[:-1]
        e_range.append(float(e.max() - e.min()))
        e_std_pa.append(float(e_unique.std()) / n_atoms[si])
        e_per_atom.append(float(e.max() - e.min()) / n_atoms[si])
        f_dev.append(float(np.abs(f - f[0]).max()))
        periodicity.append(float(abs(e[-1] - e[0])))

    metrics = {
        "equivariance_energy_range_max": max(e_range),
        "equivariance_energy_per_atom_max": max(e_per_atom),
        "equivariance_energy_std_per_atom_mean": float(np.mean(e_std_pa)),
        "equivariance_energy_std_per_atom_max": max(e_std_pa),
        "equivariance_force_dev_max": max(f_dev),
        "equivariance_periodicity_error_max": max(periodicity),
    }

    # representative (first) structure for the curve / gif: wrap every atom back
    # into the cell so the animation shows the lattice sliding through one period.
    cell0 = states[0].cell[0].detach()
    base0 = states[0].positions.detach()
    lat0 = cell0[:, axis]
    inv_cell0 = torch.linalg.inv(cell0)
    rep_frames = np.empty((num_steps, base0.shape[0], 3), dtype=np.float64)
    for ki, f in enumerate(fracs):
        pos = base0 + f * lat0
        frac = inv_cell0 @ pos.T  # (3, N) fractional coords
        rep_frames[ki] = (cell0 @ (frac - torch.floor(frac))).T.cpu().numpy()

    trace = {
        "x": thetas.numpy(),
        "energy": calc_energy[0],
        "frames": rep_frames,
        "cell": cell0.cpu().numpy(),
        "energies_all": np.stack(calc_energy),  # (n_structures, num_steps)
        "xlabel": "translation phase θ (rad)",
    }
    return CheckResult(
        "translational_equivariance",
        metrics,
        trace,
        {"axis": axis, "n_structures": len(states)},
    )
