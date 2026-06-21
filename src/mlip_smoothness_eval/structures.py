"""Build torch-sim ``SimState`` objects (and perturbations) for the probes.

Everything downstream consumes a ``SimState``; these builders are the only place
that knows how to assemble one. Atomic masses come from ``ase.data`` (lighter
than pymatgen, which the source repo used).

torch-sim convention reminders:
- ``cell`` is stored as **column vectors**, shape ``(n_systems, 3, 3)``
  (Cartesian = ``cell @ fractional``). A row-vector lattice is transposed in.
- a single structure is one "system": ``system_idx`` is all zeros.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from torch import Tensor
from torch_sim.state import SimState


def _masses_for(zs: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(
        [float(atomic_masses[int(z)]) for z in zs], dtype=dtype, device=device
    )


def from_ase(
    atoms: Atoms,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SimState:
    """Convert an :class:`ase.Atoms` to a single-system ``SimState``."""
    device = torch.device(device)
    positions = torch.tensor(np.asarray(atoms.get_positions()), dtype=dtype, device=device)
    zs = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    # ASE cell rows are lattice vectors (row-vector convention); torch-sim wants
    # column vectors, so transpose. Batch dim of 1 for the single system.
    row_cell = torch.tensor(np.asarray(atoms.get_cell()), dtype=dtype, device=device)
    cell = row_cell.mT.unsqueeze(0)
    pbc = bool(np.any(atoms.get_pbc()))
    n = positions.shape[0]
    return SimState(
        positions=positions,
        masses=_masses_for(zs, dtype, device),
        cell=cell,
        pbc=pbc,
        atomic_numbers=zs,
        system_idx=torch.zeros(n, dtype=torch.long, device=device),
    )


def diatomic(
    symbol: str,
    distance: float,
    *,
    box: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SimState:
    """Two atoms of ``symbol`` a ``distance`` apart along x, in a big aperiodic box."""
    device = torch.device(device)
    z = atomic_numbers[symbol]
    box = float(box) if box is not None else distance + 20.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=dtype, device=device
    )
    zs = torch.full((2,), z, dtype=torch.long, device=device)
    cell = (torch.eye(3, dtype=dtype, device=device) * box).unsqueeze(0)
    return SimState(
        positions=positions,
        masses=_masses_for(zs, dtype, device),
        cell=cell,
        pbc=False,
        atomic_numbers=zs,
        system_idx=torch.zeros(2, dtype=torch.long, device=device),
    )


def random_crystal(
    symbol: str = "Cu",
    *,
    repeat: int = 2,
    rattle: float = 0.05,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SimState:
    """A small rattled FCC crystal — a periodic many-body structure for the probes."""
    z = atomic_numbers[symbol]
    a = float(covalent_radii[z]) * 2.0 * 2 ** 0.5  # rough FCC lattice constant
    # FCC conventional cell basis
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    cells = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                cells.append(basis + np.array([i, j, k]))
    frac = np.concatenate(cells, axis=0) / repeat
    cell_np = np.eye(3) * a * repeat
    pos = frac @ cell_np
    rng = np.random.default_rng(seed)
    pos = pos + rng.normal(scale=rattle, size=pos.shape)
    atoms = Atoms(symbols=[symbol] * pos.shape[0], positions=pos, cell=cell_np, pbc=True)
    return from_ase(atoms, device=device, dtype=dtype)


def displaced(state: SimState, displacement: Tensor) -> SimState:
    """Return a copy of ``state`` with positions shifted by ``displacement``."""
    new = state.clone()
    new.positions = state.positions + displacement.to(state.positions)
    return new


def with_positions(state: SimState, positions: Tensor) -> SimState:
    """Return a copy of ``state`` with positions replaced by ``positions``."""
    new = state.clone()
    new.positions = positions.to(state.positions.device, state.positions.dtype)
    return new
