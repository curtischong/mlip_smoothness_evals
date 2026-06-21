"""Shared fixtures: a Lennard-Jones ground-truth model and small structures.

Lennard-Jones is analytic, smooth, and conservative, so the smoothness probes
have known-good values. The LJ model is torch-sim's built-in
``LennardJonesModel`` (an independent implementation of the contract under
test). Its exact import path is resolved at collection time so the suite tracks
the installed torch-sim version.
"""

from __future__ import annotations

import pytest
import torch

DTYPE = torch.float64
DEVICE = "cpu"


def _make_lj():
    """Build torch-sim's Lennard-Jones model with autograd-able energy."""
    from torch_sim.models.lennard_jones import LennardJonesModel

    return LennardJonesModel(
        sigma=2.0,
        epsilon=0.1,
        cutoff=6.0,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
        retain_graph=True,  # keep the energy graph so -dE/dr can be taken by autograd
    )


@pytest.fixture(scope="session")
def lj_model():
    return _make_lj()


@pytest.fixture(scope="session")
def crystal():
    from mlip_smoothness_eval.structures import random_crystal

    # 4-atom cell (repeat=1): small enough for the dense force-Jacobian probe
    # (max_atoms=24) while still being a periodic many-body structure.
    return random_crystal("Ar", repeat=1, rattle=0.05, device=DEVICE, dtype=DTYPE)


@pytest.fixture(scope="session")
def dilute_crystal():
    """A dilute FCC cell (nearest neighbour ~1.7 sigma) for the cutoff probe.

    The cutoff probe drags an atom and looks for a *discontinuity*. A dense cell
    near the LJ minimum would ram the dragged atom into a neighbour's smooth-but-
    steep r^-12 wall and inflate the max/median ratio without any discontinuity,
    so the smooth ground-truth check uses a dilute interacting structure.
    """
    import numpy as np
    from ase import Atoms

    from mlip_smoothness_eval.structures import from_ase

    a = 2.0 * 1.7 * 2 ** 0.5  # nearest neighbour ~1.7 * sigma (gentle attractive)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pos = basis @ (np.eye(3) * a)
    atoms = Atoms("Ar4", positions=pos, cell=np.eye(3) * a, pbc=True)
    return from_ase(atoms, device=DEVICE, dtype=DTYPE)
