"""Shared setup + inline-video helper for the frame-by-frame smoothness scripts.

Four of the seven probes move atoms through a sequence of frames while a metric
traces out, so they animate naturally:

- ``diatomic`` — pull a homonuclear dimer apart and watch the bond PEC,
- ``displacement_scan`` — slide every atom along one random direction,
- ``cutoff_smoothness`` — drag a single atom in a straight line,
- ``boundary_crossing`` — drag one atom a full lattice vector across a cell face.

The sibling ``# %%`` scripts in this folder (``diatomic.py``,
``displacement_scan.py``, ``cutoff.py``, ``boundary_crossing.py``) each import this module, so they all
share one reference model, the structures, and the video logic (``animate``
wraps the package renderer ``viz.gifs.make_gif``, which morphs the atoms beside
the advancing energy curve and holds the final frame). The other three probes
(nonconservativity, force-Jacobian, NVE drift) are single-state or trajectory
scalars and aren't frame animations, so they have no script here.
"""

from __future__ import annotations

import torch
from torch_sim.state import SimState

from mlip_smoothness_eval import structures
from mlip_smoothness_eval.checks import (
    CheckResult,
    boundary_crossing,
    cutoff_smoothness,
    diatomic_smoothness,
    displacement_scan,
)
from mlip_smoothness_eval.viz.gifs import make_gif

DEVICE = "cpu"
DTYPE = torch.float64


def reference_model(*, sigma: float = 2.0, epsilon: float = 0.1) -> object:
    """torch-sim's analytic Lennard-Jones model — smooth, conservative, no download.

    A known-good baseline: every metric here should look clean, which is what
    makes the animations easy to read as "this is what smooth looks like".
    """
    from torch_sim.models.lennard_jones import LennardJonesModel

    return LennardJonesModel(
        sigma=sigma,
        epsilon=epsilon,
        cutoff=3.0 * sigma,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        retain_graph=True,
    )


def dilute_crystal(*, spacing_sigma: float = 1.7, sigma: float = 2.0) -> SimState:
    """A dilute FCC Ar cell whose neighbours sit in the gentle attractive tail.

    The cutoff probe drags an atom looking for a *discontinuity*; a dense cell
    would ram it into a steep repulsive wall and inflate the spike ratio without
    any real discontinuity, so the smooth baseline uses a loosely packed cell.
    """
    import numpy as np
    from ase import Atoms

    a = sigma * spacing_sigma * 2 ** 0.5
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pos = basis @ (np.eye(3) * a)
    atoms = Atoms("Ar4", positions=pos, cell=np.eye(3) * a, pbc=True)
    return structures.from_ase(atoms, device=DEVICE, dtype=DTYPE)


def diatomic_result(model: object, symbol: str = "O") -> CheckResult:
    return diatomic_smoothness(model, symbol, device=DEVICE, dtype=DTYPE)


def displacement_result(model: object, state: SimState | None = None) -> CheckResult:
    state = state or structures.random_crystal("Ar", repeat=1, device=DEVICE, dtype=DTYPE)
    return displacement_scan(model, state)


def cutoff_result(model: object, state: SimState | None = None) -> CheckResult:
    return cutoff_smoothness(model, state or dilute_crystal())


def boundary_result(model: object, state: SimState | None = None) -> CheckResult:
    """Drag one atom a full lattice vector across a face of the dilute Ar cell.

    The same loosely-packed cell as the cutoff probe: the dragged atom passes
    through the gentle attractive tail rather than a repulsive wall, so any spike
    is a genuine boundary discontinuity, not a wall-ram.
    """
    return boundary_crossing(model, state or dilute_crystal())


def animate(
    result: CheckResult,
    path: str,
    *,
    fps: int = 12,
    freeze_last_s: float = 2.0,
    **kwargs: object,
) -> object:
    """Render ``result`` to a gif (atoms morphing beside the advancing curve, last
    frame held ``freeze_last_s`` s) and return it as an inline-displayable image."""
    from IPython.display import Image

    make_gif(result, path, fps=fps, freeze_last_s=freeze_last_s, **kwargs)
    return Image(filename=path)
