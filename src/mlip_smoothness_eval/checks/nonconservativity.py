"""Does the model's force head equal the conservative ``-dE/dr``?

A model whose forces are the true gradient of its energy (e.g. Lennard-Jones,
or any conservative MLIP like MACE) scores ~0 here. A direct-force head that
drifts from the energy gradient is the non-conservativity that causes NVE energy
drift. ``-dE/dr`` comes from autograd when the energy is differentiable, else a
finite difference (see ``checks.base``).
"""

from __future__ import annotations

from torch_sim.state import SimState

from .base import CheckResult, conservative_forces, predict


def nonconservativity(model: object, state: SimState, *, method: str = "auto") -> CheckResult:
    pred = predict(model, state)
    grad_forces, used = conservative_forces(model, state, method=method)
    diff = pred.forces - grad_forces
    rel = diff.norm() / grad_forces.norm().clamp_min(1e-8)
    metrics = {
        "nonconservativity_rmse": float(diff.pow(2).mean().sqrt()),
        "nonconservativity_rel": float(rel),
    }
    trace = {
        "model_forces": pred.forces.cpu().numpy(),
        "grad_forces": grad_forces.cpu().numpy(),
    }
    return CheckResult("nonconservativity", metrics, trace, {"method": used})
