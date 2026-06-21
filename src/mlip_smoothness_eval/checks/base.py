"""The adapter boundary: torch-sim ``model(state) -> {energy, forces}``.

Every probe reads the model's own energy and force head straight from the output
dict. Two probes (nonconservativity, force-Jacobian) also need the conservative
force ``-dE/dr`` and its derivative; the gap between the model's force head and
``-dE/dr`` *is* the non-conservativity the suite is built to detect.

There are two ways to get ``-dE/dr``:

- **autograd** (exact, fast) — when the model returns a differentiable energy
  (a graph back to positions). torch-sim built-ins do this with
  ``retain_graph=True``; a custom ModelInterface does it by not detaching.
- **finite difference** (universal fallback) — central differences of the
  model's energy/force *values*, which are always available. torch-sim's
  production models (e.g. ``MaceModel``) detach energy and forces
  unconditionally, so autograd is impossible and FD is the only option.

``method='auto'`` (the default) uses autograd when the energy is differentiable
and FD otherwise; each probe records which method produced its numbers, because
FD values are approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_sim.state import SimState

from mlip_smoothness_eval.structures import with_positions

# valid `method` values for the conservative-force / Jacobian probes
METHODS = ("auto", "autograd", "finite_difference")

_AUTOGRAD_REQUIRED_MSG = (
    "method='autograd' was requested but model(state)['energy'] is detached "
    "(requires_grad=False), so -dE/dr cannot be taken by autograd. Use "
    "method='auto' (falls back to finite differences) or method='finite_difference', "
    "or for torch-sim built-ins construct the model with retain_graph=True."
)


@dataclass
class Prediction:
    energy: Tensor  # () scalar, model energy units (eV for typical MLIPs)
    forces: Tensor  # (N, 3) the model's own force head
    grad_forces: Tensor | None  # (N, 3) conservative -dE/dr, or None if not requested


@dataclass
class CheckResult:
    """One probe's output: scalar metrics plus the raw trace used for plotting.

    ``trace`` holds numpy arrays keyed by name (e.g. ``x``, ``energy``,
    ``force``) and optionally ``frames`` (an ``(n_frames, N, 3)`` array of
    positions) so the curve / gif renderers don't have to recompute anything.
    """

    name: str
    metrics: dict[str, float]
    trace: dict[str, object]
    meta: dict[str, object]


def _as_scalar(energy: Tensor) -> Tensor:
    """Reduce a possibly per-system energy tensor to a scalar (single system)."""
    return energy.reshape(()) if energy.numel() == 1 else energy.sum()


def _grad_positions(state: SimState) -> tuple[SimState, Tensor]:
    """Clone ``state`` and swap in a grad-enabled leaf positions tensor."""
    work = state.clone()
    pos = state.positions.detach().clone().requires_grad_(True)
    work.positions = pos
    return work, pos


def predict(model: object, state: SimState) -> Prediction:
    """Energy (scalar) and the model's own force head, from a single forward.

    Positions are grad-enabled and grad is left on: torch-sim models that compute
    their force head via internal autograd need that even when we only read values.
    Outputs are detached for safe downstream use.
    """
    work, _ = _grad_positions(state)
    with torch.enable_grad():
        out = model(work)
        energy = _as_scalar(out["energy"])
    return Prediction(
        energy=energy.detach(),
        forces=out["forces"].detach().reshape(-1, 3),
        grad_forces=None,
    )


def _energy_value(model: object, state: SimState) -> float:
    """Scalar model energy at ``state`` (grad left on so force-via-autograd models run)."""
    work, _ = _grad_positions(state)
    with torch.enable_grad():
        out = model(work)
        return float(_as_scalar(out["energy"]).detach())


def conservative_forces(
    model: object,
    state: SimState,
    *,
    method: str = "auto",
    h: float = 1e-3,
) -> tuple[Tensor, str]:
    """Return ``(-dE/dr, method_used)``.

    autograd when the model's energy is differentiable; otherwise (or on
    ``method='finite_difference'``) a central finite difference of the energy.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    if method in ("auto", "autograd"):
        work, pos = _grad_positions(state)
        with torch.enable_grad():
            out = model(work)
            energy = _as_scalar(out["energy"])
            if energy.requires_grad:
                (grad,) = torch.autograd.grad(energy, pos)
                return (-grad).detach(), "autograd"
        if method == "autograd":
            raise RuntimeError(_AUTOGRAD_REQUIRED_MSG)

    return _fd_conservative_forces(model, state, h), "finite_difference"


def _fd_conservative_forces(model: object, state: SimState, h: float) -> Tensor:
    """-dE/dr by central finite differences of the model energy."""
    base = state.positions.detach()
    n = base.shape[0]
    forces = torch.zeros_like(base)
    for i in range(n):
        for d in range(3):
            plus = base.clone()
            plus[i, d] += h
            minus = base.clone()
            minus[i, d] -= h
            e_plus = _energy_value(model, with_positions(state, plus))
            e_minus = _energy_value(model, with_positions(state, minus))
            forces[i, d] = -(e_plus - e_minus) / (2.0 * h)
    return forces


def _model_forces_fn(model: object, state: SimState):
    """Return ``(flat_pos -> flat model forces, flat_pos0)`` for an autograd Jacobian.

    The Jacobian is taken of the model's *own* force head w.r.t. positions, so a
    conservative field (forces = -Hessian @ dr) yields a symmetric Jacobian.
    """
    n = state.positions.shape[0]
    template = state.clone()

    def forces_fn(flat_pos: Tensor) -> Tensor:
        work = template.clone()
        work.positions = flat_pos.view(n, 3)
        out = model(work)
        return out["forces"].reshape(-1)

    return forces_fn, template.positions.detach().reshape(-1)


def force_jacobian(
    model: object,
    state: SimState,
    *,
    method: str = "auto",
    h: float = 1e-3,
) -> tuple[Tensor, str]:
    """Return ``(J, method_used)`` for ``J = d(model forces)/d(positions)``, shape (3N, 3N).

    autograd when the model's forces are differentiable w.r.t. positions;
    otherwise a central finite difference of the force head.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    if method in ("auto", "autograd"):
        forces_fn, flat0 = _model_forces_fn(model, state)
        flat = flat0.clone().requires_grad_(True)
        with torch.enable_grad():
            f0 = forces_fn(flat)
            if f0.requires_grad:
                jac = torch.autograd.functional.jacobian(forces_fn, flat)
                return jac.detach(), "autograd"
        if method == "autograd":
            raise RuntimeError(_AUTOGRAD_REQUIRED_MSG)

    return _fd_jacobian(model, state, h), "finite_difference"


def _fd_jacobian(model: object, state: SimState, h: float) -> Tensor:
    """d(forces)/d(positions) by central finite differences of the force head."""
    base = state.positions.detach()
    n = base.shape[0]
    dim = 3 * n
    jac = torch.zeros((dim, dim), dtype=base.dtype, device=base.device)
    for k in range(dim):
        i, d = divmod(k, 3)
        plus = base.clone()
        plus[i, d] += h
        minus = base.clone()
        minus[i, d] -= h
        f_plus = predict(model, with_positions(state, plus)).forces.reshape(-1)
        f_minus = predict(model, with_positions(state, minus)).forces.reshape(-1)
        jac[:, k] = (f_plus - f_minus) / (2.0 * h)
    return jac
