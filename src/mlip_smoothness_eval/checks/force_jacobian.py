"""Force-Jacobian asymmetry: ``||J - J^T|| / ||J||`` for ``J = dF/dr``.

A conservative force field is ``F = -dE/dr``, so its Jacobian is ``-Hessian``,
which is symmetric. The Jacobian is taken of the model's *own* force head, so a
direct-force head that is not a true gradient shows up as a non-symmetric J.
``J`` comes from autograd when the forces are differentiable, else a finite
difference (see ``checks.base``).
"""

from __future__ import annotations

from torch_sim.state import SimState

from mlip_smoothness_eval.checks.base import CheckResult
from mlip_smoothness_eval.checks.base import force_jacobian as _force_jacobian


def force_jacobian_asymmetry(
    model: object,
    state: SimState,
    *,
    max_atoms: int = 24,
    method: str = "auto",
) -> CheckResult:
    n = state.positions.shape[0]
    if n > max_atoms:  # the dense (3N, 3N) Jacobian is only cheap for small cells
        return CheckResult(
            "force_jacobian", {"force_jacobian_asymmetry": float("nan")}, {}, {"n": n}
        )

    jac, used = _force_jacobian(model, state, method=method)
    asymmetry = (jac - jac.T).norm() / jac.norm().clamp_min(1e-8)
    metrics = {"force_jacobian_asymmetry": float(asymmetry)}
    trace = {"jacobian": jac.cpu().numpy()}
    return CheckResult("force_jacobian", metrics, trace, {"n": n, "method": used})
