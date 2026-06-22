"""Quantify *how smooth* the boundary-crossing energy curve actually is.

``boundary_crossing`` reports spike ratios and the periodicity error — they catch
a discontinuous wrap, but they say nothing about the shape of the curve in
between. This probe sweeps the same full turn of the periodic phase and then
fits splines to ``E(theta)`` to measure the curve's intrinsic smoothness.

The distinction that matters (and the reason for several metrics):

- **MAE to a smoothing spline** is *not* a smoothness metric. It measures how far
  the raw samples sit from a de-noised baseline — i.e. high-frequency deviation /
  noise, not the bending of the underlying curve. We keep it as that baseline.
- For the *actual* curve smoothness we integrate derivatives of an interpolating
  spline (which preserves any kink the model produces, instead of smearing it):

  - ``bending_energy`` ``= ∫ (y'')² dx`` — the classic roughness penalty for an
    ordinary ``y = f(x)`` curve. Large when the curve bends sharply.
  - ``curvature_energy`` ``= ∫ κ² ds`` with ``κ = y'' / (1 + y'²)^{3/2}`` and
    ``ds = (1 + y'²)^{1/2} dx`` — the geometric bending energy of the curve as a
    shape in the plane, invariant to how it is parametrised.
  - ``jerk_energy`` ``= ∫ (y''')² dx`` — penalises the third derivative, the
    natural smoothness measure for a motion / time-series trajectory; most
    sensitive to kinks (a corner is a jerk impulse).

Axes are normalised before fitting (``x -> [0, 1]`` over the swept phase, ``y``
by its peak-to-peak range) so the numbers are scale-free shape descriptors that
compare across models; ``mae_to_spline`` is also reported in physical eV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline
from torch_sim.state import SimState

from mlip_smoothness_eval.checks.base import CheckResult
from mlip_smoothness_eval.checks.boundary_crossing import boundary_crossing


@dataclass
class CurveSmoothness:
    """Smoothness descriptors of a 1-D curve ``y(x)`` (all but ``mae_eV`` scale-free)."""

    mae_to_spline: float  # mean |y - smoothing_spline(x)|, normalised (noise baseline)
    mae_eV: float  # same deviation in physical energy units
    bending_energy: float  # ∫ (y'')² dx           — roughness of y = f(x)
    curvature_energy: float  # ∫ κ² ds             — geometric bending of the 2-D shape
    jerk_energy: float  # ∫ (y''')² dx             — trajectory jerk, kink-sensitive


@dataclass
class _Fit:
    """Both splines fit on normalised axes, plus the scaling back to physical units."""

    xs: np.ndarray  # samples, sorted, x in [0, 1]
    ys: np.ndarray  # samples, sorted, y in [0, 1]
    y_min: float
    y_range: float
    smoothing: object  # GCV smoothing BSpline (de-noised baseline)
    interp: CubicSpline  # interpolating cubic spline (kink-preserving)


def _fit(x: np.ndarray, y: np.ndarray) -> _Fit:
    """Sort, normalise (``x -> [0, 1]``, ``y`` by range), and fit both splines once."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    xs = (x - x[0]) / (x[-1] - x[0])
    y_min, y_range = float(y.min()), float(y.max() - y.min())
    ys = (y - y_min) / y_range if y_range > 0 else np.zeros_like(y)
    return _Fit(xs, ys, y_min, y_range, make_smoothing_spline(xs, ys), CubicSpline(xs, ys))


def curve_smoothness_metrics(
    x: np.ndarray, y: np.ndarray, *, n_grid: int = 4001
) -> CurveSmoothness:
    """Spline-based smoothness metrics for samples ``(x, y)``.

    A smoothing spline (GCV-tuned) gives the de-noised baseline for the MAE; an
    interpolating cubic spline — which keeps any kink rather than smearing it —
    supplies the derivatives integrated for bending / curvature / jerk. Both are
    fit on axes normalised to ``x in [0, 1]`` and ``y`` scaled by its range.
    """
    fit = _fit(x, y)
    mae_norm = float(np.abs(fit.ys - fit.smoothing(fit.xs)).mean())

    grid = np.linspace(0.0, 1.0, n_grid)
    d1, d2, d3 = fit.interp(grid, 1), fit.interp(grid, 2), fit.interp(grid, 3)

    return CurveSmoothness(
        mae_to_spline=mae_norm,
        mae_eV=mae_norm * fit.y_range,
        bending_energy=float(np.trapezoid(d2**2, grid)),
        curvature_energy=float(np.trapezoid(d2**2 / (1.0 + d1**2) ** 2.5, grid)),
        jerk_energy=float(np.trapezoid(d3**2, grid)),
    )


def boundary_crossing_curve_smoothness(
    model: object,
    state: SimState,
    *,
    atom: int = 0,
    axis: int = 0,
    num_steps: int = 120,
) -> CheckResult:
    """Sweep an atom across the periodic boundary and score the energy curve's shape.

    Reuses ``boundary_crossing``'s sweep, then characterises ``E(theta)`` with the
    bending / curvature / jerk energies (true curve smoothness) plus the
    MAE-to-spline noise baseline. Smaller is smoother on every metric.
    """
    sweep = boundary_crossing(model, state, atom=atom, axis=axis, num_steps=num_steps)
    x = np.asarray(sweep.trace["x"], dtype=np.float64)
    energy = np.asarray(sweep.trace["energy"], dtype=np.float64)
    s = curve_smoothness_metrics(x, energy)

    metrics = {
        "boundary_curve_mae_to_spline": s.mae_to_spline,
        "boundary_curve_mae_eV": s.mae_eV,
        "boundary_curve_bending_energy": s.bending_energy,
        "boundary_curve_curvature_energy": s.curvature_energy,
        "boundary_curve_jerk_energy": s.jerk_energy,
    }

    # dense curves for the renderer: the smoothing baseline (eV) and the bending
    # density (d²y/dx²)² of the interpolating spline, whose area is bending_energy.
    fit = _fit(x, energy)
    grid = np.linspace(0.0, 1.0, 400)
    trace = dict(sweep.trace)
    trace["curve_x"] = x.min() + grid * (x.max() - x.min())
    trace["curve_energy_spline"] = fit.smoothing(grid) * fit.y_range + fit.y_min
    trace["curve_bending_density"] = fit.interp(grid, 2) ** 2

    return CheckResult("boundary_crossing2", metrics, trace, {"atom": atom, "axis": axis})
