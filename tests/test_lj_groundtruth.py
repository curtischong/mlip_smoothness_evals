"""Ground-truth assertions against Lennard-Jones.

LJ is analytic, smooth, and conservative, so each probe has a known-good value.
If the port mis-wired a check, LJ fails it here. These are the hard gate.
"""

from __future__ import annotations

import math

from mlip_smoothness_eval.checks import (
    boundary_crossing,
    cutoff_smoothness,
    diatomic_smoothness,
    displacement_scan,
    force_jacobian_asymmetry,
    nonconservativity,
    nve_energy_drift,
)


def test_nonconservativity_is_zero(lj_model, crystal):
    """LJ forces are exactly -dE/dr, so the model head matches the gradient."""
    m = nonconservativity(lj_model, crystal).metrics
    assert m["nonconservativity_rmse"] < 1e-6, m
    assert m["nonconservativity_rel"] < 1e-5, m


def test_force_jacobian_symmetric(lj_model, crystal):
    """A conservative field has J = -Hessian, so dF/dr is symmetric."""
    m = force_jacobian_asymmetry(lj_model, crystal).metrics
    assert m["force_jacobian_asymmetry"] < 1e-5, m


def test_displacement_scan_consistent(lj_model, crystal):
    """-F.d should track dE/ds; E(s) should be smooth (small roughness)."""
    m = displacement_scan(lj_model, crystal).metrics
    # finite-difference limited, but for a smooth conservative field this is tiny
    assert m["scan_force_energy_inconsistency"] < 1e-2, m
    assert math.isfinite(m["scan_energy_roughness"]), m


def test_cutoff_no_giant_spikes(lj_model, dilute_crystal):
    """Dragging an atom through a smooth potential gives no huge dE/dF spikes."""
    # gentle drag in a dilute cell: stays in the smooth regime (no wall-ram).
    m = cutoff_smoothness(lj_model, dilute_crystal, step=0.005, num_steps=60).metrics
    # a hard graph cutoff would push these into the hundreds; smooth LJ stays low
    assert m["cutoff_energy_spike_ratio"] < 50.0, m
    assert m["cutoff_force_spike_ratio"] < 50.0, m


def test_boundary_crossing_smooth_and_periodic(lj_model, dilute_crystal):
    """Dragging an atom a full lattice vector across the boundary is smooth and periodic."""
    m = boundary_crossing(lj_model, dilute_crystal).metrics
    # a discontinuous wrap would spike these; minimum-image LJ stays low
    assert m["boundary_energy_spike_ratio"] < 50.0, m
    assert m["boundary_force_spike_ratio"] < 50.0, m
    # full-period translation returns to the identical configuration
    assert m["boundary_periodicity_error"] < 1e-6, m


def test_diatomic_single_well(lj_model):
    """A clean single-well PEC: tortuosity ~ 1 and force direction flips once."""
    m = diatomic_smoothness(lj_model, "O").metrics
    assert not math.isnan(m["diatomic_tortuosity"]), m
    assert m["diatomic_tortuosity"] < 1.05, m
    assert m["diatomic_force_flips"] == 1.0, m
    assert m["diatomic_energy_jump"] < 1e-3, m


def test_nve_energy_bounded(lj_model, crystal):
    """NVE total energy stays bounded for a conservative potential."""
    m = nve_energy_drift(lj_model, crystal, steps=500, temperature_K=200.0).metrics
    assert m["nve_max_drift_per_atom"] < 0.1, m


def test_fd_fallback_matches_autograd(lj_model, crystal):
    """The finite-difference fallback reproduces the autograd answer on LJ.

    This validates the FD path used for detached-output models (e.g. MACE):
    on a conservative potential the FD nonconservativity / Jacobian asymmetry
    are still ~0 (up to finite-difference noise).
    """
    nc = nonconservativity(lj_model, crystal, method="finite_difference")
    assert nc.meta["method"] == "finite_difference"
    assert nc.metrics["nonconservativity_rmse"] < 1e-3, nc.metrics

    fj = force_jacobian_asymmetry(lj_model, crystal, method="finite_difference")
    assert fj.meta["method"] == "finite_difference"
    assert fj.metrics["force_jacobian_asymmetry"] < 1e-2, fj.metrics
