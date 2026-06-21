"""NVE energy-drift gate (the paper's headline test).

A microcanonical (constant-energy) run should keep total energy bounded. A
direct-force head that is not the gradient of the energy shows up here as
monotonic drift even when force MAE is low. This probe is torch-sim native — it
drives the user's model directly through ``nve_init`` / ``nve_step``.
"""

from __future__ import annotations

import torch
import torch_sim as ts
from torch_sim.state import SimState
from torch_sim.units import MetalUnits

from .base import CheckResult


def nve_energy_drift(
    model: object,
    state: SimState,
    *,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    steps: int = 1000,
    log_every: int = 10,
    seed: int = 0,
) -> CheckResult:
    """Run NVE and report total-energy drift per atom; bounded is good, drift is bad."""
    torch.manual_seed(seed)
    kT = temperature_K * float(MetalUnits.temperature)
    dt = timestep_fs * 1e-3 * float(MetalUnits.time)
    n_atoms = state.positions.shape[0]

    md_state = ts.nve_init(state, model, kT=kT)
    total_energies: list[float] = []
    steps_log: list[int] = []
    for step in range(steps):
        md_state = ts.nve_step(md_state, model, dt=dt)
        if step % log_every == 0:
            kinetic = ts.calc_kinetic_energy(masses=md_state.masses, momenta=md_state.momenta)
            total_energies.append(float((md_state.energy.sum() + kinetic.sum()).detach()))
            steps_log.append(step)

    energies = torch.tensor(total_energies, dtype=torch.float64)
    drift = (energies - energies[0]) / n_atoms
    metrics = {
        "nve_final_drift_per_atom": float(drift[-1].abs()),
        "nve_max_drift_per_atom": float(drift.abs().max()),
        "nve_drift_std_per_atom": float(drift.std()),
    }
    trace = {
        "x": (torch.tensor(steps_log, dtype=torch.float64) * timestep_fs).numpy(),
        "total_energy": energies.numpy(),
        "drift_per_atom": drift.numpy(),
        "xlabel": "time (fs)",
    }
    return CheckResult("nve_drift", metrics, trace, {"steps": steps})
