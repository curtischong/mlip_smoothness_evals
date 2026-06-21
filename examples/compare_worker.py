"""Run the smoothness suite for ONE model inside that model's own env, dump JSON.

Each MLIP family pins an incompatible dependency stack (see the ``conflicts``
block in pyproject.toml), so the models can never share a process. This worker
is the per-model half of the comparison: it is launched once per model under
that model's uv extra (``uv run --extra <key> python examples/compare_worker.py
<key>``), builds the model, runs ``evaluate_smoothness``, and writes the report
as plain JSON. ``compare_models.py`` then reads every JSON and plots — it needs
only plotly, never the models.

To add or fix a model, edit MODEL_LOADERS below: it is the single source of
truth for how each family is loaded. Only MACE is verified end-to-end here; the
others use the canonical torch-sim wrapper + each package's standard pretrained
entry point and may need a checkpoint path or name tweak for your setup. Pass
loader kwargs without editing via ``--opts '{"size": "medium"}'``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch

from mlip_smoothness_eval import evaluate_smoothness, structures


def _load_lj(device: str, dtype: torch.dtype, *, sigma: float = 2.0, epsilon: float = 0.1) -> object:
    # Analytic, smooth, conservative — no download. A ground-truth baseline that
    # every real MLIP should be compared against. Runs in the base env (no extra).
    from torch_sim.models.lennard_jones import LennardJonesModel

    return LennardJonesModel(sigma=sigma, epsilon=epsilon, cutoff=3.0 * sigma,
                             device=device, dtype=dtype, compute_forces=True, retain_graph=True)


def _load_mace(device: str, dtype: torch.dtype, *, size: str = "small") -> object:
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel

    raw = mace_mp(model=size, return_raw_model=True, device=device)
    return MaceModel(model=raw, device=device, dtype=dtype, compute_forces=True)


def _load_orb(device: str, dtype: torch.dtype, *, name: str = "orb_v3_conservative_inf_omat") -> object:
    from orb_models.forcefield import pretrained
    from torch_sim.models.orb import OrbModel

    orbff = getattr(pretrained, name)(device=device)
    return OrbModel(model=orbff, device=device)


def _load_mattersim(device: str, dtype: torch.dtype, *, ckpt: str = "MatterSim-v1.0.0-1M.pth") -> object:
    from mattersim.forcefield import Potential
    from torch_sim.models.mattersim import MatterSimModel

    pot = Potential.from_checkpoint(load_path=ckpt, device=device)
    return MatterSimModel(model=pot, device=device, dtype=dtype)


def _load_sevenn(device: str, dtype: torch.dtype, *, model: str = "7net-0") -> object:
    from torch_sim.models.sevennet import SevenNetModel

    return SevenNetModel(model=model, device=device, dtype=dtype)


def _load_fairchem(device: str, dtype: torch.dtype, *, name: str = "uma-s-1p1", task_name: str = "omat") -> object:
    from torch_sim.models.fairchem import FairChemModel

    return FairChemModel(model=name, neighbor_list_fn=None, device=device, dtype=dtype, task_name=task_name)


def _load_nequix(device: str, dtype: torch.dtype, *, path: str = "nequix-mp-1.nqx") -> object:
    from torch_sim.models.nequix import NequixModel

    return NequixModel.from_compiled_model(path)


def _load_nequip(device: str, dtype: torch.dtype, *, path: str) -> object:
    from torch_sim.models.nequip_framework import NequIPFrameworkModel

    return NequIPFrameworkModel.from_compiled_model(path)


def _load_metatomic(device: str, dtype: torch.dtype, *, model: str = "pet-mad") -> object:
    from torch_sim.models.metatomic import MetatomicModel

    return MetatomicModel(model, device=device)


# key -> (loader, pretty name). The key matches the pyproject optional-dependency
# extra so the orchestrator can do `uv run --extra <key>`.
MODEL_LOADERS: dict[str, tuple[Callable[..., object], str]] = {
    "lj": (_load_lj, "Lennard-Jones (reference)"),
    "mace": (_load_mace, "MACE-MP-0"),
    "orb": (_load_orb, "ORB v3"),
    "mattersim": (_load_mattersim, "MatterSim v1"),
    "sevenn": (_load_sevenn, "SevenNet-0"),
    "fairchem": (_load_fairchem, "FairChem UMA"),
    "nequix": (_load_nequix, "Nequix"),
    "nequip": (_load_nequip, "NequIP"),
    "metatomic": (_load_metatomic, "PET-MAD"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=sorted(MODEL_LOADERS))
    p.add_argument("--outdir", default="examples/comparison_results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nve-steps", type=int, default=1000)
    p.add_argument("--diatomic", default="H,O,Cu", help="comma-separated symbols")
    p.add_argument("--opts", default="{}", help="JSON kwargs forwarded to the loader")
    args = p.parse_args()

    device, dtype = args.device, torch.float64
    loader, pretty = MODEL_LOADERS[args.model]
    opts = json.loads(args.opts)

    print(f"[{args.model}] loading {pretty} on {device}")
    model = loader(device, dtype, **opts)

    report = evaluate_smoothness(
        model,
        structures=[structures.random_crystal("Cu", device=device, dtype=dtype)],
        diatomic_symbols=tuple(args.diatomic.split(",")),
        device=device,
        dtype=dtype,
        nve_steps=args.nve_steps,
        model_name=pretty,
    )
    out = report.save(Path(args.outdir) / f"{args.model}.json")
    print(f"[{args.model}] wrote {out}")


if __name__ == "__main__":
    main()
