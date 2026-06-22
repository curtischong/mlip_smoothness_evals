"""External cross-check: the Bond Smoothness Characterization Test (BSCT).

BSCT scores a potential's *Force Smoothness Deviation* (FSD) along bond-stretch
trajectories of the 485-molecule BSCT-SPICE dataset, and correlates with MD
stability while costing ~40 min of GPU instead of ~40 h of MD. It is an
independent implementation of the same "is the PES actually smooth?" question
the rest of this suite probes, so it serves as an outside check on our metrics.

BSCT drives an ASE ``Calculator``; this package's models are torch-sim
``ModelInterface`` objects, so :class:`TorchSimCalculator` bridges the two.

Upstream: https://github.com/ryanliu30/bsct (arXiv:2602.04861). Install the
optional dependency and clone its ``bsct_spice/`` dataset, then point
``data_path`` at it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_masses, atomic_numbers
from duecredit import BibTeX, due
from torch import Tensor
from torch_sim.state import SimState

from mlip_smoothness_eval.checks.base import CheckResult, predict
from mlip_smoothness_eval.structures import from_ase
from mlip_smoothness_eval.utils import resolve_path

_BSCT_REPO = "https://github.com/ryanliu30/bsct.git"
# default in-repo location for the downloaded BSCT-SPICE dataset (gitignored)
_DEFAULT_DATA_PATH = Path(resolve_path("data/bsct_spice"))
# per-system atompack cache written alongside the xyz frames (see _ensure_atompack)
_ATP_FILE = "system.atp"

_BSCT_BIBTEX = """@article{liu2026evaluation,
  title={From Evaluation to Design: Using Potential Energy Surface Smoothness Metrics to Guide Machine Learning Interatomic Potential Architectures},
  author={Liu, Ryan and Qu, Eric and Kreiman, Tobias and Blau, Samuel M and Krishnapriyan, Aditi S},
  journal={arXiv preprint arXiv:2602.04861},
  year={2026}
}"""


class TorchSimCalculator(Calculator):
    """ASE ``Calculator`` backed by a torch-sim ``ModelInterface``.

    Each ``calculate`` converts the ASE ``Atoms`` to a ``SimState`` and reads the
    model's own energy (eV) and force head (eV/Å) — the same path the native
    probes use, so BSCT scores the identical quantity.
    """

    implemented_properties = ("energy", "forces")

    def __init__(
        self,
        model: object,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.dtype = dtype

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        state = from_ase(self.atoms, device=self.device, dtype=self.dtype)
        pred = predict(self.model, state)
        self.results = {
            "energy": float(pred.energy),
            "forces": pred.forces.cpu().numpy().astype(np.float64),
        }


def _download_bsct_spice(dest: Path) -> None:
    """Fetch only the BSCT repo's ``bsct_spice/`` dataset and move it to ``dest``.

    A partial + sparse clone (``--filter=blob:none --sparse``) downloads no file
    blobs up front; ``sparse-checkout set bsct_spice`` then materializes only that
    one directory, so the rest of the repo is never transferred.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", _BSCT_REPO, tmp],
            check=True,
        )
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "bsct_spice"], check=True)
        shutil.move(str(Path(tmp) / "bsct_spice"), str(dest))


def _resolve_dataset(data_path: str | None, download_dataset: bool) -> str:
    """Pick the dataset directory: explicit path > default (download if missing)."""
    if data_path is not None:
        return data_path
    if _DEFAULT_DATA_PATH.exists():
        return str(_DEFAULT_DATA_PATH)
    if download_dataset:
        _download_bsct_spice(_DEFAULT_DATA_PATH)
        return str(_DEFAULT_DATA_PATH)
    raise ValueError(
        "BSCT-SPICE dataset not found at "
        f"{_DEFAULT_DATA_PATH}. Pass data_path=... or set download_dataset=True."
    )


def _read_xyz_frame(path: str) -> tuple[list[str], float, np.ndarray, np.ndarray]:
    """Parse one BSCT extxyz frame: species, energy (eV), positions, forces (eV/Å).

    A direct parser for the dataset's fixed ``species pos(3) forces(3)`` columns;
    ``ase.io.read`` dominates load time, so this is used for the one-time cost of
    building the atompack cache.
    """
    with open(path) as fh:
        lines = fh.read().split("\n")
    n = int(lines[0])
    energy = float(lines[1].split("energy=", 1)[1].split(None, 1)[0])
    rows = [ln.split() for ln in lines[2 : 2 + n]]
    data = np.array([r[1:7] for r in rows], dtype=np.float64)
    return [r[0] for r in rows], energy, data[:, :3], data[:, 3:6]


def _ensure_atompack(data_path: str, *, verbose: bool) -> None:
    """Cache each system's xyz frames as a per-system atompack ``.atp`` store.

    The dataset ships as ~48k single-frame extxyz files whose ase.io parsing is
    the dominant load cost; converting once to atompack lets later runs read flat
    float64 position/energy/force arrays straight into batched ``SimState``s. Run
    after download and for user-supplied paths alike; idempotent, so systems whose
    ``.atp`` already exists are skipped.
    """
    import atompack
    from tqdm import tqdm

    for dataset in sorted(os.listdir(data_path)):
        dataset_dir = os.path.join(data_path, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for system_id in tqdm(sorted(os.listdir(dataset_dir)), desc=f"atompack {dataset}", disable=not verbose):
            system_dir = os.path.join(dataset_dir, system_id)
            atp_path = os.path.join(system_dir, _ATP_FILE)
            if os.path.exists(atp_path):
                continue
            n = int(np.load(os.path.join(system_dir, "partition.npz"))["perturb_range"][2])
            db = atompack.Database(atp_path, overwrite=True)
            zs = None
            for i in range(n):
                species, energy, positions, forces = _read_xyz_frame(os.path.join(system_dir, f"atoms_{i}.xyz"))
                if zs is None:
                    zs = np.array([atomic_numbers[s] for s in species], dtype=np.uint8)
                db.add_molecule(atompack.Molecule.from_arrays(positions, zs, energy=energy, forces=forces))
            db.flush()


def _simstate_from_frames(
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SimState:
    """Pack per-frame ``(positions, atomic numbers)`` into one batched ``SimState``.

    Each frame is its own aperiodic system, mirroring the single-frame
    :func:`from_ase` path the native probes use but assembled in a single shot.
    """
    counts = [f[2].shape[0] for f in frames]
    pos = np.concatenate([f[2] for f in frames])
    zs = np.concatenate([f[3] for f in frames])
    system_idx = np.repeat(np.arange(len(frames)), counts)
    return SimState(
        positions=torch.tensor(pos, dtype=dtype, device=device),
        masses=torch.tensor(atomic_masses[zs], dtype=dtype, device=device),
        cell=torch.zeros(len(frames), 3, 3, dtype=dtype, device=device),
        pbc=False,
        atomic_numbers=torch.tensor(zs, dtype=torch.long, device=device),
        system_idx=torch.tensor(system_idx, dtype=torch.long, device=device),
    )


def _batch_predict(model: object, state: SimState) -> tuple[Tensor, Tensor]:
    """Per-system energy ``(n_systems,)`` and per-atom forces ``(n_atoms, 3)``.

    Positions are grad-enabled exactly as the single-system :func:`predict` path,
    since torch-sim models compute their force head via internal autograd even
    when only values are read.
    """
    work = state.clone()
    work.positions = state.positions.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = model(work)
    return out["energy"].detach().reshape(-1), out["forces"].detach().reshape(-1, 3)


def _bsct_systems_batched(
    model: object,
    data_path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    max_atoms_per_batch: int,
    verbose: bool,
) -> list:
    """Build BSCT ``System`` objects, batching frames into multi-system forwards.

    Ground-truth positions/energies/forces are read from each system's atompack
    cache (see :func:`_ensure_atompack`) as flat float64 arrays. Every frame is one
    "system" in a torch-sim batched ``SimState``; frames are packed greedily up to
    ``max_atoms_per_batch`` atoms and scored in a single ``model(state)`` call,
    replacing BSCT's one-frame-at-a-time ASE-calculator loop.
    """
    import atompack
    from bsct._infer import System
    from tqdm import tqdm

    raw: list[dict] = []
    frames: list[tuple[int, int, np.ndarray, np.ndarray]] = []  # (system index in `raw`, frame index, positions, atomic numbers)

    for dataset in sorted(os.listdir(data_path)):
        dataset_dir = os.path.join(data_path, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for system_id in tqdm(sorted(os.listdir(dataset_dir)), desc=dataset, disable=not verbose):
            system_dir = os.path.join(dataset_dir, system_id)
            partition = np.load(os.path.join(system_dir, "partition.npz"))
            n = int(partition["perturb_range"][2])
            lincoords = np.linspace(*partition["perturb_range"][:2], num=n)

            db = atompack.Database.open(os.path.join(system_dir, _ATP_FILE))
            flat = db.get_molecules_flat(list(range(len(db))))
            offsets = np.cumsum(np.asarray(flat["n_atoms"]))[:-1]
            positions = np.split(np.asarray(flat["positions"]), offsets)
            gt_forces = np.split(np.asarray(flat["forces"]), offsets)
            atomic_nums = np.split(np.asarray(flat["atomic_numbers"]).astype(np.int64), offsets)

            ri = len(raw)
            for i in range(n):
                frames.append((ri, i, positions[i], atomic_nums[i]))
            raw.append(
                {
                    "dataset": dataset,
                    "system_id": system_id,
                    "positions": np.array(positions),
                    "gt_energy": np.asarray(flat["energy"]),
                    "gt_forces": np.array(gt_forces),
                    "lincoords": lincoords,
                    "calc_energy": np.empty(n),
                    "calc_forces": [None] * n,
                }
            )

    device = torch.device(device)
    i = 0
    pbar = tqdm(total=len(frames), desc="BSCT forward", disable=not verbose)
    while i < len(frames):
        counts, j = [], i
        while j < len(frames) and (not counts or sum(counts) + frames[j][2].shape[0] <= max_atoms_per_batch):
            counts.append(frames[j][2].shape[0])
            j += 1
        batch = _simstate_from_frames(frames[i:j], device=device, dtype=dtype)
        energy, forces = _batch_predict(model, batch)
        for k, fk in enumerate(torch.split(forces, counts)):
            ri, fi, _, _ = frames[i + k]
            raw[ri]["calc_energy"][fi] = float(energy[k])
            raw[ri]["calc_forces"][fi] = fk.cpu().numpy().astype(np.float64)
        pbar.update(j - i)
        i = j
    pbar.close()

    systems = []
    for r in raw:
        mask = ~np.isnan(r["gt_energy"])
        systems.append(
            System(
                dataset=r["dataset"],
                system_id=r["system_id"],
                positions=r["positions"][mask],
                gt_energy=r["gt_energy"][mask],
                gt_forces=r["gt_forces"][mask],
                calc_energy=r["calc_energy"][mask],
                calc_forces=np.array(r["calc_forces"])[mask],
                lincoords=r["lincoords"][mask],
            )
        )
    return systems


@due.dcite(
    BibTeX(_BSCT_BIBTEX),
    description="Bond Smoothness Characterization Test (Force Smoothness Deviation)",
    path="mlip_smoothness_eval.checks.bsct",
)
def bsct_smoothness(
    model: object,
    *,
    data_path: str | None = None,
    download_dataset: bool = False,
    output_path: str | None = None,
    store_metrics: bool = False,
    store_plots: bool = False,
    store_xyz: bool = False,
    verbose: bool = False,
    batched: bool = True,
    max_atoms_per_batch: int = 4096,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> CheckResult:
    """Run BSCT on ``model`` and wrap its dataset-averaged metrics as a ``CheckResult``.

    The BSCT-SPICE dataset is not shipped with the pip package. Pass ``data_path``
    to point at your own copy, or leave it ``None`` to use the in-repo default
    location (``data/bsct_spice``), downloading it there first when
    ``download_dataset=True`` and it is not already present. Returns ``bsct_*`` FSD
    and energy/force-MAE metrics over the full / compression / stretch bond ranges.

    With ``batched=True`` (default) the dataset is first converted to a per-system
    atompack cache (one-time cost; see :func:`_ensure_atompack`), then inference
    packs frames into multi-system torch-sim ``SimState`` batches of up to
    ``max_atoms_per_batch`` atoms — a handful of large GPU forward passes instead of
    thousands of single-molecule ones. The ``store_*`` flags and ``output_path``
    apply only to the unbatched (``batched=False``) upstream ASE-calculator path; a
    throwaway temp dir is used there by default.
    """
    data_path = _resolve_dataset(data_path, download_dataset)

    if batched:
        from bsct._eval import METRIC_KEYS, compute_system_metrics

        _ensure_atompack(str(data_path), verbose=verbose)
        systems = _bsct_systems_batched(
            model,
            str(data_path),
            device=device,
            dtype=dtype,
            max_atoms_per_batch=max_atoms_per_batch,
            verbose=verbose,
        )
        metrics_per_system = [compute_system_metrics(s) for s in systems]
        results = {k: float(np.mean([m[k] for m in metrics_per_system])) for k in METRIC_KEYS}
    else:
        from bsct import evaluate_bsct

        calc = TorchSimCalculator(model, device=device, dtype=dtype)
        results = evaluate_bsct(
            calc=calc,
            data_path=str(data_path),
            output_path=output_path or tempfile.mkdtemp(prefix="bsct_"),
            store_metrics=store_metrics,
            store_plots=store_plots,
            store_xyz=store_xyz,
            verbose=verbose,
        )

    metrics = {f"bsct_{k}": float(v) for k, v in results.items()}
    return CheckResult("bsct", metrics, {}, {"data_path": str(data_path)})
