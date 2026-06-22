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

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from duecredit import BibTeX, due

from mlip_smoothness_eval.checks.base import CheckResult, predict
from mlip_smoothness_eval.structures import from_ase
from mlip_smoothness_eval.utils import resolve_path

_BSCT_REPO = "https://github.com/ryanliu30/bsct.git"
# default in-repo location for the downloaded BSCT-SPICE dataset (gitignored)
_DEFAULT_DATA_PATH = Path(resolve_path("data/bsct_spice"))

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
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> CheckResult:
    """Run BSCT on ``model`` and wrap its dataset-averaged metrics as a ``CheckResult``.

    The BSCT-SPICE dataset is not shipped with the pip package. Pass ``data_path``
    to point at your own copy, or leave it ``None`` to use the in-repo default
    location (``data/bsct_spice``), downloading it there first when
    ``download_dataset=True`` and it is not already present. Outputs
    (plots/xyz/per-system json) land in ``output_path`` when the corresponding
    ``store_*`` flags are set; by default a throwaway temp dir is used. Returns
    ``bsct_*`` FSD and energy/force-MAE metrics over the full / compression /
    stretch bond ranges.
    """
    from bsct import evaluate_bsct

    data_path = _resolve_dataset(data_path, download_dataset)
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
