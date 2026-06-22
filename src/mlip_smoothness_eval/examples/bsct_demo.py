# %% [markdown]
# # BSCT cross-check demo — MACE-MP-0
#
# Run the external [Bond Smoothness Characterization Test](https://github.com/ryanliu30/bsct)
# (BSCT, arXiv:2602.04861) on a real foundation MLIP and print its Force
# Smoothness Deviation alongside this suite's native scorecard. Run as a notebook
# (`# %%` cells) or top to bottom as a script.
#
# Setup:
#
#     uv sync --extra mace --extra bsct
#     uv run python src/mlip_smoothness_eval/examples/bsct_demo.py
#
# `download_bsct_dataset=True` fetches the BSCT-SPICE dataset to the in-repo
# default location (`data/bsct_spice`) on first run; pass a path as the first
# argument to use your own copy instead. Expect ~40 min on GPU.

# %%
import sys

import torch

from mlip_smoothness_eval import evaluate_smoothness

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
# first positional arg = dataset path; skip flags (e.g. Jupyter's --f=kernel.json)
data_path = next((a for a in sys.argv[1:] if not a.startswith("-")), None)  # None -> in-repo default
print("device:", DEVICE, "| dataset:", data_path or "data/bsct_spice (auto-download)")

# %% [markdown]
# ## Load MACE-MP-0 as a torch-sim model
#
# BSCT drives an ASE calculator; `evaluate_smoothness` wraps this torch-sim model
# in `TorchSimCalculator` internally, so the same model object feeds both the
# native probes and the external check.

# %%
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

raw = mace_mp(model="small", return_raw_model=True, device=DEVICE)
model = MaceModel(model=raw, device=DEVICE, dtype=DTYPE, compute_forces=True)

# %% [markdown]
# ## Run the suite with the BSCT cross-check enabled
#
# `run_bsct=True` adds the `bsct_full_*`, `bsct_compress_*`, and `bsct_stretch_*`
# metrics (FSD in Å⁻¹, energy/force MAE) to the scorecard.

# %%
report = evaluate_smoothness(
    model,
    diatomic_symbols=("H", "O"),
    device=DEVICE,
    dtype=DTYPE,
    run_bsct=True,
    bsct_data_path=data_path,
    download_bsct_dataset=True,
    model_name="MACE-MP-0 (small)",
)
report

# %% [markdown]
# ## Just the BSCT metrics

# %%
for key, value in report.metrics.items():
    if key.startswith("bsct_"):
        print(f"{key:28s} {value:.4f}")

# %%
