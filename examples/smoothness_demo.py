# %% [markdown]
# # MLIP smoothness demo — MACE-MP-0
#
# Drive the full smoothness suite through a real foundation MLIP (MACE-MP-0)
# and render the scorecard, per-check curves, a gif, and the PCA energy surface
# inline. Run this as a notebook (VS Code / Jupyter `# %%` cells) or top to
# bottom as a script.
#
# Install the demo extra first: `uv sync --extra mace`.

# %%
import torch

from mlip_smoothness_eval import evaluate_smoothness, structures

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
print("device:", DEVICE)

# %% [markdown]
# ## Load MACE-MP-0 as a torch-sim model
#
# torch-sim wraps a raw MACE module in its `MaceModel`, which implements the
# `ModelInterface` contract the suite expects. Weights fetch from the MACE
# release on first use (no HF token needed).

# %%
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

raw = mace_mp(model="small", return_raw_model=True, device=DEVICE)
model = MaceModel(model=raw, device=DEVICE, dtype=DTYPE, compute_forces=True)

# %% [markdown]
# ## Run the suite
#
# `evaluate_smoothness` runs all six probes and returns a report. The report
# renders a scorecard inline.

# %%
report = evaluate_smoothness(
    model,
    structures=[structures.random_crystal("Cu", device=DEVICE, dtype=DTYPE)],
    diatomic_symbols=("H", "O", "Cu"),
    device=DEVICE,
    dtype=DTYPE,
    nve_steps=1000,
    model_name="MACE-MP-0 (small)",
)
report

# %% [markdown]
# ## Per-check curves

# %%
report.curve("diatomic", symbol="O")

# %%
report.curve("displacement_scan")

# %%
report.curve("nve_drift")

# %% [markdown]
# ## Structure-morph gif (displacement scan)

# %%
report.gif("displacement_scan", path="displacement_scan.gif")
print("wrote displacement_scan.gif")

# %% [markdown]
# ## PCA energy surface
#
# Samples perturbed configurations, projects onto the top two PCs, and renders
# the model energy as a 3D surface. A jagged surface is a non-smooth potential.

# %%
report.pca_surface(n_samples=400)

# %%