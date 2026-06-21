# %% [markdown]
# # Diatomic bond-length sweep — frame by frame
#
# One of the three *moving-atom* smoothness probes. We pull a homonuclear dimer
# apart — from ~0.9x the covalent radius out past the van der Waals tail — and
# score the resulting 1-D potential-energy curve E(r):
#
# - **tortuosity** — total energy variation along r_min -> r_eq -> r_max; a clean
#   single-well curve scores ~1, wiggles push it up,
# - **energy jump** — magnitude-weighted sign flips in the discrete gradient,
# - **force flips** — a clean single well flips the axial force ~once.
#
# The animation shows the **two atoms separating** (left) while the **energy
# curve traces out** (right); the final frame is held for 2 s. We drive the
# analytic Lennard-Jones reference model, so this is what a smooth, conservative
# potential is *supposed* to look like.
#
# Run top-to-bottom as a script, or open in VS Code / Jupyter and run the cells.

# %%
import sys
from pathlib import Path

# the shared helper lives beside this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frame_video as fv

# %%
model = fv.reference_model()

# %% [markdown]
# ## Run the probe
# Scalar scores for the whole sweep:

# %%
result = fv.diatomic_result(model, symbol="O")
result.metrics

# %% [markdown]
# ## Watch the atoms move
# Atoms morph on the left; E(r) advances on the right; the caption holds the
# final scalar metrics.

# %%
fv.animate(result, "diatomic.gif")
# %%