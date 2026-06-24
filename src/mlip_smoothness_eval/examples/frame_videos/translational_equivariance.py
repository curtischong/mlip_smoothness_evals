# %% [markdown]
# # Translational equivariance — *does the energy move when nothing should?*
#
# Translating **every** atom of a periodic cell by the same vector relabels which
# periodic image sits in the box but leaves the physical configuration untouched,
# so the energy must stay perfectly flat. We slide each cell one full lattice
# vector (phase `θ` from `0` to `2π`, fractional shift `θ/2π`) and measure the
# spurious energy / force variation a translationally invariant model must not
# have:
#
# - **energy_range_max** `max E − min E` of the worst structure (eV),
# - **energy_per_atom_max** the same normalised by atom count (eV/atom),
# - **energy_std_per_atom_mean / _max** the average / worst per-atom std of the
#   energy across the distinct translated images (eV/atom) — how *consistent* the
#   energy stays as the cell slides; this is the headline consistency metric,
# - **force_dev_max** the largest force change from the untranslated frame (eV/Å),
# - **periodicity_error_max** the residual after a full-lattice-vector shift.
#
# Every metric is zero for a perfect model. A model with a hidden grid (mesh-based
# long-range, fixed FFT box) instead ripples as the lattice slides under the grid
# — the classic "egg-box" effect.
#
# The probe scores a **set** of structures and batches every (structure, step)
# pair into torch-sim `SimState`s; the animation below shows the first one.
#
# Run top-to-bottom as a script, or open in VS Code / Jupyter and run the cells.

# %%
from mlip_smoothness_eval.examples.frame_videos import frame_video as fv
from mlip_smoothness_eval.viz.curves import curve

# %%
model = fv.reference_model()

# %% [markdown]
# ## Run the probe
# Defaults to a few periodic Ar cells; pass your own list of periodic `states` to
# `fv.equivariance_result` to score different structures.

# %%
result = fv.equivariance_result(model)
result.metrics

# %% [markdown]
# ## See the curve
# For a translationally invariant model this is a flat line — the energy does not
# move as the whole cell slides one lattice vector.

# %%
curve(result)

# %% [markdown]
# ## Watch the cell slide one lattice vector
# Every atom shifts together and wraps through the box; a correct model keeps the
# energy curve dead flat straight through the full-period translation.

# %%
fv.animate(result, "translational_equivariance.gif")

# %%
