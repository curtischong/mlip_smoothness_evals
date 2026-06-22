# %% [markdown]
# # Boundary crossing — frame by frame
#
# One of the *moving-atom* smoothness probes, specialised to the **periodic
# boundary**. We drag a **single atom** by one full lattice vector, straight
# across a face of the cell. Under the minimum-image convention this trajectory
# is both *smooth* and *periodic*: translating an atom by a full lattice vector
# reproduces a physically identical configuration, so the energy must come back
# to exactly where it started, with no jump as the atom crosses the boundary.
#
# - **energy spike ratio** — max|dE| / median|dE| (a discontinuous wrap spikes here),
# - **force spike ratio** — same for the force on the dragged atom,
# - **periodicity error** — |E(end) − E(start)|; a model that drops PBC ramps
#   the energy and never returns, so this blows up.
#
# Ratios near 1 and a periodicity error near 0 mean the boundary is handled
# smoothly. The animation shows the dragged atom **leaving one face and
# re-entering the opposite** (positions are wrapped back into the cell) while the
# energy advances smoothly — the position teleports but the energy does not.
#
# Run top-to-bottom as a script, or open in VS Code / Jupyter and run the cells.

# %%
from mlip_smoothness_eval.examples.frame_videos import frame_video as fv

# %%
model = fv.reference_model()

# %% [markdown]
# ## Run the probe
# Defaults to the dilute FCC Ar cell; pass your own periodic `state` to
# `fv.boundary_result` to cross the boundary of a different structure.

# %%
result = fv.boundary_result(model)
result.metrics

# %% [markdown]
# ## Watch the atom cross the boundary
# The dragged atom exits one face and reappears on the opposite one; a smooth,
# periodic model keeps the energy curve continuous straight through the wrap.

# %%
fv.animate(result, "boundary_crossing.gif")

# %%
