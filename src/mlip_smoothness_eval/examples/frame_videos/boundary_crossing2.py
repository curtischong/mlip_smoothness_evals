# %% [markdown]
# # Boundary crossing — *how smooth is the curve?*
#
# Same sweep as `boundary_crossing` (drag one atom a full lattice vector across a
# cell face), but instead of spike ratios we ask what the energy curve's **shape**
# is. We fit splines to `E(θ)` and report several smoothness descriptors, because
# they measure genuinely different things:
#
# - **MAE to a smoothing spline** — deviation of the raw samples from a de-noised
#   baseline. This is *noise*, not smoothness.
# - **bending energy** `∫(E'')² dx` — roughness of `E = f(θ)`.
# - **curvature energy** `∫κ² ds` — geometric bending of the curve as a shape.
# - **jerk energy** `∫(E''')² dx` — third-derivative penalty; most sensitive to
#   kinks (a corner is a jerk impulse).
#
# A kink can hide from the MAE (low) while blowing up bending/jerk (high) — which
# is exactly why the spike-ratio probe and this one are complementary.
#
# Run top-to-bottom as a script, or open in VS Code / Jupyter and run the cells.

# %%
from mlip_smoothness_eval.examples.frame_videos import frame_video as fv
from mlip_smoothness_eval.viz.curves import curve

# %%
model = fv.reference_model()

# %% [markdown]
# ## Run the probe
# Defaults to the dilute FCC Ar cell; pass your own periodic `state` to
# `fv.boundary_curve_result` to score a different structure's crossing.

# %%
result = fv.boundary_curve_result(model)
result.metrics

# %% [markdown]
# ## See the curve
# Top: the model energy (markers) with the smoothing-spline baseline (line) — the
# gap between them is the MAE. Bottom: the bending density `(E'')²`; its shaded
# area *is* the bending energy, so spikes here mark where the curve is rough.

# %%
curve(result)

# %% [markdown]
# ## Watch the atom cross the boundary
# The dragged atom exits one face and reappears on the opposite one; for a smooth,
# periodic model the energy curve stays continuous straight through the wrap.

# %%
fv.animate(result, "boundary_crossing2.gif")

# %%
