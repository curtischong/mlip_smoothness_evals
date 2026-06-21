# %% [markdown]
# # Cutoff-crossing drag — frame by frame
#
# One of the three *moving-atom* smoothness probes. We drag a **single atom** in
# a straight line across the cell. As it crosses a neighbour-list cutoff, a
# non-smooth model gains or loses an edge and the energy/force jump. We flag
# those discontinuities as spikes relative to the median step:
#
# - **energy spike ratio** — max|dE| / median|dE|,
# - **force spike ratio** — same for |F_0|.
#
# Ratios near 1 mean no discontinuity. The animation shows the **dragged atom
# travelling through the cell** (left) while the energy advances (right); the
# final frame is held for 2 s. We use a *dilute* Ar cell so the atom moves
# through the gentle attractive tail rather than slamming into a repulsive wall.
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

# %%
result = fv.cutoff_result(model)
result.metrics

# %% [markdown]
# ## Watch the atom move
# One atom drags across the cell while the energy traces out; a smooth model
# gives a smooth curve with no spikes.

# %%
fv.animate(result, "cutoff_smoothness.gif")

# %%