# %% [markdown]
# # Displacement scan — frame by frame
#
# One of the three *moving-atom* smoothness probes. We pick a single random
# direction and slide **every atom** along it by s in [-A, A], tracing the 1-D
# cut E(s). Two things should hold for a smooth, conservative model:
#
# - **force/energy consistency** — -F.d should equal dE/ds (a direct-force head
#   that isn't a true gradient breaks this),
# - **energy roughness** — E(s) should be smooth; a large third difference flags
#   kinks or sawtooth.
#
# The animation shows the **whole crystal sliding** along the displacement
# (left) while E(s) traces out (right); the final frame is held for 2 s.
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
# Defaults to a small rattled FCC Ar cell; pass your own `state` to
# `fv.displacement_result` to scan a different structure.

# %%
result = fv.displacement_result(model)
result.metrics

# %% [markdown]
# ## Watch the atoms move
# All atoms shift together along the random direction as E(s) advances.

# %%
fv.animate(result, "displacement_scan.gif")

# %%