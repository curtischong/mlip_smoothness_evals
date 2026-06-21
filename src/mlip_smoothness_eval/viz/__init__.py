"""Visualization helpers: curves, gifs, PCA energy surface."""

from mlip_smoothness_eval.viz.curves import curve
from mlip_smoothness_eval.viz.gifs import make_gif
from mlip_smoothness_eval.viz.pca_surface import pca_energy_surface
from mlip_smoothness_eval.viz.theme import EDITORIAL_8, PLOTLY_CONFIG, SEQUENTIAL, apply_theme

__all__ = [
    "curve",
    "make_gif",
    "pca_energy_surface",
    "apply_theme",
    "EDITORIAL_8",
    "SEQUENTIAL",
    "PLOTLY_CONFIG",
]
