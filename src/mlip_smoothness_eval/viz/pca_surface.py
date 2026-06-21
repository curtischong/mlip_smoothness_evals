"""PCA energy surface: the learned PES projected onto its top two PCs.

Sample N perturbed configurations of a structure, PCA the flattened position
vectors to get PC1/PC2, evaluate the model energy at each, interpolate onto a
regular grid, and render a Plotly Surface (PC1/PC2 on the base, energy as
depth). A jagged surface is a non-smooth potential.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from torch_sim.state import SimState

from mlip_smoothness_eval.checks.base import predict
from mlip_smoothness_eval.structures import with_positions
from mlip_smoothness_eval.viz.theme import SEQUENTIAL


def pca_energy_surface(
    model: object,
    state: SimState,
    *,
    n_samples: int = 400,
    sigma: float = 0.15,
    grid: int = 60,
    seed: int = 0,
) -> go.Figure:
    """Build the PCA energy-surface figure for ``model`` around ``state``."""
    import torch

    rng = np.random.default_rng(seed)
    base = state.positions.detach().cpu().numpy()
    n_atoms = base.shape[0]
    flat0 = base.reshape(-1)

    samples = np.empty((n_samples, flat0.size), dtype=np.float64)
    energies = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        perturbed = flat0 + rng.normal(scale=sigma, size=flat0.size)
        pos_t = torch.tensor(perturbed.reshape(n_atoms, 3), dtype=state.positions.dtype)
        pred = predict(model, with_positions(state, pos_t))
        samples[i] = perturbed
        energies[i] = float(pred.energy)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(samples)  # (n_samples, 2)

    gx = np.linspace(coords[:, 0].min(), coords[:, 0].max(), grid)
    gy = np.linspace(coords[:, 1].min(), coords[:, 1].max(), grid)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    mesh_z = griddata(coords, energies, (mesh_x, mesh_y), method="cubic")
    var = pca.explained_variance_ratio_

    fig = go.Figure(
        go.Surface(
            x=mesh_x, y=mesh_y, z=mesh_z,
            colorscale=SEQUENTIAL, colorbar=dict(title="energy (eV)"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=energies, mode="markers",
            marker=dict(size=2, color="#31362E", opacity=0.4),
            name="samples", showlegend=False,
        )
    )
    fig.update_layout(
        height=560,
        paper_bgcolor="white",
        margin=dict(t=30, r=10, b=10, l=10),
        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.0%} var)",
            yaxis_title=f"PC2 ({var[1]:.0%} var)",
            zaxis_title="energy (eV)",
        ),
        font=dict(color="#1D272A", size=12),
    )
    return fig
