"""Structure-morph animations: atoms moving beside the advancing curve.

For the sweep-style checks (diatomic, displacement_scan, cutoff_smoothness) the
``CheckResult.trace`` carries the swept energy plus per-frame positions
(``diatomic`` carries only the swept radii, from which the two-atom geometry is
reconstructed). Each frame is rendered with matplotlib (3D scatter of atoms |
energy curve up to that frame) and stitched into a gif with imageio.

matplotlib (not plotly+kaleido) is used deliberately: it needs no headless
browser, so gif rendering works in any environment including notebooks and
batch nodes.
"""

from __future__ import annotations

import numpy as np

from mlip_smoothness_eval.checks.base import CheckResult

_SWEEP_CHECKS = {"diatomic", "displacement_scan", "cutoff_smoothness", "boundary_crossing"}

_EMBER = "#C4650D"
_SLATE = "#4E728A"


def _y_for(trace: dict) -> tuple[np.ndarray, str]:
    if "energy" in trace:
        return np.asarray(trace["energy"]), "energy (eV)"
    raise ValueError("trace has no energy series to animate")


def _frames_for(result: CheckResult) -> np.ndarray:
    """``(n, N, 3)`` per-frame positions for a sweep check.

    ``displacement_scan`` / ``cutoff_smoothness`` carry ``frames`` in their
    trace; ``diatomic`` carries only the swept radii, so reconstruct the two-atom
    geometry from them.
    """
    t = result.trace
    if "frames" in t:
        return np.asarray(t["frames"])
    if result.name == "diatomic":
        from mlip_smoothness_eval.checks.diatomic import diatomic_frames

        return diatomic_frames(t["symbol"], np.asarray(t["x"]))
    raise ValueError(f"check {result.name!r} carries no frames to animate")


def _caption(result: CheckResult) -> str:
    """Compact one-line summary of the check's scalar metric(s) for the title."""
    parts = [
        f"{k.split('_', 1)[1] if '_' in k else k}={v:.3g}"
        for k, v in result.metrics.items()
    ]
    return f"{result.name}  ·  " + "   ".join(parts)


def make_gif(
    result: CheckResult,
    path: str,
    *,
    fps: int = 12,
    max_frames: int = 60,
    dpi: int = 90,
    freeze_last_s: float = 2.0,
) -> str:
    """Render ``result`` to an animated gif at ``path``; returns ``path``.

    The atoms morph (left) beside the energy curve tracing out to the current
    frame (right). The final frame is held for ``freeze_last_s`` seconds so the
    end state is readable before the loop restarts.
    """
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    if result.name not in _SWEEP_CHECKS:
        raise ValueError(f"no gif renderer for check {result.name!r}")
    t = result.trace
    frames = _frames_for(result)  # (n, N, 3)
    x = np.asarray(t["x"])
    y, ylabel = _y_for(t)
    xlabel = t.get("xlabel", "step")

    n = frames.shape[0]
    idx = np.linspace(0, n - 1, min(max_frames, n)).round().astype(int)

    pos_all = frames.reshape(-1, 3)
    pad = 0.5
    ranges = [
        (float(pos_all[:, d].min()) - pad, float(pos_all[:, d].max()) + pad)
        for d in range(3)
    ]
    y_pad = 0.05 * (float(y.max()) - float(y.min()) + 1e-9)
    y_range = (float(y.min()) - y_pad, float(y.max()) + y_pad)
    x_range = (float(x.min()), float(x.max()))
    caption = _caption(result)

    images = []
    for i in idx:
        fig = plt.figure(figsize=(8, 4), dpi=dpi)
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        p = frames[i]
        ax3d.scatter(p[:, 0], p[:, 1], p[:, 2], s=120, c=_SLATE, depthshade=True)
        ax3d.set_xlim(*ranges[0])
        ax3d.set_ylim(*ranges[1])
        ax3d.set_zlim(*ranges[2])
        ax3d.set_xticklabels([])
        ax3d.set_yticklabels([])
        ax3d.set_zticklabels([])
        ax3d.set_title("structure", fontsize=10)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x[: i + 1], y[: i + 1], color=_EMBER, lw=2)
        ax.plot([x[i]], [y[i]], "o", color=_EMBER, ms=7)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, color="#E2E0D8")
        fig.suptitle(caption, fontsize=10, color=_EMBER, y=0.99)
        # explicit margins (tight_layout fights the 3D axes and warns)
        fig.subplots_adjust(left=0.04, right=0.93, bottom=0.16, top=0.88, wspace=0.28)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        images.append(buf[..., :3].copy())
        plt.close(fig)

    # Hold the last frame by repeating it, so it reads before the loop wraps.
    if images and freeze_last_s > 0:
        images.extend([images[-1]] * max(1, round(freeze_last_s * fps)))

    # imageio's pillow gif writer takes per-frame duration in ms, not fps.
    imageio.mimsave(path, images, duration=1000.0 / fps, loop=0)
    return path
