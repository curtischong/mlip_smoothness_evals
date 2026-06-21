"""``SmoothnessReport``: aggregated metrics + inline notebook rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mlip_smoothness_eval.checks.base import CheckResult


def _jsonable(value: object) -> object:
    """Recursively coerce numpy arrays/scalars into JSON-native types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value

# Metric display order for the HTML scorecard.
_METRIC_ORDER: tuple[str, ...] = (
    "nonconservativity_rmse",
    "nonconservativity_rel",
    "scan_force_energy_inconsistency",
    "scan_energy_roughness",
    "cutoff_energy_spike_ratio",
    "cutoff_force_spike_ratio",
    "force_jacobian_asymmetry",
    "diatomic_tortuosity",
    "diatomic_energy_jump",
    "diatomic_force_flips",
    "nve_final_drift_per_atom",
    "nve_max_drift_per_atom",
    "nve_drift_std_per_atom",
)


def _aggregate(results: list[CheckResult]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in results:
        for k, v in r.metrics.items():
            if v == v:  # skip NaN
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


@dataclass
class SmoothnessReport:
    """The object returned by :func:`evaluate_smoothness`.

    Holds every probe's :class:`CheckResult`, exposes aggregated ``metrics``,
    and renders a scorecard inline in a notebook via ``_repr_html_``. The
    render methods (``curve`` / ``gif`` / ``pca_surface``) return plotly figures
    or file paths.
    """

    results: list[CheckResult]
    model_name: str = "model"
    structures: list = field(default_factory=list)
    _model: object = None
    notes: list[str] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float]:
        return _aggregate(self.results)

    # ---- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """Plain-data view: aggregated metrics + each probe's metrics and curve trace.

        Model objects and the heavy ``frames`` arrays (only needed for gifs) are
        dropped so the result is small and JSON-serializable for cross-model
        comparison.
        """
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "notes": list(self.notes),
            "checks": [
                {
                    "name": r.name,
                    "metrics": _jsonable(r.metrics),
                    "meta": _jsonable(r.meta),
                    "trace": {k: _jsonable(v) for k, v in r.trace.items() if k != "frames"},
                }
                for r in self.results
            ],
        }

    def save(self, path: str | Path) -> Path:
        """Write :meth:`to_dict` to ``path`` as JSON; returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    # ---- lookups -----------------------------------------------------------
    def _find(self, name: str, symbol: str | None = None) -> CheckResult:
        for r in self.results:
            if r.name != name:
                continue
            if symbol is not None and r.trace.get("symbol") != symbol and r.meta.get("symbol") != symbol:
                continue
            return r
        raise KeyError(f"no result for check {name!r}" + (f" symbol={symbol!r}" if symbol else ""))

    # ---- render ------------------------------------------------------------
    def curve(self, name: str, *, symbol: str | None = None):
        """Plotly energy/force figure for a check (``symbol`` for diatomic)."""
        from mlip_smoothness_eval.viz.curves import curve as _curve

        return _curve(self._find(name, symbol))

    def gif(self, name: str, *, path: str | None = None, symbol: str | None = None, **kwargs) -> str:
        """Render a structure-morph gif for a sweep check; returns the path."""
        from mlip_smoothness_eval.viz.gifs import make_gif

        path = path or f"{name}.gif"
        return make_gif(self._find(name, symbol), path, **kwargs)

    def pca_surface(self, structure=None, **kwargs):
        """Plotly 3D PCA energy surface around ``structure`` (defaults to first)."""
        from mlip_smoothness_eval.viz.pca_surface import pca_energy_surface

        if self._model is None:
            raise RuntimeError("report has no model reference; cannot build PCA surface")
        if structure is None:
            if not self.structures:
                raise ValueError("no structure available for PCA surface")
            structure = self.structures[0]
        return pca_energy_surface(self._model, structure, **kwargs)

    # ---- scorecard ---------------------------------------------------------
    def to_frame(self) -> "pd.DataFrame":
        """Aggregated metrics as a single-column DataFrame, in display order."""
        import pandas as pd

        metrics = self.metrics
        ordered = [k for k in _METRIC_ORDER if k in metrics]
        return pd.DataFrame({"value": [metrics[k] for k in ordered]}, index=ordered)

    def _repr_html_(self) -> str:
        return self.to_frame()._repr_html_()

    def display(self) -> None:
        from IPython.display import display

        display(self.to_frame())
