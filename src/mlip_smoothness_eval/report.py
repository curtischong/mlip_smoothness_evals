"""``SmoothnessReport``: aggregated metrics + inline notebook rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mlip_smoothness_eval.checks.base import CheckResult

# Direction of "good" for each metric and a soft threshold for the LJ-style
# ideal. Used purely for the HTML scorecard coloring; the numbers are the truth.
_METRIC_INFO: dict[str, tuple[str, float]] = {
    "nonconservativity_rmse": ("lower", 1e-3),
    "nonconservativity_rel": ("lower", 1e-3),
    "scan_force_energy_inconsistency": ("lower", 1e-2),
    "scan_energy_roughness": ("lower", 1e3),
    "cutoff_energy_spike_ratio": ("lower", 10.0),
    "cutoff_force_spike_ratio": ("lower", 10.0),
    "force_jacobian_asymmetry": ("lower", 1e-3),
    "diatomic_tortuosity": ("one", 1.5),
    "diatomic_energy_jump": ("lower", 1e-2),
    "diatomic_force_flips": ("one", 2.0),
    "nve_final_drift_per_atom": ("lower", 1e-2),
    "nve_max_drift_per_atom": ("lower", 1e-2),
    "nve_drift_std_per_atom": ("lower", 1e-2),
}


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

    # ---- inline HTML scorecard --------------------------------------------
    def _repr_html_(self) -> str:
        return self.to_html()

    def display(self) -> None:
        from IPython.display import HTML, display

        display(HTML(self.to_html()))

    def to_html(self) -> str:
        rows = []
        for key in _METRIC_INFO:
            if key not in self.metrics:
                continue
            val = self.metrics[key]
            rows.append(self._row_html(key, val))
        notes_html = ""
        if self.notes:
            items = "".join(f"<li>{n}</li>" for n in self.notes)
            notes_html = f"<ul style='margin:8px 0 0;color:#7B7B7B;font-size:12px'>{items}</ul>"
        return (
            "<div style=\"font-family:'Suisse Intl',-apple-system,Arial,sans-serif;"
            "color:#1D272A;max-width:680px\">"
            f"<h3 style='margin:0 0 2px'>Smoothness scorecard — {self.model_name}</h3>"
            "<p style='margin:0 0 10px;color:#7B7B7B;font-size:12px'>"
            "lower is better unless noted; diatomic tortuosity / force-flips ideal ≈ 1</p>"
            "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            "<tr style='text-align:left;border-bottom:1px solid #B4B4B4'>"
            "<th style='padding:4px 8px'>metric</th>"
            "<th style='padding:4px 8px;text-align:right'>value</th>"
            "<th style='padding:4px 8px'>verdict</th></tr>"
            + "".join(rows)
            + "</table>"
            + notes_html
            + "</div>"
        )

    def _row_html(self, key: str, val: float) -> str:
        direction, thresh = _METRIC_INFO[key]
        if math.isnan(val):
            good, label, color = False, "n/a", "#7B7B7B"
        else:
            if direction == "lower":
                good = val <= thresh
            elif direction == "one":
                good = abs(val - 1.0) <= (thresh - 1.0)
            else:
                good = True
            label = "smooth" if good else "check"
            color = "#2E6E4E" if good else "#B9605B"
        return (
            "<tr style='border-bottom:1px solid #E2E0D8'>"
            f"<td style='padding:4px 8px;font-family:ui-monospace,Menlo,monospace'>{key}</td>"
            f"<td style='padding:4px 8px;text-align:right;font-family:ui-monospace,Menlo,monospace'>{val:.4g}</td>"
            f"<td style='padding:4px 8px;color:{color};font-weight:600'>{label}</td></tr>"
        )
