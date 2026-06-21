"""mlip_smoothness_eval: model-agnostic smoothness benchmark for torch-sim MLIPs.

Quickstart
----------
>>> from mlip_smoothness_eval import evaluate_smoothness
>>> report = evaluate_smoothness(model)   # model: torch_sim ModelInterface
>>> report                                 # scorecard renders inline
>>> report.curve("diatomic", symbol="O")   # plotly PEC
>>> report.pca_surface()                    # 3D PES over PC1/PC2
"""

from mlip_smoothness_eval import structures
from mlip_smoothness_eval.checks import (
    CheckResult,
    cutoff_smoothness,
    diatomic_smoothness,
    displacement_scan,
    force_jacobian_asymmetry,
    nonconservativity,
    nve_energy_drift,
    predict,
)
from mlip_smoothness_eval.report import SmoothnessReport
from mlip_smoothness_eval.runner import evaluate_smoothness

__version__ = "0.1.0"

__all__ = [
    "evaluate_smoothness",
    "SmoothnessReport",
    "CheckResult",
    "structures",
    "predict",
    "nonconservativity",
    "displacement_scan",
    "cutoff_smoothness",
    "force_jacobian_asymmetry",
    "diatomic_smoothness",
    "nve_energy_drift",
]
