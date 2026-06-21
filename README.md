# mlip_smoothness_eval

Model-agnostic smoothness & energy-conservation benchmark for any
[torch-sim](https://github.com/Radical-AI/torch-sim) MLIP (machine-learned
interatomic potential).

Static energy/force MAE can look great while a learned potential is jagged or
non-conservative — which is what actually breaks MD, relaxation, and phonons.
This package scores the potential *as a function* via six diagnostics, runs them
through a single call, and renders the results inline in a notebook.

```python
from mlip_smoothness_eval import evaluate_smoothness

report = evaluate_smoothness(model)   # model: torch_sim ModelInterface
report                                 # metric table renders inline (_repr_html_)

report.curve("diatomic", symbol="O")   # plotly potential-energy curve
report.gif("displacement_scan")        # structure morph + advancing curve
report.pca_surface(structure)          # 3D PES over PC1/PC2, z = energy
```

See `examples/smoothness_demo.py` for an end-to-end demo on MACE-MP-0.
# mlip_smoothness_evals
