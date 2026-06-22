# mlip_smoothness_eval

Model-agnostic smoothness & energy-conservation benchmark for any
[torch-sim](https://github.com/Radical-AI/torch-sim) MLIP (machine-learned
interatomic potential).

Static energy/force MAE can look great while a learned potential is jagged or
non-conservative — which is what actually breaks MD, relaxation, and phonons.
This package scores the potential *as a function* via seven diagnostics, runs them
through a single call, and renders the results inline in a notebook.

```python
from mlip_smoothness_eval import evaluate_smoothness

report = evaluate_smoothness(model)   # model: torch_sim ModelInterface
report                                 # metric table renders inline (_repr_html_)

report.curve("diatomic", symbol="O")   # plotly potential-energy curve
report.gif("displacement_scan")        # structure morph + advancing curve
report.pca_surface(structure)          # 3D PES over PC1/PC2, z = energy
```

See `src/mlip_smoothness_eval/examples/smoothness_demo.py` for an end-to-end demo on MACE-MP-0.

## External cross-check: BSCT

As an independent outside check, the suite can run the
[Bond Smoothness Characterization Test](https://github.com/ryanliu30/bsct)
(BSCT, arXiv:2602.04861) — its Force Smoothness Deviation over the BSCT-SPICE
dataset. Install the extra; the dataset is not shipped with the pip package, so
`download_bsct_dataset=True` fetches it to an in-repo default (`data/bsct_spice`)
on first use, or point `bsct_data_path` at your own copy.

```bash
uv sync --extra bsct
```

```python
report = evaluate_smoothness(model, run_bsct=True, download_bsct_dataset=True)
# or: evaluate_smoothness(model, run_bsct=True, bsct_data_path="bsct/bsct_spice")
# adds bsct_full_smoothness, bsct_compress_*, bsct_stretch_* to the scorecard
```

See `src/mlip_smoothness_eval/examples/bsct_demo.py` for a runnable end-to-end demo.

It is off by default (~40 min on GPU). Citations for the methods this suite
builds on are recorded via [duecredit](https://github.com/duecredit/duecredit);
run anything under `DUECREDIT_ENABLE=yes` and inspect with `duecredit summary`.
# mlip_smoothness_evals
