#!/usr/bin/env bash
# Cross-model smoothness comparison orchestrator.
#
# Each model family pins an incompatible dependency stack (see the `conflicts`
# block in pyproject.toml), so they can't share one environment. We run the
# worker once per model under that model's uv extra — `uv run --extra <key>`
# re-syncs to that family's resolution from the lockfile — then aggregate the
# JSON in the base env (plotly only).
#
#   examples/run_comparison.sh                 # all models below
#   examples/run_comparison.sh mace orb        # just these
#
# A model that fails to load (missing checkpoint, unsupported loader) is logged
# and skipped so the rest still produce a comparison.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(mace orb mattersim sevenn fairchem nequix nequip metatomic)
fi

OUTDIR="examples/comparison_results"
mkdir -p "$OUTDIR"

for model in "${MODELS[@]}"; do
  echo "=== $model ==="
  # `lj` is the analytic reference baseline and lives in the base env (no extra).
  if [ "$model" = "lj" ]; then
    extra_args=()
  else
    extra_args=(--extra "$model")
  fi
  if uv run "${extra_args[@]}" python examples/compare_worker.py "$model" --outdir "$OUTDIR"; then
    echo "=== $model done ==="
  else
    echo "!!! $model failed; skipping" >&2
  fi
done

echo "=== aggregating ==="
uv run python examples/compare_models.py --outdir "$OUTDIR"
