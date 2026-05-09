#!/bin/bash
# Convenience launcher for scripts/record_replays.py.
# Activates the slime_rl conda env (which has gym==0.21, slimevolleygym,
# onnxruntime), then invokes the recorder with the default fleet of pairs.
# Output JSONs land in web/replays/ and are picked up by the Replay-mode
# dropdown in the web UI.
set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl

cd /home/shuhang/YBJ/RL_test

# Default fleet: covers every trained variant vs Baseline (3 seeds for the
# headline V1-selfplay; 2 seeds for the rest), plus two control replays
# (baseline-vs-baseline and v1_selfplay self-vs-self) to verify that draws
# and noisy outcomes also serialize cleanly.
exec python scripts/record_replays.py \
    --pairs \
        baseline,v1_selfplay \
        baseline,rainbow \
        baseline,ppo \
        baseline,dueling \
        baseline,v1 \
        baseline,baseline \
        v1_selfplay,v1_selfplay \
    --n-each 2 \
    --output-dir web/replays/ \
    "$@"
