#!/usr/bin/env bash
# Build standalone Linux binaries for the slime volley demos using PyInstaller.
#
# Output:
#   dist/slime_ladder  (runs scripts/ladder_cli.py -> scripts/ladder_sim.py)
#   dist/slime_play    (runs scripts/play_local.py — pygame interactive)
#
# Run:
#   bash scripts/build_executables.sh
#
# Caveats:
#   * Linux-only — produces an ELF binary tied to the build host's glibc.
#     A binary built on an Ubuntu 22.04 box will only work on hosts with
#     glibc >= the build host's version. For maximum portability, build
#     inside a manylinux container.
#   * pygame and pyinstaller are NOT in the slime_rl env by default — this
#     script installs them on first run.
#   * The --add-data separator is ":" on Linux/macOS and ";" on Windows.
#     Hardcoded to ":" here because the slurm cluster + dev box are Linux.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="slime_rl"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

# Source conda so `conda activate` works in this non-interactive shell.
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Make sure build-time deps are present. We don't pin versions because the
# slime_rl env already pins gym/numpy and PyInstaller follows whatever it
# finds; pygame is pure-Python wheel so version doesn't matter.
python -m pip install --quiet --upgrade pip
python -m pip install --quiet pyinstaller pygame

cd "$REPO"

# Common args shared by both binaries. We deliberately collect data and
# binaries for slimevolleygym + onnxruntime since:
#   * slimevolleygym registers SlimeVolley-v0 at import time but PyInstaller's
#     static analysis can miss the side-effect imports (gym.envs.registration).
#   * onnxruntime ships .so/.dll native libs in onnxruntime/capi/ that
#     PyInstaller won't auto-bundle.
COMMON_ARGS=(
    --noconfirm
    --clean
    --onefile
    --add-data "$REPO/web/model.onnx:models"
    --add-data "$REPO/web/model_v1.onnx:models"
    --add-data "$REPO/web/model_v1_selfplay.onnx:models"
    --add-data "$REPO/web/model_rainbow.onnx:models"
    --add-data "$REPO/web/model_ppo.onnx:models"
    --hidden-import slimevolleygym
    --hidden-import slimevolleygym.slimevolley
    --hidden-import gym
    --hidden-import gym.envs.registration
    --hidden-import onnxruntime
    --collect-data slimevolleygym
    --collect-submodules slimevolleygym
    --collect-binaries onnxruntime
    --collect-data onnxruntime
)

# --- slime_ladder ----------------------------------------------------------
# Bundle ladder_sim.py as a data file so ladder_cli.py can import it at
# runtime via sys._MEIPASS (see ladder_cli._bundle_paths).
pyinstaller \
    "${COMMON_ARGS[@]}" \
    --add-data "$REPO/scripts/ladder_sim.py:scripts" \
    --name slime_ladder \
    "$REPO/scripts/ladder_cli.py"

# --- slime_play ------------------------------------------------------------
# pygame needs SDL2 native libs; --collect-binaries pygame grabs them all.
pyinstaller \
    "${COMMON_ARGS[@]}" \
    --collect-binaries pygame \
    --collect-data pygame \
    --hidden-import pygame \
    --name slime_play \
    "$REPO/scripts/play_local.py"

echo
echo "Built binaries:"
ls -lh "$REPO/dist/slime_ladder" "$REPO/dist/slime_play"
echo
echo "Try it:"
echo "  $REPO/dist/slime_ladder --n-matches 200"
echo "  $REPO/dist/slime_play --left baseline --right rainbow --auto"
