#!/usr/bin/env bash
# Idempotent setup for the slime_rl conda env.
# Pins are load-bearing — see memory/project_slime_rl.md.
set -euo pipefail

ENV_NAME="${ENV_NAME:-slime_rl}"
PY_VERSION="${PY_VERSION:-3.10}"

source /home/shuhang/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env '$ENV_NAME' exists — activating."
else
    echo "[setup] creating env '$ENV_NAME' (python=$PY_VERSION)..."
    conda create -n "$ENV_NAME" "python=$PY_VERSION" pip -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip wheel

# gym 0.21 / 0.19 ship sdist only with a malformed `extras_require`
# (`opencv-python>=3.` — invalid version spec). Modern `wheel`/`packaging`
# reject it during metadata generation regardless of build isolation.
# conda-forge has a prebuilt gym=0.21.0 binary that side-steps the whole sdist
# build chain. Use it. Pyglet too (slimevolleygym's pinned version).
conda install -c conda-forge "gym=0.21.0" -y

# Patch gym 0.21's broken METADATA — ships with `opencv-python (>=3.)` which
# is an invalid version spec; pip>=24 refuses to install ANYTHING else into the
# env while this is present. Replace `>=3.` with `>=3.0`. Surgical sed.
GYM_META="$(python -c 'import importlib.metadata as m; print(m.distribution("gym")._path)')/METADATA"
if [ -f "$GYM_META" ]; then
    sed -i 's|opencv-python (>=3\.)|opencv-python (>=3.0)|g' "$GYM_META"
    echo "[setup] patched gym METADATA at $GYM_META"
else
    echo "[setup] WARNING: gym METADATA not found at $GYM_META"
fi

# pyglet 1.5.x is a pure-python wheel on PyPI (py3-none-any), works on 3.10.
# conda-forge only built it for older python — use pip instead.
python -m pip install \
    "numpy==1.26.4" \
    "pyglet==1.5.11" \
    "opencv-python==4.10.0.84"

python -m pip install "slimevolleygym==0.1.0"

# CPU torch is enough for our 12-dim MLP DQN. Swap to cu118 build later if needed.
python -m pip install \
    "torch==2.5.1" "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
    "onnx==1.17.0" \
    "onnxruntime==1.20.1" \
    "tensorboard==2.18.0" \
    "tqdm" \
    "matplotlib"

echo "---"
echo "[setup] verifying imports..."
python - <<'PY'
import importlib, sys
mods = ["numpy", "gym", "pyglet", "cv2", "slimevolleygym", "torch", "onnx", "onnxruntime"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"  {m:18s} {getattr(mod, '__version__', '?')}")
import gym, slimevolleygym  # noqa
env = gym.make("SlimeVolleyNoFrameskip-v0")
obs = env.reset()
print(f"  env reset obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
print(f"  action_space: {env.action_space}")
print(f"  observation_space: {env.observation_space}")
env.close()
print("[setup] OK")
PY
