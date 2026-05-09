#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=dqn_rb_eps_vec
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python - <<'PY'
import torch, sys
print(f"  python={sys.version.split()[0]}  torch={torch.__version__}")
print(f"  cuda_available={torch.cuda.is_available()}  cuda_runtime={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  device_name={torch.cuda.get_device_name(0)}")
PY

# 5M transitions across 16 envs = 312_500 outer iterations.
# Train every 4 outer iters at batch 256 -> ~78k grad steps.
# Eps-greedy variant: NoisyNet was empirically broken on this small 12-d MLP
# (sigma grew to ~mu in the advantage head, drowning the action-distinguishing
# signal -- argmax bounced randomly, train_R stuck at -4.87 throughout 5M).
# Without NoisyNet's per-step jitter, training is meaningfully cheaper per
# grad step, so we drop the headroom from 6h to 4h.
python -u train/train_dqn_rainbow_eps_vec.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --save-path checkpoints/dqn_rainbow_eps_vec.pt \
    --log-dir logs/dqn_rainbow_eps_vec

echo "[$(date)] training done."
