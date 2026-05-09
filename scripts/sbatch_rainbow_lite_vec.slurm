#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=dqn_rb_lite_vec
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
# Rainbow-Lite variant: drops C51 from rainbow_eps_vec while keeping PER,
# n-step, Double DQN, Dueling, soft target. Isolation step in the Rainbow
# component ablation -- if this learns, the C51 head (or its interaction
# with PER) was the breakage; if it stalls, PER itself is the culprit.
python -u train/train_dqn_rainbow_lite_vec.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --save-path checkpoints/dqn_rainbow_lite_vec.pt \
    --log-dir logs/dqn_rainbow_lite_vec

echo "[$(date)] training done."
