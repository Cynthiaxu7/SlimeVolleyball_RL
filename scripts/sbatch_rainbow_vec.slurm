#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=dqn_rainbow_vec
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
# Rainbow is heavier per grad step than v1 (C51 + PER + Noisy + n-step + Double + Dueling)
# so we allow 6h headroom even though env throughput typically dominates.
# sigma-init 0.1 (5x lower than default 0.5). On the small 12-d MLP, σ_init=0.5
# made σ ≈ μ in magnitude after training -> noise drowned the advantage signal,
# argmax bounced randomly, train_R got stuck at -4.87 throughout. Verified post-mortem
# on the previous 5M run: deterministic Q spread was 0.003-0.07 while σ-driven
# perturbation was 0.04-0.07 — same order of magnitude. 0.1 keeps σ ~5x smaller.
python -u train/train_dqn_rainbow_vec.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --sigma-init 0.1 \
    --save-path checkpoints/dqn_rainbow_vec.pt \
    --log-dir logs/dqn_rainbow_vec

echo "[$(date)] training done."
