#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=dqn_v1_vec_sp
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

# 5M transitions across 16 envs = 312_500 outer iterations. Self-play adds
# ~one tiny per-env opponent forward per outer iter (negligible vs env step).
# Snapshot every 100k transitions -> ~50 snapshots over the run, FIFO-trimmed
# to the last 30. We resume from the v1_vec checkpoint so episodes have a
# meaningful initial opponent and learning starts from a competent policy.
python -u train/train_dqn_v1_vec_selfplay.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --device cuda \
    --seed 0 \
    --resume-from checkpoints/dqn_v1_vec.pt \
    --save-path checkpoints/dqn_v1_vec_selfplay.pt \
    --log-dir logs/dqn_v1_vec_selfplay \
    --pool-dir checkpoints/pool_v1_selfplay/

echo "[$(date)] training done."
