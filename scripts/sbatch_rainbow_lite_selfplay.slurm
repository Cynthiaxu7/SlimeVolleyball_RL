#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=rb_lite_sp
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY'
import torch
print(f"  torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}")
PY

# Rainbow-Lite self-play: PER + Double + n-step + Dueling + soft-tau on top
# of the v1-style self-play scaffolding (SelfPlayVecEnv, OpponentPool,
# baseline-mixing, two-eval). Resumes from the proven Rainbow-Lite v2
# checkpoint and seeds the pool with v0/v1/v1-selfplay so the agent fights
# a diverse cohort from the very first iteration.
python -u train/train_dqn_rainbow_lite_vec_selfplay.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --lr 2.5e-4 \
    --epsilon-fraction 0.30 \
    --baseline-opponent-prob 0.4 \
    --resume-from checkpoints/dqn_rainbow_lite_v2.pt \
    --pool-dir checkpoints/pool_rainbow_lite_selfplay/ \
    --save-path checkpoints/dqn_rainbow_lite_selfplay.pt \
    --log-dir logs/dqn_rainbow_lite_selfplay \
    --seed-pool-from \
        checkpoints/dqn_vec.pt \
        checkpoints/dqn_v1_vec.pt \
        checkpoints/dqn_v1_vec_selfplay_v2_20m.pt \
        checkpoints/dqn_v1_vec_selfplay_v3_20m.pt

echo "[$(date)] training done."
