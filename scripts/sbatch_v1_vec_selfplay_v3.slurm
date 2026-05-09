#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=dqn_v1_vec_sp3
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Self-play v3: same baseline-mixing as v2 BUT with v1's normal LR (2.5e-4).
# Goal: isolate whether the slow learning in v2 was caused by lr=5e-5 or by
# the baseline-mixing itself. Same arch, same save/log path naming as v2 but
# with _v3 suffix to avoid clobbering.
python -u train/train_dqn_v1_vec_selfplay.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --lr 2.5e-4 \
    --baseline-opponent-prob 0.4 \
    --resume-from checkpoints/dqn_v1_vec.pt \
    --pool-dir checkpoints/pool_v1_selfplay_v3/ \
    --save-path checkpoints/dqn_v1_vec_selfplay_v3.pt \
    --log-dir logs/dqn_v1_vec_selfplay_v3

echo "[$(date)] training done."
