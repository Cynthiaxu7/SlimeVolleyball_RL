#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=rb_lite_sp_fixed
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

# Re-train Rainbow Lite selfplay 20M after the obs ×10 bug fix in
# BaselineOpponentAdapter. Resume from rainbow_lite_v2 (5M, trained against
# CORRECT baseline via single-arg env). Seed pool with only clean-baseline
# ckpts to keep the bug-fix experiment uncontaminated.
python -u train/train_dqn_rainbow_lite_vec_selfplay.py \
    --total-timesteps 20000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --lr 2.5e-4 \
    --epsilon-fraction 0.30 \
    --baseline-opponent-prob 0.4 \
    --resume-from checkpoints/dqn_rainbow_lite_v2.pt \
    --pool-dir checkpoints/pool_rainbow_lite_sp_20m_fixed/ \
    --save-path checkpoints/dqn_rainbow_lite_sp_20m_fixed.pt \
    --log-dir logs/dqn_rainbow_lite_sp_20m_fixed \
    --seed-pool-from \
        checkpoints/dqn_vec.pt \
        checkpoints/dqn_v1_vec.pt \
        checkpoints/dqn_rainbow_lite_v2.pt

echo "[$(date)] done."
