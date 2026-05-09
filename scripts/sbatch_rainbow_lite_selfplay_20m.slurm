#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=rb_lite_sp_20m
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

# Rainbow Lite + self-play, 20M timesteps, seeded with all our trained models
# (v0, v1 cold, v1 selfplay v2_20m, v3_20m, AND the 5M rainbow_lite_selfplay).
# Goal: see how far Rainbow Lite + diverse seeded pool can push past v1 selfplay's +0.2.
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
    --resume-from checkpoints/dqn_rainbow_lite_selfplay.pt \
    --pool-dir checkpoints/pool_rainbow_lite_sp_20m/ \
    --save-path checkpoints/dqn_rainbow_lite_sp_20m.pt \
    --log-dir logs/dqn_rainbow_lite_sp_20m \
    --seed-pool-from \
        checkpoints/dqn_vec.pt \
        checkpoints/dqn_v1_vec.pt \
        checkpoints/dqn_v1_vec_selfplay_v2_20m.pt \
        checkpoints/dqn_v1_vec_selfplay_v3_20m.pt \
        checkpoints/dqn_rainbow_lite_v2.pt

echo "[$(date)] done."
