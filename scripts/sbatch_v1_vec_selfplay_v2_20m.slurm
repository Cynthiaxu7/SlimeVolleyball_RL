#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=v2_20m
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# v2 long: baseline_prob=0.4, lr=5e-5, 20M transitions (~3.5h at 1670 sps).
# Distinct save/log/pool dirs from the 5M v2 run so they don't clobber.
python -u train/train_dqn_v1_vec_selfplay.py \
    --total-timesteps 20000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --lr 5e-5 \
    --baseline-opponent-prob 0.4 \
    --resume-from checkpoints/dqn_v1_vec.pt \
    --pool-dir checkpoints/pool_v2_20m/ \
    --save-path checkpoints/dqn_v1_vec_selfplay_v2_20m.pt \
    --log-dir logs/dqn_v1_vec_selfplay_v2_20m

echo "[$(date)] done."
