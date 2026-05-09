#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=v1_sp_seeded
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# v1 self-play, seeded variant: same hyperparams as v2_20m (lr=5e-5,
# baseline_prob=0.4, 20M transitions) but the OpponentPool starts with a
# diverse cohort of historical checkpoints (v0, v1 cold, v2/v3 self-play
# 20M). Distinct save/log/pool dirs (suffix _seeded) so this run doesn't
# collide with v2_20m / v3_20m.
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
    --pool-dir checkpoints/pool_v1_selfplay_seeded/ \
    --save-path checkpoints/dqn_v1_vec_selfplay_seeded.pt \
    --log-dir logs/dqn_v1_vec_selfplay_seeded \
    --seed-pool-from \
        checkpoints/dqn_vec.pt \
        checkpoints/dqn_v1_vec.pt \
        checkpoints/dqn_v1_vec_selfplay_v2_20m.pt \
        checkpoints/dqn_v1_vec_selfplay_v3_20m.pt

echo "[$(date)] done."
