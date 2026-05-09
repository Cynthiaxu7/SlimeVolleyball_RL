#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=v1_sp_fixed
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Re-train V1 selfplay 20M after the obs ×10 bug fix in
# BaselineOpponentAdapter (now feeds normalized obs directly, matching env's
# auto-baseline). Same hyperparams as 17332 v1_sp_seeded but seed pool
# limited to ckpts trained against the CORRECT baseline (dqn_vec, dqn_v1_vec
# — both single-arg-env trained, no bug). Distinct save/log/pool dirs.
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
    --pool-dir checkpoints/pool_v1_selfplay_fixed/ \
    --save-path checkpoints/dqn_v1_vec_selfplay_fixed.pt \
    --log-dir logs/dqn_v1_vec_selfplay_fixed \
    --seed-pool-from \
        checkpoints/dqn_vec.pt \
        checkpoints/dqn_v1_vec.pt

echo "[$(date)] done."
