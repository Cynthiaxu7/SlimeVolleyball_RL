#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=plain_dqn_5m
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Ablation cell A: plain DQN (no Dueling, no Double, no N-step). The floor
# baseline that lets us claim "Dueling helps" by comparing with v0
# (dqn_vec.pt = same trainer, default = Dueling on). Same hyperparams (lr,
# target-update-every, eps schedule) and same total steps so the only
# variable is the network architecture.
python -u train/train_dqn_vec.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --target-update-every 500 \
    --device cuda \
    --seed 0 \
    --lr 2.5e-4 \
    --no-dueling \
    --save-path checkpoints/dqn_plain_vec.pt \
    --log-dir logs/dqn_plain_vec

echo "[$(date)] done."
