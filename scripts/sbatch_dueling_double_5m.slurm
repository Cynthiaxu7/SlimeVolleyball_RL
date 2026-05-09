#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=duel_double_5m
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Ablation cell C: Dueling + Double DQN (NO N-step). Uses the v1 trainer
# with --n-step 1 so the only difference vs v1 cold (which has n-step=3)
# is the bootstrap horizon. Combined with v0 (Dueling alone, no Double, no
# N-step), this isolates Double's contribution.
python -u train/train_dqn_v1_vec.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --soft-tau 0.005 \
    --n-step 1 \
    --device cuda \
    --seed 0 \
    --lr 2.5e-4 \
    --save-path checkpoints/dqn_v1_dd_vec.pt \
    --log-dir logs/dqn_v1_dd_vec

echo "[$(date)] done."
