#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=rb_lite_v2
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

# Rainbow-Lite v2: rerun with v1_vec's hyperparams to control for the LR /
# eps-fraction / warmup mismatch flagged by the PER reviewer. If it learns now,
# the previous run was just slow, not broken; PER is fine.
python -u train/train_dqn_rainbow_lite_vec.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --lr 2.5e-4 \
    --epsilon-fraction 0.30 \
    --save-path checkpoints/dqn_rainbow_lite_v2.pt \
    --log-dir logs/dqn_rainbow_lite_v2

echo "[$(date)] done."
