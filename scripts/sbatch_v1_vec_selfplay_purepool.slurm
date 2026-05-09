#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=v1_sp_purepool
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# V1 self-play "pure pool" experiment: NO baseline as opponent during training
# (baseline_opp_prob=0). The OpponentPool is seeded with the ladder-king
# v1_selfplay (1765 ELO, the buggy-baseline-trained v1_vec_selfplay.pt) so
# the agent sees a strong & well-rounded opponent from step 0 — instead of
# being thrown at the real baseline, which the prior fixed run had to grind
# against.
#
# Hypothesis: the prior v1_sp_fixed run lost -111 ELO vs the buggy v1_selfplay
# in ladder because (a) it overfit to baseline-specific play and (b) its pool
# only had clean-baseline ckpts (less varied). Seeding with v1_selfplay
# imports the buggy-lineage's playstyle into the pool while keeping the
# trainer itself bug-free (the BaselineOpponentAdapter ×10 patch is in
# place — it just never fires here because baseline_opp_prob=0).
#
# Eval vs baseline still uses the env's auto-baseline (single-arg env.step
# in evaluate_vs_baseline) so we still get an honest base_R measurement.
python -u train/train_dqn_v1_vec_selfplay.py \
    --total-timesteps 20000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --lr 5e-5 \
    --baseline-opponent-prob 0.0 \
    --resume-from checkpoints/dqn_v1_vec.pt \
    --pool-dir checkpoints/pool_v1_sp_purepool/ \
    --save-path checkpoints/dqn_v1_vec_selfplay_purepool.pt \
    --log-dir logs/dqn_v1_vec_selfplay_purepool \
    --seed-pool-from \
        checkpoints/dqn_v1_vec.pt \
        checkpoints/dqn_v1_vec_selfplay.pt

echo "[$(date)] done."
