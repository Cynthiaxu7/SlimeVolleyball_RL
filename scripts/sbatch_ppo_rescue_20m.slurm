#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=ppo_rescue
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# PPO rescue run after diagnosing why ppo_lowent_20m_fixed (final base_R=-3.4)
# failed: entropy collapsed by step 1.4M (H=0.108) so the policy committed
# early to a strategy that beats weak self-snapshots (pool_R=+1) but never
# learns to handle the strong baseline (base_R=-3.4 throughout).
#
# Changes vs ppo_lowent_20m_fixed:
#   ent_coef     0.001 -> 0.01     (10x: keep entropy alive longer)
#   rollout_len  128   -> 256      (better advantage estimate; episode is 3000 frames)
#   --lr-anneal               NEW: linear decay to 0 (standard PPO recipe)
#   --target-kl 0.02          NEW: early-stop update epochs to prevent over-update
#   baseline_opp_prob 0.7 -> 0.7   (UNCHANGED — pool only has self-snapshots,
#                                    so HIGH baseline ratio is correct; the
#                                    bug was entropy, not opponent mix)
#   baseline ×10 bug          ALREADY FIXED in the trainer file
python -u train/train_ppo_selfplay_vec.py \
    --total-timesteps 20000000 \
    --num-envs 16 \
    --rollout-len 256 \
    --update-epochs 4 \
    --minibatch-size 256 \
    --clip-eps 0.2 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --vf-coef 0.5 \
    --ent-coef 0.01 \
    --lr 3e-4 \
    --lr-anneal \
    --target-kl 0.02 \
    --max-grad-norm 0.5 \
    --baseline-opponent-prob 0.7 \
    --snapshot-every 100000 \
    --pool-max-size 30 \
    --eval-every 25000 \
    --eval-episodes 5 \
    --device cuda \
    --seed 0 \
    --pool-dir checkpoints/pool_ppo_rescue_20m/ \
    --save-path checkpoints/ppo_rescue_20m.pt \
    --log-dir logs/ppo_rescue_20m

echo "[$(date)] done."
