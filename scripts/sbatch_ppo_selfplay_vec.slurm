#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=ppo_sp
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - <<'PY'
import torch
print(f"  torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}")
PY

# PPO self-play. PPO is on-policy so no replay buffer / no warm-start required;
# we train from scratch by default. Same self-play scaffolding as DQN-selfplay-v2:
#   --baseline-opponent-prob 0.4 : 40% of episodes use the scripted baseline,
#       60% use a pool snapshot. Mixing keeps the baseline distribution in
#       on-policy rollouts and prevents the policy drifting away from it.
#   --num-envs 16 + --rollout-len 128 : 2048 transitions per PPO iteration
#       (standard CleanRL-style sizing for 16 envs).
#   --update-epochs 4 + --minibatch-size 256 : 8 minibatches/epoch * 4 epochs
#       = 32 grad steps per rollout. Standard PPO budget.
#   --lr 3e-4 + --max-grad-norm 0.5 : PPO defaults; resilient to self-play
#       non-stationarity in our experience.
python -u train/train_ppo_selfplay_vec.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --rollout-len 128 \
    --update-epochs 4 \
    --minibatch-size 256 \
    --clip-eps 0.2 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --vf-coef 0.5 \
    --ent-coef 0.01 \
    --lr 3e-4 \
    --max-grad-norm 0.5 \
    --baseline-opponent-prob 0.4 \
    --snapshot-every 100000 \
    --pool-max-size 30 \
    --eval-every 25000 \
    --eval-episodes 5 \
    --device cuda \
    --seed 0 \
    --pool-dir checkpoints/pool_ppo_selfplay/ \
    --save-path checkpoints/ppo_selfplay_vec.pt \
    --log-dir logs/ppo_selfplay_vec

echo "[$(date)] training done."
