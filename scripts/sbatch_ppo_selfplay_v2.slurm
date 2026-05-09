#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=ppo_sp_v2
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

# PPO self-play v2: stronger baseline anchoring (0.7 vs 0.4 in v1) since the
# first run drifted catastrophically against baseline (final base_R = -4.8
# despite pool_R = +3.4). Higher mix forces 70% of episodes to feature the
# scripted baseline, keeping that distribution dominant in on-policy rollouts.
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
    --baseline-opponent-prob 0.7 \
    --snapshot-every 100000 \
    --pool-max-size 30 \
    --eval-every 25000 \
    --eval-episodes 5 \
    --device cuda \
    --seed 0 \
    --pool-dir checkpoints/pool_ppo_selfplay_v2/ \
    --save-path checkpoints/ppo_selfplay_v2.pt \
    --log-dir logs/ppo_selfplay_v2

echo "[$(date)] done."
