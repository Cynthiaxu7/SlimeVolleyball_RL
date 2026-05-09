#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=ppo_lowent_20m
#SBATCH --output=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.out
#SBATCH --error=/home/shuhang/YBJ/RL_test/logs/sbatch_%j.err

set -euo pipefail

source /home/shuhang/miniconda3/etc/profile.d/conda.sh
conda activate slime_rl
cd /home/shuhang/YBJ/RL_test

# PPO long-run with lower entropy coefficient.
# v1 (5M, ent=0.01) finished at base_R=-1.0 with H=1.30 (still 73% uniform).
# v2 (20M, ent=0.001) lets policy converge harder so greedy argmax is reliable.
# Goal: close the train/eval greedy gap by driving entropy below ~0.5.
python -u train/train_ppo_selfplay_vec.py \
    --total-timesteps 20000000 \
    --num-envs 16 \
    --rollout-len 128 \
    --update-epochs 4 \
    --minibatch-size 256 \
    --clip-eps 0.2 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --vf-coef 0.5 \
    --ent-coef 0.001 \
    --lr 3e-4 \
    --max-grad-norm 0.5 \
    --baseline-opponent-prob 0.7 \
    --snapshot-every 100000 \
    --pool-max-size 30 \
    --eval-every 25000 \
    --eval-episodes 5 \
    --device cuda \
    --seed 0 \
    --pool-dir checkpoints/pool_ppo_lowent_20m/ \
    --save-path checkpoints/ppo_lowent_20m.pt \
    --log-dir logs/ppo_lowent_20m

echo "[$(date)] done."
