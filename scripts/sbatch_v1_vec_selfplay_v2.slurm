#!/bin/bash
#SBATCH --partition=a6k
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=dqn_v1_vec_sp2
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

# Self-play v2: addresses catastrophic forgetting observed in v1 (job 17319).
#   --baseline-opponent-prob 0.4 : 40% of episodes use scripted baseline,
#       60% use a pool snapshot. Keeps baseline distribution in the replay.
#   --lr 5e-5 : 5x lower than initial training (2.5e-4). Continuation should
#       fine-tune, not overwrite, the 5M-trained weights.
#   Different save/log/pool dirs so this run doesn't clobber 17319.
python -u train/train_dqn_v1_vec_selfplay.py \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --batch-size 256 \
    --train-freq 4 \
    --device cuda \
    --seed 0 \
    --lr 5e-5 \
    --baseline-opponent-prob 0.4 \
    --resume-from checkpoints/dqn_v1_vec.pt \
    --pool-dir checkpoints/pool_v1_selfplay_v2/ \
    --save-path checkpoints/dqn_v1_vec_selfplay_v2.pt \
    --log-dir logs/dqn_v1_vec_selfplay_v2

echo "[$(date)] training done."
