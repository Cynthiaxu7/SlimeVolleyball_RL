"""Single-file Dueling DQN trainer for SlimeVolleyNoFrameskip-v0 (old gym 0.19).

Trains against the built-in baseline opponent (single-agent mode). Designed for
later ONNX export: the model's forward returns Q-values of shape (B, 6).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import gym  # noqa: F401  -- old gym 0.19, registers SlimeVolley envs upon slimevolleygym import
import numpy as np
import slimevolleygym  # noqa: F401  -- registers "SlimeVolleyNoFrameskip-v0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DuelingMLP(nn.Module):
    """Dueling MLP: shared trunk -> separate value & advantage heads.

    Shapes: trunk (12->128->128), value head (128->64->1), advantage head
    (128->64->6). Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)). The mean-subtraction
    formulation is the standard Dueling form (Wang et al. 2016); max-subtraction
    is an alternative from the same paper but is less commonly used in practice.
    """

    def __init__(self, obs_dim: int = 12, n_actions: int = 6) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        value = self.value_head(features)                     # (B, 1)
        advantage = self.advantage_head(features)             # (B, n_actions)
        adv_mean = advantage.mean(dim=-1, keepdim=True)       # (B, 1)
        return value + (advantage - adv_mean)                 # (B, n_actions)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Plain ring buffer with uniform sampling. Stores float32 obs."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        ids = np.random.randint(0, self.size, size=batch_size)
        return Batch(
            obs=torch.from_numpy(self.obs[ids]).to(device),
            actions=torch.from_numpy(self.actions[ids]).to(device),
            rewards=torch.from_numpy(self.rewards[ids]).to(device),
            next_obs=torch.from_numpy(self.next_obs[ids]).to(device),
            dones=torch.from_numpy(self.dones[ids]).to(device),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# slimevolleygym registers no state+discrete env directly: SlimeVolley-v0 gives
# 12-d state with MultiBinary(3); SlimeVolleyNoFrameskip-v0 gives pixels with
# Discrete(6). We want state+discrete, so wrap SlimeVolley-v0 and translate
# Discrete(6) -> MultiBinary(3) via the same action_table the env uses internally.
ACTION_TABLE = [
    [0, 0, 0],  # NOOP
    [1, 0, 0],  # LEFT
    [1, 0, 1],  # UPLEFT
    [0, 0, 1],  # UP
    [0, 1, 1],  # UPRIGHT
    [0, 1, 0],  # RIGHT
]


class DiscreteSlimeWrapper(gym.ActionWrapper):
    """Expose Discrete(6) on top of SlimeVolley-v0's MultiBinary(3) actions."""

    def __init__(self, env: "gym.Env") -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))

    def action(self, action: int) -> list[int]:
        return ACTION_TABLE[int(action)]


def make_env(seed: int) -> "gym.Env":
    env = gym.make("SlimeVolley-v0")
    env = DiscreteSlimeWrapper(env)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def linear_epsilon(step: int, total_steps: int, start: float, end: float, fraction: float) -> float:
    decay_steps = max(1, int(total_steps * fraction))
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def evaluate(
    model: DuelingMLP,
    seed: int,
    episodes: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Greedy eval on a fresh env. Returns (mean_return, std_return, mean_length)."""
    env = make_env(seed + 10_000)  # offset seed so eval env != train env
    returns: list[float] = []
    lengths: list[int] = []
    model.eval()
    with torch.no_grad():
        for ep in range(episodes):
            obs = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            ep_return = 0.0
            ep_length = 0
            done = False
            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                q = model(obs_t)
                action = int(q.argmax(dim=-1).item())
                obs, reward, done, _info = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                ep_return += float(reward)
                ep_length += 1
            returns.append(ep_return)
            lengths.append(ep_length)
    env.close()
    model.train()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", help='"cpu", "cuda", or "auto"')
    p.add_argument(
        "--save-path",
        type=str,
        default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_smoke.pt",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default="/home/shuhang/YBJ/RL_test/logs/dqn_smoke",
    )
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--eval-episodes", type=int, default=5)
    return p.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def save_checkpoint(
    path: str,
    model: DuelingMLP,
    total_timesteps_trained: int,
    final_eval_reward: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": 12,
            "n_actions": 6,
            "arch": "dueling_mlp_128_128_64",
            "total_timesteps_trained": int(total_timesteps_trained),
            "final_eval_reward": float(final_eval_reward),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- seeding & determinism ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = resolve_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # --- env ---
    env = make_env(args.seed)
    obs_dim = int(np.array(env.observation_space.shape).prod())  # 12
    n_actions = int(env.action_space.n)                          # 6
    assert obs_dim == 12 and n_actions == 6, (
        f"unexpected env shapes: obs_dim={obs_dim}, n_actions={n_actions}"
    )

    # --- nets, optim, buffer ---
    online = DuelingMLP(obs_dim, n_actions).to(device)
    target = DuelingMLP(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=2.5e-4)
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim)

    # --- hyperparams ---
    gamma = 0.99
    batch_size = 64
    warmup_steps = 1_000
    train_freq = 4
    target_update_every_grad_steps = 500
    grad_clip_norm = 10.0
    eps_start, eps_end, eps_fraction = 1.0, 0.02, 0.30

    # --- run-state ---
    grad_steps = 0
    best_eval_reward = -float("inf")
    recent_losses: deque[float] = deque(maxlen=100)
    recent_qmeans: deque[float] = deque(maxlen=100)
    episode_return = 0.0
    episode_length = 0

    obs = np.asarray(env.reset(), dtype=np.float32)
    start_time = time.time()

    for global_step in range(1, args.total_timesteps + 1):
        # --- epsilon-greedy action selection ---
        epsilon = linear_epsilon(global_step, args.total_timesteps, eps_start, eps_end, eps_fraction)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = int(online(obs_t).argmax(dim=-1).item())

        next_obs, reward, done, _info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        reward_f = float(reward)

        buffer.add(obs, int(action), reward_f, next_obs, bool(done))
        episode_return += reward_f
        episode_length += 1

        if done:
            writer.add_scalar("rollout/episode_return", episode_return, global_step)
            writer.add_scalar("rollout/episode_length", episode_length, global_step)
            obs = np.asarray(env.reset(), dtype=np.float32)
            episode_return = 0.0
            episode_length = 0
        else:
            obs = next_obs

        # --- learn ---
        should_train = (
            buffer.size >= max(warmup_steps, batch_size)
            and global_step % train_freq == 0
        )
        if should_train:
            batch = buffer.sample(batch_size, device)
            with torch.no_grad():
                next_q = target(batch.next_obs).max(dim=-1).values
                td_target = batch.rewards + gamma * next_q * (1.0 - batch.dones)
            q_all = online(batch.obs)
            q_pred = q_all.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_pred, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            grad_steps += 1
            recent_losses.append(float(loss.item()))
            recent_qmeans.append(float(q_pred.mean().item()))

            if grad_steps % target_update_every_grad_steps == 0:
                target.load_state_dict(online.state_dict())

            if grad_steps % 100 == 0:
                writer.add_scalar("train/loss", float(np.mean(recent_losses)), global_step)
                writer.add_scalar("train/q_mean", float(np.mean(recent_qmeans)), global_step)
                writer.add_scalar("train/epsilon", epsilon, global_step)

        # --- periodic evaluation ---
        if global_step % args.eval_every == 0:
            eval_mean, eval_std, eval_len = evaluate(
                online, args.seed, args.eval_episodes, device
            )
            writer.add_scalar("eval/reward_mean", eval_mean, global_step)
            writer.add_scalar("eval/reward_std", eval_std, global_step)
            writer.add_scalar("eval/episode_length_mean", eval_len, global_step)
            elapsed = time.time() - start_time
            sps = global_step / max(elapsed, 1e-6)
            print(
                f"[step {global_step:>7d}] eps={epsilon:.3f} "
                f"eval_R={eval_mean:+.3f}+-{eval_std:.3f} "
                f"eval_len={eval_len:.1f} sps={sps:.1f}",
                flush=True,
            )
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                save_checkpoint(args.save_path, online, global_step, eval_mean)

    # --- final eval + checkpoint ---
    final_mean, final_std, final_len = evaluate(
        online, args.seed, args.eval_episodes, device
    )
    writer.add_scalar("eval/reward_mean", final_mean, args.total_timesteps)
    writer.add_scalar("eval/reward_std", final_std, args.total_timesteps)
    writer.add_scalar("eval/episode_length_mean", final_len, args.total_timesteps)
    print(
        f"[final {args.total_timesteps}] "
        f"eval_R={final_mean:+.3f}+-{final_std:.3f} eval_len={final_len:.1f}",
        flush=True,
    )
    save_checkpoint(args.save_path, online, args.total_timesteps, final_mean)
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
