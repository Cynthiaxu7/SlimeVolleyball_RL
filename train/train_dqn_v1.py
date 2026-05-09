"""Single-file Dueling DQN trainer (v1) for SlimeVolleyNoFrameskip-v0.

Adds three Rainbow ingredients on top of the v0 baseline (train_dqn.py):

  1. Double DQN:    online-net picks argmax action at s', target-net evaluates.
  2. N-step return: replay stores n-step transitions (default n=3) with
                    correct truncation at episode boundaries.
  3. Soft target update: Polyak averaging each grad step (default tau=0.005)
                         instead of a periodic hard copy.

Architecture, env wrapper, eps schedule, eval loop, optimizer, and TB scalar
names are unchanged from v0.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import gym  # noqa: F401  -- old gym 0.19, registers SlimeVolley envs upon slimevolleygym import
import numpy as np
import slimevolleygym  # noqa: F401  -- registers "SlimeVolleyNoFrameskip-v0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Model (architecturally identical to v0's DuelingMLP)
# ---------------------------------------------------------------------------
class DuelingMLPv1(nn.Module):
    """Dueling MLP: trunk 12->128->128, value head 128->64->1, advantage head
    128->64->6, Q = V + (A - mean A)."""

    def __init__(self, obs_dim: int = 12, n_actions: int = 6) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


# ---------------------------------------------------------------------------
# Replay buffer (now stores n-step transitions; "reward" is the discounted
# n-step return R^(n), "next_obs" is s_{t+n_actual}, "done" is the terminal
# flag at or before t+n_actual, and "n_actual" is the discount exponent we
# must apply to the bootstrap term at training time).
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor       # n-step discounted return R_t^(n)
    next_obs: torch.Tensor      # s_{t+n_actual}
    dones: torch.Tensor         # 1.0 if terminal reached within the window
    n_actual: torch.Tensor      # discount exponent for the bootstrap term


class ReplayBuffer:
    """Plain ring buffer with uniform sampling. Stores float32 obs and the
    n-step bookkeeping fields needed for Double-DQN with n-step targets."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.n_actual = np.zeros((capacity,), dtype=np.int64)
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool, n_actual: int) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.n_actual[i] = int(n_actual)
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
            n_actual=torch.from_numpy(self.n_actual[ids]).to(device),
        )


# ---------------------------------------------------------------------------
# N-step buffer (per-episode FIFO that emits n-step transitions)
# ---------------------------------------------------------------------------
class NStepBuffer:
    """Maintains a per-episode FIFO of the last <= n one-step transitions and
    emits properly-truncated n-step entries to the replay buffer.

    Emission rules:
      - When the FIFO is full (length == n) on a non-terminal step, emit one
        entry: n_actual = n, done_terminal = False, next_obs = s_{t+n}.
      - On a terminal step we first append the terminal one-step record, then
        flush ALL remaining items in the FIFO. Each flushed item gets
        n_actual = (its remaining horizon), done_terminal = True, and
        next_obs = s_terminal. We do NOT bootstrap past terminal: the trainer
        zeroes the bootstrap term when done_terminal is True.
    """

    def __init__(self, n: int, gamma: float) -> None:
        assert n >= 1
        self.n = n
        self.gamma = gamma
        # each item: (obs, action, reward, next_obs, done_one_step)
        self._fifo: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n)

    def reset(self) -> None:
        self._fifo.clear()

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool,
            ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]]:
        """Append a one-step transition; return any n-step entries now ready.
        Each emitted tuple is (obs, action, R_n, next_obs_end, done_term, n_actual)."""
        self._fifo.append((obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)))
        emitted: List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]] = []

        def accumulate(items, start: int) -> float:
            R, discount = 0.0, 1.0
            for j in range(start, len(items)):
                R += discount * items[j][2]
                discount *= self.gamma
            return R

        if done:
            # Flush every remaining FIFO item with done_term=True so the
            # trainer drops the bootstrap (we never bootstrap past terminal).
            fifo_list = list(self._fifo)
            window_next_obs = fifo_list[-1][3]  # s_terminal
            for k in range(len(fifo_list)):
                obs_k, act_k = fifo_list[k][0], fifo_list[k][1]
                emitted.append((obs_k, act_k, accumulate(fifo_list, k),
                                window_next_obs, True, len(fifo_list) - k))
            self._fifo.clear()
            return emitted

        if len(self._fifo) == self.n:
            # FIFO full mid-episode: emit only the oldest; newer items still
            # need more rewards. popleft so state matches deque maxlen behavior.
            fifo_list = list(self._fifo)
            obs_0, act_0 = fifo_list[0][0], fifo_list[0][1]
            emitted.append((obs_0, act_0, accumulate(fifo_list, 0),
                            fifo_list[-1][3], False, self.n))
            self._fifo.popleft()
            return emitted

        return emitted


# ---------------------------------------------------------------------------
# Env helpers (identical to v0)
# ---------------------------------------------------------------------------
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


def evaluate(model: DuelingMLPv1, seed: int, episodes: int,
             device: torch.device) -> Tuple[float, float, float]:
    """Greedy eval on a fresh env. Returns (mean_return, std_return, mean_length)."""
    env = make_env(seed + 10_000)
    returns: List[float] = []
    lengths: List[int] = []
    model.eval()
    with torch.no_grad():
        for _ep in range(episodes):
            obs = np.asarray(env.reset(), dtype=np.float32)
            ep_return, ep_length, done = 0.0, 0, False
            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = int(model(obs_t).argmax(dim=-1).item())
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
    p.add_argument("--save-path", type=str,
                   default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_v1.pt")
    p.add_argument("--log-dir", type=str,
                   default="/home/shuhang/YBJ/RL_test/logs/dqn_v1")
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--soft-tau", type=float, default=0.005,
                   help="Polyak averaging factor for soft target update.")
    p.add_argument("--n-step", type=int, default=3,
                   help="N-step return horizon for replay storage.")
    return p.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def save_checkpoint(
    path: str, model: DuelingMLPv1, total_timesteps_trained: int,
    final_eval_reward: float, n_step: int, soft_tau: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": 12,
        "n_actions": 6,
        "arch": "dueling_v1_mlp_128_128_64",
        "total_timesteps_trained": int(total_timesteps_trained),
        "final_eval_reward": float(final_eval_reward),
        "n_step": int(n_step),
        "double": True,
        "soft_tau": float(soft_tau),
    }, path)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """In-place Polyak averaging: target <- (1 - tau) * target + tau * online."""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


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
    online = DuelingMLPv1(obs_dim, n_actions).to(device)
    target = DuelingMLPv1(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=2.5e-4)
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim)

    # --- hyperparams ---
    gamma = 0.99
    batch_size = 64
    warmup_steps = 1_000
    train_freq = 4
    grad_clip_norm = 10.0
    eps_start, eps_end, eps_fraction = 1.0, 0.02, 0.30
    n_step = int(args.n_step)
    soft_tau = float(args.soft_tau)

    nstep_buf = NStepBuffer(n=n_step, gamma=gamma)

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

        # Push the one-step transition into the per-episode FIFO; it returns
        # any n-step entries that are now ready (zero, one, or — on terminal
        # — up to n entries when flushing the remaining tail).
        for entry in nstep_buf.add(obs, int(action), reward_f, next_obs, bool(done)):
            e_obs, e_act, e_R, e_next_obs, e_done_term, e_n_actual = entry
            buffer.add(e_obs, e_act, e_R, e_next_obs, e_done_term, e_n_actual)

        episode_return += reward_f
        episode_length += 1

        if done:
            writer.add_scalar("rollout/episode_return", episode_return, global_step)
            writer.add_scalar("rollout/episode_length", episode_length, global_step)
            obs = np.asarray(env.reset(), dtype=np.float32)
            episode_return = 0.0
            episode_length = 0
            # FIFO was already flushed inside nstep_buf.add() on done=True.
            nstep_buf.reset()
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
                # Double-DQN target: ONLINE picks the argmax action at s_{t+n};
                # TARGET evaluates that action's Q-value. Both branches are
                # under no_grad and the gathered tensor is a leaf w.r.t. loss,
                # so no gradients flow into the target path.
                next_q_online = online(batch.next_obs)
                next_actions = next_q_online.argmax(dim=-1, keepdim=True)
                next_q_target_all = target(batch.next_obs)
                next_q = next_q_target_all.gather(1, next_actions).squeeze(1)

                # n-step bootstrap: reward is already R_t^(n) and the discount
                # exponent is n_actual (which equals the FIFO length when
                # flushed at terminal, so it is correct even for short eps).
                discount = torch.pow(
                    torch.full_like(batch.rewards, gamma), batch.n_actual.to(batch.rewards.dtype)
                )
                td_target = batch.rewards + discount * next_q * (1.0 - batch.dones)

            q_all = online(batch.obs)
            q_pred = q_all.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_pred, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            # Soft target update happens AFTER each grad step, so the target
            # net always lags the post-update online net by Polyak factor tau.
            soft_update(target, online, soft_tau)

            grad_steps += 1
            recent_losses.append(float(loss.item()))
            recent_qmeans.append(float(q_pred.mean().item()))

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
                save_checkpoint(args.save_path, online, global_step, eval_mean, n_step, soft_tau)

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
    save_checkpoint(args.save_path, online, args.total_timesteps, final_mean, n_step, soft_tau)
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
