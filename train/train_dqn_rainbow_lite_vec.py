"""Vector-env Rainbow-Lite DQN trainer for SlimeVolley-v0.

Forked from ``train_dqn_rainbow_eps_vec.py``. **C51 has been REMOVED** and
the model returns a scalar Q value per action (Dueling combination), trained
with a Huber TD loss (smooth_l1) just like ``train_dqn_v1_vec.py``.

This is the next isolation step in the Rainbow-component ablation. We have
already empirically established that:

  * ``train_dqn_rainbow_vec.py`` (full Rainbow with NoisyNet+C51+PER): does
    NOT learn (train_R stuck at -4.87 over 4M steps).
  * ``train_dqn_rainbow_eps_vec.py`` (Rainbow without NoisyNet, KEEPS C51 +
    PER): does NOT learn (train_R still -4.83 at 1M steps).
  * ``train_dqn_v1_vec.py`` (Dueling+Double+N-step+eps-greedy, no PER, no
    C51): LEARNS (eval_R reaches -0.2).

So the breakage is in PER, C51, or their interaction. THIS trainer drops C51
while keeping PER, isolating PER's contribution. If this trainer also fails
to learn, PER itself is the root cause; if it learns, C51 (or C51+PER
interaction) is the culprit.

What stays from rainbow_eps_vec
-------------------------------
  * Prioritized Experience Replay (sum-tree, alpha=0.5, beta annealed
    0.4 -> 1.0 across training, IS-weighted Huber loss, priorities = |TD|).
  * Per-env n-step (n=3) FIFO buffer with terminal_observation handling.
  * Soft Polyak target update (tau=0.005) every gradient step.
  * Double DQN target: action chosen by online net, value taken from target.
  * SubprocVecEnv with N=16 workers; per-env n-step buffer.
  * batch_size=256, train_freq=4, lr=1e-4 (same as rainbow_eps_vec).
  * Linear epsilon-greedy schedule (1.0 -> 0.02 over first 20% of training).
  * spawn-safety, ``__main__`` guard.

What changes from rainbow_eps_vec
---------------------------------
  * Model: ``RainbowLiteDuelingMLP`` instead of ``RainbowEpsilonDuelingC51``.
    Architecture is **identical to v1_vec's DuelingMLPv1**: trunk
    12->128->128, value_head 128->64->1, advantage_head 128->64->n_actions,
    Q = V + (A - mean A). ``forward(obs) -> (B, n_actions)`` scalar Q.
  * No ``support`` buffer, no ``n_atoms``/``v_min``/``v_max`` CLI args, no
    ``categorical_projection`` function, no log_softmax over atoms.
  * Loss: ``(weights * smooth_l1(q_taken, td_target)).mean()`` (IS-weighted
    Huber). Mirrors v1_vec's loss except for the IS weighting.
  * Priorities: ``(|td_error| + eps) ** alpha``, computed from the scalar
    TD error (not per-sample CE).
  * Checkpoint metadata: ``arch="rainbow_lite_dueling"``. Importantly the
    state-dict layout matches DuelingMLPv1's structure exactly, so
    ``train/export_onnx.py`` should load this checkpoint without modification
    (it uses ``strict=True`` against ``DuelingQNet`` which has the same
    parameter shapes; the ``arch`` string is only printed, not validated).

Replay-buffer schema is also simplified back to v1_vec's: we store
``(obs, action, R_n, next_obs_end, done_terminal, n_actual)`` and let the
trainer compute ``gamma_eff = gamma ** n_actual * (1 - done)`` at sample
time, exactly as v1_vec does. The PER wrapping (priorities + IS weights) is
layered on top.
"""

from __future__ import annotations

import argparse
import functools
import multiprocessing as _mp
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import gym  # noqa: F401  -- old gym 0.19; needed for SlimeVolley registration
import numpy as np
import slimevolleygym  # noqa: F401  -- registers "SlimeVolley-v0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Make `train/` importable regardless of CWD so spawn children can re-import
# sibling modules (DiscreteSlimeWrapper, vec_env) when this module is the
# trainer entrypoint.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Sibling imports: the wrapper and action table are shared with v1; we
# DELIBERATELY copy the model / replay / n-step code inline below (rather
# than importing) so that the file is self-contained and free of spawn-time
# import surprises.
from train_dqn_v1 import ACTION_TABLE, DiscreteSlimeWrapper  # noqa: E402, F401
from vec_env import SubprocVecEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Rainbow-Lite Dueling MLP (scalar Q; architecturally identical to
# DuelingMLPv1 in train_dqn_v1_vec.py).
# ---------------------------------------------------------------------------
class RainbowLiteDuelingMLP(nn.Module):
    """Dueling MLP with mean-subtracted advantage; returns scalar Q per action.

    Architecture (intentionally byte-for-byte the same as ``DuelingMLPv1``):
      * trunk:          Linear(12, 128) -> ReLU -> Linear(128, 128) -> ReLU
      * value_head:     Linear(128, 64) -> ReLU -> Linear(64, 1)
      * advantage_head: Linear(128, 64) -> ReLU -> Linear(64, n_actions)

    forward(obs) -> Q with Q = V + (A - mean_a A), shape ``(B, n_actions)``.

    Keeping the layout identical means the existing ``train/export_onnx.py``
    can load this checkpoint with ``strict=True`` (it ignores the ``arch``
    string -- only the parameter shapes / names matter).
    """

    def __init__(self, obs_dim: int = 12, n_actions: int = 6) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
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
        v = self.value_head(features)
        a = self.advantage_head(features)
        # Dueling combination: subtract mean advantage for V identifiability.
        return v + (a - a.mean(dim=-1, keepdim=True))


# ---------------------------------------------------------------------------
# Per-env n-step buffer (identical to v1_vec / rainbow_eps_vec).
# ---------------------------------------------------------------------------
class NStepBuffer:
    """Per-env FIFO emitting properly-truncated n-step entries.

    One instance per parallel env. Emission contract:

      * Non-terminal step + FIFO full -> emit oldest with
        n_actual = n, done_terminal = False, next_obs = s_{t+n}.
      * Terminal step -> append the terminal one-step record FIRST, then
        flush every remaining FIFO entry with done_terminal = True,
        n_actual = remaining-horizon, next_obs = s_terminal. The trainer
        zeroes the bootstrap when done_terminal is True so we never bootstrap
        past a terminal.
    """

    def __init__(self, n: int, gamma: float) -> None:
        assert n >= 1
        self.n = int(n)
        self.gamma = float(gamma)
        # Each item: (obs, action, reward, next_obs, done_one_step).
        self._fifo: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.n)

    def reset(self) -> None:
        self._fifo.clear()

    def _accumulate(self, items, start: int) -> float:
        R, discount = 0.0, 1.0
        for j in range(start, len(items)):
            R += discount * items[j][2]
            discount *= self.gamma
        return R

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]]:
        """Append a one-step transition; return any n-step entries now ready.

        Each emitted tuple is
        ``(obs, action, R_n, next_obs_end, done_terminal, n_actual)``.
        """
        self._fifo.append((obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)))
        emitted: List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]] = []

        if done:
            fifo_list = list(self._fifo)
            window_next_obs = fifo_list[-1][3]  # s_terminal (true terminal obs).
            for k in range(len(fifo_list)):
                obs_k, act_k = fifo_list[k][0], fifo_list[k][1]
                emitted.append((
                    obs_k, act_k, self._accumulate(fifo_list, k),
                    window_next_obs, True, len(fifo_list) - k,
                ))
            self._fifo.clear()
            return emitted

        if len(self._fifo) == self.n:
            fifo_list = list(self._fifo)
            obs_0, act_0 = fifo_list[0][0], fifo_list[0][1]
            emitted.append((
                obs_0, act_0, self._accumulate(fifo_list, 0),
                fifo_list[-1][3], False, self.n,
            ))
            self._fifo.popleft()
            return emitted

        return emitted


# ---------------------------------------------------------------------------
# SumTree (segment tree backing PER) -- identical to rainbow_eps_vec.
# ---------------------------------------------------------------------------
class SumTree:
    """Array-backed sum tree for O(log N) prioritized sampling.

    Layout: complete binary tree of size 2*N - 1 with N rounded up to the next
    power of two. Internal nodes occupy [0, N-1); leaves occupy [N-1, 2N-1);
    leaf k corresponds to data slot k. For N=4: indices 0 (root), 1-2 (level 1),
    3-6 (leaves), so data slot i lives at tree index i + 3.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        n = 1
        while n < self.capacity:
            n *= 2
        self.n_leaves = n
        self.tree = np.zeros(2 * n - 1, dtype=np.float64)

    def _leaf_index(self, data_idx: int) -> int:
        return data_idx + self.n_leaves - 1

    def update(self, data_idx: int, priority: float) -> None:
        idx = self._leaf_index(data_idx)
        delta = float(priority) - self.tree[idx]
        self.tree[idx] = float(priority)
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def total(self) -> float:
        return float(self.tree[0])

    def get(self, value: float) -> Tuple[int, float]:
        """Find the leaf whose cumulative-sum range contains ``value``.

        Returns ``(data_idx, priority)``.
        """
        idx = 0
        while idx < self.n_leaves - 1:
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        priority = float(self.tree[idx])
        data_idx = idx - (self.n_leaves - 1)
        return data_idx, priority


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer (n-step transitions stored in v1_vec's schema)
# ---------------------------------------------------------------------------
@dataclass
class PERBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor       # n-step discounted return R_t^(n)
    next_obs: torch.Tensor      # s_{t + n_actual} (true terminal obs if done)
    dones: torch.Tensor         # 1.0 if terminal reached within the window
    n_actuals: torch.Tensor     # discount exponent for the bootstrap term
    weights: torch.Tensor       # IS weights (B,)
    indices: np.ndarray         # data-buffer indices, for priority update


class PrioritizedReplayBuffer:
    """Sum-tree backed prioritized replay over n-step transitions.

    Storage schema (mirrors v1_vec's ReplayBuffer):
        (obs, action, R_n, next_obs_end, done_terminal, n_actual)

    Trainer reconstructs ``gamma_eff = gamma ** n_actual * (1 - done)`` at
    sample time. Priority for a fresh transition = ``max_priority``; on
    update, ``priority_i = (|td_error_i| + eps) ** alpha``.
    """

    def __init__(self, capacity: int, obs_dim: int, alpha: float, eps: float = 1e-6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.n_actuals = np.zeros((capacity,), dtype=np.int64)
        self.tree = SumTree(capacity)
        self.idx = 0
        self.size = 0
        self.max_priority = 1.0  # newcomers get this; ensures they get sampled.

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        n_actual: int,
    ) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.n_actuals[i] = int(n_actual)
        # New transitions get max-priority so they are sampled at least once
        # before their real TD error overwrites the priority.
        priority = float(self.max_priority) ** self.alpha
        self.tree.update(i, priority)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float, device: torch.device) -> PERBatch:
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        total = self.tree.total()
        if total <= 0.0:
            # Degenerate: fall back to uniform sampling over filled slots.
            indices = np.random.randint(0, self.size, size=batch_size).astype(np.int64)
            priorities[:] = 1.0
        else:
            segment = total / batch_size
            for k in range(batch_size):
                lo = segment * k
                hi = segment * (k + 1)
                value = np.random.uniform(lo, hi)
                data_idx, priority = self.tree.get(value)
                # Guard against rare sum-tree edge case where data_idx points
                # past the filled portion (when capacity isn't a power of two).
                if data_idx >= self.size:
                    data_idx = np.random.randint(0, self.size)
                    priority = max(self.tree.tree[self.tree._leaf_index(data_idx)], 1e-12)
                indices[k] = data_idx
                priorities[k] = priority

        sampling_probs = priorities / max(total, 1e-12)
        # IS weight w_i = (N * P(i))^(-beta), normalized by max for stability.
        weights = (self.size * np.maximum(sampling_probs, 1e-12)) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)
        weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)

        return PERBatch(
            obs=torch.from_numpy(self.obs[indices]).to(device),
            actions=torch.from_numpy(self.actions[indices]).to(device),
            rewards=torch.from_numpy(self.rewards[indices]).to(device),
            next_obs=torch.from_numpy(self.next_obs[indices]).to(device),
            dones=torch.from_numpy(self.dones[indices]).to(device),
            n_actuals=torch.from_numpy(self.n_actuals[indices]).to(device),
            weights=weights_t,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """priority_i = (|td_i| + eps)^alpha. Also tracks max for newcomers."""
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self.tree.update(int(idx), p)
            if abs(float(err)) + self.eps > self.max_priority:
                self.max_priority = abs(float(err)) + self.eps


# ---------------------------------------------------------------------------
# Soft target update (Polyak averaging on parameters).
# ---------------------------------------------------------------------------
@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


# ---------------------------------------------------------------------------
# Linear epsilon schedule (copied inline from train_dqn_v1_vec.py).
# ---------------------------------------------------------------------------
def linear_epsilon(step: int, total_steps: int,
                   start: float, end: float, fraction: float) -> float:
    """eps decays linearly from `start` to `end` over the first
    `fraction * total_steps` transitions; flat at `end` afterwards."""
    decay_steps = max(1, int(total_steps * fraction))
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


# ---------------------------------------------------------------------------
# Env helpers (single env for eval; vec env for training).
# ---------------------------------------------------------------------------
def make_env(seed: int) -> "gym.Env":
    """Module-level factory: SubprocVecEnv must be able to pickle this for
    the spawn start method, so we cannot use a closure-bound lambda. The
    trainer wraps this with ``functools.partial(make_env, seed_base + i)``.
    """
    env = gym.make("SlimeVolley-v0")
    env = DiscreteSlimeWrapper(env)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def evaluate(
    model: RainbowLiteDuelingMLP,
    seed: int,
    episodes: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Greedy eval on a fresh single env. Plain argmax over scalar Q.

    No noise toggling needed (no NoisyNet, no distribution): mirrors v1_vec's
    eval path exactly -- model.eval() + torch.no_grad() + argmax forward.
    """
    env = make_env(seed + 10_000)
    returns: List[float] = []
    lengths: List[int] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(episodes):
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
    finally:
        if was_training:
            model.train()
    env.close()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--total-timesteps", type=int, default=5_000_000,
        help=("Total transitions ingested across ALL envs. Outer loop "
              "iterations = total-timesteps / num-envs."),
    )
    p.add_argument("--num-envs", type=int, default=16,
                   help="Number of parallel envs (one subprocess each).")
    p.add_argument("--batch-size", type=int, default=256,
                   help="PER minibatch size (raised from 64 due to N-fold ingestion).")
    p.add_argument(
        "--train-freq", type=int, default=4,
        help=("Outer iterations between gradient steps. With num-envs=N "
              "and train-freq=K, one grad step happens every N*K transitions."),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--save-path", type=str,
        default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_rainbow_lite_vec.pt",
    )
    p.add_argument(
        "--log-dir", type=str,
        default="/home/shuhang/YBJ/RL_test/logs/dqn_rainbow_lite_vec",
    )
    p.add_argument("--eval-every", type=int, default=25_000,
                   help="Eval cadence in transitions (NOT iterations).")
    p.add_argument("--eval-episodes", type=int, default=5)

    # Optimizer / loss knobs
    # lr default 1e-4 mirrors rainbow_eps_vec (a touch higher than rainbow's
    # 6.25e-5; without NoisyNet we can push lr a bit and stay stable on this
    # small env).
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--warmup-steps", type=int, default=2_000,
                   help="Min replay size before any gradient step (transitions).")
    p.add_argument("--tau", type=float, default=0.005,
                   help="Polyak coefficient for soft target update each grad step.")
    p.add_argument("--grad-clip-norm", type=float, default=10.0)

    # PER knobs
    p.add_argument("--alpha", type=float, default=0.5,
                   help="PER prioritization exponent. 0 = uniform; 1 = full.")
    p.add_argument("--beta-start", type=float, default=0.4)
    p.add_argument("--beta-end", type=float, default=1.0)
    p.add_argument("--per-eps", type=float, default=1e-6)

    # Epsilon-greedy schedule
    p.add_argument("--epsilon-start", type=float, default=1.0,
                   help="Initial exploration probability for linear-eps schedule.")
    p.add_argument("--epsilon-end", type=float, default=0.02,
                   help="Final exploration probability (held flat after decay).")
    p.add_argument("--epsilon-fraction", type=float, default=0.2,
                   help="Fraction of total-timesteps over which eps decays linearly.")

    return p.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda" and not torch.cuda.is_available():
        # Match v1_vec / rainbow_eps_vec behavior: explicit "cuda" without CUDA
        # soft-fails to CPU.
        print("[warn] --device cuda requested but CUDA unavailable; using CPU.",
              flush=True)
        return torch.device("cpu")
    return torch.device(flag)


def save_checkpoint(
    path: str,
    model: RainbowLiteDuelingMLP,
    total_timesteps_trained: int,
    final_eval_reward: float,
    args: argparse.Namespace,
) -> None:
    """Save state_dict on CPU for portability across machines/devices.

    Schema:
      * arch="rainbow_lite_dueling" -- distinguishes from v1's
        "dueling_v1_mlp_128_128_64" and from rainbow_eps's
        "rainbow_eps_dueling_c51". Parameter SHAPES match DuelingMLPv1 so
        ``train/export_onnx.py`` can load this checkpoint with strict=True
        (it only checks the state_dict layout, not the arch string).
      * No v_min/v_max/n_atoms (C51 removed).
      * PER + n-step + soft-tau + epsilon schedule fields preserved so the
        training-time configuration is recoverable from the checkpoint alone.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": cpu_state,
            "obs_dim": 12,
            "n_actions": 6,
            "arch": "rainbow_lite_dueling",
            "gamma": float(args.gamma),
            "n_step": int(args.n_step),
            "alpha": float(args.alpha),
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
            "soft_tau": float(args.tau),
            "epsilon_start": float(args.epsilon_start),
            "epsilon_end": float(args.epsilon_end),
            "epsilon_fraction": float(args.epsilon_fraction),
            "total_timesteps_trained": int(total_timesteps_trained),
            "final_eval_reward": float(final_eval_reward),
            "num_envs": int(args.num_envs),
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

    num_envs = int(args.num_envs)
    if args.total_timesteps % num_envs != 0:
        # Round down to a multiple of num_envs so iterations align cleanly.
        args.total_timesteps = (args.total_timesteps // num_envs) * num_envs
    n_iterations = args.total_timesteps // num_envs

    print(
        f"[config] device={device} num_envs={num_envs} "
        f"total_timesteps={args.total_timesteps} (transitions across all envs) "
        f"iterations={n_iterations} batch_size={args.batch_size} "
        f"train_freq={args.train_freq} (outer-iters per grad) "
        f"n_step={args.n_step} tau={args.tau} lr={args.lr} "
        f"alpha={args.alpha} beta_start={args.beta_start} "
        f"eps_start={args.epsilon_start} eps_end={args.epsilon_end} "
        f"eps_fraction={args.epsilon_fraction} seed={args.seed}",
        flush=True,
    )

    # --- vec env: per-worker seeds args.seed + i ---
    # ``functools.partial(make_env, seed_base + i)`` is picklable under spawn
    # (unlike a closure-bound lambda); SubprocVecEnv ships these to children.
    seed_base = int(args.seed)
    env_fns = [functools.partial(make_env, seed_base + i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    try:
        obs_dim = int(np.array(vec_env.observation_space.shape).prod())
        n_actions = int(vec_env.action_space.n)
        assert obs_dim == 12 and n_actions == 6, (
            f"unexpected env shapes: obs_dim={obs_dim}, n_actions={n_actions}"
        )

        # --- nets, optim, buffer ---
        online = RainbowLiteDuelingMLP(obs_dim=obs_dim, n_actions=n_actions).to(device)
        target = RainbowLiteDuelingMLP(obs_dim=obs_dim, n_actions=n_actions).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()
        for p in target.parameters():
            p.requires_grad = False

        # Default Adam epsilon (1e-8). The rainbow eps=1.5e-4 was a C51
        # numerical-stability concession; no longer needed without softmax.
        optimizer = torch.optim.Adam(online.parameters(), lr=float(args.lr))

        buffer = PrioritizedReplayBuffer(
            capacity=args.replay_capacity,
            obs_dim=obs_dim,
            alpha=args.alpha,
            eps=args.per_eps,
        )

        # --- hyperparams already grouped on args ---
        gamma = float(args.gamma)
        batch_size = int(args.batch_size)
        warmup_steps = int(args.warmup_steps)
        train_freq = int(args.train_freq)
        grad_clip_norm = float(args.grad_clip_norm)
        n_step = int(args.n_step)
        tau = float(args.tau)
        eps_start = float(args.epsilon_start)
        eps_end = float(args.epsilon_end)
        eps_fraction = float(args.epsilon_fraction)

        # one n-step FIFO per env
        nstep_bufs: List[NStepBuffer] = [
            NStepBuffer(n=n_step, gamma=gamma) for _ in range(num_envs)
        ]

        # --- run-state ---
        grad_steps = 0
        best_eval_reward = -float("inf")
        recent_losses: deque[float] = deque(maxlen=100)
        recent_qmeans: deque[float] = deque(maxlen=100)
        recent_td: deque[float] = deque(maxlen=100)
        # per-env episode bookkeeping
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        # rolling buffer of recent finished-episode returns (across all envs)
        recent_ep_returns: deque[float] = deque(maxlen=100)
        recent_ep_lengths: deque[float] = deque(maxlen=100)

        # initial reset; per-worker seeds were already applied at worker init
        obs = vec_env.reset()  # (N, 12)

        global_step = 0          # total transitions ingested across all envs
        last_eval_step = 0
        start_time = time.time()

        for it in range(1, n_iterations + 1):
            # --- linear-eps action selection.
            # Single coin flip per outer iteration applies to all N envs (same
            # convention as v1_vec); cheap and avoids forwarding a batch we
            # won't use. Eps decays linearly start->end over the first
            # `eps_fraction * total_timesteps` transitions, then flat at end.
            epsilon = linear_epsilon(
                global_step, args.total_timesteps,
                eps_start, eps_end, eps_fraction,
            )
            if random.random() < epsilon:
                actions = np.random.randint(0, n_actions, size=num_envs).astype(np.int64)
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).to(device)
                    q_all = online(obs_t)  # (N, n_actions) scalar Q
                    actions = q_all.argmax(dim=-1).cpu().numpy().astype(np.int64)

            next_obs, rewards, dones, infos = vec_env.step(actions)
            global_step += num_envs

            # --- per-env n-step ingestion ---
            # When dones[i] is True, next_obs[i] from SubprocVecEnv is the
            # AUTO-RESET post-reset obs. The TRUE terminal next-obs is in
            # infos[i]["terminal_observation"]; we MUST use that for the
            # n-step transition's `next_obs` (otherwise the bootstrap target
            # would be conditioned on a state from a totally unrelated
            # episode, which is exactly the bug this code path exists to
            # avoid).
            for i in range(num_envs):
                if dones[i]:
                    term_obs = infos[i].get("terminal_observation", next_obs[i])
                    term_obs = np.asarray(term_obs, dtype=np.float32)
                    emitted = nstep_bufs[i].add(
                        obs[i], int(actions[i]), float(rewards[i]),
                        term_obs, True,
                    )
                else:
                    emitted = nstep_bufs[i].add(
                        obs[i], int(actions[i]), float(rewards[i]),
                        np.asarray(next_obs[i], dtype=np.float32), False,
                    )
                for entry in emitted:
                    e_obs, e_act, e_R, e_next_obs, e_done_term, e_n_actual = entry
                    buffer.add(
                        e_obs, int(e_act), float(e_R), e_next_obs,
                        bool(e_done_term), int(e_n_actual),
                    )

                # episode bookkeeping
                episode_returns[i] += float(rewards[i])
                episode_lengths[i] += 1
                if dones[i]:
                    recent_ep_returns.append(float(episode_returns[i]))
                    recent_ep_lengths.append(int(episode_lengths[i]))
                    writer.add_scalar(
                        "rollout/episode_return", float(episode_returns[i]),
                        global_step,
                    )
                    writer.add_scalar(
                        "rollout/episode_length", int(episode_lengths[i]),
                        global_step,
                    )
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0
                    nstep_bufs[i].reset()

            # advance obs to the auto-reset obs returned by the vec env
            obs = next_obs

            # --- learn (one grad step every `train_freq` ITERATIONS, i.e.
            # every train_freq*num_envs transitions) ---
            should_train = (
                buffer.size >= max(warmup_steps, batch_size)
                and (it % train_freq == 0)
            )
            if should_train:
                # Anneal beta from beta_start to beta_end linearly across training.
                frac = min(1.0, global_step / max(1, args.total_timesteps))
                beta = args.beta_start + (args.beta_end - args.beta_start) * frac

                batch = buffer.sample(batch_size, beta, device)

                # --- predicted Q at (s, a) under the online net ---
                q_online_all = online(batch.obs)                              # (B, n_actions)
                q_taken = q_online_all.gather(
                    1, batch.actions.long().view(-1, 1)
                ).squeeze(1)                                                  # (B,)

                # --- Double DQN target: action by online, value by target ---
                with torch.no_grad():
                    q_next_online = online(batch.next_obs)                    # (B, n_actions)
                    next_actions = q_next_online.argmax(dim=-1)               # (B,)
                    q_next_target_all = target(batch.next_obs)                # (B, n_actions)
                    q_next = q_next_target_all.gather(
                        1, next_actions.view(-1, 1)
                    ).squeeze(1)                                              # (B,)

                    # gamma_eff = gamma ** n_actual; (1 - done) zeros the
                    # bootstrap when the n-step window terminated.
                    gamma_eff = torch.pow(
                        torch.full_like(batch.rewards, gamma),
                        batch.n_actuals.to(batch.rewards.dtype),
                    )
                    y = batch.rewards + gamma_eff * q_next * (1.0 - batch.dones)

                # --- IS-weighted Huber loss ---
                td_error = q_taken - y
                per_sample_loss = F.smooth_l1_loss(q_taken, y, reduction="none")
                loss = (batch.weights * per_sample_loss).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                # --- priority update: |TD| (raw; PER buffer applies alpha) ---
                with torch.no_grad():
                    td_abs = td_error.detach().abs().cpu().numpy().astype(np.float64)
                buffer.update_priorities(batch.indices, td_abs)

                # --- soft target update (tau blend each gradient step) ---
                soft_update(target, online, tau)

                grad_steps += 1
                recent_losses.append(float(loss.item()))
                recent_qmeans.append(float(q_taken.mean().item()))
                recent_td.append(float(td_abs.mean()))

                if grad_steps % 100 == 0:
                    writer.add_scalar("train/loss", float(np.mean(recent_losses)), global_step)
                    writer.add_scalar("train/q_mean", float(np.mean(recent_qmeans)), global_step)
                    writer.add_scalar("train/td_abs_mean", float(np.mean(recent_td)), global_step)
                    writer.add_scalar("train/beta", beta, global_step)
                    writer.add_scalar("train/epsilon", epsilon, global_step)
                    writer.add_scalar("train/replay_size", float(buffer.size), global_step)

            # --- periodic eval (cadence in transitions, not iterations) ---
            if (global_step - last_eval_step) >= int(args.eval_every):
                last_eval_step = global_step
                eval_mean, eval_std, eval_len = evaluate(
                    online, args.seed, args.eval_episodes, device,
                )
                writer.add_scalar("eval/reward_mean", eval_mean, global_step)
                writer.add_scalar("eval/reward_std", eval_std, global_step)
                writer.add_scalar("eval/episode_length_mean", eval_len, global_step)
                elapsed = time.time() - start_time
                sps = global_step / max(elapsed, 1e-6)
                rolling_R = (float(np.mean(recent_ep_returns))
                             if recent_ep_returns else float("nan"))
                print(
                    f"[step {global_step:>9d}] eps={epsilon:.3f} "
                    f"eval_R={eval_mean:+.3f}+-{eval_std:.3f} "
                    f"eval_len={eval_len:.1f} train_R={rolling_R:+.3f} "
                    f"sps={sps:.1f} grad_steps={grad_steps}",
                    flush=True,
                )
                if eval_mean > best_eval_reward:
                    best_eval_reward = eval_mean
                    save_checkpoint(args.save_path, online, global_step, eval_mean, args)

        # --- final eval + checkpoint ---
        final_mean, final_std, final_len = evaluate(
            online, args.seed, args.eval_episodes, device,
        )
        writer.add_scalar("eval/reward_mean", final_mean, args.total_timesteps)
        writer.add_scalar("eval/reward_std", final_std, args.total_timesteps)
        writer.add_scalar("eval/episode_length_mean", final_len, args.total_timesteps)
        print(
            f"[final {args.total_timesteps}] "
            f"eval_R={final_mean:+.3f}+-{final_std:.3f} "
            f"eval_len={final_len:.1f} grad_steps={grad_steps}",
            flush=True,
        )
        save_checkpoint(args.save_path, online, args.total_timesteps, final_mean, args)
        writer.close()

    finally:
        # Always shut down workers, even on KeyboardInterrupt or exception.
        try:
            vec_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    # spawn requires that the entrypoint be guarded by __main__ so children
    # can re-import this module without re-running the trainer. Setting the
    # start method here is a no-op if already set elsewhere; we rely on
    # SubprocVecEnv getting its own spawn context internally.
    try:
        _mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # Start method already set by the embedding process; harmless.
        pass
    main()
