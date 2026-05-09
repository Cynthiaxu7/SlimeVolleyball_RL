"""Vector-env Rainbow DQN trainer for SlimeVolley-v0 (old gym 0.19).

Full Rainbow stack (Dueling + Double DQN + n-step=3 + PER + NoisyNet + C51)
on top of ``SubprocVecEnv``. Algorithmically identical to the single-env
Rainbow trainer in ``train_dqn_rainbow.py``; only the data-collection loop
and a few cadences differ:

  * N parallel envs feed a single shared PrioritizedReplayBuffer.
  * Per-env ``NStepBuffer`` (one FIFO per worker) so that n-step transitions
    never bridge across episode boundaries on different envs.
  * On ``done[i] = True`` the worker has already auto-reset; the *true*
    terminal next-obs comes from ``info[i]["terminal_observation"]``.
  * NoisyNet noise resampled once per outer iteration (action selection)
    and once per gradient step on both online + target nets. NOT every
    env step — that would be wasted compute given how cheap an iteration is.
  * PER ingest: every n-step transition added with current ``max_priority``
    so it is sampled at least once before its real TD error is computed
    (standard Rainbow; identical to the single-env trainer).
  * After each gradient step, ``replay.update_priorities`` is called with the
    per-sample cross-entropy loss (rank-equivalent to KL; standard Rainbow).

Defaults vs. single-env Rainbow:

    --num-envs        16
    --batch-size      256   (was 64; data ingestion is N-fold)
    --train-freq      4     (env-iters per grad step; 64 transitions / grad)
    --total-timesteps 5_000_000  (transitions across ALL envs)

Checkpoint metadata is byte-identical to the single-env Rainbow checkpoint
(same ``arch="rainbow_dueling_noisy_c51"`` and same key set) plus an extra
``num_envs`` field, so ``export_rainbow_onnx.py`` works on the resulting
checkpoint without modification.
"""

from __future__ import annotations

import argparse
import functools
import math
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
# DELIBERATELY copy the model / replay / n-step / projection code inline
# below (rather than importing from train_dqn_rainbow) so that the file is
# self-contained and free of spawn-time import surprises.
from train_dqn_v1 import ACTION_TABLE, DiscreteSlimeWrapper  # noqa: E402, F401
from vec_env import SubprocVecEnv  # noqa: E402


# ---------------------------------------------------------------------------
# NoisyLinear (Fortunato et al. 2018, factorized Gaussian noise)
# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise on weights and biases.

    Parameters (trainable): weight_mu, weight_sigma, bias_mu, bias_sigma.
    Buffers (non-trainable): weight_epsilon, bias_epsilon.

    Factorized noise: epsilon_w = f(eps_out) outer f(eps_in) with
    f(x) = sgn(x) * sqrt(|x|), eps_{in,out} ~ N(0, I); bias eps = f(eps_out).
    mu ~ U(-1/sqrt(in), 1/sqrt(in)); sigma init = sigma0 / sqrt(in).
    `set_noise_disabled(True)` short-circuits forward to plain Linear(mu_w, mu_b);
    this is what the ONNX export path relies on.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_init = float(sigma_init)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        # Plain bool (not buffer) so the ONNX tracer never captures it as a graph input.
        self.noise_disabled: bool = False

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        sigma_fill = self.sigma_init / math.sqrt(self.in_features)
        nn.init.constant_(self.weight_sigma, sigma_fill)
        nn.init.constant_(self.bias_sigma, sigma_fill)

    @staticmethod
    def _f_noise(x: torch.Tensor) -> torch.Tensor:
        # f(x) = sgn(x) * sqrt(|x|) -- the factorized-noise transform.
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        device = self.weight_mu.device
        eps_in = self._f_noise(torch.randn(self.in_features, device=device))
        eps_out = self._f_noise(torch.randn(self.out_features, device=device))
        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_epsilon.copy_(eps_out)

    def set_noise_disabled(self, flag: bool) -> None:
        self.noise_disabled = bool(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_disabled:
            return F.linear(x, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Rainbow Dueling C51 model
# ---------------------------------------------------------------------------
class RainbowDuelingC51(nn.Module):
    """Dueling C51 with NoisyNet heads.

      forward_train(obs) -> (B, n_actions, n_atoms) log-probabilities (train path).
      forward(obs)       -> (B, n_actions)          expected Q (ONNX path).

    Per-atom Dueling combination in logit space:
        logits(s, a, z) = V(s, z) + (A(s, a, z) - mean_a A(s, a, z))
        log_p(s, a, .)  = log_softmax over z.
    """

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 6,
        n_atoms: int = 51,
        v_min: float = -5.0,
        v_max: float = 5.0,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            NoisyLinear(128, 64, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(64, n_atoms, sigma_init=sigma_init),
        )
        self.advantage_head = nn.Sequential(
            NoisyLinear(128, 64, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(64, n_actions * n_atoms, sigma_init=sigma_init),
        )

    # --- noise-management helpers ---------------------------------------
    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def set_all_noise_disabled(self, flag: bool) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.set_noise_disabled(flag)

    # --- core distributional forward ------------------------------------
    def _logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Return per-(action, atom) logits before the softmax.

        Shape: (B, n_actions, n_atoms).
        """
        features = self.trunk(obs)
        v = self.value_head(features).view(-1, 1, self.n_atoms)
        a = self.advantage_head(features).view(-1, self.n_actions, self.n_atoms)
        a_centered = a - a.mean(dim=1, keepdim=True)
        return v + a_centered

    def forward_train(self, obs: torch.Tensor) -> torch.Tensor:
        """Training-time forward: returns log-probabilities (B, n_actions, n_atoms)."""
        logits = self._logits(obs)
        return F.log_softmax(logits, dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """ONNX/inference forward: returns expected Q-values (B, n_actions).

        Caller must call `set_all_noise_disabled(True)` for deterministic
        outputs (e.g. during ONNX export and evaluation).
        """
        logits = self._logits(obs)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.support.view(1, 1, -1)).sum(dim=-1)

    def expected_q_from_log_p(self, log_p: torch.Tensor) -> torch.Tensor:
        """Helper used in the train loop to avoid a second forward pass."""
        probs = log_p.exp()
        return (probs * self.support.view(1, 1, -1)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Per-env n-step buffer
# ---------------------------------------------------------------------------
class NStepBuffer:
    """Per-env FIFO emitting properly-truncated n-step entries.

    One instance per parallel env. Emission contract (matches single-env
    Rainbow's NStepBuffer semantics, but expressed in the
    ``train_dqn_v1_vec.py`` style: emit-list per call so the master can
    drain bridging windows on a terminal step in one shot).

      * Non-terminal step + FIFO full -> emit oldest with
        n_actual = n, done_terminal = False, next_obs = s_{t+n}.
      * Terminal step -> append the terminal one-step record FIRST, then
        flush every remaining FIFO entry with done_terminal = True,
        n_actual = remaining-horizon, next_obs = s_terminal. The trainer
        translates this into ``gamma_eff = 0`` so we never bootstrap past
        a terminal.
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
        ``(obs, action, R_n, next_obs_end, done_terminal, n_actual)`` where
        ``n_actual`` is the actual horizon used (n on the steady-state path,
        less when the window was clipped by a terminal).
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
# SumTree (segment tree backing PER)
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
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------
@dataclass
class PERBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    n_step_returns: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    gamma_effs: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedReplayBuffer:
    """Sum-tree backed prioritized replay buffer.

    Stores n-step transitions:
        (obs, action, R_n, next_obs_{t+n}, terminal_within_window, gamma_eff)
    where ``gamma_eff = gamma ** n_actual`` if the window did not terminate,
    else ``0`` so the bootstrap term disappears.
    """

    def __init__(self, capacity: int, obs_dim: int, alpha: float, eps: float = 1e-6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.n_step_returns = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.gamma_effs = np.zeros((capacity,), dtype=np.float32)
        self.tree = SumTree(capacity)
        self.idx = 0
        self.size = 0
        self.max_priority = 1.0  # newcomers get this; ensures they get sampled.

    def add(
        self,
        obs: np.ndarray,
        action: int,
        n_step_return: float,
        next_obs: np.ndarray,
        done: bool,
        gamma_eff: float,
    ) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.n_step_returns[i] = float(n_step_return)
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.gamma_effs[i] = float(gamma_eff)
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
            n_step_returns=torch.from_numpy(self.n_step_returns[indices]).to(device),
            next_obs=torch.from_numpy(self.next_obs[indices]).to(device),
            dones=torch.from_numpy(self.dones[indices]).to(device),
            gamma_effs=torch.from_numpy(self.gamma_effs[indices]).to(device),
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
# C51 categorical projection
# ---------------------------------------------------------------------------
def categorical_projection(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    gamma_effs: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
    n_atoms: int,
    delta_z: float,
) -> torch.Tensor:
    """Project Bellman-shifted target distribution onto the fixed support.

    next_dist: (B, n_atoms) target probabilities at next_obs for the action
    chosen by the online net (Double DQN). rewards: aggregated n-step return.
    gamma_effs: gamma^K if no terminal in window, else 0. Returns m (B, n_atoms)
    summing to 1 per row. For each source atom j: Tz = r + gamma_eff * z_j,
    clamp to [V_min, V_max], distribute p[j] between bins l = floor(b),
    u = ceil(b) with weights (u - b) and (b - l) respectively.
    """
    batch_size = next_dist.shape[0]
    device = next_dist.device

    tz = rewards.unsqueeze(1) + gamma_effs.unsqueeze(1) * support.unsqueeze(0)
    tz = tz.clamp(v_min, v_max)
    b = (tz - v_min) / delta_z  # fractional atom index in [0, n_atoms - 1].
    l = b.floor().long().clamp(0, n_atoms - 1)
    u = b.ceil().long().clamp(0, n_atoms - 1)

    m = torch.zeros(batch_size, n_atoms, device=device, dtype=next_dist.dtype)
    lower_mass = next_dist * (u.float() - b)
    upper_mass = next_dist * (b - l.float())

    # Edge case: l == u means Tz landed exactly on an atom; both weights are 0
    # so the standard split would lose the mass. Route full p[j] to that atom.
    eq_mask = (l == u)
    if eq_mask.any():
        lower_mass = torch.where(eq_mask, next_dist, lower_mass)
        upper_mass = torch.where(eq_mask, torch.zeros_like(upper_mass), upper_mass)

    m.scatter_add_(1, l, lower_mass)
    m.scatter_add_(1, u, upper_mass)
    return m


# ---------------------------------------------------------------------------
# Soft target update
# ---------------------------------------------------------------------------
@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    # Polyak averaging on parameters only; buffers (atom support, NoisyLinear
    # epsilons) stay as-is on the target.
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


# ---------------------------------------------------------------------------
# Env helpers (single env for eval; vec env for training)
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
    model: RainbowDuelingC51,
    seed: int,
    episodes: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Greedy eval on a fresh single env with NoisyNet noise DISABLED.

    Matches the ONNX inference path exactly: we set ``noise_disabled=True``
    on every NoisyLinear, run argmax over expected Q, then restore in a
    finally block (so a thrown exception during eval cannot leave the model
    in noise-disabled mode).
    """
    env = make_env(seed + 10_000)
    returns: List[float] = []
    lengths: List[int] = []

    was_training = model.training
    model.eval()
    model.set_all_noise_disabled(True)
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
        model.set_all_noise_disabled(False)
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
        default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_rainbow_vec.pt",
    )
    p.add_argument(
        "--log-dir", type=str,
        default="/home/shuhang/YBJ/RL_test/logs/dqn_rainbow_vec",
    )
    p.add_argument("--eval-every", type=int, default=25_000,
                   help="Eval cadence in transitions (NOT iterations).")
    p.add_argument("--eval-episodes", type=int, default=5)

    # Optimizer / loss knobs
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

    # C51 / NoisyNet knobs
    p.add_argument("--n-atoms", type=int, default=51)
    p.add_argument("--v-min", type=float, default=-5.0)
    p.add_argument("--v-max", type=float, default=5.0)
    p.add_argument("--sigma-init", type=float, default=0.5)

    return p.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda" and not torch.cuda.is_available():
        # Match v1_vec behavior: explicit "cuda" without CUDA soft-fails to CPU.
        print("[warn] --device cuda requested but CUDA unavailable; using CPU.",
              flush=True)
        return torch.device("cpu")
    return torch.device(flag)


def save_checkpoint(
    path: str,
    model: RainbowDuelingC51,
    total_timesteps_trained: int,
    final_eval_reward: float,
    args: argparse.Namespace,
) -> None:
    """Save state_dict on CPU for portability across machines/devices.

    Schema is byte-identical to the single-env Rainbow checkpoint plus
    ``num_envs`` so ``export_rainbow_onnx.py`` works without modification.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": cpu_state,
            "obs_dim": 12,
            "n_actions": 6,
            "arch": "rainbow_dueling_noisy_c51",
            "v_min": float(args.v_min),
            "v_max": float(args.v_max),
            "n_atoms": int(args.n_atoms),
            "n_step": int(args.n_step),
            "alpha": float(args.alpha),
            "beta_start": float(args.beta_start),
            "sigma_init": float(args.sigma_init),
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
        f"v_min={args.v_min} v_max={args.v_max} n_atoms={args.n_atoms} "
        f"sigma_init={args.sigma_init} seed={args.seed}",
        flush=True,
    )

    # --- vec env: per-worker seeds args.seed + i ---
    # `functools.partial(make_env, seed_base + i)` is picklable under spawn
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
        online = RainbowDuelingC51(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            sigma_init=args.sigma_init,
        ).to(device)
        target = RainbowDuelingC51(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            sigma_init=args.sigma_init,
        ).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()
        for p in target.parameters():
            p.requires_grad = False

        # eps=1.5e-4 matches the standard Rainbow Adam epsilon; helps with
        # numerical stability under C51's softmax outputs.
        optimizer = torch.optim.Adam(online.parameters(), lr=float(args.lr), eps=1.5e-4)

        replay = PrioritizedReplayBuffer(
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
            # --- NoisyNet noise resampling for the action-selection forward.
            # Resample once per outer iteration (cheap; one sample of factorized
            # noise per NoisyLinear). Resampling per env-step would be wasted
            # compute -- the same minibatch of N envs uses one noise sample.
            with torch.no_grad():
                online.reset_noise()
                online.set_all_noise_disabled(False)
                obs_t = torch.from_numpy(obs).to(device)
                q_all = online(obs_t)
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
                    # gamma_eff = 0 if terminal in window; else gamma^n_actual.
                    gamma_eff = 0.0 if e_done_term else (gamma ** int(e_n_actual))
                    replay.add(
                        e_obs, int(e_act), float(e_R), e_next_obs,
                        bool(e_done_term), float(gamma_eff),
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
                replay.size >= max(warmup_steps, batch_size)
                and (it % train_freq == 0)
            )
            if should_train:
                # Anneal beta from beta_start to beta_end linearly across training.
                frac = min(1.0, global_step / max(1, args.total_timesteps))
                beta = args.beta_start + (args.beta_end - args.beta_start) * frac

                batch = replay.sample(batch_size, beta, device)

                # Resample NoisyNet noise on BOTH nets for this gradient step.
                # (Action-selection noise above used a separate sample.)
                online.reset_noise()
                target.reset_noise()
                online.set_all_noise_disabled(False)
                target.set_all_noise_disabled(False)

                # --- target distribution (Double DQN + categorical projection) ---
                with torch.no_grad():
                    # Double DQN: action chosen by online net's expected Q at
                    # next_obs, value taken from target net's distribution at
                    # that same action.
                    next_log_p_online = online.forward_train(batch.next_obs)
                    next_q_online = online.expected_q_from_log_p(next_log_p_online)
                    next_actions = next_q_online.argmax(dim=-1)  # (B,)

                    next_log_p_target = target.forward_train(batch.next_obs)
                    next_dist_target = next_log_p_target.exp()
                    # Gather the target distribution at the action selected by
                    # the online net: shape (B, n_atoms).
                    idx = next_actions.view(-1, 1, 1).expand(-1, 1, args.n_atoms)
                    next_dist = next_dist_target.gather(1, idx).squeeze(1)

                    # If the n-step window terminated, gamma_eff is 0 and the
                    # bootstrap mass collapses to r exactly (as expected).
                    m = categorical_projection(
                        next_dist=next_dist,
                        rewards=batch.n_step_returns,
                        gamma_effs=batch.gamma_effs,
                        support=online.support,
                        v_min=args.v_min,
                        v_max=args.v_max,
                        n_atoms=args.n_atoms,
                        delta_z=online.delta_z,
                    )

                # --- predicted distribution at (s, a) under the online net ---
                log_p_all = online.forward_train(batch.obs)  # (B, n_actions, n_atoms)
                a_idx = batch.actions.view(-1, 1, 1).expand(-1, 1, args.n_atoms)
                log_p_sa = log_p_all.gather(1, a_idx).squeeze(1)  # (B, n_atoms)

                # KL(m || p) drops the m*log(m) constant -> cross-entropy
                # -sum_z m[z]*log p[z], weighted by IS weights.
                ce_per_sample = -(m * log_p_sa).sum(dim=-1)  # (B,)
                loss = (batch.weights * ce_per_sample).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                # Priority = per-sample CE loss (rank-equivalent to KL; standard
                # Rainbow). Cap to keep the sum tree numerically sane.
                with torch.no_grad():
                    td_priorities = ce_per_sample.detach().abs().cpu().numpy().astype(np.float64)
                    td_priorities = np.clip(td_priorities, 0.0, 1e3)
                replay.update_priorities(batch.indices, td_priorities)

                # --- soft target update (tau blend each gradient step) ---
                soft_update(target, online, tau)

                grad_steps += 1
                recent_losses.append(float(loss.item()))
                with torch.no_grad():
                    expected_q_pred = online.expected_q_from_log_p(log_p_all)
                    q_taken = expected_q_pred.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
                    recent_qmeans.append(float(q_taken.mean().item()))
                    recent_td.append(float(td_priorities.mean()))

                if grad_steps % 100 == 0:
                    writer.add_scalar("train/loss", float(np.mean(recent_losses)), global_step)
                    writer.add_scalar("train/q_mean", float(np.mean(recent_qmeans)), global_step)
                    writer.add_scalar("train/td_priority_mean", float(np.mean(recent_td)), global_step)
                    writer.add_scalar("train/beta", beta, global_step)
                    writer.add_scalar("train/replay_size", float(replay.size), global_step)

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
                    f"[step {global_step:>9d}] "
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
