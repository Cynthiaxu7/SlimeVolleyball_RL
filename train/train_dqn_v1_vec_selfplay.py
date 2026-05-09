"""Vector-env Dueling DQN trainer with self-play (v1-selfplay).

Forked from ``train_dqn_v1_vec.py``. Same algorithm (Dueling MLP + Double
DQN target + n-step return + soft Polyak target update) and same Q-net
checkpoint format (``arch="dueling_v1_mlp_128_128_64"``) so
``export_onnx.py`` works on the produced checkpoint.

What changes vs v1_vec
----------------------
* SubprocVecEnv -> SelfPlayVecEnv. Each parallel env serves the
  2-player slimevolley API: we pass BOTH the learner's action and the
  opponent's action per step, and we get back BOTH sides' obs.
* Resume: at startup, optionally load an existing checkpoint into the
  online net (and copy to target). Used to continue training a v1 policy
  via self-play instead of reset-from-scratch.
* Opponent pool: in-memory FIFO of past policy snapshots (persisted to
  disk for traceability). At episode boundaries we sample a snapshot per
  env and use its greedy action as the opponent's move. The pool grows
  every ``--snapshot-every`` transitions.
* Two evals per cadence:
    1. eval-vs-baseline (single-env, slimevolley's built-in baseline) —
       same code path as v1_vec; this is the TRUE skill metric we use to
       gate "best" checkpoint saves.
    2. eval-vs-pool (sample 5 opponents from the pool, full episodes) —
       a sanity check that we still beat past selves; logged but not
       used for gating.

Algorithm preservation
----------------------
* DuelingMLPv1 architecture identical (``dueling_v1_mlp_128_128_64``).
* Replay buffer stores AGENT-side transitions only (we never train on the
  opponent's view; the opponent is a frozen snapshot).
* Per-env n-step buffer is reset on done[i] independently, same as v1_vec.

Notes for forks (v0 / Rainbow)
------------------------------
* The "online net architecture" lives in three places: (a) the
  ``DuelingMLPv1`` class below, (b) the ``factory_fn`` closure passed to
  ``OpponentPool``, (c) the ``arch`` field in ``save_checkpoint``. v0
  forks should pick a different ``arch`` string; Rainbow forks should
  swap in their distributional Q-net but keep the same factory pattern.
* The replay buffer has no algorithm-specific assumptions baked in;
  Rainbow can replace ``ReplayBuffer`` with ``PrioritizedReplay`` without
  touching the self-play loop.
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
from typing import Deque, List, Optional, Tuple

import gym  # noqa: F401  -- registers SlimeVolley-v0 alongside slimevolleygym
import numpy as np
import slimevolleygym  # noqa: F401  -- registers "SlimeVolley-v0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Make `train/` importable for the spawned children + for `python train/...` runs.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# Reverse of ACTION_TABLE: (forward, backward, jump) -> Discrete index.
# Used to map slimevolley's BaselinePolicy [bin, bin, bin] output -> Discrete(6).
INVERSE_ACTION_TABLE = {
    (0, 0, 0): 0, (1, 0, 0): 1, (1, 0, 1): 2,
    (0, 0, 1): 3, (0, 1, 1): 4, (0, 1, 0): 5,
}


class BaselineOpponentAdapter:
    """Wraps slimevolleygym's built-in 120-param tanh-RNN baseline as a 'pool
    member' look-alike. Each instance owns an INDEPENDENT recurrent hidden
    state (so per-env instances don't cross-contaminate).

    Important obs-scale detail: the env's INTERNAL auto-baseline (single-arg
    env.step path) calls ``self.policy.predict(obs)`` on the SAME normalized
    obs the agent sees — see SlimeVolleyEnv.step in slimevolleygym/slimevolley.py.
    So the BaselinePolicy weights were effectively trained against normalized
    inputs. Multiplying back by 10 here saturates the tanh and crippled the
    baseline (a long-standing bug; trained agents looked stronger than they
    actually were against the real baseline).
    """

    def __init__(self) -> None:
        from slimevolleygym.slimevolley import BaselinePolicy
        self.policy = BaselinePolicy()

    def reset(self) -> None:
        self.policy.reset()

    def predict_action(self, obs12_normalized: np.ndarray) -> int:
        binary = self.policy.predict(obs12_normalized)
        return INVERSE_ACTION_TABLE.get(
            (int(binary[0]), int(binary[1]), int(binary[2])), 0,
        )

# Sibling-module imports.
from train_dqn_v1 import ACTION_TABLE, DiscreteSlimeWrapper  # noqa: E402, F401
from selfplay import OpponentPool  # noqa: E402
from selfplay_vec_env import SelfPlayVecEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Q-net (architecturally identical to v1's DuelingMLPv1; same `arch` string)
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
# Replay buffer (single-process: only the master writes/reads)
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
    """Plain ring buffer with uniform sampling. Stores n-step transitions.

    AGENT-SIDE ONLY. The opponent's transitions are never stored; the
    opponent is a frozen snapshot whose policy we don't train.
    """

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
# Per-env n-step buffer (identical to v1_vec)
# ---------------------------------------------------------------------------
class NStepBuffer:
    """Per-env FIFO of one-step transitions emitting n-step entries.

    Emission rules:
      * Non-terminal step + FIFO full: emit oldest with n_actual = n,
        done_terminal = False, next_obs = s_{t+n}.
      * Terminal step: append the terminal one-step record FIRST, then flush
        every remaining FIFO entry with done_terminal = True,
        n_actual = remaining-horizon, next_obs = s_terminal.
    """

    def __init__(self, n: int, gamma: float) -> None:
        assert n >= 1
        self.n = n
        self.gamma = gamma
        self._fifo: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n)

    def reset(self) -> None:
        self._fifo.clear()

    def _accumulate(self, items, start: int) -> float:
        R, discount = 0.0, 1.0
        for j in range(start, len(items)):
            R += discount * items[j][2]
            discount *= self.gamma
        return R

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool,
            ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]]:
        self._fifo.append((obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)))
        emitted: List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]] = []

        if done:
            fifo_list = list(self._fifo)
            window_next_obs = fifo_list[-1][3]  # s_terminal
            for k in range(len(fifo_list)):
                obs_k, act_k = fifo_list[k][0], fifo_list[k][1]
                emitted.append((obs_k, act_k, self._accumulate(fifo_list, k),
                                window_next_obs, True, len(fifo_list) - k))
            self._fifo.clear()
            return emitted

        if len(self._fifo) == self.n:
            fifo_list = list(self._fifo)
            obs_0, act_0 = fifo_list[0][0], fifo_list[0][1]
            emitted.append((obs_0, act_0, self._accumulate(fifo_list, 0),
                            fifo_list[-1][3], False, self.n))
            self._fifo.popleft()
            return emitted

        return emitted


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------
def make_raw_env(seed: int):
    """Module-level RAW slimevolley env factory (NO DiscreteSlimeWrapper).

    SelfPlayVecEnv workers handle the Discrete(6) -> MultiBinary(3)
    conversion themselves via ACTION_TABLE; we keep the raw env so we
    have direct access to ``env.step(action, otherAction=...)``.

    .unwrapped strips OrderEnforcing/TimeLimit which don't forward the
    `otherAction` kwarg. slimevolley has its own t_limit=3000, so it's safe.
    """
    env = gym.make("SlimeVolley-v0").unwrapped
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def make_eval_env(seed: int):
    """Single-env eval factory: wraps with DiscreteSlimeWrapper so the
    single-arg ``env.step(int)`` invokes the built-in baseline opponent.

    This is the SAME path used by v1 / v1_vec for eval-vs-baseline.
    """
    env = gym.make("SlimeVolley-v0")
    env = DiscreteSlimeWrapper(env)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def linear_epsilon(step: int, total_steps: int,
                   start: float, end: float, fraction: float) -> float:
    decay_steps = max(1, int(total_steps * fraction))
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def evaluate_vs_baseline(model: DuelingMLPv1, seed: int, episodes: int,
                         device: torch.device) -> Tuple[float, float, float]:
    """Greedy eval vs slimevolley's built-in baseline (single env).

    Identical to v1_vec's eval — uses ``DiscreteSlimeWrapper`` over a fresh
    single env; the wrapper's single-arg step delegates to the env's
    internal baseline opponent. Returns ``(mean_R, std_R, mean_len)``.
    """
    env = make_eval_env(seed + 10_000)
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


def evaluate_vs_pool(model: DuelingMLPv1, pool: OpponentPool,
                     seed: int, n_opponents: int, device: torch.device,
                     rng: np.random.Generator) -> Tuple[float, float]:
    """Greedy eval vs sampled pool opponents (single raw env, full episodes).

    Returns ``(mean_R, std_R)`` averaged over ``n_opponents`` episodes,
    each played to completion against a different sampled snapshot.
    Returns ``(nan, nan)`` if the pool is empty.
    """
    if len(pool) == 0:
        return float("nan"), float("nan")

    returns: List[float] = []
    model.eval()
    with torch.no_grad():
        for ep_i in range(n_opponents):
            opp = pool.sample(rng)
            if opp is None:  # extremely unlikely after the len check above
                break
            env = make_raw_env(seed + 20_000 + ep_i)
            obs = env.reset()
            # noop-noop step to read both sides' initial obs (same trick as
            # SelfPlayVecEnv worker, see selfplay_vec_env._do_reset).
            obs2, _r, done, info = env.step(ACTION_TABLE[0],
                                            otherAction=ACTION_TABLE[0])
            if done:
                # Vanishingly rare; just skip this episode.
                env.close()
                continue
            obs = np.asarray(obs2, dtype=np.float32)
            other_obs = np.asarray(info["otherObs"], dtype=np.float32)
            ep_return = 0.0
            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                opp_t = torch.from_numpy(other_obs).unsqueeze(0).to(device)
                my_idx = int(model(obs_t).argmax(dim=-1).item())
                opp_idx = int(opp(opp_t).argmax(dim=-1).item())
                obs_n, reward, done, info = env.step(
                    ACTION_TABLE[my_idx], otherAction=ACTION_TABLE[opp_idx],
                )
                ep_return += float(reward)
                obs = np.asarray(obs_n, dtype=np.float32)
                other_obs = np.asarray(info["otherObs"], dtype=np.float32)
            env.close()
            returns.append(ep_return)
    model.train()
    if not returns:
        return float("nan"), float("nan")
    return float(np.mean(returns)), float(np.std(returns))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Identical to v1_vec ----------------------------------------------------
    p.add_argument("--total-timesteps", type=int, default=5_000_000,
                   help="Total transitions ingested across ALL envs.")
    p.add_argument("--num-envs", type=int, default=16,
                   help="Number of parallel envs (one subprocess each).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-freq", type=int, default=4,
                   help="Env steps PER WORKER between gradient steps.")
    p.add_argument("--soft-tau", type=float, default=0.005)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=2.5e-4)
    # Self-play continuation tends to forget baseline-specific play. Mixing
    # the built-in baseline as a sometimes-opponent keeps that distribution
    # in the replay buffer. 0.0 = pure self-play (orig spec), 1.0 = pure
    # baseline opponent (no self-play happens). 0.4 is a reasonable mix.
    p.add_argument("--baseline-opponent-prob", type=float, default=0.0,
                   help="Probability that a fresh episode's opponent is the "
                        "scripted baseline rather than a pool snapshot.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--save-path", type=str,
                   default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_v1_vec_selfplay.pt")
    p.add_argument("--log-dir", type=str,
                   default="/home/shuhang/YBJ/RL_test/logs/dqn_v1_vec_selfplay")
    p.add_argument("--eval-every", type=int, default=25_000,
                   help="Eval cadence in transitions.")
    p.add_argument("--eval-episodes", type=int, default=5)
    # New for self-play ------------------------------------------------------
    p.add_argument("--resume-from", type=str,
                   default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_v1_vec.pt",
                   help=("Path to an existing checkpoint to resume from. If "
                         "the file is missing we silently start from scratch."))
    p.add_argument("--pool-dir", type=str,
                   default="/home/shuhang/YBJ/RL_test/checkpoints/pool_v1_selfplay/",
                   help="Directory to persist opponent snapshots.")
    p.add_argument("--snapshot-every", type=int, default=100_000,
                   help=("Add a frozen snapshot to the opponent pool every "
                         "this many transitions across all envs."))
    p.add_argument("--pool-max-size", type=int, default=30,
                   help="FIFO cap on the in-memory opponent pool.")
    p.add_argument("--pool-eval-opponents", type=int, default=5,
                   help="Number of pool opponents sampled per eval-vs-pool.")
    p.add_argument("--seed-pool-from", nargs="*", default=[],
                   help="Optional list of checkpoint paths to seed the OpponentPool "
                        "with (in addition to the resume snapshot). Architecture must "
                        "match this trainer's model.")
    return p.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but CUDA unavailable; using CPU.",
              flush=True)
        return torch.device("cpu")
    return torch.device(flag)


def save_checkpoint(
    path: str, model: DuelingMLPv1, total_timesteps_trained: int,
    final_eval_reward: float, n_step: int, soft_tau: float, num_envs: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": 12,
        "n_actions": 6,
        # Unchanged so export_onnx.py works on this checkpoint as-is.
        "arch": "dueling_v1_mlp_128_128_64",
        "total_timesteps_trained": int(total_timesteps_trained),
        "final_eval_reward": float(final_eval_reward),
        "n_step": int(n_step),
        "double": True,
        "soft_tau": float(soft_tau),
        "num_envs": int(num_envs),
        "selfplay": True,
    }, path)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
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
    rng = np.random.default_rng(args.seed)  # for opponent sampling

    device = resolve_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    num_envs = int(args.num_envs)
    if args.total_timesteps % num_envs != 0:
        args.total_timesteps = (args.total_timesteps // num_envs) * num_envs
    n_iterations = args.total_timesteps // num_envs

    print(
        f"[config] device={device} num_envs={num_envs} "
        f"total_timesteps={args.total_timesteps} (transitions across all envs) "
        f"iterations={n_iterations} batch_size={args.batch_size} "
        f"train_freq={args.train_freq} (per-worker) n_step={args.n_step} "
        f"soft_tau={args.soft_tau} lr={args.lr} seed={args.seed} "
        f"snapshot_every={args.snapshot_every} pool_max_size={args.pool_max_size}",
        flush=True,
    )

    # --- nets, optim, buffer ---
    obs_dim, n_actions = 12, 6  # known from spec; we double-check via vec_env below
    online = DuelingMLPv1(obs_dim, n_actions).to(device)
    target = DuelingMLPv1(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=float(args.lr))
    buffer = ReplayBuffer(capacity=200_000, obs_dim=obs_dim)

    # --- resume (BEFORE building the pool, so we can seed the pool with this) ---
    prior_eval_reward: Optional[float] = None
    resumed = False
    if args.resume_from and os.path.isfile(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        online.load_state_dict(ckpt["model_state_dict"], strict=True)
        prior_eval_reward = float(ckpt.get("final_eval_reward", float("nan")))
        print(f"[resume] loaded from {args.resume_from}, "
              f"prior eval_reward={prior_eval_reward}", flush=True)
        resumed = True
    else:
        if args.resume_from:
            print(f"[resume] {args.resume_from} not found; starting from scratch.",
                  flush=True)

    target.load_state_dict(online.state_dict())
    target.eval()

    # --- opponent pool ---
    # factory_fn must build a fresh module on `device` so eval-mode inference
    # in each iteration runs on the same device as the online net (avoids
    # per-step cross-device copies).
    def _opp_factory() -> nn.Module:
        return DuelingMLPv1(obs_dim, n_actions).to(device)

    pool = OpponentPool(
        pool_dir=args.pool_dir,
        factory_fn=_opp_factory,
        device=str(device),
        max_size=int(args.pool_max_size),
    )
    # Load any prior snapshots BEFORE adding the resume snapshot so we don't
    # double-count the same checkpoint if a previous run already snapshotted it.
    pool.load_existing()
    if resumed:
        # Seed the pool with the resumed checkpoint so we ALWAYS have an
        # opponent (even before the first --snapshot-every threshold).
        pool.add(online.state_dict(), step=0)
    # Optional external seed checkpoints (e.g., v0, v1 cold, prior selfplay
    # runs). These are kept in-memory only; they don't write into pool_dir.
    for seed_path in args.seed_pool_from:
        if os.path.isfile(seed_path):
            pool.add_seed_checkpoint(seed_path, label=os.path.basename(seed_path))
        else:
            print(f"[seed] WARNING: {seed_path} not found, skipping", flush=True)
    print(f"[pool] dir={args.pool_dir} initial_size={len(pool)}", flush=True)

    # --- vec env ---
    seed_base = int(args.seed)
    env_fns = [functools.partial(make_raw_env, seed_base + i) for i in range(num_envs)]
    vec_env = SelfPlayVecEnv(env_fns)

    try:
        # Sanity-check spaces (raw env: Box(12) obs, MultiBinary(3) actions —
        # the discrete wrapping is on the trainer's side, not the env's).
        env_obs_dim = int(np.array(vec_env.observation_space.shape).prod())
        assert env_obs_dim == 12, f"unexpected env obs_dim={env_obs_dim}"

        # --- hyperparams ---
        gamma = float(args.gamma)
        batch_size = int(args.batch_size)
        warmup_steps = 1_000
        train_freq = int(args.train_freq)
        grad_clip_norm = 10.0
        eps_start, eps_end, eps_fraction = 1.0, 0.02, 0.30
        n_step = int(args.n_step)
        soft_tau = float(args.soft_tau)

        nstep_bufs: List[NStepBuffer] = [
            NStepBuffer(n=n_step, gamma=gamma) for _ in range(num_envs)
        ]

        # --- run-state ---
        grad_steps = 0
        best_eval_reward = -float("inf")
        recent_losses: deque[float] = deque(maxlen=100)
        recent_qmeans: deque[float] = deque(maxlen=100)
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        recent_ep_returns: deque[float] = deque(maxlen=100)
        recent_ep_lengths: deque[float] = deque(maxlen=100)

        # Per-env opponent assignments. Each slot is one of:
        #   - nn.Module (pool snapshot) — we forward(obs) and argmax
        #   - BaselineOpponentAdapter — we call .predict_action(obs)
        #   - None — NOOP fallback (only if pool is empty AND baseline is off)
        # Refreshed whenever that env hits done.
        baseline_opponents = [BaselineOpponentAdapter() for _ in range(num_envs)]

        def _sample_opponent(env_idx: int):
            if rng.random() < args.baseline_opponent_prob:
                baseline_opponents[env_idx].reset()
                return baseline_opponents[env_idx]
            return pool.sample(rng)

        opp_assignments: List[object] = [
            _sample_opponent(i) for i in range(num_envs)
        ]

        # initial reset; per-worker seeds were applied at worker init
        obs, opp_obs = vec_env.reset()  # each (N, 12)

        global_step = 0
        last_eval_step = 0
        last_snapshot_step = 0
        start_time = time.time()

        for it in range(1, n_iterations + 1):
            epsilon = linear_epsilon(global_step, args.total_timesteps,
                                     eps_start, eps_end, eps_fraction)

            # --- vectorized epsilon-greedy (agent side) ---
            if random.random() < epsilon:
                actions = np.random.randint(0, n_actions,
                                            size=num_envs).astype(np.int64)
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).to(device)
                    q_all = online(obs_t)
                    actions = q_all.argmax(dim=-1).cpu().numpy().astype(np.int64)

            # --- opponent actions (greedy from each env's assigned snapshot) ---
            # We do one forward per env; opponents may differ across envs so
            # we don't trivially batch them. With pool size ~30 and many envs
            # likely sharing the same snapshot, a smarter dispatcher could
            # group-by-opponent and forward each group once — premature
            # optimization for now (O(N) tiny forwards on 12-dim inputs is
            # cheap relative to env step time).
            opp_actions = np.zeros(num_envs, dtype=np.int64)
            with torch.no_grad():
                for i in range(num_envs):
                    opp = opp_assignments[i]
                    if opp is None:
                        opp_actions[i] = 0  # NOOP fallback (pool empty)
                    elif isinstance(opp, BaselineOpponentAdapter):
                        opp_actions[i] = opp.predict_action(opp_obs[i])
                    else:
                        opp_t = torch.from_numpy(opp_obs[i]).unsqueeze(0).to(device)
                        opp_actions[i] = int(opp(opp_t).argmax(dim=-1).item())

            next_obs, next_opp_obs, rewards, dones, infos = vec_env.step(
                actions, opp_actions,
            )
            global_step += num_envs

            # --- per-env n-step ingestion (AGENT side only) ---
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
                    buffer.add(e_obs, e_act, e_R, e_next_obs, e_done_term, e_n_actual)

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
                    # Sample a fresh opponent for the new episode in this env.
                    opp_assignments[i] = _sample_opponent(i)

            # advance both sides' obs for the next iteration
            obs = next_obs
            opp_obs = next_opp_obs

            # --- learn ---
            should_train = (
                buffer.size >= max(warmup_steps, batch_size)
                and (it % train_freq == 0)
            )
            if should_train:
                batch = buffer.sample(batch_size, device)

                with torch.no_grad():
                    next_q_online = online(batch.next_obs)
                    next_actions = next_q_online.argmax(dim=-1, keepdim=True)
                    next_q_target_all = target(batch.next_obs)
                    next_q = next_q_target_all.gather(1, next_actions).squeeze(1)

                    discount = torch.pow(
                        torch.full_like(batch.rewards, gamma),
                        batch.n_actual.to(batch.rewards.dtype),
                    )
                    td_target = batch.rewards + discount * next_q * (1.0 - batch.dones)

                q_all = online(batch.obs)
                q_pred = q_all.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q_pred, td_target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                soft_update(target, online, soft_tau)

                grad_steps += 1
                recent_losses.append(float(loss.item()))
                recent_qmeans.append(float(q_pred.mean().item()))

                if grad_steps % 100 == 0:
                    writer.add_scalar("train/loss", float(np.mean(recent_losses)), global_step)
                    writer.add_scalar("train/q_mean", float(np.mean(recent_qmeans)), global_step)
                    writer.add_scalar("train/epsilon", epsilon, global_step)

            # --- periodic snapshot ---
            if (global_step - last_snapshot_step) >= int(args.snapshot_every):
                last_snapshot_step = global_step
                pool.add(online.state_dict(), step=global_step)
                writer.add_scalar("pool/size", len(pool), global_step)
                print(f"[snapshot] step={global_step} pool_size={len(pool)}",
                      flush=True)

            # --- periodic eval ---
            if (global_step - last_eval_step) >= int(args.eval_every):
                last_eval_step = global_step
                base_mean, base_std, base_len = evaluate_vs_baseline(
                    online, args.seed, args.eval_episodes, device,
                )
                pool_mean, pool_std = evaluate_vs_pool(
                    online, pool, args.seed, int(args.pool_eval_opponents),
                    device, rng,
                )
                writer.add_scalar("eval_baseline/reward_mean", base_mean, global_step)
                writer.add_scalar("eval_baseline/reward_std", base_std, global_step)
                writer.add_scalar("eval_baseline/episode_length_mean", base_len, global_step)
                if not np.isnan(pool_mean):
                    writer.add_scalar("eval_pool/reward_mean", pool_mean, global_step)
                    writer.add_scalar("eval_pool/reward_std", pool_std, global_step)
                writer.add_scalar("pool/size", len(pool), global_step)

                elapsed = time.time() - start_time
                sps = global_step / max(elapsed, 1e-6)
                rolling_R = (float(np.mean(recent_ep_returns))
                             if recent_ep_returns else float("nan"))
                pool_str = (f"pool_R={pool_mean:+.3f}+-{pool_std:.3f}"
                            if not np.isnan(pool_mean) else "pool_R=n/a")
                print(
                    f"[step {global_step:>9d}] eps={epsilon:.3f} "
                    f"base_R={base_mean:+.3f}+-{base_std:.3f} "
                    f"base_len={base_len:.1f} {pool_str} "
                    f"train_R={rolling_R:+.3f} sps={sps:.1f} "
                    f"grad_steps={grad_steps} pool_size={len(pool)}",
                    flush=True,
                )
                # Save checkpoint on best eval-vs-baseline (the true skill metric).
                if base_mean > best_eval_reward:
                    best_eval_reward = base_mean
                    save_checkpoint(
                        args.save_path, online, global_step, base_mean,
                        n_step, soft_tau, num_envs,
                    )

        # --- final eval + checkpoint ---
        final_base_mean, final_base_std, final_base_len = evaluate_vs_baseline(
            online, args.seed, args.eval_episodes, device,
        )
        final_pool_mean, final_pool_std = evaluate_vs_pool(
            online, pool, args.seed, int(args.pool_eval_opponents),
            device, rng,
        )
        writer.add_scalar("eval_baseline/reward_mean", final_base_mean, args.total_timesteps)
        writer.add_scalar("eval_baseline/reward_std", final_base_std, args.total_timesteps)
        writer.add_scalar("eval_baseline/episode_length_mean", final_base_len, args.total_timesteps)
        if not np.isnan(final_pool_mean):
            writer.add_scalar("eval_pool/reward_mean", final_pool_mean, args.total_timesteps)
            writer.add_scalar("eval_pool/reward_std", final_pool_std, args.total_timesteps)
        writer.add_scalar("pool/size", len(pool), args.total_timesteps)

        final_pool_str = (
            f"pool_R={final_pool_mean:+.3f}+-{final_pool_std:.3f}"
            if not np.isnan(final_pool_mean) else "pool_R=n/a"
        )
        print(
            f"[final {args.total_timesteps}] "
            f"base_R={final_base_mean:+.3f}+-{final_base_std:.3f} "
            f"base_len={final_base_len:.1f} {final_pool_str} "
            f"grad_steps={grad_steps} pool_size={len(pool)}",
            flush=True,
        )
        save_checkpoint(
            args.save_path, online, args.total_timesteps, final_base_mean,
            n_step, soft_tau, num_envs,
        )
        writer.close()

    finally:
        try:
            vec_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    # `spawn` requires that the entrypoint be guarded by __main__ so children
    # can re-import this module without re-running the trainer. SelfPlayVecEnv
    # gets its own spawn context internally; this guard is the standard
    # belt-and-braces.
    try:
        _mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
