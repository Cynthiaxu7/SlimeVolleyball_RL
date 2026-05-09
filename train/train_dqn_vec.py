"""Vector-env Dueling DQN trainer (v0, vanilla) for SlimeVolley-v0.

Vanilla DQN baseline for ablation against ``train_dqn_v1_vec.py``:
Dueling-MLP architecture ONLY. No Double, no N-step, hard target copy.
Same vec-env infrastructure (SubprocVecEnv, N parallel envs feeding a
single shared replay buffer and a single GPU/CPU learner).

Algorithmic deltas vs v1_vec:
  * 1-step transitions stored directly (no per-env NStepBuffer).
  * Target = r + gamma * max_a Q_target(s', a) * (1 - done) — no online net
    involvement in the bootstrap.
  * Hard target copy every ``--target-update-every`` grad steps (no Polyak).

Q-net architecture is byte-for-byte identical to ``DuelingMLPv1`` (renamed
``DuelingMLPv0`` here for clarity); checkpoint metadata records the v0 arch
string so ``export_onnx.py`` can dispatch on it.

Key knobs (see ``parse_args`` for full list):

    --num-envs            N parallel envs (default 16).
    --total-timesteps     Total transitions ingested across ALL envs (NOT per-worker).
                          Outer loop runs total_timesteps / num_envs iterations.
                          Default 5_000_000 (~312_500 outer iterations at N=16).
    --train-freq          Env steps PER WORKER between gradient steps (default 4).
                          Wall-clock grad steps == total_timesteps / num_envs / train_freq.
    --batch-size          256 (raised from 64 because data ingestion is N-fold).
    --target-update-every Hard-copy cadence in grad steps (default 500).
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
from typing import List, Tuple

import gym  # noqa: F401  -- old gym 0.19; needed for env registration via slimevolleygym
import numpy as np
import slimevolleygym  # noqa: F401  -- registers SlimeVolley-v0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Make `train/` importable regardless of CWD (so both
# `python train/train_dqn_vec.py` from the repo root and
# `python train_dqn_vec.py` from inside `train/` work). We also need
# this for the `spawn` child re-import to find sibling modules.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Sibling-module imports: reuse the wrapper / action table from v1 to keep
# the env stack identical across ablations.
from train_dqn_v1 import ACTION_TABLE, DiscreteSlimeWrapper  # noqa: E402, F401
from vec_env import SubprocVecEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Q-net (architecturally identical to v1's DuelingMLPv1; only the class name
# and the checkpoint `arch` string differ).
# ---------------------------------------------------------------------------
class DuelingMLPv0(nn.Module):
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


class PlainQMLPv0(nn.Module):
    """Vanilla single-head Q net (no Dueling decomposition). Used by the
    --no-dueling ablation cell to isolate Dueling's contribution. Same trunk
    + same total parameter count as DuelingMLPv0 so the comparison is
    architecture-decomposition vs raw-Q, not capacity.
    """

    def __init__(self, obs_dim: int = 12, n_actions: int = 6) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.trunk(obs))


# ---------------------------------------------------------------------------
# Replay buffer (single-process: only the master writes/reads). Plain 1-step
# transitions only — schema simplified vs v1_vec (no n_actual field).
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor       # immediate one-step reward r_t
    next_obs: torch.Tensor      # s_{t+1}
    dones: torch.Tensor         # 1.0 if s_{t+1} is terminal


class ReplayBuffer:
    """Plain ring buffer with uniform sampling over 1-step transitions."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool) -> None:
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
# Env helpers (single env for eval; vec env for training)
# ---------------------------------------------------------------------------
def make_env(seed: int):
    """Module-level factory: SubprocVecEnv must be able to pickle this when
    using the spawn start method, so we cannot use a closure-bound lambda
    that references ``args``. The trainer wraps this with a per-worker seed
    via a default-arg trick (see main)."""
    env = gym.make("SlimeVolley-v0")
    env = DiscreteSlimeWrapper(env)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def linear_epsilon(step: int, total_steps: int,
                   start: float, end: float, fraction: float) -> float:
    decay_steps = max(1, int(total_steps * fraction))
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def evaluate(model: DuelingMLPv0, seed: int, episodes: int,
             device: torch.device) -> Tuple[float, float, float]:
    """Greedy eval on a fresh single env. Returns (mean_R, std_R, mean_len)."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--total-timesteps", type=int, default=5_000_000,
        help=("Total transitions ingested across ALL envs (apples-to-apples "
              "with the single-env trainers). Outer loop iterations = "
              "total-timesteps / num-envs."),
    )
    p.add_argument("--num-envs", type=int, default=16,
                   help="Number of parallel envs (one subprocess each).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--train-freq", type=int, default=4,
        help=("Env steps PER WORKER between gradient steps. With num-envs=N "
              "and train-freq=K, one grad step happens every N*K transitions."),
    )
    p.add_argument(
        "--target-update-every", type=int, default=500,
        help=("Hard target-network copy cadence, measured in gradient steps. "
              "Vanilla DQN uses a hard refresh rather than Polyak averaging."),
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--no-dueling", action="store_true", default=False,
                   help="Use a plain single-head Q net (PlainQMLPv0) instead "
                        "of the Dueling decomposition. Used for the Dueling "
                        "vs no-Dueling ablation cell.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--save-path", type=str,
                   default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_vec.pt")
    p.add_argument("--log-dir", type=str,
                   default="/home/shuhang/YBJ/RL_test/logs/dqn_vec")
    p.add_argument("--eval-every", type=int, default=25_000,
                   help="Eval cadence in transitions (NOT iterations).")
    p.add_argument("--eval-episodes", type=int, default=5)
    return p.parse_args()


def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda" and not torch.cuda.is_available():
        # Match v0/v1 behavior: explicit "cuda" without CUDA is an error case
        # but we soft-fail to CPU to keep CI/devboxes happy. The header logs
        # the resolved device.
        print("[warn] --device cuda requested but CUDA unavailable; using CPU.",
              flush=True)
        return torch.device("cpu")
    return torch.device(flag)


def save_checkpoint(
    path: str, model: nn.Module, total_timesteps_trained: int,
    final_eval_reward: float, target_update_every: int, num_envs: int,
    no_dueling: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": 12,
        "n_actions": 6,
        # v0 arch string: vanilla DQN ablation. Plain Q is the --no-dueling
        # ablation cell; otherwise byte-for-byte identical to v1's DuelingMLP.
        "arch": ("plain_q_v0_mlp_128_128_64" if no_dueling
                 else "dueling_v0_mlp_128_128_64"),
        "total_timesteps_trained": int(total_timesteps_trained),
        "final_eval_reward": float(final_eval_reward),
        "double": False,
        "target_update_every": int(target_update_every),
        "num_envs": int(num_envs),
    }, path)


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
        # Round down to a multiple of num_envs so iterations align.
        args.total_timesteps = (args.total_timesteps // num_envs) * num_envs
    n_iterations = args.total_timesteps // num_envs

    print(
        f"[config] device={device} num_envs={num_envs} "
        f"total_timesteps={args.total_timesteps} (transitions across all envs) "
        f"iterations={n_iterations} batch_size={args.batch_size} "
        f"train_freq={args.train_freq} (per-worker) "
        f"target_update_every={args.target_update_every} "
        f"lr={args.lr} seed={args.seed}",
        flush=True,
    )

    # --- vec env: per-worker seeds args.seed + i ---
    # NOTE: with multiprocessing's `spawn` start method (which we use to be
    # safe with numpy/torch), `env_fns` are pickled and shipped to the child
    # processes. A closure-bound lambda defined inside `main()` is NOT
    # picklable, so we use `functools.partial` over the module-level
    # `make_env` instead.
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
        net_cls = PlainQMLPv0 if args.no_dueling else DuelingMLPv0
        online = net_cls(obs_dim, n_actions).to(device)
        target = net_cls(obs_dim, n_actions).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()
        optimizer = torch.optim.Adam(online.parameters(), lr=float(args.lr))
        # 200k capacity: with N-fold ingestion the buffer rolls fast; this
        # gives us a few minutes of recent history at typical throughput.
        buffer = ReplayBuffer(capacity=200_000, obs_dim=obs_dim)

        # --- hyperparams ---
        gamma = float(args.gamma)
        batch_size = int(args.batch_size)
        warmup_steps = 1_000          # transitions in buffer before first grad
        train_freq = int(args.train_freq)  # env steps PER WORKER between grads
        grad_clip_norm = 10.0
        eps_start, eps_end, eps_fraction = 1.0, 0.02, 0.30
        target_update_every = int(args.target_update_every)

        # --- run-state ---
        grad_steps = 0
        best_eval_reward = -float("inf")
        recent_losses: deque[float] = deque(maxlen=100)
        recent_qmeans: deque[float] = deque(maxlen=100)
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
            # --- per-iteration epsilon (linear; same schedule as v0/v1, but
            # parameterized over total_timesteps so the curve aligns) ---
            epsilon = linear_epsilon(global_step, args.total_timesteps,
                                     eps_start, eps_end, eps_fraction)

            # --- vectorized epsilon-greedy action selection ---
            if random.random() < epsilon:
                # Pure random per env (cheap; avoids forwarding a batch we
                # won't use). We keep epsilon as a SINGLE-COIN switch (rather
                # than per-env) for simplicity; works fine in practice.
                actions = np.random.randint(0, n_actions, size=num_envs).astype(np.int64)
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).to(device)
                    q_all = online(obs_t)
                    actions = q_all.argmax(dim=-1).cpu().numpy().astype(np.int64)

            next_obs, rewards, dones, infos = vec_env.step(actions)
            global_step += num_envs

            # --- per-env 1-step ingestion ---
            # IMPORTANT: when dones[i] is True, next_obs[i] is the AUTO-RESET
            # post-reset obs from worker i. The TRUE terminal obs for the
            # transition is in infos[i]["terminal_observation"]. We must use
            # that for the buffer's `next_obs` field; otherwise the bootstrap
            # target would be conditioned on a state from a totally unrelated
            # episode (but we also mask with `done=True`, so the bootstrap is
            # zeroed anyway — this is still the correct stored value).
            for i in range(num_envs):
                if dones[i]:
                    term_obs = infos[i].get("terminal_observation", next_obs[i])
                    term_obs = np.asarray(term_obs, dtype=np.float32)
                    buffer.add(
                        obs[i], int(actions[i]), float(rewards[i]),
                        term_obs, True,
                    )
                else:
                    buffer.add(
                        obs[i], int(actions[i]), float(rewards[i]),
                        np.asarray(next_obs[i], dtype=np.float32), False,
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

            # advance obs to the auto-reset obs returned by the vec env
            obs = next_obs

            # --- learn (one grad step every `train_freq` ITERATIONS, i.e.
            # every `train_freq` env-steps PER WORKER == train_freq*num_envs
            # transitions) ---
            should_train = (
                buffer.size >= max(warmup_steps, batch_size)
                and (it % train_freq == 0)
            )
            if should_train:
                batch = buffer.sample(batch_size, device)

                # Vanilla DQN target: y = r + gamma * max_a Q_target(s', a) * (1 - done).
                # NO online net involvement in the bootstrap (cf. Double DQN).
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

                # Hard target copy on a fixed cadence (vs v1's Polyak averaging).
                if grad_steps % target_update_every == 0:
                    target.load_state_dict(online.state_dict())

                if grad_steps % 100 == 0:
                    writer.add_scalar("train/loss", float(np.mean(recent_losses)), global_step)
                    writer.add_scalar("train/q_mean", float(np.mean(recent_qmeans)), global_step)
                    writer.add_scalar("train/epsilon", epsilon, global_step)

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
                    save_checkpoint(
                        args.save_path, online, global_step, eval_mean,
                        target_update_every, num_envs,
                        no_dueling=bool(args.no_dueling),
                    )

        # --- final eval + checkpoint ---
        final_mean, final_std, final_len = evaluate(
            online, args.seed, args.eval_episodes, device,
        )
        writer.add_scalar("eval/reward_mean", final_mean, args.total_timesteps)
        writer.add_scalar("eval/reward_std", final_std, args.total_timesteps)
        writer.add_scalar("eval/episode_length_mean", final_len, args.total_timesteps)
        print(
            f"[final {args.total_timesteps}] "
            f"eval_R={final_mean:+.3f}+-{final_std:.3f} eval_len={final_len:.1f} "
            f"grad_steps={grad_steps}",
            flush=True,
        )
        save_checkpoint(
            args.save_path, online, args.total_timesteps, final_mean,
            target_update_every, num_envs,
            no_dueling=bool(args.no_dueling),
        )
        writer.close()

    finally:
        # Always shut down workers, even on KeyboardInterrupt or exception.
        try:
            vec_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    # `spawn` requires that the entrypoint be guarded by __main__ so children
    # can re-import this module without re-running the trainer. Setting the
    # start method here is a no-op if already set elsewhere; we rely on
    # SubprocVecEnv getting its own spawn context internally.
    try:
        _mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # Start method already set by the embedding process; harmless.
        pass
    main()
