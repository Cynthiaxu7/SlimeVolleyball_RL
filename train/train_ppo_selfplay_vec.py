"""Vector-env PPO trainer with self-play (ppo-selfplay-vec).

Forked from ``train_dqn_v1_vec_selfplay.py``. Replaces Dueling-DQN with PPO
(actor-critic + GAE + clipped surrogate). Same self-play scaffolding:
``SelfPlayVecEnv`` for parallel 2-player rollouts, ``OpponentPool`` for FIFO
snapshots, and ``BaselineOpponentAdapter`` to mix the scripted baseline as a
sometimes-opponent.

Why PPO here
------------
The hypothesis we want to validate empirically is that a policy-gradient
method (PPO) is more *stable* than off-policy DQN in adversarial self-play:
DQN's bootstrapped Q-targets get pulled around by the moving opponent
distribution, whereas PPO's on-policy clipped updates are local in policy
space and don't rely on a Q-function tracking a non-stationary env.

Differences from the DQN self-play trainer
------------------------------------------
* Architecture: ``ActorCriticMLP`` (Tanh trunk; orthogonal init; small-std
  init on actor's last layer to keep the initial policy near-uniform).
  Same obs_dim=12, n_actions=6.
* Rollout: synchronous on-policy collection of ``rollout_len * num_envs``
  transitions per iteration (default 128*16 = 2048). No replay buffer.
* Update: PPO with GAE-lambda advantages, multi-epoch SGD over minibatches,
  clipped policy loss, value loss (clipped — see note below), entropy bonus.
* Action selection: stochastic (``Categorical(logits=...).sample()``) during
  training rollouts; greedy (argmax) for opponent inference and eval.
* Checkpoint: ``arch="ppo_actor_critic_mlp_128"``. The ONNX exporter
  reconstructs the actor-critic and exports the actor's logits path
  (``forward(obs) -> logits``), which is argmax-equivalent to softmax
  probabilities — matches the web frontend's argmax contract.

Design choices, documented
--------------------------
* Activation in trunk: Tanh. PPO convention pairs Tanh with orthogonal init
  (standard in CleanRL / SB3). ReLU works too but Tanh + orthogonal is the
  canonical PPO recipe and tends to be slightly more stable empirically.
* Value loss: CLIPPED (PPO2 style). ``v_pred`` clipped to ``v_old +-
  clip_eps``; loss is max(MSE(v_pred, returns), MSE(v_clipped, returns)).
  Helps prevent value function from drifting too far during multi-epoch
  updates. CLI ``--no-clip-vloss`` falls back to plain MSE if you want to
  ablate.
* Advantage normalization: per-minibatch (mean=0, std=1, eps=1e-8). Common
  PPO practice; improves optimization conditioning.
* Opponent inference: greedy (argmax of logits). Could be made stochastic
  for diversity but greedy gives a deterministic snapshot, matching how DQN
  self-play handled it and keeping eval comparable.
* BaselineOpponentAdapter: copied INLINE rather than imported from the DQN
  trainer. The DQN trainer is a self-contained script, not a library; the
  adapter is small (~25 lines) and depending on it would be a fragile
  cross-script import. Same INVERSE_ACTION_TABLE rationale.
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
from typing import Deque, List, Optional, Tuple

import gym  # noqa: F401  -- registers SlimeVolley-v0 alongside slimevolleygym
import numpy as np
import slimevolleygym  # noqa: F401  -- registers "SlimeVolley-v0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
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

    NOTE: copied inline from train_dqn_v1_vec_selfplay.py rather than imported
    to keep this script independent of the DQN trainer module.
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
# Actor-critic network (PPO convention: Tanh trunk + orthogonal init)
# ---------------------------------------------------------------------------
def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0),
                bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init (std) + constant bias. Standard PPO recipe.

    The actor's final layer is initialized with std=0.01 so the initial
    policy is near-uniform over actions (logits all near zero) — preventing
    the network from committing to an arbitrary action distribution at start.
    The critic's final layer uses std=1.0 so the value head's outputs aren't
    artificially squashed near zero.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticMLP(nn.Module):
    """Shared-trunk actor-critic with Tanh activations.

    trunk:  obs_dim -> hidden -> hidden  (Tanh between Linears)
    actor:  hidden -> n_actions          (logits; argmax = greedy policy)
    critic: hidden -> 1                  (state value V(s))

    forward(obs) -> logits (B, n_actions). This is the ONNX-facing path so
    the web frontend's argmax contract works unchanged: argmax of logits =
    argmax of softmax(logits), so no softmax layer is exported.

    forward_train(obs) -> (logits, values) used inside the rollout/update
    loops where we need both heads in a single pass.
    """

    def __init__(self, obs_dim: int = 12, n_actions: int = 6,
                 hidden: int = 128) -> None:
        super().__init__()
        # Trunk: two Linear+Tanh blocks. Orthogonal-init each linear with the
        # PPO-conventional std=sqrt(2) for hidden Tanh layers.
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        # Actor head: small-std init keeps the initial policy near-uniform.
        self.actor = _layer_init(nn.Linear(hidden, n_actions), std=0.01)
        # Critic head: std=1 so V(s) isn't artificially close to zero at start.
        self.critic = _layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """ONNX-facing path: logits only. Argmax over logits is the greedy
        policy and matches the web frontend's argmax-over-model-output
        contract — no softmax layer needed (argmax is invariant under softmax).
        """
        h = self.trunk(obs)
        return self.actor(h)

    def forward_train(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, values) for use during rollout/update."""
        h = self.trunk(obs)
        logits = self.actor(h)
        values = self.critic(h).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def act(self, obs: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action, log_prob, value).

        action and log_prob are (B,) int64/float32; value is (B,) float32.
        Used during rollout collection only; the update path recomputes log
        probs and values under-grad in ``forward_train``.
        """
        logits, values = self.forward_train(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values


# ---------------------------------------------------------------------------
# Rollout buffer (on-policy storage; one rollout = rollout_len * num_envs steps)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Fixed-size on-policy rollout storage.

    Layout is (rollout_len, num_envs, ...) so we can compute GAE in a single
    backwards pass over the time axis (per env). After computing returns and
    advantages we ``flatten`` the buffer into (rollout_len * num_envs, ...)
    and shuffle/minibatch over that flat axis for SGD updates.

    We DO NOT store ``next_obs`` per step — only the obs at step t. The
    bootstrap value at the end of the rollout is computed once from the
    final next-obs and passed to ``compute_gae`` directly. ``dones`` masks
    the GAE recursion at episode boundaries.
    """

    def __init__(self, rollout_len: int, num_envs: int, obs_dim: int,
                 device: torch.device) -> None:
        self.rollout_len = int(rollout_len)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.device = device

        T, N = self.rollout_len, self.num_envs
        # Float32 except for actions (int64). Stored on CPU as numpy to avoid
        # GPU memory fragmentation; we batch-move to device at update time.
        self.obs = np.zeros((T, N, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        # Filled by compute_gae:
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

        self._t = 0  # next slot to write

    def reset(self) -> None:
        self._t = 0

    def add(self, obs: np.ndarray, action: np.ndarray, log_prob: np.ndarray,
            reward: np.ndarray, done: np.ndarray, value: np.ndarray) -> None:
        """Append one timestep across all envs.

        Shapes (all ``num_envs``-leading): obs (N, obs_dim), action (N,),
        log_prob (N,), reward (N,), done (N,), value (N,).
        """
        t = self._t
        if t >= self.rollout_len:
            raise RuntimeError(
                f"RolloutBuffer overflow: t={t}, rollout_len={self.rollout_len}"
            )
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self._t += 1

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray,
                    gamma: float, gae_lambda: float) -> None:
        """Standard GAE-lambda + bootstrapped returns.

        Time-axis recurrence (per env, scalar):
          delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
          A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
          R_t     = A_t + V(s_t)

        We mask by ``done_t`` (the done flag at the END of step t — meaning the
        env terminated AFTER taking action_t) to zero the bootstrap whenever
        the next step is the start of a new episode. ``last_values`` /
        ``last_dones`` provide the bootstrap from beyond the rollout edge.
        """
        T = self.rollout_len
        adv = np.zeros((T, self.num_envs), dtype=np.float32)
        gae = np.zeros((self.num_envs,), dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values.astype(np.float32)
            else:
                # Buffer stores POST-step dones: dones[t] = "did step t end?".
                # We need "is s_{t+1} a fresh post-reset obs?" which is dones[t],
                # NOT dones[t+1] (CleanRL's pre-step convention). The boundary
                # case above already uses last_dones consistently with this.
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            delta = (self.rewards[t]
                     + gamma * next_values * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values

    def flatten(self) -> dict:
        """Return flattened (T*N, ...) tensors on the trainer's device.

        Called once per rollout, immediately before the SGD epoch loop. The
        returned dict is consumed by an indexing-based shuffler in the update
        loop (no DataLoader; the rollout is small enough to fit in RAM).
        """
        TN = self.rollout_len * self.num_envs
        return {
            "obs": torch.from_numpy(self.obs.reshape(TN, self.obs_dim)).to(self.device),
            "actions": torch.from_numpy(self.actions.reshape(TN)).to(self.device),
            "log_probs": torch.from_numpy(self.log_probs.reshape(TN)).to(self.device),
            "advantages": torch.from_numpy(self.advantages.reshape(TN)).to(self.device),
            "returns": torch.from_numpy(self.returns.reshape(TN)).to(self.device),
            "values": torch.from_numpy(self.values.reshape(TN)).to(self.device),
        }


# ---------------------------------------------------------------------------
# Env factories (identical to DQN self-play trainer)
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
    """
    env = gym.make("SlimeVolley-v0")
    env = DiscreteSlimeWrapper(env)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Eval (vs scripted baseline + vs sampled pool opponents)
# ---------------------------------------------------------------------------
def evaluate_vs_baseline(model: ActorCriticMLP, seed: int, episodes: int,
                         device: torch.device) -> Tuple[float, float, float]:
    """Greedy eval vs slimevolley's built-in baseline (single env).

    Greedy = argmax over actor logits (no Categorical sampling). Mirrors the
    DQN trainer's eval-vs-baseline path. Returns (mean_R, std_R, mean_len).
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


def evaluate_vs_pool(model: ActorCriticMLP, pool: OpponentPool,
                     seed: int, n_opponents: int, device: torch.device,
                     rng: np.random.Generator) -> Tuple[float, float]:
    """Greedy eval vs sampled pool opponents. Returns (mean_R, std_R).

    Each episode plays a different sampled snapshot. Returns (nan, nan) if
    the pool is empty. Same structure as the DQN trainer's evaluate_vs_pool
    but using ActorCriticMLP's forward (logits) for both sides.
    """
    if len(pool) == 0:
        return float("nan"), float("nan")

    returns: List[float] = []
    model.eval()
    with torch.no_grad():
        for ep_i in range(n_opponents):
            opp = pool.sample(rng)
            if opp is None:
                break
            env = make_raw_env(seed + 20_000 + ep_i)
            obs = env.reset()
            obs2, _r, done, info = env.step(ACTION_TABLE[0],
                                            otherAction=ACTION_TABLE[0])
            if done:
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
    # Global rollout sizing -------------------------------------------------
    p.add_argument("--total-timesteps", type=int, default=5_000_000,
                   help="Total transitions ingested across ALL envs.")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--rollout-len", type=int, default=128,
                   help="Env steps per env per rollout (one PPO iteration).")
    # PPO update -----------------------------------------------------------
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-anneal", action="store_true", default=False,
                   help="Linearly decay lr to 0 across the full training run "
                        "(standard PPO recipe). Off by default for backwards "
                        "compatibility with prior runs.")
    p.add_argument("--target-kl", type=float, default=None,
                   help="If set, stop the PPO update epochs early when the "
                        "running mean of approx_kl in the current epoch exceeds "
                        "this. Typical values 0.015-0.03; 0.02 is a safe default. "
                        "Prevents over-updating on a stale rollout (which causes "
                        "entropy collapse).")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--no-clip-vloss", action="store_true",
                   help="Disable PPO2-style clipped value loss; use plain MSE.")
    p.add_argument("--norm-adv", action="store_true", default=True,
                   help="Normalize advantages per minibatch (default on).")
    # Self-play ------------------------------------------------------------
    p.add_argument("--snapshot-every", type=int, default=100_000,
                   help="Add a frozen snapshot to the opponent pool every "
                        "this many transitions across all envs.")
    p.add_argument("--pool-max-size", type=int, default=30)
    p.add_argument("--baseline-opponent-prob", type=float, default=0.4,
                   help="Probability that a fresh episode's opponent is the "
                        "scripted baseline rather than a pool snapshot.")
    p.add_argument("--pool-eval-opponents", type=int, default=5)
    # Resume / IO ----------------------------------------------------------
    p.add_argument("--resume-from", type=str, default="",
                   help="Optional checkpoint to warm-start from (PPO from "
                        "scratch is the typical setup; resume is for "
                        "continuation experiments).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--save-path", type=str,
                   default="/home/shuhang/YBJ/RL_test/checkpoints/ppo_selfplay_vec.pt")
    p.add_argument("--log-dir", type=str,
                   default="/home/shuhang/YBJ/RL_test/logs/ppo_selfplay_vec")
    p.add_argument("--pool-dir", type=str,
                   default="/home/shuhang/YBJ/RL_test/checkpoints/pool_ppo_selfplay/")
    p.add_argument("--eval-every", type=int, default=25_000,
                   help="Eval cadence in transitions across all envs.")
    p.add_argument("--eval-episodes", type=int, default=5)
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
    path: str, model: ActorCriticMLP, total_timesteps_trained: int,
    final_eval_reward: float, args: argparse.Namespace,
) -> None:
    """Save actor-critic + PPO hyperparam metadata.

    The ``arch`` string is "ppo_actor_critic_mlp_128" so future tooling can
    distinguish PPO checkpoints from DQN ones at a glance. The exporter
    (``export_ppo_onnx.py``) reconstructs ActorCriticMLP from these fields.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": 12,
        "n_actions": 6,
        "hidden": 128,
        "arch": "ppo_actor_critic_mlp_128",
        "algorithm": "ppo",
        "total_timesteps_trained": int(total_timesteps_trained),
        "final_eval_reward": float(final_eval_reward),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_eps": float(args.clip_eps),
        "ent_coef": float(args.ent_coef),
        "vf_coef": float(args.vf_coef),
        "baseline_opponent_prob": float(args.baseline_opponent_prob),
        "rollout_len": int(args.rollout_len),
        "update_epochs": int(args.update_epochs),
        "minibatch_size": int(args.minibatch_size),
        "lr": float(args.lr),
        "max_grad_norm": float(args.max_grad_norm),
        "num_envs": int(args.num_envs),
        "selfplay": True,
        "clip_vloss": (not args.no_clip_vloss),
    }, path)


# ---------------------------------------------------------------------------
# Update step (one PPO update over the freshly collected rollout)
# ---------------------------------------------------------------------------
def ppo_update(model: ActorCriticMLP, optimizer: torch.optim.Optimizer,
               flat: dict, args: argparse.Namespace
               ) -> dict:
    """Run ``update_epochs`` SGD epochs over minibatches of the flat rollout.

    Implements:
      * Clipped policy loss: min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).
      * Value loss: clipped (PPO2) or plain MSE (toggled via --no-clip-vloss).
      * Entropy bonus: ent_coef * H(pi).
      * Per-minibatch advantage normalization.
      * Grad clip max-norm = max_grad_norm (default 0.5).

    Returns a dict of mean diagnostic scalars over all minibatches in this
    update (so the trainer can log them once per iteration).
    """
    obs_all = flat["obs"]
    actions_all = flat["actions"]
    old_log_probs_all = flat["log_probs"]
    advantages_all = flat["advantages"]
    returns_all = flat["returns"]
    old_values_all = flat["values"]

    total = obs_all.shape[0]
    mb = int(args.minibatch_size)
    if mb <= 0 or mb > total:
        # Defensive: if minibatch_size > rollout, just use the whole rollout
        # as one minibatch (still runs `update_epochs` epochs over it).
        mb = total
    # Indices for shuffling. We use np for the shuffle (deterministic w.r.t.
    # the trainer's np.random seed) and convert per minibatch.
    indices = np.arange(total)

    losses: List[float] = []
    pg_losses: List[float] = []
    v_losses: List[float] = []
    entropies: List[float] = []
    approx_kls: List[float] = []
    clip_fractions: List[float] = []

    clip_eps = float(args.clip_eps)
    vf_coef = float(args.vf_coef)
    ent_coef = float(args.ent_coef)
    max_grad_norm = float(args.max_grad_norm)
    clip_vloss = not bool(args.no_clip_vloss)

    target_kl = args.target_kl if args.target_kl is not None else None
    early_stopped_epoch: Optional[int] = None
    for _epoch in range(int(args.update_epochs)):
        np.random.shuffle(indices)
        # Per-epoch KL accumulator for the target-kl early-stop check. We
        # decide AT THE END of an epoch (CleanRL convention) whether to skip
        # the remaining epochs — checking inside the minibatch loop is too
        # noisy for small minibatches.
        epoch_kls: List[float] = []
        for start in range(0, total, mb):
            end = min(start + mb, total)
            mb_idx = torch.from_numpy(indices[start:end]).to(obs_all.device)

            mb_obs = obs_all.index_select(0, mb_idx)
            mb_actions = actions_all.index_select(0, mb_idx)
            mb_old_log_probs = old_log_probs_all.index_select(0, mb_idx)
            mb_advantages = advantages_all.index_select(0, mb_idx)
            mb_returns = returns_all.index_select(0, mb_idx)
            mb_old_values = old_values_all.index_select(0, mb_idx)

            # Per-minibatch advantage normalization. PPO is sensitive to
            # advantage scale; normalizing here is the stable choice (vs.
            # normalizing once at the rollout level).
            if args.norm_adv and mb_advantages.numel() > 1:
                mb_advantages = ((mb_advantages - mb_advantages.mean())
                                 / (mb_advantages.std() + 1e-8))

            new_logits, new_values = model.forward_train(mb_obs)
            dist = Categorical(logits=new_logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # Probability ratio under-grad. exp(log new - log old).
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            # Clipped surrogate objective. We MAXIMIZE this (so we negate it
            # before adding to the loss).
            unclipped = ratio * mb_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Value loss. Two flavors:
            #  - plain MSE (--no-clip-vloss): ((V_pred - R)^2).mean()
            #  - clipped (default): max( (V_pred - R)^2,
            #                            (V_clipped - R)^2 ).mean()/2
            #    where V_clipped = V_old + clip(V_pred - V_old, +-clip_eps).
            #    The /2 follows the original PPO2 convention; we keep it for
            #    parity with CleanRL/SB3.
            if clip_vloss:
                v_unclipped = (new_values - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -clip_eps, clip_eps,
                )
                v_clipped_loss = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
            else:
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

            entropy_bonus = entropy
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            # Diagnostics. ``approx_kl`` and ``clip_fraction`` are the
            # standard PPO health-check scalars (see CleanRL): a sudden
            # spike in approx_kl or clip_fraction usually means lr too high
            # or the rollout went stale.
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()

            losses.append(float(loss.item()))
            pg_losses.append(float(policy_loss.item()))
            v_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kls.append(float(approx_kl.item()))
            epoch_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))
        if target_kl is not None and epoch_kls:
            mean_kl = float(np.mean(epoch_kls))
            if mean_kl > target_kl:
                early_stopped_epoch = _epoch + 1
                break

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "policy_loss": float(np.mean(pg_losses)) if pg_losses else float("nan"),
        "value_loss": float(np.mean(v_losses)) if v_losses else float("nan"),
        "entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else float("nan"),
        "clip_fraction": (float(np.mean(clip_fractions))
                          if clip_fractions else float("nan")),
        "early_stopped_epoch": (early_stopped_epoch if early_stopped_epoch
                                is not None else int(args.update_epochs)),
    }


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
    rollout_len = int(args.rollout_len)
    transitions_per_iter = num_envs * rollout_len
    if args.total_timesteps % transitions_per_iter != 0:
        # Round DOWN to the nearest full rollout. Keeps the iteration count
        # an integer; we'd rather drop a partial rollout than emit one with
        # padded zeros.
        args.total_timesteps = (
            (args.total_timesteps // transitions_per_iter) * transitions_per_iter
        )
    n_iterations = max(1, args.total_timesteps // transitions_per_iter)

    print(
        f"[config] device={device} num_envs={num_envs} "
        f"rollout_len={rollout_len} transitions_per_iter={transitions_per_iter} "
        f"total_timesteps={args.total_timesteps} iterations={n_iterations} "
        f"update_epochs={args.update_epochs} minibatch_size={args.minibatch_size} "
        f"clip_eps={args.clip_eps} gamma={args.gamma} gae_lambda={args.gae_lambda} "
        f"vf_coef={args.vf_coef} ent_coef={args.ent_coef} lr={args.lr} "
        f"max_grad_norm={args.max_grad_norm} clip_vloss={not args.no_clip_vloss} "
        f"baseline_opponent_prob={args.baseline_opponent_prob} "
        f"snapshot_every={args.snapshot_every} pool_max_size={args.pool_max_size} "
        f"seed={args.seed}",
        flush=True,
    )

    # --- net + optim ---
    obs_dim, n_actions, hidden = 12, 6, 128
    model = ActorCriticMLP(obs_dim, n_actions, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), eps=1e-5)

    # --- optional warm start ---
    prior_eval_reward: Optional[float] = None
    resumed = False
    if args.resume_from and os.path.isfile(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        # We accept a checkpoint produced by this script (matching arch).
        # Cross-arch resume (e.g. from DQN) is not supported and would
        # silently break — so we strict-load.
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        prior_eval_reward = float(ckpt.get("final_eval_reward", float("nan")))
        print(f"[resume] loaded from {args.resume_from}, "
              f"prior eval_reward={prior_eval_reward}", flush=True)
        resumed = True
    elif args.resume_from:
        print(f"[resume] {args.resume_from} not found; starting from scratch.",
              flush=True)

    # --- opponent pool ---
    # factory_fn must build a fresh module on `device` so eval-mode inference
    # in each iteration runs on the same device as the online net (avoids
    # per-step cross-device copies).
    def _opp_factory() -> nn.Module:
        return ActorCriticMLP(obs_dim, n_actions, hidden).to(device)

    pool = OpponentPool(
        pool_dir=args.pool_dir,
        factory_fn=_opp_factory,
        device=str(device),
        max_size=int(args.pool_max_size),
    )
    # Restore any prior snapshots BEFORE seeding from a resume so we don't
    # double-count if a previous run already snapshotted this checkpoint.
    pool.load_existing()
    if resumed:
        pool.add(model.state_dict(), step=0)
    print(f"[pool] dir={args.pool_dir} initial_size={len(pool)}", flush=True)

    # --- vec env ---
    seed_base = int(args.seed)
    env_fns = [functools.partial(make_raw_env, seed_base + i) for i in range(num_envs)]
    vec_env = SelfPlayVecEnv(env_fns)

    try:
        # Sanity-check spaces (raw env: Box(12) obs, MultiBinary(3) actions —
        # the discrete wrapping is on the trainer's side, not the env's).
        env_obs_dim = int(np.array(vec_env.observation_space.shape).prod())
        assert env_obs_dim == obs_dim, (
            f"unexpected env obs_dim={env_obs_dim} (expected {obs_dim})"
        )

        # --- per-env opponent assignments (refreshed on episode boundary) ---
        baseline_opponents = [BaselineOpponentAdapter() for _ in range(num_envs)]

        def _sample_opponent(env_idx: int):
            """Return either a pool snapshot (nn.Module) or a baseline adapter.

            Mirrors the DQN trainer's mixing strategy: with probability
            ``baseline_opponent_prob`` we use the scripted baseline
            (resetting its hidden state first); otherwise we sample uniformly
            from the pool. If the pool is empty AND we didn't pick baseline,
            we fall back to baseline anyway — this keeps the early-training
            phase from playing against a NOOP opponent.
            """
            if rng.random() < args.baseline_opponent_prob:
                baseline_opponents[env_idx].reset()
                return baseline_opponents[env_idx]
            sampled = pool.sample(rng)
            if sampled is None:
                # Pool empty: fall back to baseline so we never pass NOOP.
                baseline_opponents[env_idx].reset()
                return baseline_opponents[env_idx]
            return sampled

        opp_assignments: List[object] = [
            _sample_opponent(i) for i in range(num_envs)
        ]

        # --- rollout buffer (allocated once; reused across iterations) ---
        rb = RolloutBuffer(rollout_len, num_envs, obs_dim, device)

        # --- run-state ---
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        recent_ep_returns: Deque[float] = deque(maxlen=100)
        recent_ep_lengths: Deque[float] = deque(maxlen=100)
        best_eval_reward = -float("inf")

        # Initial reset; per-worker seeds were applied at worker init.
        obs, opp_obs = vec_env.reset()  # each (N, 12)

        global_step = 0
        last_eval_step = 0
        last_snapshot_step = 0
        start_time = time.time()

        for it in range(1, n_iterations + 1):
            # ------- Optional linear lr decay (standard PPO recipe) -------
            # frac goes 1.0 -> 0.0 across the run; ``it`` is 1-indexed so the
            # final iteration gets lr ~ args.lr / n_iterations (not exactly 0,
            # to avoid a silent zero-update final step).
            if args.lr_anneal:
                frac = max(0.0, 1.0 - (it - 1) / float(n_iterations))
                lr_now = float(args.lr) * frac
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
            # ------- Rollout collection -------
            rb.reset()
            for t in range(rollout_len):
                # Agent action: stochastic Categorical sample. We keep
                # log_prob and value for the PPO update.
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).to(device)
                    actions_t, log_probs_t, values_t = model.act(obs_t)
                    actions_np = actions_t.cpu().numpy().astype(np.int64)
                    log_probs_np = log_probs_t.cpu().numpy().astype(np.float32)
                    values_np = values_t.cpu().numpy().astype(np.float32)

                # Opponent actions: greedy argmax of each env's assigned
                # snapshot (or baseline). We don't batch across envs because
                # opponents may differ across slots; with N=16 and small
                # 12-dim inputs, the per-env forwards are cheap relative to
                # env step time.
                opp_actions = np.zeros(num_envs, dtype=np.int64)
                with torch.no_grad():
                    for i in range(num_envs):
                        opp = opp_assignments[i]
                        if isinstance(opp, BaselineOpponentAdapter):
                            opp_actions[i] = opp.predict_action(opp_obs[i])
                        elif opp is None:
                            opp_actions[i] = 0  # NOOP fallback (very rare)
                        else:
                            opp_t = torch.from_numpy(opp_obs[i]).unsqueeze(0).to(device)
                            opp_actions[i] = int(opp(opp_t).argmax(dim=-1).item())

                next_obs, next_opp_obs, rewards, dones, infos = vec_env.step(
                    actions_np, opp_actions,
                )

                # Store the transition. ``done`` here is the post-step done
                # flag — the GAE recursion uses it to mask the bootstrap.
                rb.add(
                    obs=obs.astype(np.float32),
                    action=actions_np,
                    log_prob=log_probs_np,
                    reward=rewards.astype(np.float32),
                    done=dones.astype(np.float32),
                    value=values_np,
                )

                # Episode bookkeeping + per-env opponent refresh on done.
                # NOTE: we DO NOT swap ``obs`` to a "terminal observation"
                # here. ``next_obs`` from SelfPlayVecEnv is already the
                # post-reset obs on done indices, which is what we want
                # for the next-iteration policy forward. The GAE bootstrap
                # is masked correctly by ``done`` regardless.
                global_step += num_envs
                for i in range(num_envs):
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
                        opp_assignments[i] = _sample_opponent(i)

                obs = next_obs
                opp_obs = next_opp_obs

            # ------- Bootstrap value at the end of the rollout -------
            # We need V(s_{T}) and the corresponding done flags to seed the
            # GAE backward recursion. ``obs`` is already the post-rollout
            # state; we forward it through the critic (no grad).
            with torch.no_grad():
                last_obs_t = torch.from_numpy(obs).to(device)
                _last_logits, last_values_t = model.forward_train(last_obs_t)
                last_values = last_values_t.cpu().numpy().astype(np.float32)
            # last_dones: the FINAL step's done flags. If the env auto-reset
            # already handled the terminal, ``obs`` is the new-episode obs and
            # we must mask the bootstrap accordingly. ``rb.dones[-1]`` is
            # exactly that (the post-step done at the last rollout step).
            last_dones = rb.dones[-1].copy()

            rb.compute_gae(
                last_values=last_values,
                last_dones=last_dones,
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
            )

            # ------- PPO update -------
            flat = rb.flatten()
            metrics = ppo_update(model, optimizer, flat, args)

            # ------- TB logging (per-iteration; aggregated minibatch means) -------
            writer.add_scalar("train/loss", metrics["loss"], global_step)
            writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("train/entropy", metrics["entropy"], global_step)
            writer.add_scalar("train/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], global_step)
            # Rollout-level scalars (advantage statistics; useful for diagnosing
            # saturation in the value function).
            writer.add_scalar(
                "rollout/advantage_mean", float(rb.advantages.mean()), global_step,
            )
            writer.add_scalar(
                "rollout/advantage_std", float(rb.advantages.std()), global_step,
            )
            writer.add_scalar(
                "rollout/return_mean", float(rb.returns.mean()), global_step,
            )
            writer.add_scalar(
                "rollout/value_mean", float(rb.values.mean()), global_step,
            )

            # --- periodic snapshot ---
            if (global_step - last_snapshot_step) >= int(args.snapshot_every):
                last_snapshot_step = global_step
                pool.add(model.state_dict(), step=global_step)
                writer.add_scalar("pool/size", len(pool), global_step)
                print(f"[snapshot] step={global_step} pool_size={len(pool)}",
                      flush=True)

            # --- periodic eval ---
            if (global_step - last_eval_step) >= int(args.eval_every):
                last_eval_step = global_step
                base_mean, base_std, base_len = evaluate_vs_baseline(
                    model, args.seed, args.eval_episodes, device,
                )
                pool_mean, pool_std = evaluate_vs_pool(
                    model, pool, args.seed, int(args.pool_eval_opponents),
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
                    f"[step {global_step:>9d}] "
                    f"base_R={base_mean:+.3f}+-{base_std:.3f} "
                    f"base_len={base_len:.1f} {pool_str} "
                    f"train_R={rolling_R:+.3f} sps={sps:.1f} "
                    f"loss={metrics['loss']:+.4f} "
                    f"pl={metrics['policy_loss']:+.4f} "
                    f"vl={metrics['value_loss']:.4f} "
                    f"H={metrics['entropy']:.3f} "
                    f"kl={metrics['approx_kl']:.4f} "
                    f"clipfrac={metrics['clip_fraction']:.3f} "
                    f"pool_size={len(pool)}",
                    flush=True,
                )
                # Save checkpoint on best eval-vs-baseline.
                if base_mean > best_eval_reward:
                    best_eval_reward = base_mean
                    save_checkpoint(
                        args.save_path, model, global_step, base_mean, args,
                    )

        # --- final eval + checkpoint ---
        final_base_mean, final_base_std, final_base_len = evaluate_vs_baseline(
            model, args.seed, args.eval_episodes, device,
        )
        final_pool_mean, final_pool_std = evaluate_vs_pool(
            model, pool, args.seed, int(args.pool_eval_opponents),
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
            f"pool_size={len(pool)}",
            flush=True,
        )
        save_checkpoint(
            args.save_path, model, args.total_timesteps, final_base_mean, args,
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
