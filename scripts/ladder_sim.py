"""Python ladder ground-truth simulator.

Reproduces the browser's ELO-ladder logic (web/elo.js + web/game.js fast-sim
path) using slimevolleygym's native Python physics. The browser runs the same
roster (12 AI = 6 variants x 2 clones) with ELO K=32 and start=1500; this
script does the same matchmaking + ELO update so its final standings can be
compared against the in-browser ladder as a sanity check.

Each match:
  * env = gym.make("SlimeVolley-v0").unwrapped (raw 2-arg step API).
  * one agent on RIGHT, one on LEFT, randomized side-swap per match.
  * noop-noop initial step to populate info["otherObs"] (mirrors
    selfplay_vec_env._do_reset's behavior).
  * loop: each side picks an action from its observation; we call
    env.step(action_R, otherAction=action_L) and split the returned
    (obs_R, info["otherObs"]) for the next step.
  * ONNX agents: argmax over the single output tensor (dueling Q, raw Q,
    or PPO logits — argmax is correct for all three, since for both Q-values
    and logits the greedy action is the argmax).
  * Baseline: per-match fresh BaselinePolicy() with its own RNN hidden state;
    feed the normalized obs DIRECTLY (env's internal auto-baseline does the
    same — predict(obs) without rescaling), then map (forward, backward, jump)
    -> Discrete(6) via INVERSE_ACTION_TABLE.

Outcome convention (matches web/elo.js recordResult):
  left_life > right_life  => left wins (S_left = 1)
  left_life < right_life  => right wins (S_left = 0)
  left_life == right_life => draw (S_left = 0.5)

The slimevolley unwrapped env exposes agent_left.life / agent_right.life on
env.game (mirroring the JS World).

Self-contained: imports only stdlib + numpy + onnxruntime + gym +
slimevolleygym. Does NOT import anything from train/ to keep this script
runnable without the trainer modules on the path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

import gym  # noqa: F401  -- registers SlimeVolley-v0 alongside slimevolleygym
import slimevolleygym  # noqa: F401  -- registers "SlimeVolley-v0"

try:
    import onnxruntime as ort
except ImportError as e:
    print("[fatal] onnxruntime not installed in this conda env.", file=sys.stderr)
    raise

try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# Discrete(6) -> MultiBinary(3): (forward, backward, jump). Matches the
# ACTION_TABLE in web/physics.js and train/train_dqn_v1.py.
ACTION_TABLE: List[List[int]] = [
    [0, 0, 0],  # 0 NOOP
    [1, 0, 0],  # 1 forward
    [1, 0, 1],  # 2 forward + jump
    [0, 0, 1],  # 3 jump
    [0, 1, 1],  # 4 backward + jump
    [0, 1, 0],  # 5 backward
]
# Inverse: (f, b, j) -> Discrete index. Used to map BaselinePolicy's binary
# output back to a Discrete(6) so all paths share the same dispatch.
INVERSE_ACTION_TABLE: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0, (1, 0, 0): 1, (1, 0, 1): 2,
    (0, 0, 1): 3, (0, 1, 1): 4, (0, 1, 0): 5,
}


# ---------------------------------------------------------------------------
# Roster (matches web/elo.js, sans Human)
# ---------------------------------------------------------------------------
# Each tuple: (id, name, variant_key, model_filename or None for baseline).
ROSTER_SPEC: List[Tuple[str, str, str, Optional[str]]] = [
    ("plain_a",         "Plain DQN A",     "plain",       "model_plain.onnx"),
    ("plain_b",         "Plain DQN B",     "plain",       "model_plain.onnx"),
    ("dueling_a",       "Dueling A",       "dueling",     "model.onnx"),
    ("dueling_b",       "Dueling B",       "dueling",     "model.onnx"),
    ("duel_double_a",   "Duel+Double A",   "duel_double", "model_duel_double.onnx"),
    ("duel_double_b",   "Duel+Double B",   "duel_double", "model_duel_double.onnx"),
    ("duel_nstep_a",    "Duel+Nstep A",    "duel_nstep",  "model_duel_nstep.onnx"),
    ("duel_nstep_b",    "Duel+Nstep B",    "duel_nstep",  "model_duel_nstep.onnx"),
    ("v1_a",            "V1 cold A",       "v1",          "model_v1.onnx"),
    ("v1_b",            "V1 cold B",       "v1",          "model_v1.onnx"),
    ("v1_selfplay_a",   "V1 selfplay A",   "v1_selfplay", "model_v1_selfplay.onnx"),
    ("v1_selfplay_b",   "V1 selfplay B",   "v1_selfplay", "model_v1_selfplay.onnx"),
    ("v1_purepool_a",   "V1 purepool A",   "v1_purepool", "model_v1_purepool.onnx"),
    ("v1_purepool_b",   "V1 purepool B",   "v1_purepool", "model_v1_purepool.onnx"),
    ("v1_sp_seeded_a",  "V1 sp seeded A",  "v1_selfplay_seeded", "model_v1_selfplay_seeded.onnx"),
    ("v1_sp_seeded_b",  "V1 sp seeded B",  "v1_selfplay_seeded", "model_v1_selfplay_seeded.onnx"),
    ("rainbow_a",        "Rainbow 20M sp A", "rainbow",       "model_rainbow.onnx"),
    ("rainbow_b",        "Rainbow 20M sp B", "rainbow",       "model_rainbow.onnx"),
    ("rainbow_5m_a",     "Rainbow 5M A",     "rainbow_5m",    "model_rainbow_5m.onnx"),
    ("rainbow_5m_b",     "Rainbow 5M B",     "rainbow_5m",    "model_rainbow_5m.onnx"),
    ("rainbow_5m_sp_a",  "Rainbow 5M sp A",  "rainbow_5m_sp", "model_rainbow_5m_sp.onnx"),
    ("rainbow_5m_sp_b",  "Rainbow 5M sp B",  "rainbow_5m_sp", "model_rainbow_5m_sp.onnx"),
    ("rainbow_sp_fixed_a", "Rainbow sp fixed A", "rainbow_sp_fixed", "model_rainbow_sp_fixed.onnx"),
    ("rainbow_sp_fixed_b", "Rainbow sp fixed B", "rainbow_sp_fixed", "model_rainbow_sp_fixed.onnx"),
    ("rainbow_full_a",   "Rainbow full A",   "rainbow_full", "model_rainbow_full.onnx"),
    ("rainbow_full_b",   "Rainbow full B",   "rainbow_full", "model_rainbow_full.onnx"),
    ("rainbow_eps_a",    "Rainbow eps A",    "rainbow_eps",  "model_rainbow_eps.onnx"),
    ("rainbow_eps_b",    "Rainbow eps B",    "rainbow_eps",  "model_rainbow_eps.onnx"),
    ("v1_sp_fixed_a",   "V1 sp fixed A",   "v1_sp_fixed", "model_v1_selfplay_fixed.onnx"),
    ("v1_sp_fixed_b",   "V1 sp fixed B",   "v1_sp_fixed", "model_v1_selfplay_fixed.onnx"),
    ("ppo_a",           "PPO A",           "ppo",         "model_ppo.onnx"),
    ("ppo_b",           "PPO B",           "ppo",         "model_ppo.onnx"),
    ("ppo_fixed_a",     "PPO fixed A",     "ppo_fixed",   "model_ppo_fixed.onnx"),
    ("ppo_fixed_b",     "PPO fixed B",     "ppo_fixed",   "model_ppo_fixed.onnx"),
    ("ppo_rescue_a",    "PPO rescue A",    "ppo_rescue",  "model_ppo_rescue.onnx"),
    ("ppo_rescue_b",    "PPO rescue B",    "ppo_rescue",  "model_ppo_rescue.onnx"),
    ("baseline_a",      "Baseline A",      "baseline",    None),
    ("baseline_b",      "Baseline B",      "baseline",    None),
]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class OnnxAgent:
    """Stateless ONNX greedy-argmax agent. One session shared across all
    clones of the same variant (deterministic policy => same output for same
    obs, so cloning the entry doesn't change behavior; ELO drift between
    A and B is purely env stochasticity / side asymmetry / opponent order).
    """

    def __init__(self, model_path: str) -> None:
        # CPU EP is plenty fast for a 12-d MLP; avoids a CUDA spin-up cost
        # per session and keeps the script runnable on login nodes.
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name  # "observation"

    def predict_action(self, obs: np.ndarray) -> int:
        out = self.session.run(None, {self.input_name: obs[None].astype(np.float32)})[0]
        return int(np.argmax(out[0]))


class BaselineAgent:
    """slimevolleygym BaselinePolicy wrapper. Stateful (tanh-RNN hidden
    state); a fresh instance is created per match (and per side) so the
    two clones don't share hidden state.
    """

    def __init__(self) -> None:
        from slimevolleygym.slimevolley import BaselinePolicy
        self.policy = BaselinePolicy()

    def reset(self) -> None:
        self.policy.reset()

    def predict_action(self, obs: np.ndarray) -> int:
        # The env's INTERNAL auto-baseline calls self.policy.predict(obs)
        # with the SAME normalized obs (post /10) — see SlimeVolleyEnv.step
        # in slimevolleygym/slimevolley.py. Feeding `obs * 10` here would
        # saturate tanh and cripple baseline.
        binary = self.policy.predict(obs)
        return INVERSE_ACTION_TABLE.get(
            (int(binary[0]), int(binary[1]), int(binary[2])), 0,
        )


# ---------------------------------------------------------------------------
# Player record
# ---------------------------------------------------------------------------
class Player:
    def __init__(self, pid: str, name: str, variant_key: str, model_path: Optional[str]) -> None:
        self.id = pid
        self.name = name
        self.variant_key = variant_key
        self.model_path = model_path  # None for baseline
        self.rating = 1500.0
        self.games = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0


def build_roster(web_dir: str) -> Tuple[List[Player], Dict[str, OnnxAgent]]:
    """Construct the 12-player roster + per-variant ONNX session cache.
    Baseline variants get fresh per-match instances (handled in run_match)
    so we don't cache them here.
    """
    players: List[Player] = []
    onnx_cache: Dict[str, OnnxAgent] = {}
    missing: List[str] = []
    for pid, name, vk, fname in ROSTER_SPEC:
        path = os.path.join(web_dir, fname) if fname else None
        if path is not None and not os.path.isfile(path):
            missing.append(path)
        players.append(Player(pid, name, vk, path))
    if missing:
        unique = sorted(set(missing))
        print("[warn] missing ONNX model files (matches involving these will be skipped):",
              file=sys.stderr)
        for m in unique:
            print(f"  - {m}", file=sys.stderr)
    # Pre-load ONNX sessions per unique variant model path (one per file,
    # not one per clone).
    for p in players:
        if p.model_path and p.model_path not in onnx_cache and os.path.isfile(p.model_path):
            onnx_cache[p.model_path] = OnnxAgent(p.model_path)
    return players, onnx_cache


def make_agent_for(player: Player, onnx_cache: Dict[str, OnnxAgent]):
    """Return an agent instance for one side of one match.
    ONNX sessions are shared (stateless). Baseline gets a fresh instance per
    side per match (RNN state must be per-slot, see web/game.js
    getOrCreateAgent).
    """
    if player.variant_key == "baseline":
        return BaselineAgent()
    if player.model_path is None or player.model_path not in onnx_cache:
        return None
    return onnx_cache[player.model_path]


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------
def run_match(left: Player, right: Player, onnx_cache: Dict[str, OnnxAgent],
              env, env_seed: int) -> Tuple[int, int]:
    """Play one match between left and right. Returns (left_life, right_life).
    Returns (-1, -1) sentinel if either agent could not be constructed
    (e.g. missing ONNX file).
    """
    left_agent  = make_agent_for(left,  onnx_cache)
    right_agent = make_agent_for(right, onnx_cache)
    if left_agent is None or right_agent is None:
        return (-1, -1)
    if hasattr(left_agent, "reset"):  left_agent.reset()
    if hasattr(right_agent, "reset"): right_agent.reset()

    env.seed(env_seed)
    env.reset()
    # Noop-noop initial step to populate info["otherObs"] before the main
    # loop. Same trick as selfplay_vec_env._do_reset and the eval-vs-pool
    # path in train_dqn_v1_vec_selfplay.py.
    obs_R, _r, done, info = env.step(ACTION_TABLE[0], otherAction=ACTION_TABLE[0])
    obs_R = np.asarray(obs_R, dtype=np.float32)
    obs_L = np.asarray(info["otherObs"], dtype=np.float32)

    # Slimevolley caps episodes at t_limit=3000. Add a small safety margin
    # so a misbehaving combo never wedges the run.
    max_steps = 3100
    step = 0
    while not done and step < max_steps:
        a_L = left_agent.predict_action(obs_L)
        a_R = right_agent.predict_action(obs_R)
        obs_R, _r, done, info = env.step(
            ACTION_TABLE[a_R], otherAction=ACTION_TABLE[a_L],
        )
        obs_R = np.asarray(obs_R, dtype=np.float32)
        obs_L = np.asarray(info["otherObs"], dtype=np.float32)
        step += 1

    # The unwrapped slimevolley env stores the world on env.game.
    return int(env.game.agent_left.life), int(env.game.agent_right.life)


# ---------------------------------------------------------------------------
# Matchmaking + ELO
# ---------------------------------------------------------------------------
K = 32
INIT_RATING = 1500.0


def expected(my_r: float, opp_r: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opp_r - my_r) / 400.0))


def match_key(a_id: str, b_id: str) -> str:
    # Order-independent (matches elo.js _matchKey).
    return f"{a_id}|{b_id}" if a_id < b_id else f"{b_id}|{a_id}"


def pick_pair(players: List[Player], rng: random.Random,
              recent: List[str], recent_max: int = 3) -> Tuple[Player, Player]:
    """Random pair, avoiding pairs played in the last `recent_max` matches.
    Mirrors elo.js nextMatch's random branch (10 attempts, then accept the
    last fallback). Both clones of the same variant CAN play each other —
    the only filter is the recent-pairs window.
    """
    n = len(players)
    fallback: Optional[Tuple[Player, Player]] = None
    for _attempt in range(10):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        a, b = players[i], players[j]
        fallback = (a, b)
        if match_key(a.id, b.id) not in recent:
            return (a, b)
    assert fallback is not None
    return fallback


def update_elo(left: Player, right: Player, left_life: int, right_life: int) -> float:
    """Apply ELO update; return s_left so callers can log it.
    Identical math to web/elo.js recordResult.
    """
    if left_life > right_life:
        s_left = 1.0
    elif left_life < right_life:
        s_left = 0.0
    else:
        s_left = 0.5
    s_right = 1.0 - s_left
    e_left = expected(left.rating, right.rating)
    e_right = 1.0 - e_left
    left.rating  += K * (s_left  - e_left)
    right.rating += K * (s_right - e_right)
    for p, s in ((left, s_left), (right, s_right)):
        p.games += 1
        if   s == 1.0: p.wins   += 1
        elif s == 0.5: p.draws  += 1
        else:          p.losses += 1
    return s_left


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_table(players: List[Player], n_matches: int) -> None:
    ranked = sorted(players, key=lambda p: (-p.rating, p.name))
    print(f"\n# Python Ladder Ground Truth (N={n_matches} matches)\n")
    print("| Rank | Player | ELO | Games | W-D-L | win-rate |")
    print("|------|--------|-----|-------|-------|----------|")
    for i, p in enumerate(ranked, start=1):
        wr = (p.wins + 0.5 * p.draws) / p.games if p.games > 0 else 0.0
        print(f"| {i} | {p.name} | {p.rating:.0f} | {p.games} | "
              f"{p.wins}-{p.draws}-{p.losses} | {wr*100:.0f}% |")

    # Per-variant aggregate (mean ELO across A/B clones).
    by_variant: Dict[str, List[Player]] = defaultdict(list)
    for p in players:
        by_variant[p.variant_key].append(p)
    print("\n## Per-variant aggregate (mean ELO across A and B clones)\n")
    print("| Variant | Mean ELO | Games (sum) | W-D-L (sum) |")
    print("|---------|----------|-------------|-------------|")
    agg_rows = []
    for vk, ps in by_variant.items():
        mean_elo = sum(p.rating for p in ps) / len(ps)
        agg_rows.append((mean_elo, vk, ps))
    agg_rows.sort(key=lambda r: -r[0])
    for mean_elo, vk, ps in agg_rows:
        g = sum(p.games for p in ps)
        w = sum(p.wins for p in ps)
        d = sum(p.draws for p in ps)
        l = sum(p.losses for p in ps)
        print(f"| {vk} | {mean_elo:.0f} | {g} | {w}-{d}-{l} |")


def save_json(path: str, players: List[Player], n_matches: int, seed: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "n_matches": n_matches,
        "seed": seed,
        "K": K,
        "init_rating": INIT_RATING,
        "players": [
            {"id": p.id, "name": p.name, "variant_key": p.variant_key,
             "rating": p.rating, "games": p.games,
             "wins": p.wins, "draws": p.draws, "losses": p.losses}
            for p in players
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[json] wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matches", type=int, default=1000)
    ap.add_argument("--web-dir", type=str, default="/home/shuhang/YBJ/RL_test/web/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None,
                    help="If set, also dump JSON results to this path. If 'auto', "
                         "writes to logs/ladder_sim_<timestamp>.json.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    players, onnx_cache = build_roster(args.web_dir)

    # One env reused across all matches (env.seed before each reset gives
    # reproducibility while saving the per-match construction overhead).
    env = gym.make("SlimeVolley-v0").unwrapped

    recent: List[str] = []
    recent_max = 3

    iterator = range(args.n_matches)
    if _HAS_TQDM:
        iterator = tqdm(iterator, desc="matches", smoothing=0.05)

    t0 = time.time()
    skipped = 0
    for i in iterator:
        a, b = pick_pair(players, rng, recent, recent_max=recent_max)
        # Randomize side assignment, like elo.js nextMatch's swap.
        if rng.random() < 0.5:
            left_p, right_p = b, a
        else:
            left_p, right_p = a, b
        recent.append(match_key(left_p.id, right_p.id))
        while len(recent) > recent_max:
            recent.pop(0)

        env_seed = args.seed + 1 + i  # +1 so the first match doesn't reuse args.seed exactly
        l_life, r_life = run_match(left_p, right_p, onnx_cache, env, env_seed)
        if l_life < 0:
            skipped += 1
            continue
        update_elo(left_p, right_p, l_life, r_life)

        if not _HAS_TQDM and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"[{i+1}/{args.n_matches}] {rate:.1f} matches/s "
                  f"elapsed={elapsed:.1f}s", flush=True)

    env.close()

    if skipped > 0:
        print(f"\n[note] skipped {skipped} matches due to missing models.",
              file=sys.stderr)

    print_table(players, args.n_matches - skipped)

    if args.output:
        if args.output == "auto":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"/home/shuhang/YBJ/RL_test/logs/ladder_sim_{ts}.json"
        else:
            out_path = args.output
        save_json(out_path, players, args.n_matches - skipped, args.seed)


if __name__ == "__main__":
    main()
