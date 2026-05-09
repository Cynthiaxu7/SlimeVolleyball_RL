"""Replay recorder for the slime-volleyball web visualizer.

Drives matches in slimevolleygym's native Python physics (the same env as
scripts/ladder_sim.py uses) and dumps a frame-by-frame JSON trace per match.
The web UI's Replay mode (web/replay.js) loads these JSONs and replays them
on the existing canvas WITHOUT re-running physics — that's the whole point:
it isolates whether visual/render bugs are in the JS physics or elsewhere by
showing the user exactly what the Python ladder simulator saw.

Per-frame state captured:
  * ball   = [x, y, vx, vy]                (world coords, NOT /10 normalized)
  * left   = [x, y, vx, vy]                (agent_left  — dir=-1)
  * right  = [x, y, vx, vy]                (agent_right — dir=+1)
  * lives  = [left_life, right_life]
  * actions= [a_left, a_right]              (Discrete(6) indices)
  * obs_*  = the obs each side's MODEL SAW (already /10 normalized, len 12)
  * q_*    = ONNX output (len 6) if that side is trained, else None

The first frame (t=0) is the world state immediately after env.reset() and
the noop-noop init step (which populates info["otherObs"] mirroring the
ladder_sim.py setup); subsequent frames are post-step state.

Run:
  python scripts/record_replays.py \
    --pairs v1_selfplay,baseline rainbow,baseline ppo,baseline \
    --n-each 3 --output-dir web/replays/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

import gym  # noqa: F401  -- registers SlimeVolley-v0 alongside slimevolleygym
import slimevolleygym  # noqa: F401  -- registers "SlimeVolley-v0"

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    print("[fatal] onnxruntime not installed in this conda env.", file=sys.stderr)
    raise


# Discrete(6) -> MultiBinary(3): (forward, backward, jump). Matches
# web/physics.js ACTION_TABLE and train/train_dqn_v1*.
ACTION_TABLE: List[List[int]] = [
    [0, 0, 0],  # 0 NOOP
    [1, 0, 0],  # 1 forward
    [1, 0, 1],  # 2 forward + jump
    [0, 0, 1],  # 3 jump
    [0, 1, 1],  # 4 backward + jump
    [0, 1, 0],  # 5 backward
]
INVERSE_ACTION_TABLE: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0, (1, 0, 0): 1, (1, 0, 1): 2,
    (0, 0, 1): 3, (0, 1, 1): 4, (0, 1, 0): 5,
}


# Variant key -> ONNX file under web/. None = baseline (pure-Python policy,
# no Q-values exposed). These keys are the same strings the web UI uses in
# AI_VARIANTS / dropdowns / replay JSON metadata.
VARIANT_TO_ONNX: Dict[str, Optional[str]] = {
    "plain":          "model_plain.onnx",
    "dueling":        "model.onnx",
    "duel_double":    "model_duel_double.onnx",
    "duel_nstep":     "model_duel_nstep.onnx",
    "v1":             "model_v1.onnx",
    "v1_cold":        "model_v1.onnx",         # alias used in some sample lists
    "v1_selfplay":    "model_v1_selfplay.onnx",
    "v1_purepool":    "model_v1_purepool.onnx",
    "v1_selfplay_seeded": "model_v1_selfplay_seeded.onnx",
    "ppo":            "model_ppo.onnx",
    "rainbow":        "model_rainbow.onnx",
    "rainbow_5m":     "model_rainbow_5m.onnx",
    "rainbow_5m_sp":  "model_rainbow_5m_sp.onnx",
    "rainbow_sp_fixed": "model_rainbow_sp_fixed.onnx",
    "rainbow_full":   "model_rainbow_full.onnx",
    "rainbow_eps":    "model_rainbow_eps.onnx",
    "v1_sp_fixed":    "model_v1_selfplay_fixed.onnx",
    "ppo_fixed":      "model_ppo_fixed.onnx",
    "ppo_rescue":     "model_ppo_rescue.onnx",
    "baseline":       None,
}


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
class OnnxAgent:
    """Stateless ONNX greedy-argmax agent. Returns (action, q_values).
    q_values are the raw model output (Q for DQN/Dueling/Rainbow, logits for
    PPO — argmax is correct for both, but we expose the vector so the web UI
    can plot the per-action time-series consistent with what game.js does
    for live play).
    """

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, obs: np.ndarray) -> Tuple[int, List[float]]:
        out = self.session.run(None, {self.input_name: obs[None].astype(np.float32)})[0]
        q = out[0].astype(np.float32)
        a = int(np.argmax(q))
        return a, [float(v) for v in q]


class BaselineAgent:
    """slimevolleygym BaselinePolicy wrapper. Stateful (tanh-RNN hidden);
    a fresh instance per match per side so left/right don't share state.
    Does NOT expose Q-values — Baseline is a hand-tuned tanh-RNN, not a
    Q-network. Returns (action, None).
    """

    def __init__(self, warmup_steps: int = 0) -> None:
        from slimevolleygym.slimevolley import BaselinePolicy
        self.policy = BaselinePolicy()
        # Warm up the RNN hidden state so the first real-match frames don't
        # see a cold (all-zero) hidden state. The env's auto-baseline
        # carries hidden state across episodes, so cold start in our
        # 2-arg env loop puts baseline at a disadvantage. Drive the policy
        # through ~N typical-magnitude observations to evolve its state.
        if warmup_steps > 0:
            rng = np.random.default_rng(0)
            for _ in range(warmup_steps):
                # Mimic in-distribution NORMALIZED obs (scaleFactor=10
                # already applied — matches what env feeds the policy).
                fake_obs = rng.normal(0.0, 0.5, size=12).astype(np.float32)
                self.policy.predict(fake_obs)

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, obs: np.ndarray) -> Tuple[int, None]:
        # The env's INTERNAL auto-baseline calls self.policy.predict(obs)
        # with the SAME normalized obs (post /10) — see
        # SlimeVolleyEnv.step in slimevolleygym/slimevolley.py. Feeding
        # `obs * 10` here would saturate tanh and cripple baseline; that
        # was a long-standing bug that made trained agents look stronger
        # than they actually are.
        binary = self.policy.predict(obs)
        return INVERSE_ACTION_TABLE.get(
            (int(binary[0]), int(binary[1]), int(binary[2])), 0,
        ), None


def make_agent(variant: str, web_dir: str, onnx_cache: Dict[str, OnnxAgent], baseline_warmup: int = 0):
    """Build (or look up) the agent for a side. Baseline gets a fresh
    instance per call (per-side RNN state); ONNX agents share a session
    across calls (deterministic; per-call inputs are independent).
    """
    if variant == "baseline":
        return BaselineAgent(warmup_steps=baseline_warmup), None
    fname = VARIANT_TO_ONNX.get(variant)
    if fname is None:
        return None, f"unknown variant '{variant}'"
    path = os.path.join(web_dir, fname)
    if not os.path.isfile(path):
        return None, f"missing ONNX file: {path}"
    if path not in onnx_cache:
        onnx_cache[path] = OnnxAgent(path)
    return onnx_cache[path], None


# --------------------------------------------------------------------------
# Match recording
# --------------------------------------------------------------------------
def _agent_quad(a) -> List[float]:
    """[x, y, vx, vy] in world coords — NOT divided by 10."""
    return [float(a.x), float(a.y), float(a.vx), float(a.vy)]


def _ball_quad(b) -> List[float]:
    return [float(b.x), float(b.y), float(b.vx), float(b.vy)]


def record_match(left_variant: str, right_variant: str, seed: int,
                 web_dir: str, onnx_cache: Dict[str, OnnxAgent],
                 env, baseline_warmup: int = 0) -> Optional[dict]:
    """Run one match; return the full replay payload (meta + frames).
    Returns None if either agent could not be constructed.
    """
    left_agent,  err_l = make_agent(left_variant,  web_dir, onnx_cache, baseline_warmup)
    right_agent, err_r = make_agent(right_variant, web_dir, onnx_cache, baseline_warmup)
    if left_agent is None or right_agent is None:
        print(f"  skipped {left_variant} vs {right_variant} seed={seed}: "
              f"left_err={err_l} right_err={err_r}", file=sys.stderr)
        return None

    # Stateful agents (Baseline RNN) need a fresh hidden state per match.
    if hasattr(left_agent,  "reset"): left_agent.reset()
    if hasattr(right_agent, "reset"): right_agent.reset()

    env.seed(seed)
    env.reset()

    # Noop-noop init step to populate info["otherObs"] (mirrors
    # selfplay_vec_env._do_reset and ladder_sim.py's run_match).
    obs_R, _r, done, info = env.step(ACTION_TABLE[0], otherAction=ACTION_TABLE[0])
    obs_R = np.asarray(obs_R, dtype=np.float32)
    obs_L = np.asarray(info["otherObs"], dtype=np.float32)

    game = env.game  # the underlying World instance

    frames: List[dict] = []

    # Frame t=0: post-init state (after the noop-noop init step). Actions
    # for this frame default to 0 (no action was *chosen* yet — the init
    # step used noop-noop). Q-values are None because no model was queried
    # for this frame.
    frames.append({
        "t": 0,
        "ball":  _ball_quad(game.ball),
        "left":  _agent_quad(game.agent_left),
        "right": _agent_quad(game.agent_right),
        "lives": [int(game.agent_left.life), int(game.agent_right.life)],
        "actions": [0, 0],
        "obs_left":  obs_L.tolist(),
        "obs_right": obs_R.tolist(),
        "q_left":  None,
        "q_right": None,
    })

    # Match the slimevolley episode cap of 3000 + safety margin (same as
    # ladder_sim.py).
    max_steps = 3100
    step = 0
    while not done and step < max_steps:
        a_L, q_L = left_agent.predict(obs_L)
        a_R, q_R = right_agent.predict(obs_R)
        # Snapshot the obs the model SAW for this step (pre-step), then step.
        seen_obs_L = obs_L.tolist()
        seen_obs_R = obs_R.tolist()
        obs_R, _r, done, info = env.step(
            ACTION_TABLE[a_R], otherAction=ACTION_TABLE[a_L],
        )
        obs_R = np.asarray(obs_R, dtype=np.float32)
        obs_L = np.asarray(info["otherObs"], dtype=np.float32)
        step += 1
        frames.append({
            "t": step,
            "ball":  _ball_quad(game.ball),
            "left":  _agent_quad(game.agent_left),
            "right": _agent_quad(game.agent_right),
            "lives": [int(game.agent_left.life), int(game.agent_right.life)],
            "actions": [int(a_L), int(a_R)],
            "obs_left":  seen_obs_L,
            "obs_right": seen_obs_R,
            "q_left":  q_L,
            "q_right": q_R,
        })

    l_life = int(game.agent_left.life)
    r_life = int(game.agent_right.life)
    if l_life > r_life:
        winner = "left"
    elif l_life < r_life:
        winner = "right"
    else:
        winner = "draw"

    meta = {
        "left_variant": left_variant,
        "right_variant": right_variant,
        "left_model":  VARIANT_TO_ONNX.get(left_variant),   # may be None for baseline
        "right_model": VARIANT_TO_ONNX.get(right_variant),
        "seed": int(seed),
        "outcome": {
            "left_lives":  l_life,
            "right_lives": r_life,
            "winner": winner,
            "steps": step,  # number of post-init steps actually taken
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {"meta": meta, "frames": frames}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_pairs(items: List[str]) -> List[Tuple[str, str]]:
    """Each pair is 'left,right' (left-side variant first, right-side second)."""
    out: List[Tuple[str, str]] = []
    for it in items:
        if "," not in it:
            raise SystemExit(f"--pairs entry '{it}' must be 'LEFT,RIGHT'")
        a, b = it.split(",", 1)
        a, b = a.strip(), b.strip()
        if not a or not b:
            raise SystemExit(f"--pairs entry '{it}' has empty side")
        out.append((a, b))
    return out


def fmt_outcome(meta: dict) -> str:
    o = meta["outcome"]
    return f"{o['winner']} wins {o['left_lives']}-{o['right_lives']} ({o['steps']} steps)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, nargs="+", required=True,
                    help="One or more 'LEFT,RIGHT' variant pairs, e.g. "
                         "v1_selfplay,baseline. LEFT goes on agent_left "
                         "(dir=-1); RIGHT on agent_right (dir=+1).")
    ap.add_argument("--n-each", type=int, default=3,
                    help="Number of seeds per pair (each gets its own JSON file).")
    ap.add_argument("--seed-base", type=int, default=42,
                    help="Base seed; per-replay seed is seed_base + i*1000 + pair_idx.")
    ap.add_argument("--output-dir", type=str, default="/home/shuhang/YBJ/RL_test/web/replays/")
    ap.add_argument("--web-dir", type=str, default="/home/shuhang/YBJ/RL_test/web/",
                    help="Where to find the .onnx files referenced by VARIANT_TO_ONNX.")
    ap.add_argument("--baseline-warmup", type=int, default=0,
                    help="Steps to warm up baseline RNN hidden state before "
                         "match start. 0 = cold start (default; mimics fresh "
                         "BaselinePolicy()). 100+ = pre-evolved state, more "
                         "representative of how the env's auto-baseline plays "
                         "after a few prior episodes.")
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    os.makedirs(args.output_dir, exist_ok=True)

    onnx_cache: Dict[str, OnnxAgent] = {}
    env = gym.make("SlimeVolley-v0").unwrapped

    index_entries: List[dict] = []
    summary: List[Tuple[str, str, str]] = []
    t0 = time.time()
    written = 0
    skipped = 0

    for pair_idx, (left_v, right_v) in enumerate(pairs):
        for i in range(args.n_each):
            # Stagger seeds across pairs so we don't keep replaying the same
            # initial-ball state for every pair.
            seed = args.seed_base + pair_idx * 1000 + i
            payload = record_match(
                left_variant=left_v, right_variant=right_v,
                seed=seed, web_dir=args.web_dir, onnx_cache=onnx_cache, env=env,
                baseline_warmup=args.baseline_warmup,
            )
            if payload is None:
                skipped += 1
                continue
            fname = f"{right_v}_vs_{left_v}_seed{seed}.json"
            out_path = os.path.join(args.output_dir, fname)
            with open(out_path, "w") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            outcome_str = fmt_outcome(payload["meta"])
            print(f"  [{written + 1}] {fname}: {outcome_str}", flush=True)
            summary.append((fname, f"{left_v} vs {right_v}", outcome_str))
            index_entries.append({
                "file": fname,
                "left":  left_v,
                "right": right_v,
                "outcome": outcome_str,
                "steps":  payload["meta"]["outcome"]["steps"],
                "winner": payload["meta"]["outcome"]["winner"],
                "left_lives":  payload["meta"]["outcome"]["left_lives"],
                "right_lives": payload["meta"]["outcome"]["right_lives"],
                "seed": seed,
            })
            written += 1

    env.close()

    # web/replays/index.json: lists every replay we just wrote so the JS
    # dropdown can populate without a directory listing endpoint.
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w") as fh:
        json.dump({"replays": index_entries}, fh, indent=2)
    print(f"\n[index] wrote {index_path} ({len(index_entries)} replays)")

    elapsed = time.time() - t0
    print(f"\n[summary] wrote {written} replays, skipped {skipped} "
          f"in {elapsed:.1f}s")

    # Aggregate per-pair outcomes (winner counts) for a quick eyeball check.
    by_pair: Dict[str, Dict[str, int]] = {}
    by_pair_steps: Dict[str, List[int]] = {}
    for entry in index_entries:
        key = f"{entry['left']} vs {entry['right']}"
        by_pair.setdefault(key, {"left": 0, "right": 0, "draw": 0})
        by_pair[key][entry["winner"]] += 1
        by_pair_steps.setdefault(key, []).append(entry["steps"])
    print("\n[per-pair aggregate]")
    for k, counts in by_pair.items():
        steps = by_pair_steps[k]
        mean_steps = sum(steps) / len(steps) if steps else 0
        print(f"  {k}: left={counts['left']} right={counts['right']} "
              f"draw={counts['draw']} mean_steps={mean_steps:.0f}")


if __name__ == "__main__":
    main()
