"""Head-to-head A vs B over N matches with alternating sides.

Prints W-D-L from A's perspective + life-margin distribution. Useful for
isolating *why* one model dominates another (e.g. "lost on volleys but
won on serves" by inspecting outcome bins).

Each match runs slimevolleygym's native physics in single-process. Both
agents share an OnnxRuntime CPU session (deterministic argmax over the
single output tensor — Q for DQN/Dueling/Rainbow, logits for PPO).
Baseline ('baseline') uses the inline BaselineAgent from record_replays.py.

Run:
  python scripts/head2head.py \
    --a checkpoints/dqn_v1_vec_selfplay_fixed.pt \
    --b web/model_v1_selfplay.onnx \
    --n 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

import gym  # noqa: F401
import slimevolleygym  # noqa: F401
import onnxruntime as ort

# Reuse same ACTION_TABLE / INVERSE_ACTION_TABLE / BaselineAgent as the
# recorder so behavior is identical to ladder_sim and record_replays.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_replays import ACTION_TABLE, INVERSE_ACTION_TABLE, BaselineAgent  # noqa: E402


class OnnxAgent:
    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, obs: np.ndarray) -> int:
        out = self.session.run(None, {self.input_name: obs[None].astype(np.float32)})[0]
        return int(np.argmax(out[0]))


def make_agent(spec: str):
    """spec is either 'baseline' or a path to an .onnx file. .pt is rejected
    (we only support exported models for fair comparison with the web demo).
    """
    if spec == "baseline":
        return BaselineAgent(warmup_steps=0)
    if spec.endswith(".onnx") and os.path.isfile(spec):
        return OnnxAgent(spec)
    raise SystemExit(f"unsupported agent spec: {spec!r} (use 'baseline' or path/to/model.onnx)")


def predict_action(agent, obs: np.ndarray) -> int:
    if isinstance(agent, BaselineAgent):
        a, _ = agent.predict(obs)
        return a
    return agent.predict(obs)


def run_match(agent_left, agent_right, env, seed: int) -> Tuple[int, int, int]:
    """Returns (left_lives, right_lives, steps)."""
    if hasattr(agent_left,  "reset"): agent_left.reset()
    if hasattr(agent_right, "reset"): agent_right.reset()
    env.seed(seed)
    env.reset()
    obs_R, _r, done, info = env.step(ACTION_TABLE[0], otherAction=ACTION_TABLE[0])
    obs_R = np.asarray(obs_R, dtype=np.float32)
    obs_L = np.asarray(info["otherObs"], dtype=np.float32)
    step = 0
    max_steps = 3100
    while not done and step < max_steps:
        a_L = predict_action(agent_left,  obs_L)
        a_R = predict_action(agent_right, obs_R)
        obs_R, _r, done, info = env.step(ACTION_TABLE[a_R], otherAction=ACTION_TABLE[a_L])
        obs_R = np.asarray(obs_R, dtype=np.float32)
        obs_L = np.asarray(info["otherObs"], dtype=np.float32)
        step += 1
    return int(env.game.agent_left.life), int(env.game.agent_right.life), step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Agent A: 'baseline' or path/to/A.onnx")
    ap.add_argument("--b", required=True, help="Agent B: 'baseline' or path/to/B.onnx")
    ap.add_argument("--n", type=int, default=200,
                    help="Total matches (split half A-on-left, half A-on-right).")
    ap.add_argument("--seed", type=int, default=20260507)
    args = ap.parse_args()

    agent_A = make_agent(args.a)
    agent_B = make_agent(args.b)

    env = gym.make("SlimeVolley-v0").unwrapped

    # Half of matches: A on LEFT, B on RIGHT.
    # Other half: A on RIGHT, B on LEFT.
    n_each = args.n // 2
    results = {"A_win": 0, "B_win": 0, "draw": 0}
    margin_hist: Counter = Counter()  # margin = A_life - B_life
    A_life_total = 0
    B_life_total = 0
    steps_list: List[int] = []

    t0 = time.time()
    for i in range(n_each):
        seed = args.seed + i
        l_life, r_life, steps = run_match(agent_A, agent_B, env, seed)
        A_life, B_life = l_life, r_life
        margin = A_life - B_life
        margin_hist[margin] += 1
        A_life_total += A_life; B_life_total += B_life; steps_list.append(steps)
        if margin > 0:   results["A_win"] += 1
        elif margin < 0: results["B_win"] += 1
        else:            results["draw"] += 1

    for i in range(n_each):
        seed = args.seed + 1_000_000 + i  # disjoint seed range
        l_life, r_life, steps = run_match(agent_B, agent_A, env, seed)
        A_life, B_life = r_life, l_life
        margin = A_life - B_life
        margin_hist[margin] += 1
        A_life_total += A_life; B_life_total += B_life; steps_list.append(steps)
        if margin > 0:   results["A_win"] += 1
        elif margin < 0: results["B_win"] += 1
        else:            results["draw"] += 1

    elapsed = time.time() - t0
    n_total = n_each * 2

    print(f"\n# Head-to-head: A={args.a}  vs  B={args.b}")
    print(f"#   N={n_total} matches ({n_each} A-left + {n_each} A-right), seed_base={args.seed}\n")
    print(f"A wins: {results['A_win']:4d}  ({100*results['A_win']/n_total:.1f}%)")
    print(f"B wins: {results['B_win']:4d}  ({100*results['B_win']/n_total:.1f}%)")
    print(f"Draws : {results['draw']:4d}  ({100*results['draw']/n_total:.1f}%)")
    print(f"\nMean lives: A={A_life_total/n_total:.2f}  B={B_life_total/n_total:.2f}")
    print(f"Mean steps: {np.mean(steps_list):.0f}  (median {int(np.median(steps_list))})")
    print(f"\nLife-margin (A - B) distribution:")
    for m in sorted(margin_hist):
        bar = "#" * margin_hist[m]
        sign = "+" if m > 0 else ""
        print(f"  {sign}{m:>3d} : {margin_hist[m]:4d}  {bar}")
    print(f"\n[time] {elapsed:.1f}s ({n_total/elapsed:.1f} matches/s)")


if __name__ == "__main__":
    main()
