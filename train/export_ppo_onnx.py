"""Export a PPO actor-critic checkpoint to ONNX (actor logits only).

Loads a checkpoint produced by ``train_ppo_selfplay_vec.py``, reconstructs the
``ActorCriticMLP`` inline (so this script is independent of the trainer
module), exports the actor's logits path (``forward(obs) -> logits``) to
ONNX, and validates ONNX Runtime parity against PyTorch on random inputs.

Why export logits (not softmax / argmax)
----------------------------------------
The web frontend takes ``argmax`` over the model's output to pick a discrete
action. Argmax is invariant under softmax (argmax of logits == argmax of
probabilities), so exporting logits gives the same deterministic policy with
strictly fewer ONNX ops. The eval/inference path in the trainer also uses
``argmax(logits)``; this keeps the exported model bit-identical-up-to-fp-noise
to the PyTorch greedy policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


OBS_DIM: int = 12
N_ACTIONS: int = 6
HIDDEN: int = 128
MAX_ABS_DIFF_TOL: float = 1e-5


# ---------------------------------------------------------------------------
# Model (reconstructed inline; mirrors trainer's ActorCriticMLP exactly)
# ---------------------------------------------------------------------------
class ActorCriticMLP(nn.Module):
    """PPO actor-critic with a Tanh trunk and a logits head.

    NOTE: this is a stand-alone reconstruction of the trainer's class. The
    ``forward`` method intentionally returns ONLY actor logits (the ONNX
    target). The critic head exists in the state_dict but is unused here.
    Initialization scheme is irrelevant since we strict-load the trained
    weights immediately after construction.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = HIDDEN) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Actor logits only — the web frontend's argmax contract works as-is.
        h = self.trunk(obs)
        return self.actor(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PPO actor-critic checkpoint to ONNX (actor logits)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/shuhang/YBJ/RL_test/checkpoints/ppo_selfplay_vec.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/shuhang/YBJ/RL_test/web/model_ppo.onnx",
    )
    parser.add_argument("--n-validate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path: Path) -> Tuple[ActorCriticMLP, dict]:
    """Reconstruct ActorCriticMLP from checkpoint metadata + strict-load weights.

    Reads ``obs_dim``, ``n_actions``, and ``hidden`` from the checkpoint dict
    (with sensible defaults matching the trainer). Verifies the dims match
    what the exporter / web frontend expect (12, 6) so we fail fast on a
    stale/incompatible checkpoint rather than silently exporting garbage.
    """
    ckpt = torch.load(str(path), map_location="cpu")
    obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
    n_actions = int(ckpt.get("n_actions", N_ACTIONS))
    hidden = int(ckpt.get("hidden", HIDDEN))

    if obs_dim != OBS_DIM or n_actions != N_ACTIONS:
        raise ValueError(
            f"Checkpoint dims (obs={obs_dim}, act={n_actions}) do not match "
            f"expected ({OBS_DIM}, {N_ACTIONS})."
        )

    arch = ckpt.get("arch", "<unknown>")
    if arch and arch != "ppo_actor_critic_mlp_128":
        # Not a hard error: some experiments may use a different hidden size.
        # We warn rather than abort, but still strict-load below.
        print(f"[warn] unexpected arch={arch!r} (expected "
              f"'ppo_actor_critic_mlp_128'); proceeding with hidden={hidden}.")

    model = ActorCriticMLP(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def export_to_onnx(model: ActorCriticMLP, output_path: Path) -> None:
    """Export ``model.forward`` (actor logits) with a dynamic batch axis.

    We use opset 17 (matches export_onnx.py) and constant-folding so the
    exported graph is small and the resulting file is portable to ONNX
    Runtime Web (the frontend's runtime).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=17,
            input_names=["observation"],
            output_names=["logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
            export_params=True,
        )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def validate(
    model: ActorCriticMLP,
    onnx_path: Path,
    n_validate: int,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Forward 100 random obs through both runtimes; return diff statistics.

    On any per-sample max-abs-diff >= MAX_ABS_DIFF_TOL we print the first
    few mismatches and exit non-zero. Tight tolerance (1e-5) — the network
    is small and CPU-only inference is deterministic; meaningful drift here
    would indicate a real export bug (e.g. operator-version mismatch).
    """
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    diffs = np.zeros(n_validate, dtype=np.float64)
    mismatches: list[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    first_obs: np.ndarray | None = None
    first_pt: np.ndarray | None = None
    first_ox: np.ndarray | None = None

    max_abs = 0.0
    sum_abs = 0.0
    count_elems = 0

    for i in range(n_validate):
        obs_np = rng.uniform(-1.0, 1.0, size=(1, OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(obs_np)).cpu().numpy()
        ox_out = session.run(["logits"], {input_name: obs_np})[0]

        abs_diff = np.abs(pt_out - ox_out)
        sample_max = float(abs_diff.max())
        diffs[i] = sample_max
        max_abs = max(max_abs, sample_max)
        sum_abs += float(abs_diff.sum())
        count_elems += abs_diff.size

        if sample_max >= MAX_ABS_DIFF_TOL and len(mismatches) < 3:
            mismatches.append((i, obs_np, pt_out, ox_out))

        if i == 0:
            first_obs = obs_np
            first_pt = pt_out
            first_ox = ox_out

    mean_abs = sum_abs / max(count_elems, 1)

    if max_abs >= MAX_ABS_DIFF_TOL:
        print(f"[FAIL] max_abs_diff={max_abs:.3e} >= tol={MAX_ABS_DIFF_TOL:.0e}")
        print("First mismatches:")
        for idx, obs_np, pt_out, ox_out in mismatches:
            print(f"  sample {idx}: obs={obs_np.ravel().tolist()}")
            print(f"    pt = {pt_out.ravel().tolist()}")
            print(f"    ox = {ox_out.ravel().tolist()}")
            print(f"    diff = {(pt_out - ox_out).ravel().tolist()}")
        sys.exit(1)

    assert first_obs is not None and first_pt is not None and first_ox is not None
    return max_abs, mean_abs, first_obs, first_pt, first_ox


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.is_file():
        print(f"[FAIL] checkpoint not found: {checkpoint_path}")
        return 1

    model, ckpt = load_checkpoint(checkpoint_path)

    arch = ckpt.get("arch", "<unknown>")
    algorithm = ckpt.get("algorithm", "<unknown>")
    timesteps = ckpt.get("total_timesteps_trained", "<unknown>")
    final_reward = ckpt.get("final_eval_reward", float("nan"))
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  algorithm:                {algorithm}")
    print(f"  arch:                     {arch}")
    print(f"  total_timesteps_trained:  {timesteps}")
    print(f"  final_eval_reward:        {final_reward}")

    export_to_onnx(model, output_path)
    size_kb = output_path.stat().st_size / 1024.0
    print(f"Exported ONNX: {output_path} ({size_kb:.2f} KB)")

    max_abs, mean_abs, first_obs, first_pt, first_ox = validate(
        model, output_path, args.n_validate, rng
    )
    print(f"Validation over {args.n_validate} random samples:")
    print(f"  max_abs_diff:  {max_abs:.3e}")
    print(f"  mean_abs_diff: {mean_abs:.3e}")
    print("Sample 0:")
    print(f"  observation:   {first_obs.ravel().tolist()}")
    print(f"  pytorch logits: {first_pt.ravel().tolist()}")
    print(f"  onnx logits:    {first_ox.ravel().tolist()}")
    print(f"  argmax (pt={int(first_pt.argmax())}, ox={int(first_ox.argmax())})")
    print("[OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
