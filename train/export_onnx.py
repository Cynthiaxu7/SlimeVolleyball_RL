"""Export a Dueling DQN PyTorch checkpoint to ONNX and validate numerical parity.

Loads a checkpoint produced by the trainer, reconstructs the Q-network, exports
it to ONNX (opset 17, dynamic batch axis), and validates that ONNX Runtime
output matches PyTorch output to within a tight tolerance.
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
MAX_ABS_DIFF_TOL: float = 1e-5


class DuelingQNet(nn.Module):
    """Dueling DQN: shared trunk feeding value and advantage heads.

    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        v = self.value_head(features)
        a = self.advantage_head(features)
        # Dueling combination: subtract mean advantage for identifiability of V.
        return v + (a - a.mean(dim=-1, keepdim=True))


class PlainQNet(nn.Module):
    """Plain single-head Q net (matches PlainQMLPv0 in train_dqn_vec.py).
    Same trunk + same total parameter shape as DuelingQNet but no V/A split.
    Used for the --no-dueling ablation cell.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Dueling DQN to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_smoke.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/shuhang/YBJ/RL_test/web/model.onnx",
    )
    parser.add_argument("--n-validate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path: Path) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(str(path), map_location="cpu")
    obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
    n_actions = int(ckpt.get("n_actions", N_ACTIONS))
    if obs_dim != OBS_DIM or n_actions != N_ACTIONS:
        raise ValueError(
            f"Checkpoint dims (obs={obs_dim}, act={n_actions}) do not match "
            f"expected ({OBS_DIM}, {N_ACTIONS})."
        )
    # Auto-detect Dueling vs plain Q from the state_dict keys. Dueling has
    # value_head.* + advantage_head.* layers; plain has q_head.*. The arch
    # string is informative but not authoritative — checkpoints from older
    # runs may not have it set.
    sd = ckpt["model_state_dict"]
    arch = ckpt.get("arch", "")
    is_plain = ("q_head.0.weight" in sd) or arch.startswith("plain_q")
    if is_plain:
        model = PlainQNet(obs_dim=obs_dim, n_actions=n_actions)
    else:
        model = DuelingQNet(obs_dim=obs_dim, n_actions=n_actions)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ckpt


def export_to_onnx(model: DuelingQNet, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=17,
            input_names=["observation"],
            output_names=["q_values"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "q_values": {0: "batch_size"},
            },
            do_constant_folding=True,
            export_params=True,
        )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def validate(
    model: DuelingQNet,
    onnx_path: Path,
    n_validate: int,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
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
        ox_out = session.run(["q_values"], {input_name: obs_np})[0]

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
    timesteps = ckpt.get("total_timesteps_trained", "<unknown>")
    final_reward = ckpt.get("final_eval_reward", float("nan"))
    print(f"Loaded checkpoint: {checkpoint_path}")
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
    print(f"  pytorch Q:     {first_pt.ravel().tolist()}")
    print(f"  onnx Q:        {first_ox.ravel().tolist()}")
    print("[OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
