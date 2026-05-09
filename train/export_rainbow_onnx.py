"""Export a Rainbow Dueling C51 (NoisyNet) PyTorch checkpoint to ONNX and validate.

Loads a checkpoint produced by `train_dqn_rainbow.py`, reconstructs the network
inline (NoisyLinear + RainbowDuelingC51 copied verbatim from the trainer to keep
this script self-contained), disables NoisyNet noise so the export path is pure
deterministic Linear(weight_mu, bias_mu), and exports to ONNX (opset 17, dynamic
batch axis). Then validates that ONNX Runtime output matches PyTorch output to
within a tight tolerance.

ONNX-friendliness notes:
  - NoisyLinear.forward short-circuits to F.linear(x, weight_mu, bias_mu) when
    `noise_disabled=True`. We toggle that via set_all_noise_disabled(True) prior
    to torch.onnx.export, so the trace never sees the noise branch's
    register_buffer epsilons -- only plain weight_mu/bias_mu Linear ops.
  - The C51 expected-Q reduction is just softmax over the atom axis followed by
    a contraction with self.support; both are well-supported in opset 17.
  - The dueling combination uses tensor.mean(dim=1, keepdim=True), which is
    standard ReduceMean and exports cleanly.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F


OBS_DIM: int = 12
N_ACTIONS: int = 6
MAX_ABS_DIFF_TOL: float = 1e-5
DEFAULT_V_MIN: float = -5.0
DEFAULT_V_MAX: float = 5.0
DEFAULT_N_ATOMS: int = 51


# --- NoisyLinear (Fortunato et al. 2018, factorized Gaussian noise) ---
# Copied verbatim from train_dqn_rainbow.py to keep this script self-contained.
class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise on weights and biases.

    Parameters (trainable): weight_mu, weight_sigma, bias_mu, bias_sigma.
    Buffers (non-trainable): weight_epsilon, bias_epsilon.

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


# --- Rainbow Dueling C51 model ---
# Copied verbatim from train_dqn_rainbow.py to keep this script self-contained.
class RainbowDuelingC51(nn.Module):
    """Dueling C51 with NoisyNet heads.

      forward_train(obs) -> (B, n_actions, n_atoms) log-probabilities (train path).
      forward(obs)       -> (B, n_actions)          expected Q (ONNX path).
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
        """Return per-(action, atom) logits before the softmax. Shape: (B, n_actions, n_atoms)."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Rainbow Dueling C51 to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/shuhang/YBJ/RL_test/checkpoints/dqn_rainbow.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/shuhang/YBJ/RL_test/web/model_rainbow.onnx",
    )
    parser.add_argument("--n-validate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class RainbowEpsDuelingC51(nn.Module):
    """C51 + Dueling but with PLAIN Linear heads (no NoisyNet). Used for the
    epsilon-greedy Rainbow variant — eval/argmax greedy, no exploration in the
    network. Expected Q identical to RainbowDuelingC51 (same softmax-over-atoms
    contraction with self.support).
    """

    def __init__(self, obs_dim=12, n_actions=6, n_atoms=51,
                 v_min=DEFAULT_V_MIN, v_max=DEFAULT_V_MAX) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_atoms),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions * n_atoms),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        f = self.trunk(obs)
        v = self.value_head(f).view(-1, 1, self.n_atoms)
        a = self.advantage_head(f).view(-1, self.n_actions, self.n_atoms)
        logits = v + (a - a.mean(dim=1, keepdim=True))
        probs = F.softmax(logits, dim=-1)
        return (probs * self.support.view(1, 1, -1)).sum(dim=-1)


def load_checkpoint(path: Path) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
    n_actions = int(ckpt.get("n_actions", N_ACTIONS))
    if obs_dim != OBS_DIM or n_actions != N_ACTIONS:
        raise ValueError(
            f"Checkpoint dims (obs={obs_dim}, act={n_actions}) do not match "
            f"expected ({OBS_DIM}, {N_ACTIONS})."
        )
    v_min = float(ckpt.get("v_min", DEFAULT_V_MIN))
    v_max = float(ckpt.get("v_max", DEFAULT_V_MAX))
    n_atoms = int(ckpt.get("n_atoms", DEFAULT_N_ATOMS))

    # Auto-detect NoisyNet (weight_mu / weight_sigma keys) vs plain Linear
    # (weight key). Eps-greedy Rainbow uses plain Linear; full Rainbow uses
    # NoisyLinear.
    sd = ckpt["model_state_dict"]
    is_noisy = any(k.endswith(".weight_mu") for k in sd.keys())
    if is_noisy:
        model = RainbowDuelingC51(
            obs_dim=obs_dim, n_actions=n_actions, n_atoms=n_atoms,
            v_min=v_min, v_max=v_max,
        )
        model.load_state_dict(sd, strict=True)
        # Disable NoisyNet noise so forward(x) becomes deterministic
        # Linear(weight_mu, bias_mu). MUST happen before tracing.
        model.set_all_noise_disabled(True)
    else:
        model = RainbowEpsDuelingC51(
            obs_dim=obs_dim, n_actions=n_actions, n_atoms=n_atoms,
            v_min=v_min, v_max=v_max,
        )
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ckpt


def export_to_onnx(model: RainbowDuelingC51, output_path: Path) -> None:
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
    model: RainbowDuelingC51,
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
            pt_out = model(torch.from_numpy(obs_np)).cpu().numpy().astype(np.float32)
        ox_out = session.run(["q_values"], {input_name: obs_np})[0].astype(np.float32)

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
    v_min = ckpt.get("v_min", DEFAULT_V_MIN)
    v_max = ckpt.get("v_max", DEFAULT_V_MAX)
    n_atoms = ckpt.get("n_atoms", DEFAULT_N_ATOMS)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  arch:                     {arch}")
    print(f"  total_timesteps_trained:  {timesteps}")
    print(f"  final_eval_reward:        {final_reward}")
    print(f"  v_min / v_max / n_atoms:  {v_min} / {v_max} / {n_atoms}")

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
