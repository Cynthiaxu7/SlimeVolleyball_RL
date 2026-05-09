"""Generate publication-quality matplotlib figures for the Slime Volleyball RL paper.

Outputs four PDFs into report/figures/:
  - training_curves.pdf  (Fig 1: training reward across 5 representative runs)
  - ablation_bars.pdf    (Fig 2: 5-cell component ablation)
  - elo_ladder.pdf       (Fig 3: 19-variant aggregate ELO horizontal bars)
  - ppo_entropy.pdf      (Fig 4: PPO entropy collapse vs rescue)

Usage (from project root):
    conda activate slime_rl
    python scripts/plot_report_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tbparse import SummaryReader

# ---------------------------------------------------------------------------
# Paths and global style
# ---------------------------------------------------------------------------
ROOT = Path("/home/shuhang/YBJ/RL_test")
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,  # embed TrueType so reviewers can copy text
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_scalar(run_dir: Path, tag: str):
    """Return (steps, values) numpy arrays for a single TB scalar tag."""
    df = SummaryReader(str(run_dir)).scalars
    sub = df[df["tag"] == tag].sort_values("step")
    return sub["step"].to_numpy(), sub["value"].to_numpy()


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered-ish trailing rolling mean. Pads start so output length matches."""
    if len(x) == 0 or window <= 1:
        return x
    w = min(window, len(x))
    kernel = np.ones(w) / w
    smoothed = np.convolve(x, kernel, mode="valid")
    pad = np.full(len(x) - len(smoothed), smoothed[0])
    return np.concatenate([pad, smoothed])


# ---------------------------------------------------------------------------
# Figure 1: training reward curves
# ---------------------------------------------------------------------------
def figure_training_curves():
    runs = [
        ("dqn_v1_vec", "V1 cold (5M)", "#1f77b4"),
        ("dqn_v1_vec_selfplay_fixed", "V1 self-play (fixed)", "#2ca02c"),
        ("dqn_rainbow_lite_sp_20m_fixed", "Rainbow Lite self-play", "#9467bd"),
        ("ppo_lowent_20m_fixed", "PPO (entropy collapse)", "#d62728"),
        ("ppo_rescue_20m", "PPO rescue", "#ff7f0e"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, label, color in runs:
        steps, vals = load_scalar(LOG_DIR / name, "rollout/episode_return")
        if len(steps) == 0:
            print(f"[Fig1] WARNING: no rollout/episode_return for {name}")
            continue
        # rollout/episode_return is per-episode; smooth heavily to declutter.
        win = max(50, len(vals) // 200)
        smooth = rolling_mean(vals, win)
        ax.plot(steps / 1e6, smooth, label=label, color=color, linewidth=1.4, alpha=0.9)

    ax.set_xlabel("global step (M)")
    ax.set_ylabel("episode return (rolling mean)")
    ax.set_title("Training reward over time")
    ax.axhline(0.0, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out = OUT_DIR / "training_curves.pdf"
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"[Fig1] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: component ablation bars
# ---------------------------------------------------------------------------
def figure_ablation_bars():
    # Ground truth from docs/ABLATIONS.md Section 1.
    cells = [
        ("A: Plain DQN", -0.6, 0.5),
        ("B: Dueling (v0)", -0.4, None),
        ("C: Dueling + Double", 0.2, 0.4),
        ("D: Dueling + N-step", -2.2, 1.2),
        ("E: Full v1 (D+Dbl+N)", -0.4, None),
    ]

    labels = [c[0] for c in cells]
    values = np.array([c[1] for c in cells])
    errs = np.array([c[2] if c[2] is not None else 0.0 for c in cells])

    # Color-code: green=best, red=worst, neutral=blue-ish for the rest.
    best_idx = int(np.argmax(values))
    worst_idx = int(np.argmin(values))
    colors = []
    for i in range(len(cells)):
        if i == best_idx:
            colors.append("#2ca02c")
        elif i == worst_idx:
            colors.append("#d62728")
        else:
            colors.append("#4c72b0")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.arange(len(cells))
    ax.bar(
        xs,
        values,
        yerr=errs,
        color=colors,
        ecolor="#444444",
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
    )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"eval $R$ vs scripted baseline")
    ax.set_title("Component ablation: Dueling x Double x N-step (5M steps)")

    # Annotate each bar with its value (above for positive, below for negative).
    for i, (v, e) in enumerate(zip(values, errs)):
        offset = 0.18 + e
        if v >= 0:
            ax.text(i, v + offset, f"{v:+.1f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(i, v - offset, f"{v:+.1f}", ha="center", va="top", fontsize=9)

    ymin = min(values - errs) - 0.8
    ymax = max(values + errs) + 0.8
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    out = OUT_DIR / "ablation_bars.pdf"
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"[Fig2] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: ELO ladder
# ---------------------------------------------------------------------------
def figure_elo_ladder():
    with open(LOG_DIR / "ladder_sim_20260508_102033.json") as f:
        data = json.load(f)

    by_variant: dict[str, list[float]] = {}
    for p in data["players"]:
        by_variant.setdefault(p["variant_key"], []).append(p["rating"])
    means = {k: float(np.mean(v)) for k, v in by_variant.items()}
    items = sorted(means.items(), key=lambda kv: kv[1])  # ascending so top of bar plot is best

    def family_color(key: str) -> str:
        if key == "baseline":
            return "#7f7f7f"
        if key.startswith("v1"):
            return "#1f77b4"
        if key.startswith("rainbow"):
            return "#9467bd"
        if key.startswith("ppo"):
            return "#ff7f0e"
        if key in {"plain", "dueling"} or key.startswith("duel_"):
            return "#7fbfe0"
        return "#bcbd22"

    keys = [k for k, _ in items]
    elos = np.array([v for _, v in items])
    colors = [family_color(k) for k in keys]

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ys = np.arange(len(keys))
    ax.barh(ys, elos, color=colors, edgecolor="black", linewidth=0.4, alpha=0.92)
    ax.set_yticks(ys)
    ax.set_yticklabels(keys)
    ax.axvline(1500, color="black", ls=":", lw=0.7, alpha=0.7, label="init = 1500")
    ax.set_xlabel("mean ELO (1500 ladder matches, K=32)")
    ax.set_title("Per-variant aggregate ELO")

    # Annotate ELO numbers to the right of each bar.
    xmin, xmax = elos.min(), elos.max()
    span = xmax - xmin
    pad = span * 0.012
    for y, v in zip(ys, elos):
        ax.text(v + pad, y, f"{v:.0f}", va="center", ha="left", fontsize=8)

    # Bold the baseline tick label so the reader sees the "scripted" reference fast.
    for tick in ax.get_yticklabels():
        if tick.get_text() == "baseline":
            tick.set_fontweight("bold")
            tick.set_color("#444444")

    # Family legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="#1f77b4", label="V1 family"),
        Patch(color="#9467bd", label="Rainbow family"),
        Patch(color="#ff7f0e", label="PPO family"),
        Patch(color="#7fbfe0", label="Plain / Dueling / N-step"),
        Patch(color="#7f7f7f", label="Scripted baseline"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    ax.set_xlim(xmin - span * 0.02, xmax + span * 0.10)
    fig.tight_layout()
    out = OUT_DIR / "elo_ladder.pdf"
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"[Fig3] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: PPO entropy
# ---------------------------------------------------------------------------
def figure_ppo_entropy():
    runs = [
        ("ppo_lowent_20m_fixed", "PPO (entropy collapsed)", "#d62728"),
        ("ppo_rescue_20m", "PPO rescue (ent_coef 0.01)", "#ff7f0e"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    collapsed_steps = collapsed_vals = None
    for name, label, color in runs:
        steps, vals = load_scalar(LOG_DIR / name, "train/entropy")
        if len(steps) == 0:
            print(f"[Fig4] WARNING: no train/entropy for {name}")
            continue
        smooth = rolling_mean(vals, max(20, len(vals) // 200))
        ax.plot(steps / 1e6, smooth, label=label, color=color, linewidth=1.4, alpha=0.9)
        if name == "ppo_lowent_20m_fixed":
            collapsed_steps, collapsed_vals = steps, smooth

    # Max-entropy reference line for 6-action discrete policy.
    max_ent = float(np.log(6))  # ~1.7918
    ax.axhline(max_ent, color="gray", ls=":", lw=1.0, label=f"max entropy = log(6) approx {max_ent:.2f}")

    # Annotate the entropy-collapse cliff on the failed run.
    if collapsed_steps is not None and len(collapsed_vals) > 0:
        threshold = 0.5
        below = np.where(collapsed_vals < threshold)[0]
        if len(below):
            i_cliff = int(below[0])
            x_cliff = collapsed_steps[i_cliff] / 1e6
            y_cliff = collapsed_vals[i_cliff]
            ax.annotate(
                "entropy collapse",
                xy=(x_cliff, y_cliff),
                xytext=(x_cliff + 4.0, y_cliff + 0.55),
                fontsize=9,
                color="#a00000",
                arrowprops=dict(arrowstyle="->", color="#a00000", lw=0.9),
            )

    ax.set_xlabel("global step (M)")
    ax.set_ylabel("policy entropy H")
    ax.set_title("PPO policy entropy: collapse vs rescue")
    ax.set_ylim(-0.05, max_ent + 0.20)
    ax.legend(loc="center right", framealpha=0.9)
    fig.tight_layout()
    out = OUT_DIR / "ppo_entropy.pdf"
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"[Fig4] wrote {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    figure_training_curves()
    figure_ablation_bars()
    figure_elo_ladder()
    figure_ppo_entropy()
    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
