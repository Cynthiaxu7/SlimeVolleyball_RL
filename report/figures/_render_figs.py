"""Render the three architecture / system diagrams as PDFs (and PNG previews).

Fallback for environments without LaTeX. The matching .tex sources live alongside
this script and remain the canonical, re-editable definitions. This script aims
to reproduce them with publication-quality matplotlib output.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# style
# ----------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GRAY = "#e7e7e7"
GRAY_E = "#888888"
BLUE = "#cfe0f5"
BLUE_E = "#3a78b8"
ORANGE = "#fde2c4"
ORANGE_E = "#d68a3a"
GREEN = "#cfe9d2"
GREEN_E = "#3f8a48"
PURPLE = "#e1d4ef"
PURPLE_E = "#7b50b3"
LIGHT_BLUE = "#e6f0fb"
LIGHT_ORANGE = "#feefdd"
LIGHT_GREEN = "#e7f5ea"
LIGHT_PURPLE = "#f0e8f8"


def add_box(
    ax,
    xy,
    w,
    h,
    text,
    *,
    fill=GRAY,
    edge=GRAY_E,
    fontsize=9.5,
    weight="normal",
    italic=False,
    pad=0.02,
    radius=0.03,
):
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=1.3,
        facecolor=fill,
        edgecolor=edge,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        style="italic" if italic else "normal",
        zorder=3,
    )
    return (x, y, w, h)


def arrow(ax, src, dst, *, color="#222", lw=1.0, style="-", ls="-", rad=0.0):
    a = FancyArrowPatch(
        src,
        dst,
        arrowstyle="->",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(a)


def edge_top(b):
    x, y, w, h = b
    return (x, y + h / 2)


def edge_bot(b):
    x, y, w, h = b
    return (x, y - h / 2)


def edge_left(b):
    x, y, w, h = b
    return (x - w / 2, y)


def edge_right(b):
    x, y, w, h = b
    return (x + w / 2, y)


def save(fig, name):
    pdf = HERE / f"{name}.pdf"
    png = HERE / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05, dpi=180)
    plt.close(fig)
    print(f"  wrote {pdf.name} + {png.name}")


# ======================================================================
# Diagram 1: Dueling DQN
# ======================================================================
def draw_dueling():
    fig, ax = plt.subplots(figsize=(3.9, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    ax.set_aspect("equal")
    ax.axis("off")

    BW, BH = 4.4, 0.85
    cx = 5.0

    # input
    ino = add_box(ax, (cx, 16.3), BW, BH, r"obs $\in \mathbb{R}^{12}$", fill="#f0f0f0")

    # trunk
    fc1 = add_box(ax, (cx, 15.05), BW, BH, r"Linear  $12 \to 128$", fill=GRAY)
    rl1 = add_box(ax, (cx, 13.95), BW, BH, "ReLU", fill=GRAY)
    fc2 = add_box(ax, (cx, 12.85), BW, BH, r"Linear  $128 \to 128$", fill=GRAY)
    rl2 = add_box(ax, (cx, 11.75), BW, BH, "ReLU", fill=GRAY)

    # split
    splitx_l, splitx_r = 2.5, 7.5

    # value head (left, blue)
    ax.text(splitx_l, 10.6, "value head", ha="center", va="center",
            fontsize=8, style="italic", color="#444")
    vfc1 = add_box(ax, (splitx_l, 9.85), BW, BH, r"Linear  $128 \to 64$",
                   fill=BLUE, edge=BLUE_E)
    vrl = add_box(ax, (splitx_l, 8.75), BW, BH, "ReLU", fill=BLUE, edge=BLUE_E)
    vfc2 = add_box(ax, (splitx_l, 7.65), BW, BH, r"Linear  $64 \to 1$",
                   fill=BLUE, edge=BLUE_E)
    vout = add_box(ax, (splitx_l, 6.4), BW, BH, r"$V(s) \in \mathbb{R}$",
                   fill=LIGHT_BLUE, edge=BLUE_E)

    # advantage head (right, orange)
    ax.text(splitx_r, 10.6, "advantage head", ha="center", va="center",
            fontsize=8, style="italic", color="#444")
    afc1 = add_box(ax, (splitx_r, 9.85), BW, BH, r"Linear  $128 \to 64$",
                   fill=ORANGE, edge=ORANGE_E)
    arl = add_box(ax, (splitx_r, 8.75), BW, BH, "ReLU", fill=ORANGE, edge=ORANGE_E)
    afc2 = add_box(ax, (splitx_r, 7.65), BW, BH, r"Linear  $64 \to 6$",
                   fill=ORANGE, edge=ORANGE_E)
    aout = add_box(ax, (splitx_r, 6.4), BW, BH, r"$A(s,\cdot) \in \mathbb{R}^{6}$",
                   fill=LIGHT_ORANGE, edge=ORANGE_E)

    # combine block
    comb = add_box(
        ax,
        (cx, 4.0),
        9.4,
        1.25,
        r"$Q(s,a) \,=\, V(s) \,+\, \left[ A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a') \right]$",
        fill=LIGHT_GREEN,
        edge=GREEN_E,
        fontsize=9,
    )

    # output
    out = add_box(ax, (cx, 1.85), BW, BH, r"$Q \in \mathbb{R}^{6}$",
                  fill=LIGHT_GREEN, edge=GREEN_E)

    # arrows: trunk
    for a, b in [(ino, fc1), (fc1, rl1), (rl1, fc2), (fc2, rl2)]:
        arrow(ax, edge_bot(a), edge_top(b))

    # split: rl2 -> vfc1 and rl2 -> afc1 (via a Y)
    rl2_b = edge_bot(rl2)
    yfork = rl2_b[1] - 0.35
    ax.plot([rl2_b[0], rl2_b[0]], [rl2_b[1], yfork], color="#222", lw=1.3, zorder=3)
    ax.plot([splitx_l, splitx_r], [yfork, yfork], color="#222", lw=1.3, zorder=3)
    arrow(ax, (splitx_l, yfork), edge_top(vfc1))
    arrow(ax, (splitx_r, yfork), edge_top(afc1))

    # value chain
    for a, b in [(vfc1, vrl), (vrl, vfc2), (vfc2, vout)]:
        arrow(ax, edge_bot(a), edge_top(b))
    # advantage chain
    for a, b in [(afc1, arl), (arl, afc2), (afc2, aout)]:
        arrow(ax, edge_bot(a), edge_top(b))

    # both heads -> combine
    # value out (down then right)
    vout_b = edge_bot(vout)
    aout_b = edge_bot(aout)
    comb_l = edge_left(comb)
    comb_r = edge_right(comb)
    # bend toward combine
    ymid = (vout_b[1] + comb_l[1]) / 2 + 0.1
    ax.plot([vout_b[0], vout_b[0]], [vout_b[1], ymid], color="#222", lw=1.3, zorder=3)
    arrow(ax, (vout_b[0], ymid), comb_l)
    ax.plot([aout_b[0], aout_b[0]], [aout_b[1], ymid], color="#222", lw=1.3, zorder=3)
    arrow(ax, (aout_b[0], ymid), comb_r)

    # combine -> output
    arrow(ax, edge_bot(comb), edge_top(out))

    save(fig, "arch_dueling")


# ======================================================================
# Diagram 2: PPO actor-critic
# ======================================================================
def draw_ppo():
    fig, ax = plt.subplots(figsize=(4.4, 6.6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal")
    ax.axis("off")

    BW, BH = 5.0, 0.85
    HW = 4.6  # head box width (wider so Categorical fits)
    cx = 5.0

    # input
    ino = add_box(ax, (cx, 17.2), BW, BH, r"obs $\in \mathbb{R}^{12}$", fill="#f0f0f0")

    # trunk
    fc1 = add_box(ax, (cx, 16.0), BW, BH, r"Linear  $12 \to 128$", fill=GRAY)
    th1 = add_box(ax, (cx, 14.9), BW, BH, "Tanh", fill=GRAY)
    fc2 = add_box(ax, (cx, 13.8), BW, BH, r"Linear  $128 \to 128$", fill=GRAY)
    th2 = add_box(ax, (cx, 12.7), BW, BH, "Tanh", fill=GRAY)

    splitx_l, splitx_r = 2.4, 7.6

    # actor head
    ax.text(splitx_l, 11.55, "actor head", ha="center", va="center",
            fontsize=8, style="italic", color="#444")
    afc = add_box(ax, (splitx_l, 10.8), HW, BH, r"Linear  $128 \to 6$",
                  fill=BLUE, edge=BLUE_E)
    alog = add_box(ax, (splitx_l, 9.7), HW, BH, r"logits $\in \mathbb{R}^{6}$",
                   fill=LIGHT_BLUE, edge=BLUE_E)
    acat = add_box(ax, (splitx_l, 8.6), HW, BH, "Categorical(softmax)",
                   fill=BLUE, edge=BLUE_E, fontsize=8.5)
    aact = add_box(ax, (splitx_l, 7.4), HW, BH, r"action  $a \sim \pi_{\theta}(\cdot|s)$",
                   fill=LIGHT_BLUE, edge=BLUE_E, fontsize=9)

    # critic head
    ax.text(splitx_r, 11.55, "critic head", ha="center", va="center",
            fontsize=8, style="italic", color="#444")
    cfc = add_box(ax, (splitx_r, 10.8), HW, BH, r"Linear  $128 \to 1$",
                  fill=ORANGE, edge=ORANGE_E)
    vout = add_box(ax, (splitx_r, 9.7), HW, BH, r"$V(s) \in \mathbb{R}$",
                   fill=LIGHT_ORANGE, edge=ORANGE_E)

    # losses
    lpi = add_box(
        ax, (splitx_l, 4.6), 4.8, 1.4,
        "PPO clipped surrogate\n" + r"$\mathcal{L}^{\mathsf{CLIP}}(\theta)$",
        fill=LIGHT_GREEN, edge=GREEN_E, fontsize=9,
    )
    lv = add_box(
        ax, (splitx_r, 4.6), 4.8, 1.4,
        "value loss\n" + r"$\mathcal{L}^{V}(\phi) = (V_{\phi}(s)-\hat{R})^{2}$",
        fill=LIGHT_GREEN, edge=GREEN_E, fontsize=8.5,
    )

    # arrows: trunk
    for a, b in [(ino, fc1), (fc1, th1), (th1, fc2), (fc2, th2)]:
        arrow(ax, edge_bot(a), edge_top(b))

    # split
    th2_b = edge_bot(th2)
    yfork = th2_b[1] - 0.35
    ax.plot([th2_b[0], th2_b[0]], [th2_b[1], yfork], color="#222", lw=1.3, zorder=3)
    ax.plot([splitx_l, splitx_r], [yfork, yfork], color="#222", lw=1.3, zorder=3)
    arrow(ax, (splitx_l, yfork), edge_top(afc))
    arrow(ax, (splitx_r, yfork), edge_top(cfc))

    # actor chain
    for a, b in [(afc, alog), (alog, acat), (acat, aact)]:
        arrow(ax, edge_bot(a), edge_top(b))
    # critic chain
    arrow(ax, edge_bot(cfc), edge_top(vout))

    # dashed arrows to losses
    arrow(ax, edge_bot(aact), edge_top(lpi), ls="--", color="#444")
    arrow(ax, edge_bot(vout), edge_top(lv), ls="--", color="#444")

    ax.text(
        cx, 3.55,
        "dashed = training-only loss path",
        ha="center", va="center",
        fontsize=7.5, style="italic", color="#666",
    )

    save(fig, "arch_ppo")


# ======================================================================
# Diagram 3: Self-play training loop
# ======================================================================
def draw_selfplay():
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-0.5, 14.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # vec env (top center)
    env = add_box(
        ax, (9.5, 12.4), 6.6, 1.6,
        "Vec env (16 parallel)\n" + r"SlimeVolley-v0",
        fill=GREEN, edge=GREEN_E, fontsize=10, weight="bold",
    )

    # online policy (center left)
    agent = add_box(
        ax, (3.8, 8.4), 6.4, 1.7,
        "Online policy\n" + r"$\pi_{\theta}$  (agent being trained)",
        fill=BLUE, edge=BLUE_E, fontsize=10, weight="bold",
    )

    # opponent assignment (center right) - widened so all text fits
    opp = add_box(
        ax, (14.6, 8.4), 8.4, 2.7,
        "Opponent assignment\n"
        r"with prob. $p$:  scripted Baseline" + "\n"
        r"else:  sample from FIFO pool of past" + "\n"
        r"snapshots (max size 30)",
        fill=PURPLE, edge=PURPLE_E, fontsize=8.5, weight="bold",
    )

    # buffer (bottom right)
    buf = add_box(
        ax, (14.6, 3.6), 6.4, 1.7,
        "Replay buffer (DQN)  /\nRollout buffer (PPO)",
        fill=ORANGE, edge=ORANGE_E, fontsize=10, weight="bold",
    )

    # loss + grad (bottom left)
    loss = add_box(
        ax, (3.8, 3.6), 6.4, 1.5,
        "Loss + grad step\n" + r"SGD on $\theta$",
        fill="#dcdcdc", edge="#444", fontsize=10, weight="bold",
    )

    # adapter inset (bottom center)
    adapter = add_box(
        ax, (9.5, 0.85), 7.4, 1.3,
        "BaselineOpponentAdapter\n"
        "wraps slimevolleygym scripted RNN baseline",
        fill=LIGHT_PURPLE, edge=PURPLE_E, fontsize=8.5, italic=True,
    )

    # ---------- arrows ----------

    # env -> agent (obs)
    arrow(ax, (env[0] - 2.4, edge_bot(env)[1]), (agent[0] + 1.7, edge_top(agent)[1]),
          rad=0.2)
    ax.text(5.6, 11.0, r"obs $o_t$", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # env -> opp (opp obs)
    arrow(ax, (env[0] + 2.4, edge_bot(env)[1]), (opp[0] - 2.5, edge_top(opp)[1]),
          rad=-0.2)
    ax.text(12.9, 11.0, "opp. obs", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # agent -> env (action)  curve up on the right side of agent
    arrow(ax, (agent[0] + 1.7, edge_top(agent)[1] + 0.1),
          (env[0] - 1.7, edge_bot(env)[1] - 0.05), rad=-0.25)
    ax.text(6.7, 10.05, r"$a_t \sim \pi_{\theta}$", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # opp -> env (opp action)
    arrow(ax, (opp[0] - 2.5, edge_top(opp)[1] + 0.1),
          (env[0] + 1.7, edge_bot(env)[1] - 0.05), rad=0.25)
    ax.text(12.1, 10.05, "opp. action", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # env -> buffer (transitions) - down the right side
    src = (env[0] + 3.3, edge_right(env)[1])
    # route: env right -> down to buffer
    ax.plot([src[0], 18.6, 18.6], [src[1], src[1], buf[1]],
            color="#222", lw=1.0, zorder=3)
    arrow(ax, (18.6, buf[1]), (edge_right(buf)[0] + 0.01, buf[1]))
    ax.text(18.95, 8.0, r"$(o_t,a_t,r_t,o_{t+1})$",
            fontsize=8.5, rotation=90, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # buffer -> loss (batch)
    arrow(ax, edge_left(buf), edge_right(loss))
    ax.text(9.1, 3.95, "batch", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # loss -> agent (gradient)
    arrow(ax, edge_top(loss), edge_bot(agent))
    ax.text(4.0, 6.0, r"$\nabla_{\theta}\mathcal{L}$", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # snapshot save: agent -> opponent pool (curved purple, dashed, below)
    arrow(
        ax,
        (edge_right(agent)[0], edge_right(agent)[1] - 0.45),
        (edge_left(opp)[0], edge_left(opp)[1] - 0.55),
        rad=0.35, ls="--", color=PURPLE_E, lw=1.4,
    )
    ax.text(9.2, 6.05, "snapshot every 100k steps",
            fontsize=8.5, style="italic", color="#5b3a8a",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # adapter -> opp (vertical dashed)
    arrow(ax, edge_top(adapter), (opp[0], edge_bot(opp)[1]),
          ls="--", color=PURPLE_E, lw=1.2)

    # legend (top-left small)
    handles = [
        mpatches.Patch(facecolor=BLUE, edgecolor=BLUE_E, label="agent"),
        mpatches.Patch(facecolor=GREEN, edgecolor=GREEN_E, label="environment"),
        mpatches.Patch(facecolor=PURPLE, edgecolor=PURPLE_E, label="opponent pool"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        fontsize=8,
        handlelength=1.2,
    )

    save(fig, "system_selfplay")


# ======================================================================
if __name__ == "__main__":
    print("Rendering figures into", HERE)
    draw_dueling()
    draw_ppo()
    draw_selfplay()
    print("done.")
