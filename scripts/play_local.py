"""Interactive pygame visualization of slime volleyball.

Lets you play human-vs-AI or watch AI-vs-AI on any combination of trained
variants. Designed to be runnable standalone or bundled via PyInstaller.

CLI:
    python scripts/play_local.py --left human --right v1_selfplay
    python scripts/play_local.py --left baseline --right rainbow --auto

Variants:
    human, dueling, v1, v1_selfplay, rainbow, ppo, baseline

Controls:
    Left human:  A=forward (toward net), D=backward, W=jump
    Right human: J=forward (toward net), L=backward, I=jump
    Esc=quit, R=restart

The "forward" direction for an agent is "toward the opponent" because the
env uses a self.dir flag and bakes the agent's perspective into the action,
so the same Discrete(6) actions work for both sides — but at the keyboard
level we want left/right intuition. We mirror the keymap so A/J both push
the slime toward the centre net regardless of which side it's on.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

import gym  # noqa: F401  -- registers SlimeVolley-v0 via slimevolleygym
import slimevolleygym  # noqa: F401


# ---------------------------------------------------------------------------
# Discrete(6) action table — same as scripts/ladder_sim.py / web/physics.js.
# Order is (forward, backward, jump) where "forward" is the agent's own
# notion of forward (toward the opponent).
# ---------------------------------------------------------------------------
ACTION_TABLE: List[List[int]] = [
    [0, 0, 0],  # 0 NOOP
    [1, 0, 0],  # 1 forward
    [1, 0, 1],  # 2 forward + jump
    [0, 0, 1],  # 3 jump
    [0, 1, 1],  # 4 backward + jump
    [0, 1, 0],  # 5 backward
]
INVERSE_ACTION_TABLE = {
    (0, 0, 0): 0, (1, 0, 0): 1, (1, 0, 1): 2,
    (0, 0, 1): 3, (0, 1, 1): 4, (0, 1, 0): 5,
}


VARIANT_TO_FILE = {
    "dueling":      "model.onnx",
    "v1":           "model_v1.onnx",
    "v1_selfplay":  "model_v1_selfplay.onnx",
    "rainbow":      "model_rainbow.onnx",
    "ppo":          "model_ppo.onnx",
}
VALID_AGENTS = list(VARIANT_TO_FILE.keys()) + ["baseline", "human"]


def model_dir() -> str:
    """Resolve where bundled ONNX models live.

    When frozen by PyInstaller (--onefile), data files are unpacked into
    sys._MEIPASS/<add-data destination>. Otherwise we fall back to the
    repo's web/ folder so the script is runnable from source too.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "models")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class OnnxAgent:
    """Stateless ONNX greedy-argmax agent. Lazily imports onnxruntime so
    that --left human --right human still works without it installed."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_action(self, obs: np.ndarray) -> int:
        out = self.session.run(
            None, {self.input_name: obs[None].astype(np.float32)},
        )[0]
        return int(np.argmax(out[0]))

    def reset(self) -> None:  # match BaselineAgent's interface
        pass


class BaselineAgent:
    """Wrap slimevolleygym BaselinePolicy. Stateful (tanh-RNN), so each
    side gets its own instance."""

    def __init__(self) -> None:
        from slimevolleygym.slimevolley import BaselinePolicy
        self.policy = BaselinePolicy()

    def reset(self) -> None:
        self.policy.reset()

    def predict_action(self, obs: np.ndarray) -> int:
        # Env scales obs by 1/10; BaselinePolicy was trained on raw scale.
        binary = self.policy.predict(obs * 10.0)
        return INVERSE_ACTION_TABLE.get(
            (int(binary[0]), int(binary[1]), int(binary[2])), 0,
        )


class HumanAgent:
    """Reads pygame key state. Maps the configured 3-key set to one of
    the 6 Discrete actions. The ``side`` flag flips A/D semantics so the
    keymap matches each player's screen-relative intuition.

    Keymap (matches the spec):
        Left player:  A=move screen-left, D=move screen-right, W=jump
        Right player: J=move screen-left, L=move screen-right, I=jump

    The env's "forward" means "toward the opponent" because Agent.update
    flips vx by self.dir. So for the LEFT slime, screen-right (D) is
    forward; for the RIGHT slime, screen-left (J) is forward.
    """

    def __init__(self, side: str) -> None:
        import pygame
        if side == "left":
            self._fwd_key  = pygame.K_d  # toward the centre net
            self._back_key = pygame.K_a
            self._jump_key = pygame.K_w
        else:
            self._fwd_key  = pygame.K_j
            self._back_key = pygame.K_l
            self._jump_key = pygame.K_i

    def predict_action(self, obs: np.ndarray) -> int:  # obs unused
        import pygame
        keys = pygame.key.get_pressed()
        f = 1 if keys[self._fwd_key] else 0
        b = 1 if keys[self._back_key] else 0
        j = 1 if keys[self._jump_key] else 0
        return INVERSE_ACTION_TABLE.get((f, b, j), 0)

    def reset(self) -> None:
        pass


def make_agent(name: str, side: str) -> object:
    if name == "human":
        return HumanAgent(side)
    if name == "baseline":
        return BaselineAgent()
    if name not in VARIANT_TO_FILE:
        raise ValueError(f"Unknown agent variant: {name}")
    path = os.path.join(model_dir(), VARIANT_TO_FILE[name])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing ONNX model for {name}: {path}. "
            f"If running from source, ensure web/{VARIANT_TO_FILE[name]} exists.",
        )
    return OnnxAgent(path)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
# Pulled from slimevolleygym/slimevolley.py (the env's own constants). We
# don't import them because we want this script to work even if the env
# code drifts; the geometry is fixed by the trained policies' obs space.
REF_W = 24 * 2
REF_H = REF_W
REF_U = 1.5
REF_WALL_WIDTH = 1.0
REF_WALL_HEIGHT = 3.5
WIN_W, WIN_H = 1200, 500
FACTOR = WIN_W / REF_W


def to_screen(x: float, y: float) -> Tuple[int, int]:
    """World coords -> pygame pixel coords.
    World x is centred at 0 (range ~[-REF_W/2, REF_W/2]); world y is up.
    Pygame y grows downward, so we flip.
    """
    sx = int((x + REF_W / 2) * FACTOR)
    sy = int(WIN_H - y * FACTOR)
    return sx, sy


def draw_world(surface, game) -> None:
    import pygame
    BG     = (255, 255, 255)
    GROUND = (128, 227, 153)
    FENCE  = (240, 210, 130)
    BALL   = (255, 200, 20)
    LEFT_C = (240,  75,   0)
    RIGHT_C = (0, 150, 255)

    surface.fill(BG)

    # Ground (a wall at y=0..REF_U, full width).
    gx, gy = to_screen(-REF_W / 2, REF_U)
    pygame.draw.rect(surface, GROUND, (0, gy, WIN_W, WIN_H - gy))

    # Fence (centred, REF_WALL_WIDTH wide, REF_WALL_HEIGHT tall, sitting on ground).
    fx, fy = to_screen(-REF_WALL_WIDTH / 2, REF_U + REF_WALL_HEIGHT)
    pygame.draw.rect(
        surface, FENCE,
        (fx, fy, int(REF_WALL_WIDTH * FACTOR),
         int(REF_WALL_HEIGHT * FACTOR)),
    )

    # Agents drawn as half-circles (top half only — slimes sit on ground).
    for agent, colour in (
        (game.agent_left,  LEFT_C),
        (game.agent_right, RIGHT_C),
    ):
        cx, cy = to_screen(agent.x, agent.y)
        r = int(agent.r * FACTOR)
        # Half-circle: clip to a rect that ends at the agent's centre y.
        rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
        prev_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(cx - r, cy - r, 2 * r, r))
        pygame.draw.ellipse(surface, colour, rect)
        surface.set_clip(prev_clip)
        # Eye (white blob + black pupil tracking the ball).
        ex_offset = (0.6 * agent.r) * (1 if agent.dir == -1 else -1)
        eye_world_x = agent.x + ex_offset
        eye_world_y = agent.y + 0.6 * agent.r
        ex, ey = to_screen(eye_world_x, eye_world_y)
        pygame.draw.circle(surface, (255, 255, 255), (ex, ey), max(2, int(r * 0.3)))
        # Pupil tracks the ball direction.
        bx_world, by_world = game.ball.x, game.ball.y
        dx = bx_world - eye_world_x
        dy = by_world - eye_world_y
        d  = (dx * dx + dy * dy) ** 0.5 or 1.0
        pupil_off = 0.15 * agent.r
        px, py = to_screen(eye_world_x + dx / d * pupil_off,
                           eye_world_y + dy / d * pupil_off)
        pygame.draw.circle(surface, (0, 0, 0), (px, py), max(1, int(r * 0.1)))

    # Ball.
    bx, by = to_screen(game.ball.x, game.ball.y)
    pygame.draw.circle(surface, BALL, (bx, by), int(game.ball.r * FACTOR))

    # Lives as little coins above the playing field, one per remaining life
    # beyond the current point.
    for agent in (game.agent_left, game.agent_right):
        for i in range(1, agent.life):
            cx_world = agent.dir * (REF_W / 2 + 0.5 - i * 2.0)
            cx, cy = to_screen(cx_world, REF_H - 1.5)
            pygame.draw.circle(surface, FENCE, (cx, cy), max(2, int(0.5 * FACTOR)))


def draw_hud(surface, font, left_name: str, right_name: str,
             left_life: int, right_life: int,
             banner: Optional[str]) -> None:
    import pygame
    label = font.render(
        f"{left_name} ({left_life})   vs   {right_name} ({right_life})",
        True, (40, 40, 40),
    )
    surface.blit(label, (WIN_W // 2 - label.get_width() // 2, 10))
    if banner:
        big = pygame.font.SysFont(None, 64)
        text = big.render(banner, True, (200, 30, 30))
        surface.blit(
            text,
            (WIN_W // 2 - text.get_width() // 2,
             WIN_H // 2 - text.get_height() // 2),
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def play(left_name: str, right_name: str, auto: bool, fps: int = 30) -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption(f"Slime Volley — {left_name} vs {right_name}")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    env = gym.make("SlimeVolley-v0").unwrapped

    def new_episode():
        agents = (make_agent(left_name, "left"), make_agent(right_name, "right"))
        for ag in agents:
            ag.reset()
        env.reset()
        # Noop-noop priming step — same trick as ladder_sim/run_match: makes
        # info["otherObs"] available before the first real action choice.
        obs_R, _r, done, info = env.step(ACTION_TABLE[0], otherAction=ACTION_TABLE[0])
        return agents, np.asarray(info["otherObs"], dtype=np.float32), \
               np.asarray(obs_R, dtype=np.float32), done

    agents, obs_L, obs_R, done = new_episode()
    banner: Optional[str] = None
    last_left_life = env.game.agent_left.life
    last_right_life = env.game.agent_right.life

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    agents, obs_L, obs_R, done = new_episode()
                    banner = None
                    last_left_life = env.game.agent_left.life
                    last_right_life = env.game.agent_right.life

        if not done:
            a_L = agents[0].predict_action(obs_L)
            a_R = agents[1].predict_action(obs_R)
            obs_R_new, _r, done, info = env.step(
                ACTION_TABLE[a_R], otherAction=ACTION_TABLE[a_L],
            )
            obs_R = np.asarray(obs_R_new, dtype=np.float32)
            obs_L = np.asarray(info["otherObs"], dtype=np.float32)

            # Detect a life lost on either side and flash a short banner.
            ll = env.game.agent_left.life
            rl = env.game.agent_right.life
            if ll < last_left_life:
                banner = "Point for right!"
            elif rl < last_right_life:
                banner = "Point for left!"
            last_left_life, last_right_life = ll, rl

            if done:
                if env.game.agent_left.life > env.game.agent_right.life:
                    banner = (
                        "Player wins!" if left_name == "human" else
                        ("AI wins!" if right_name == "human" else f"{left_name} wins!")
                    )
                elif env.game.agent_right.life > env.game.agent_left.life:
                    banner = (
                        "Player wins!" if right_name == "human" else
                        ("AI wins!" if left_name == "human" else f"{right_name} wins!")
                    )
                else:
                    banner = "Draw!"
        elif auto:
            # Auto-restart in spectator mode for a continuous loop demo.
            agents, obs_L, obs_R, done = new_episode()
            banner = None
            last_left_life = env.game.agent_left.life
            last_right_life = env.game.agent_right.life

        draw_world(screen, env.game)
        draw_hud(
            screen, font, left_name, right_name,
            env.game.agent_left.life, env.game.agent_right.life,
            banner,
        )
        pygame.display.flip()
        clock.tick(fps)

    env.close()
    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive slime volley demo.")
    ap.add_argument("--left",  default="human",   choices=VALID_AGENTS)
    ap.add_argument("--right", default="rainbow", choices=VALID_AGENTS)
    ap.add_argument("--auto",  action="store_true",
                    help="In AI-vs-AI mode, auto-restart after each game.")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    play(args.left, args.right, args.auto, fps=args.fps)


if __name__ == "__main__":
    main()
