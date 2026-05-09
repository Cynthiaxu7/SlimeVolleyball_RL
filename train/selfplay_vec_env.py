"""SelfPlayVecEnv: parallel slimevolley envs exposing the 2-player API.

Mirrors the contract of ``vec_env.SubprocVecEnv`` but each step accepts
BOTH players' actions (right-side learner + left-side opponent) and returns
BOTH players' obs after the step. The right side is the "agent" we are
training; the left side is the snapshot opponent. Reward is the agent's.

Key design notes
----------------
* Worker holds the RAW slimevolley env, NOT the ``DiscreteSlimeWrapper``.
  We need access to ``env.step(action, otherAction=...)`` directly, which
  the action wrapper would hide. The worker converts ``Discrete(6)`` ints
  to MultiBinary via the shared ``ACTION_TABLE`` from ``train_dqn_v1``.

* slimevolley's ``env.reset()`` returns ONLY the right-side obs. To get
  the left side's initial observation we take ONE noop-noop step right
  after reset and read ``info['otherObs']``. This advances physics by 1
  frame (~0.03% of an average ~3000-frame episode — negligible). We
  considered computing the mirror manually, but the env's internal
  multiagent flag controls obs format and we avoid coupling to that. If
  the noop-noop step returns ``done=True`` (extremely unlikely on frame
  1), we recursively retry the reset.

* ``observation_space`` / ``action_space`` reported are the RAW env's
  (Box(12) and MultiBinary(3)). The trainer is expected to know the
  Discrete(6) wrapping and pass int actions; the worker performs the
  ACTION_TABLE lookup.

* Spawn start method, same picklability rules as ``SubprocVecEnv``: pass
  module-level callables (or ``functools.partial`` over them) for
  ``env_fns``; closures defined inside ``main()`` will not pickle.

* Auto-reset on done: when ``done[i]`` is True the worker emits both
  ``info['terminal_observation']`` (agent) and ``info['terminal_other_obs']``
  (opponent), then resets and runs the noop-noop step internally so the
  next ``step()`` already has both sides' fresh obs.
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import sys
from multiprocessing.connection import Connection
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np


# Make `train/` importable in the spawned children regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(remote: Connection, parent_remote: Connection,
            env_fn: Callable[[], Any], seed: Optional[int]) -> None:
    """Subprocess entrypoint. Owns one RAW slimevolley env.

    Commands (cmd, data):
      * ('step',   (my_idx, opp_idx))  -> sends (obs, other_obs, reward,
                                                 done, info); auto-resets.
      * ('reset',  None)               -> sends (obs, other_obs).
      * ('seed',   seed)               -> reseeds env + action space.
      * ('spaces', None)               -> sends (obs_space, act_space).
      * ('close',  None)               -> closes env, returns.
    """
    # Imports happen inside the child so the parent doesn't need to have
    # already loaded slimevolley/torch when workers spawn.
    import gym  # noqa: F401 -- registers SlimeVolley-v0 alongside slimevolleygym
    import slimevolleygym  # noqa: F401
    from train_dqn_v1 import ACTION_TABLE  # discrete -> [forward, backward, jump]

    parent_remote.close()
    # gym.make wraps slimevolley in OrderEnforcing/TimeLimit, neither of which
    # forwards the `otherAction` kwarg. slimevolley has its own t_limit=3000
    # internally, so .unwrapped is safe and necessary for the 2-player API.
    env = env_fn().unwrapped
    if seed is not None:
        try:
            env.seed(seed)
        except AttributeError:
            pass
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass

    def _do_reset() -> Tuple[np.ndarray, np.ndarray]:
        """Reset and run ONE noop-noop step to populate the opponent's obs.

        Rationale: ``env.reset()`` only hands back the right-side obs.
        slimevolley exposes the opponent's obs via ``info['otherObs']`` on
        step. Taking a single noop-noop step advances physics by 1 frame
        (~0.03% overhead per episode) but gives us both sides' state in a
        principled way. The alternative (manually mirroring the agent obs)
        couples to internal env state and would silently break if the env
        changes its multiagent obs encoding.
        """
        obs = env.reset()
        noop = ACTION_TABLE[0]  # [0, 0, 0]
        obs2, _r, done, info = env.step(noop, otherAction=noop)
        if done:
            # Extremely unlikely on frame 1; retry to avoid emitting a
            # terminal initial state which the master can't represent.
            return _do_reset()
        return (np.asarray(obs2, dtype=np.float32),
                np.asarray(info["otherObs"], dtype=np.float32))

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                my_idx, opp_idx = data
                my_mb = ACTION_TABLE[int(my_idx)]
                opp_mb = ACTION_TABLE[int(opp_idx)]
                obs, reward, done, info = env.step(my_mb, otherAction=opp_mb)
                obs = np.asarray(obs, dtype=np.float32)
                other_obs = np.asarray(info["otherObs"], dtype=np.float32)
                if bool(done):
                    # Stash terminals BEFORE auto-reset so the master can
                    # build the correct n-step transition for the agent.
                    info = dict(info) if info is not None else {}
                    info["terminal_observation"] = obs
                    info["terminal_other_obs"] = other_obs
                    # Auto-reset (does its own noop-noop internally).
                    obs, other_obs = _do_reset()
                remote.send((obs, other_obs, float(reward), bool(done), info))
            elif cmd == "reset":
                obs, other_obs = _do_reset()
                remote.send((obs, other_obs))
            elif cmd == "seed":
                if data is not None:
                    try:
                        env.seed(int(data))
                    except AttributeError:
                        pass
                    try:
                        env.action_space.seed(int(data))
                    except AttributeError:
                        pass
                remote.send(None)
            elif cmd == "spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "close":
                try:
                    env.close()
                finally:
                    remote.close()
                return
            else:
                raise RuntimeError(f"SelfPlayVecEnv worker: unknown cmd {cmd!r}")
    except (EOFError, KeyboardInterrupt):
        try:
            env.close()
        except Exception:
            pass
        return


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------
class SelfPlayVecEnv:
    """Run N raw slimevolley envs in parallel and serve a 2-player API.

    Parameters
    ----------
    env_fns:
        Sequence of zero-arg callables; each returns a fresh RAW slimevolley
        env (NOT ``DiscreteSlimeWrapper``). The worker converts Discrete(6)
        ints into MultiBinary actions via ``ACTION_TABLE`` before stepping.
    """

    def __init__(self, env_fns: Sequence[Callable[[], Any]]) -> None:
        if len(env_fns) < 1:
            raise ValueError("SelfPlayVecEnv requires at least one env_fn.")
        self._closed = False
        self._num_envs = len(env_fns)

        ctx = _mp.get_context("spawn")
        pipes = [ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        self._remotes: List[Connection] = [p[0] for p in pipes]
        self._work_remotes: List[Connection] = [p[1] for p in pipes]

        self._procs: List[_mp.Process] = []
        for i, (fn, work_remote, remote) in enumerate(
            zip(env_fns, self._work_remotes, self._remotes)
        ):
            proc = ctx.Process(
                target=_worker,
                args=(work_remote, remote, fn, None),
                daemon=True,
                name=f"SelfPlayVecEnv-Worker-{i}",
            )
            proc.start()
            work_remote.close()
            self._procs.append(proc)

        # Handshake: pull spaces from worker 0. Note: these are the RAW env
        # spaces (Box(12) obs, MultiBinary(3) actions). The trainer wraps
        # action selection in Discrete(6) on its side.
        self._remotes[0].send(("spaces", None))
        self._observation_space, self._action_space = self._remotes[0].recv()

    # ----- properties -----
    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def observation_space(self):  # noqa: D401 -- gym-style attribute
        return self._observation_space

    @property
    def action_space(self):  # noqa: D401 -- gym-style attribute
        return self._action_space

    # ----- lifecycle -----
    def reset(self, seeds: Optional[Sequence[int]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all envs.

        Each worker performs an ``env.reset()`` followed by one noop-noop
        step (so we get both sides' initial obs from the same physics
        rollout). See worker ``_do_reset`` for the rationale.

        Returns
        -------
        (obs_agent, obs_opp)
            Each ``np.ndarray`` of shape ``(N, 12)``, dtype float32.
        """
        if seeds is not None:
            if len(seeds) != self._num_envs:
                raise ValueError(
                    f"len(seeds)={len(seeds)} != num_envs={self._num_envs}"
                )
            for r, s in zip(self._remotes, seeds):
                r.send(("seed", int(s)))
            for r in self._remotes:
                r.recv()  # drain "seed" acks

        for r in self._remotes:
            r.send(("reset", None))
        agent_list: List[np.ndarray] = []
        opp_list: List[np.ndarray] = []
        for r in self._remotes:
            o, oo = r.recv()
            agent_list.append(np.asarray(o))
            opp_list.append(np.asarray(oo))
        obs_agent = np.asarray(np.stack(agent_list, axis=0), dtype=np.float32)
        obs_opp = np.asarray(np.stack(opp_list, axis=0), dtype=np.float32)
        return obs_agent, obs_opp

    def step(self, my_acts: np.ndarray, opp_acts: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step every env with both sides' actions.

        Parameters
        ----------
        my_acts:
            Shape ``(N,)``, ``Discrete(6)`` indices for the right-side learner.
        opp_acts:
            Shape ``(N,)``, ``Discrete(6)`` indices for the left-side opponent.

        Returns
        -------
        (obs_agent, obs_opp, rewards_agent, dones, infos)
            * ``obs_agent``     ``(N, 12)`` float32 — auto-reset obs on done.
            * ``obs_opp``       ``(N, 12)`` float32 — auto-reset obs on done.
            * ``rewards_agent`` ``(N,)``    float32 — learner's per-step reward.
            * ``dones``         ``(N,)``    bool.
            * ``infos``         ``list[dict]``. On done indices we set
              ``info['terminal_observation']`` (agent's true terminal obs) and
              ``info['terminal_other_obs']`` (opponent's). The agent obs/opp
              obs returned in the tuple are the POST-RESET obs; the trainer
              must consult ``info['terminal_observation']`` when constructing
              the n-step transition that ends on this done.
        """
        if len(my_acts) != self._num_envs or len(opp_acts) != self._num_envs:
            raise ValueError(
                f"action length mismatch: my={len(my_acts)}, opp={len(opp_acts)}, "
                f"num_envs={self._num_envs}"
            )

        for r, a, b in zip(self._remotes, my_acts, opp_acts):
            # Cast to plain int defensively; numpy scalars pickle but are
            # cheaper to send as native ints, and the worker re-casts anyway.
            ai = int(a.item()) if isinstance(a, np.ndarray) and a.shape == () else int(a)
            bi = int(b.item()) if isinstance(b, np.ndarray) and b.shape == () else int(b)
            r.send(("step", (ai, bi)))

        agent_list: List[np.ndarray] = []
        opp_list: List[np.ndarray] = []
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        dones = np.zeros(self._num_envs, dtype=bool)
        infos: List[dict] = []
        for i, r in enumerate(self._remotes):
            o, oo, rew, done, info = r.recv()
            agent_list.append(np.asarray(o))
            opp_list.append(np.asarray(oo))
            rewards[i] = rew
            dones[i] = done
            infos.append(info if isinstance(info, dict) else {})

        obs_agent = np.asarray(np.stack(agent_list, axis=0), dtype=np.float32)
        obs_opp = np.asarray(np.stack(opp_list, axis=0), dtype=np.float32)
        return obs_agent, obs_opp, rewards, dones, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in self._remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for r in self._remotes:
            try:
                r.close()
            except Exception:
                pass
        for p in self._procs:
            if p.is_alive():
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)

    def __del__(self) -> None:  # pragma: no cover -- best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
