"""SubprocVecEnv: run N gym-0.19 envs in parallel via spawned subprocesses.

Mirrors the well-known SB3 / CleanRL vector-env contract:

  * ``reset(seeds=None)``  -> ``np.ndarray`` of shape ``(N, *obs_shape)``.
  * ``step(actions)``      -> ``(obs, rewards, dones, infos)`` with auto-reset.
  * ``close()``            -> sends ``"close"`` to every worker and joins it.

Auto-reset semantics (read carefully — the trainer relies on this):

  When env ``i`` returns ``done=True`` on a step, the worker immediately calls
  ``env.reset()`` and substitutes the post-reset obs into the returned tuple.
  The TRUE terminal observation (the one the agent should bootstrap from) is
  stashed in ``info[i]["terminal_observation"]``. Therefore, when constructing
  an n-step transition that ends on a done flag, the trainer must use
  ``info[i]["terminal_observation"]`` as ``s_{t+1}``, NOT the auto-reset obs.

We use ``multiprocessing.get_context("spawn")`` rather than the default fork
start method on Linux. slimevolleygym + numpy + (eventually) torch don't
always survive ``fork`` cleanly (locks held by the parent, lazy-loaded C
extensions that re-init poorly post-fork, CUDA contexts, etc.). Spawn is
slower at startup but is the only safe choice for mixed numpy/torch
workloads. We deliberately do NOT call ``set_start_method`` at import time:
that would mutate global multiprocessing state and could collide with other
modules (e.g. a DataLoader that already configured its own start method).
Instead we obtain a private context via ``get_context("spawn")`` and use it
to construct ``Process`` and ``Pipe`` only inside this class.
"""

from __future__ import annotations

import multiprocessing as _mp
from multiprocessing.connection import Connection
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(remote: Connection, parent_remote: Connection,
            env_fn: Callable[[], Any], seed: Optional[int]) -> None:
    """Subprocess entrypoint. Owns one env and serves commands over `remote`.

    Commands (cmd, data):
      * ('step', action)    -> sends (obs, reward, done, info); auto-resets.
      * ('reset', None)     -> sends post-reset obs.
      * ('seed',  seed)     -> reseeds env + action space; sends None.
      * ('spaces', None)    -> sends (observation_space, action_space).
      * ('close', None)     -> closes env, closes remote, returns.
    """
    # Close the parent's end of the pipe in the child to avoid a stray FD
    # keeping the pipe open after the parent crashes.
    parent_remote.close()
    env = env_fn()
    if seed is not None:
        try:
            env.seed(seed)
        except AttributeError:
            pass
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if bool(done):
                    # Stash terminal obs so the master can build the correct
                    # n-step transition; the obs we *return* is the auto-reset
                    # post-reset state for the new episode.
                    info = dict(info) if info is not None else {}
                    info["terminal_observation"] = np.asarray(obs)
                    obs = env.reset()
                remote.send((np.asarray(obs), float(reward), bool(done), info))
            elif cmd == "reset":
                obs = env.reset()
                remote.send(np.asarray(obs))
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
                raise RuntimeError(f"SubprocVecEnv worker: unknown cmd {cmd!r}")
    except (EOFError, KeyboardInterrupt):
        # Pipe closed unexpectedly (master died). Clean up and exit quietly.
        try:
            env.close()
        except Exception:
            pass
        return


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------
class SubprocVecEnv:
    """Run a list of env factories in parallel subprocesses.

    Parameters
    ----------
    env_fns:
        Sequence of zero-arg callables; each returns a fresh gym env. Capture
        per-worker seed via default-arg trick:
        ``[lambda i=i: make_env(seed + i) for i in range(N)]``.
    """

    def __init__(self, env_fns: Sequence[Callable[[], Any]]) -> None:
        if len(env_fns) < 1:
            raise ValueError("SubprocVecEnv requires at least one env_fn.")
        self._closed = False
        self._num_envs = len(env_fns)

        ctx = _mp.get_context("spawn")
        # One bidirectional pipe per worker.
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
                name=f"SubprocVecEnv-Worker-{i}",
            )
            proc.start()
            # Master keeps only its end of the pipe; close the worker's end on
            # the master side so an unexpected worker death is detectable as
            # EOF on `recv`.
            work_remote.close()
            self._procs.append(proc)

        # Handshake: pull spaces from worker 0.
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
    def reset(self, seeds: Optional[Sequence[int]] = None) -> np.ndarray:
        """Reset all envs. If `seeds` provided, must have length num_envs.

        Returns batched obs of shape ``(N, *obs_shape)``, dtype ``float32``.
        """
        if seeds is not None:
            if len(seeds) != self._num_envs:
                raise ValueError(
                    f"len(seeds)={len(seeds)} != num_envs={self._num_envs}"
                )
            for r, s in zip(self._remotes, seeds):
                r.send(("seed", int(s)))
            for r in self._remotes:
                r.recv()  # drain ack from each "seed" command

        for r in self._remotes:
            r.send(("reset", None))
        obs_list = [r.recv() for r in self._remotes]
        return np.asarray(np.stack(obs_list, axis=0), dtype=np.float32)

    def step(self, actions: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step every env. ``actions`` shape ``(N,)`` for Discrete envs.

        Returns ``(obs, rewards, dones, infos)``:
          * ``obs``     ``(N, *obs_shape)`` float32 (AUTO-RESET obs on done).
          * ``rewards`` ``(N,)`` float32.
          * ``dones``   ``(N,)`` bool.
          * ``infos``   ``list[dict]`` length N. ``info["terminal_observation"]``
                        is set on indices where ``dones`` is True.
        """
        if len(actions) != self._num_envs:
            raise ValueError(
                f"len(actions)={len(actions)} != num_envs={self._num_envs}"
            )

        # Send all step commands first, then drain. This is the trivial form
        # of pipelining — workers run in parallel during the round-trip.
        for r, a in zip(self._remotes, actions):
            # Most envs accept python ints for Discrete; numpy scalars work too.
            # Convert defensively to a plain int to avoid pickling numpy types.
            if isinstance(a, np.ndarray) and a.shape == ():
                payload = a.item()
            elif isinstance(a, np.ndarray):
                payload = a  # MultiBinary etc.
            else:
                payload = int(a)
            r.send(("step", payload))

        obs_list: List[np.ndarray] = []
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        dones = np.zeros(self._num_envs, dtype=bool)
        infos: List[dict] = []
        for i, r in enumerate(self._remotes):
            obs_i, reward_i, done_i, info_i = r.recv()
            obs_list.append(np.asarray(obs_i))
            rewards[i] = reward_i
            dones[i] = done_i
            infos.append(info_i if isinstance(info_i, dict) else {})

        obs = np.asarray(np.stack(obs_list, axis=0), dtype=np.float32)
        return obs, rewards, dones, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Best-effort: tell each worker to close. If a remote is already
        # broken (worker crashed) just skip it.
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
