"""OpponentPool: in-memory pool of past policy snapshots for self-play.

Each snapshot is one ``.pt`` file at ``pool_dir/snapshot_{step:09d}.pt`` plus
an in-memory eval-mode copy on ``device``. FIFO eviction beyond ``max_size``.

Disk persistence is for traceability (post-hoc inspection, ablations, manual
re-loads); the in-memory copies are what the trainer queries each iteration.
On restart we rebuild memory from disk via ``load_existing()``.

Caller responsibilities
-----------------------
  * Provide a ``factory_fn`` (zero-arg) that constructs a fresh ``nn.Module``
    on the desired device. The pool calls this for every snapshot it adds.
  * Run inference itself: ``with torch.no_grad(): model(obs_t).argmax(-1)``.
    The pool keeps models in eval mode and detached from any optimizer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


_SNAPSHOT_RE = re.compile(r"^snapshot_(\d{9})\.pt$")
# Sentinel step for seeded snapshots: external checkpoints loaded via
# ``add_seed_checkpoint`` get this so they never collide with real training
# snapshot indices (which are non-negative global_step values).
_SEED_STEP: int = -1


class OpponentPool:
    """FIFO pool of frozen policy snapshots.

    Parameters
    ----------
    pool_dir:
        Directory for ``.pt`` files. Created if missing.
    factory_fn:
        Zero-arg callable returning a fresh ``nn.Module`` already on
        ``device``. The pool loads ``state_dict`` into the result with
        ``strict=True``.
    device:
        Device string used for in-memory inference (e.g. ``"cuda"``,
        ``"cpu"``). Snapshots are saved to disk as CPU tensors but kept in
        memory on this device for fast eval.
    max_size:
        Maximum number of in-memory snapshots. FIFO-evicted (oldest by step)
        when exceeded; the corresponding file is also deleted.
    """

    def __init__(self, pool_dir: str, factory_fn: Callable[[], nn.Module],
                 device: str = "cpu", max_size: int = 30) -> None:
        self.pool_dir = pool_dir
        self.factory_fn = factory_fn
        self.device = device
        self.max_size = int(max_size)
        os.makedirs(self.pool_dir, exist_ok=True)
        # Parallel lists, ordered oldest-first by `step`. We don't expect more
        # than `max_size` entries (~30) so list operations are fine.
        self._steps: List[int] = []
        self._models: List[nn.Module] = []

    # ------------------------------------------------------------------
    def add(self, online_state_dict: dict, step: int) -> None:
        """Persist ``online_state_dict`` to disk and add an eval-mode copy.

        FIFO-evicts the oldest entry if over capacity (deleting its file).
        """
        # CPU view of the state_dict so the saved file is portable.
        cpu_state = {k: v.detach().cpu() for k, v in online_state_dict.items()}
        path = os.path.join(self.pool_dir, f"snapshot_{int(step):09d}.pt")
        torch.save(cpu_state, path)

        model = self.factory_fn()
        model.load_state_dict(cpu_state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self._steps.append(int(step))
        self._models.append(model)

        while len(self._models) > self.max_size:
            old_step = self._steps.pop(0)
            self._models.pop(0)
            old_path = os.path.join(
                self.pool_dir, f"snapshot_{int(old_step):09d}.pt"
            )
            try:
                os.remove(old_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    def add_seed_checkpoint(self, ckpt_path: Union[str, Path],
                            label: str = "") -> None:
        """Load a state_dict from disk and add it as a permanent pool member.

        Used at trainer startup to seed the pool with externally trained
        checkpoints (e.g., v0, v1 cold, v1 selfplay) before training begins.
        Seed snapshots are stored in-memory but NOT written into ``pool_dir``
        (to avoid mixing with in-training snapshots). They DO count against
        ``max_size`` for FIFO eviction; their step is tagged with a sentinel
        ``-1`` so they sort before any real-step snapshot.
        """
        path = str(ckpt_path)
        # CPU-side load so we don't pin GPU memory just to forward into the
        # factory-built module. weights_only=False mirrors the trainer's own
        # checkpoint loaders (these files contain dicts with metadata, not
        # bare tensors).
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Trainer-style checkpoints wrap state_dict under "model_state_dict";
        # raw state_dicts are accepted as-is for forward compatibility.
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            cpu_state = ckpt["model_state_dict"]
        else:
            cpu_state = ckpt

        model = self.factory_fn()
        model.load_state_dict(cpu_state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Use the sentinel so seed entries sort before in-training snapshots
        # (helpful if both ever co-exist in `_steps`); FIFO eviction on
        # add() then prefers to drop seeds first as they are oldest.
        self._steps.append(_SEED_STEP)
        self._models.append(model)

        # Capacity enforcement: same FIFO rule as `add`, but no on-disk file
        # to remove for seed entries (the resume snapshot still has one if it
        # was added via `add(...)` previously).
        while len(self._models) > self.max_size:
            old_step = self._steps.pop(0)
            self._models.pop(0)
            if old_step >= 0:
                old_path = os.path.join(
                    self.pool_dir, f"snapshot_{int(old_step):09d}.pt",
                )
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        display = label if label else os.path.basename(path)
        print(f"[seed] loaded {display} into pool (size={len(self)})",
              flush=True)

    # ------------------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> Optional[nn.Module]:
        """Uniformly sample one snapshot. Returns ``None`` if empty."""
        if not self._models:
            return None
        i = int(rng.integers(0, len(self._models)))
        return self._models[i]

    def latest(self) -> Optional[nn.Module]:
        """Return the most recently added snapshot, or ``None`` if empty."""
        if not self._models:
            return None
        return self._models[-1]

    def __len__(self) -> int:
        return len(self._models)

    # ------------------------------------------------------------------
    def load_existing(self) -> None:
        """Walk ``pool_dir`` for ``snapshot_*.pt``, sort by step, load in.

        Loads up to ``max_size`` entries (the newest by step if there are
        more files than capacity); the rest are left untouched on disk.
        Idempotent w.r.t. files already represented in memory: this method
        is intended to be called once at startup, BEFORE any ``add`` calls.

        Only files matching the strict ``snapshot_{step:09d}.pt`` pattern are
        loaded; any other ``.pt`` files in ``pool_dir`` (e.g. operator-dropped
        seed checkpoints with arbitrary names) are silently skipped here, so
        ``add_seed_checkpoint`` is the only intended path for non-snapshot
        members.
        """
        if not os.path.isdir(self.pool_dir):
            return
        entries: List[int] = []
        for name in os.listdir(self.pool_dir):
            m = _SNAPSHOT_RE.match(name)
            if m is not None:
                entries.append(int(m.group(1)))
        entries.sort()
        # If more files than capacity, keep the newest `max_size`.
        if len(entries) > self.max_size:
            entries = entries[-self.max_size:]

        for step in entries:
            path = os.path.join(self.pool_dir, f"snapshot_{step:09d}.pt")
            try:
                cpu_state = torch.load(path, map_location="cpu")
            except (OSError, RuntimeError):
                # Corrupt/unreadable file — skip without polluting memory.
                continue
            model = self.factory_fn()
            model.load_state_dict(cpu_state, strict=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._steps.append(step)
            self._models.append(model)
