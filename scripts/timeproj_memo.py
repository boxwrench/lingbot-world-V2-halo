"""Session-level memo for the DiT timestep projection (P2.4).

Upstream proof (`model_fast.py`, accepted main): `e0 = time_projection(e)`
where `e` derives solely from the timestep tensor `t`. Only a handful of
distinct `t` values exist per session (999/899/702 + clean 0), repeating
every chunk and action — so the projection is pure recomputation after the
first encounter per timestep.

Design (no upstream edit, mainline default-off):

- `install_timeproj_memo(model)` swaps `model.time_projection` for an
  exact-replay wrapper. Called from `prepare_session`, gated on
  `LINGBOT_TIMEPROJ_MEMO=1`.
- The wrapper is keyed by the current timestep, published by the caller
  (`generate_chunk` per denoise forward, `commit_clean_kv` for the clean
  commit) via `set_current_timestep`. Without a published timestep the
  wrapper passes through untouched (safe default).
- Hits return `cached.clone()` so no downstream in-place op can corrupt
  the stored value; misses compute, store a detached clone, and return
  the computed tensor. Replay is bitwise-exact by construction on the
  deterministic stack (C2 A1/A2 precedent).
"""
from __future__ import annotations

import torch

_current_timestep: list[float | None] = [None]


def set_current_timestep(value: float | None) -> None:
    _current_timestep[0] = None if value is None else float(value)


def current_timestep() -> float | None:
    return _current_timestep[0]


class MemoTimeProjection(torch.nn.Module):
    def __init__(self, wrapped: torch.nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.cache: dict[tuple, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def forward(self, e: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        t = _current_timestep[0]
        if t is None:
            return self.wrapped(e)
        key = (t, tuple(e.shape), str(e.dtype), str(e.device))
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit.clone()
        out = self.wrapped(e)
        self.cache[key] = out.detach().clone()
        self.misses += 1
        return out


def install_timeproj_memo(model: torch.nn.Module) -> MemoTimeProjection:
    existing = getattr(model, "time_projection", None)
    if isinstance(existing, MemoTimeProjection):
        return existing
    if existing is None:
        raise AttributeError("model has no time_projection to memoize")
    memo = MemoTimeProjection(existing)
    model.time_projection = memo
    return memo


def memo_stats(model: torch.nn.Module) -> dict[str, int]:
    memo = getattr(model, "time_projection", None)
    if not isinstance(memo, MemoTimeProjection):
        return {"hits": 0, "misses": 0, "keys": 0, "installed": 0}
    return {"hits": memo.hits, "misses": memo.misses,
            "keys": len(memo.cache), "installed": 1}
