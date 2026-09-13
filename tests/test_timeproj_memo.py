from __future__ import annotations

import torch

from timeproj_memo import (
    MemoTimeProjection,
    current_timestep,
    install_timeproj_memo,
    memo_stats,
    set_current_timestep,
)


def make_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.time_projection = torch.nn.Sequential(
        torch.nn.SiLU(), torch.nn.Linear(16, 96))
    return model


def test_miss_then_exact_replay() -> None:
    model = make_model()
    memo = install_timeproj_memo(model)
    assert isinstance(model.time_projection, MemoTimeProjection)
    assert install_timeproj_memo(model) is memo  # idempotent
    set_current_timestep(999.0)
    try:
        x = torch.randn(1, 8, 16)
        first = memo(x)
        assert memo.misses == 1 and memo.hits == 0
        second = memo(x)
        assert memo.hits == 1
        assert torch.equal(first, second)
        cached = memo.cache[(999.0, (1, 8, 16), "torch.float32", "cpu")]
        assert second.data_ptr() != cached.data_ptr()  # clone-on-hit protects the store
        # A new timestep misses again; a repeat hits.
        set_current_timestep(899.0)
        memo(x)
        assert memo.misses == 2
        set_current_timestep(999.0)
        memo(x)
        assert memo.hits == 2
        stats = memo_stats(model)
        assert stats == {"hits": 2, "misses": 2, "keys": 2, "installed": 1}
    finally:
        set_current_timestep(None)


def test_passthrough_without_timestep() -> None:
    model = make_model()
    memo = install_timeproj_memo(model)
    set_current_timestep(None)
    assert current_timestep() is None
    x = torch.randn(1, 8, 16)
    out = memo(x)
    assert torch.equal(out, model.time_projection.wrapped(x))
    assert memo.hits == 0 and memo.misses == 0


def test_stats_uninstalled() -> None:
    assert memo_stats(torch.nn.Module()) == {
        "hits": 0, "misses": 0, "keys": 0, "installed": 0}
