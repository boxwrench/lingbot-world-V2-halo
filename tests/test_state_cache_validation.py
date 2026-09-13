from __future__ import annotations

from state_cache_validation import reset_cursors_at_bootstrap


def test_bootstrap_cursors_ignore_capacity_key() -> None:
    # cache_positions() reports cache_capacity_tokens alongside the two
    # cursors; a correct bootstrap must still pass.
    reset_pos = {
        "global_end_index": 1008,
        "local_end_index": 1008,
        "cache_capacity_tokens": 12096,
    }
    assert reset_cursors_at_bootstrap(reset_pos, 1008) is True


def test_bootstrap_cursors_reject_wrong_positions() -> None:
    assert reset_cursors_at_bootstrap(
        {"global_end_index": 0, "local_end_index": 0, "cache_capacity_tokens": 12096},
        1008,
    ) is False
    assert reset_cursors_at_bootstrap(
        {"global_end_index": 1008, "local_end_index": 0},
        1008,
    ) is False
