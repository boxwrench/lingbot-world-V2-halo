#!/usr/bin/env python3
"""Fast, model-free checks for the bounded live-viewer input policy."""

from __future__ import annotations

import time

from run_live import PendingInputState


def test_latest_valid_replaces_previous() -> None:
    state = PendingInputState(time.perf_counter())
    state.begin_action("forward")
    state.observe("w")
    state.observe("d")
    state.observe("s")
    state.observe("unsupported")
    assert state.pending_key == "s"
    assert len(state._active_events) == 3
    assert state._active_events[-1]["replaced_pending"] is True
    report = state.finish_action()
    assert report["pending_key_after_action"] == "s"


def test_repeats_are_one_pending_command() -> None:
    state = PendingInputState(time.perf_counter())
    state.begin_action("forward")
    state.observe("w")
    state.observe("w")
    state.observe("w")
    assert state.pending_key == "w"
    assert state.take_pending() == "w"
    assert state.pending_key is None


def test_invalid_input_does_not_erase_pending() -> None:
    state = PendingInputState(time.perf_counter())
    state.begin_action("forward")
    state.observe("l")
    state.observe("not-a-command")
    assert state.pending_key == "l"


def test_quit_is_sticky_and_not_queued() -> None:
    state = PendingInputState(time.perf_counter())
    state.begin_action("forward")
    assert state.observe("d") == "d"
    assert state.observe("q") == "escape"
    assert state.quit_requested is True
    assert state.pending_key == "d"
    assert state.observe("s") is None
    assert state.pending_key == "d"


def test_waiting_input_is_promoted_when_action_starts() -> None:
    state = PendingInputState(time.perf_counter())
    state.observe("w")
    assert state.waiting_key == "w"
    state.begin_action("bootstrap")
    assert state.waiting_key is None
    assert state.pending_key == "w"


if __name__ == "__main__":
    for test in (
        test_latest_valid_replaces_previous,
        test_repeats_are_one_pending_command,
        test_invalid_input_does_not_erase_pending,
        test_quit_is_sticky_and_not_queued,
        test_waiting_input_is_promoted_when_action_starts,
    ):
        test()
    print("live input queue tests: PASS")
