from __future__ import annotations

import time

from browser_protocol import (
    ActionRequest,
    BoundedPresentationQueue,
    PendingActionMailbox,
    PresentationFrame,
    TelemetryLedger,
    pack_frame_envelope,
    unpack_frame_envelope,
)


def request(action_id: int, key: str) -> ActionRequest:
    return ActionRequest(action_id, key, float(action_id), time.perf_counter())


def test_latest_valid_replaces_and_invalid_does_not_erase() -> None:
    mailbox = PendingActionMailbox()
    assert mailbox.submit(request(1, "w"))["status"] == "queued"
    assert mailbox.submit(request(2, "d"))["status"] == "replaced"
    assert mailbox.submit(request(3, "unsupported"))["status"] == "ignored"
    current = mailbox.take(timeout=0.01)
    assert current is not None and current.action_id == 2 and current.key == "d"
    mailbox.complete()


def test_busy_replacement_and_repeated_key_are_one_slot() -> None:
    mailbox = PendingActionMailbox()
    mailbox.submit(request(1, "w"))
    current = mailbox.take(timeout=0.01)
    assert current is not None
    assert current.action_id == 1 and current.key == "w"
    mailbox.submit(request(2, "w"))
    mailbox.submit(request(3, "w"))
    mailbox.submit(request(4, "s"))
    assert mailbox.snapshot()["pending_action_id"] == 4
    mailbox.complete()
    assert mailbox.take(timeout=0.01).action_id == 4


def test_quit_has_priority() -> None:
    mailbox = PendingActionMailbox()
    mailbox.submit(request(1, "w"))
    mailbox.submit(request(2, "escape"))
    assert mailbox.quit_requested is True
    assert mailbox.take(timeout=0.01) is None


def test_frame_envelope_round_trip() -> None:
    header = {
        "type": "frame",
        "action_id": 17,
        "frame_index": 0,
        "frame_count": 4,
        "width": 672,
        "height": 384,
        "bootstrap": False,
    }
    parsed, payload = unpack_frame_envelope(pack_frame_envelope(header, b"jpeg"))
    assert parsed == header
    assert payload == b"jpeg"


def test_new_action_drops_stale_tail_frames() -> None:
    queue = BoundedPresentationQueue(max_frames=8)
    for index in range(1, 4):
        queue.submit(PresentationFrame(1, index, 4, b"x", False, {}))
    dropped = queue.submit(PresentationFrame(2, 0, 4, b"x", False, {}))
    assert dropped == 3
    assert queue.take(timeout=0.01).action_id == 2
    assert queue.dropped_tail_frames == 3


def test_telemetry_uses_null_until_measured() -> None:
    ledger = TelemetryLedger()
    ledger.start(7, "turn_right")
    ledger.mark_server(7, "base_rgb_ready_ms", 938.0)
    record = ledger.finish(7, status="complete")
    assert record["server"]["base_rgb_ready_ms"] == 938.0
    assert record["server"]["clean_kv_complete_ms"] is None
