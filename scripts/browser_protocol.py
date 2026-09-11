"""Model-free contracts shared by the local browser server and its tests."""

from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


ACTION_KEYS = frozenset("wsadjlik ")
QUIT_KEYS = frozenset(("q", "escape", "esc"))
KEY_ALIASES = {
    "arrowup": "i",
    "arrowdown": "k",
    "arrowleft": "j",
    "arrowright": "l",
    "up": "i",
    "down": "k",
    "left": "j",
    "right": "l",
    "space": " ",
    "esc": "escape",
}


def normalize_key(value: object) -> str:
    key = str(value).lower()
    return KEY_ALIASES.get(key, key)


def is_action_key(value: object) -> bool:
    key = normalize_key(value)
    return len(key) == 1 and key in ACTION_KEYS


def is_quit_key(value: object) -> bool:
    return normalize_key(value) in QUIT_KEYS


@dataclass(frozen=True)
class ActionRequest:
    action_id: int
    key: str
    client_keydown_ms: float | None
    received_ms: float


class PendingActionMailbox:
    """Thread-safe one-slot latest-valid action mailbox.

    The model owner is the only consumer. WebSocket handlers only call
    ``submit``; they never touch model or GPU state.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._current: ActionRequest | None = None
        self._pending: ActionRequest | None = None
        self._quit = False

    def submit(self, request: ActionRequest) -> dict[str, Any]:
        with self._condition:
            if is_quit_key(request.key):
                self._quit = True
                self._condition.notify_all()
                return {"status": "quit_requested", "action_id": request.action_id}
            if not is_action_key(request.key):
                return {"status": "ignored", "action_id": request.action_id}
            key = normalize_key(request.key)
            request = ActionRequest(
                request.action_id,
                key,
                request.client_keydown_ms,
                request.received_ms,
            )
            replaced = self._pending
            self._pending = request
            self._condition.notify_all()
            result: dict[str, Any] = {
                "status": "replaced" if replaced is not None else "queued",
                "action_id": request.action_id,
            }
            if replaced is not None:
                result["replaced_action_id"] = replaced.action_id
            return result

    def request_quit(self) -> None:
        with self._condition:
            self._quit = True
            self._condition.notify_all()

    def take(self, timeout: float | None = None) -> ActionRequest | None:
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._quit or self._pending is not None,
                timeout=timeout,
            ):
                return None
            if self._quit:
                return None
            request = self._pending
            self._pending = None
            self._current = request
            return request

    def complete(self) -> None:
        with self._condition:
            self._current = None
            self._condition.notify_all()

    @property
    def quit_requested(self) -> bool:
        with self._condition:
            return self._quit

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "current_action_id": self._current.action_id if self._current else None,
                "pending_action_id": self._pending.action_id if self._pending else None,
                "quit_requested": self._quit,
            }


@dataclass
class PresentationFrame:
    action_id: int
    frame_index: int
    frame_count: int
    frame: Any
    bootstrap: bool
    server: dict[str, Any]


class BoundedPresentationQueue:
    """Bounded frame queue where a new action's RGB0 invalidates stale tails."""

    def __init__(self, max_frames: int = 8) -> None:
        self.max_frames = max_frames
        self._condition = threading.Condition()
        self._items: deque[PresentationFrame] = deque()
        self.dropped_tail_frames = 0
        self.dropped_by_action: dict[int, int] = {}

    def submit(self, item: PresentationFrame) -> int:
        with self._condition:
            dropped = 0
            if item.frame_index == 0:
                kept: deque[PresentationFrame] = deque()
                for old in self._items:
                    if old.action_id != item.action_id:
                        if old.frame_index > 0:
                            dropped += 1
                    else:
                        kept.append(old)
                self._items = kept
            while len(self._items) >= self.max_frames:
                old = self._items.popleft()
                if old.frame_index > 0:
                    dropped += 1
            self._items.append(item)
            self.dropped_tail_frames += dropped
            if dropped:
                self.dropped_by_action[item.action_id] = (
                    self.dropped_by_action.get(item.action_id, 0) + dropped
                )
            self._condition.notify()
            return dropped

    def take(self, timeout: float | None = None) -> PresentationFrame | None:
        with self._condition:
            if not self._condition.wait_for(lambda: bool(self._items), timeout=timeout):
                return None
            return self._items.popleft()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


def pack_frame_envelope(header: dict[str, Any], jpeg_bytes: bytes) -> bytes:
    """Pack a frame as big-endian uint32 JSON length, JSON, then JPEG bytes."""
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded + jpeg_bytes


def unpack_frame_envelope(packet: bytes) -> tuple[dict[str, Any], bytes]:
    if len(packet) < 4:
        raise ValueError("frame packet is shorter than its length prefix")
    header_len = struct.unpack(">I", packet[:4])[0]
    end = 4 + header_len
    if end > len(packet):
        raise ValueError("frame packet header exceeds packet length")
    header = json.loads(packet[4:end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("frame packet header must be a JSON object")
    return header, packet[end:]


class TelemetryLedger:
    """Thread-safe action telemetry with null for unobserved boundaries."""

    SERVER_FIELDS = (
        "input_received_ms",
        "input_selected_ms",
        "generation_start_ms",
        "accepted_x0_ms",
        "base_rgb_ready_ms",
        "clean_kv_start_ms",
        "clean_kv_complete_ms",
        "all_rgb_ready_ms",
        "next_ready_ms",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[int, dict[str, Any]] = {}

    def _ensure_locked(self, action_id: int) -> dict[str, Any]:
        record = self._records.get(action_id)
        if record is None:
            record = {
                "action_id": action_id,
                "action": "unknown",
                "bootstrap": False,
                "server": {field: None for field in self.SERVER_FIELDS},
                "browser": {
                    "keydown_ms": None,
                    "frame_received_ms": None,
                    "frame_decoded_ms": None,
                    "frame_presented_ms": None,
                },
                "status": "running",
            }
            self._records[action_id] = record
        return record

    def start(self, action_id: int, action: str, bootstrap: bool = False) -> dict[str, Any]:
        with self._lock:
            record = {
                "action_id": action_id,
                "action": action,
                "bootstrap": bootstrap,
                "server": {field: None for field in self.SERVER_FIELDS},
                "browser": {
                    "keydown_ms": None,
                    "frame_received_ms": None,
                    "frame_decoded_ms": None,
                    "frame_presented_ms": None,
                },
                "status": "running",
            }
            self._records[action_id] = record
            return _copy_record(record)

    def mark_server(self, action_id: int, field: str, value_ms: float) -> None:
        with self._lock:
            record = self._ensure_locked(action_id)
            record["server"][field] = float(value_ms)

    def mark_browser(self, action_id: int, field: str, value_ms: float) -> None:
        with self._lock:
            record = self._ensure_locked(action_id)
            record["browser"][field] = float(value_ms)

    def finish(self, action_id: int, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = self._ensure_locked(action_id)
            record.update(fields)
            record["status"] = fields.get("status", "complete")
            return _copy_record(record)

    def replace(self, action_id: int, replaced_by: int) -> dict[str, Any]:
        return self.finish(action_id, status="replaced", replaced_by=replaced_by)

    def get(self, action_id: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(action_id)
            return _copy_record(record) if record else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_copy_record(self._records[key]) for key in sorted(self._records)]


def _copy_record(record: dict[str, Any] | None) -> dict[str, Any]:
    return json.loads(json.dumps(record)) if record is not None else {}


def monotonic_ms(origin: float) -> float:
    return (time.perf_counter() - origin) * 1000.0
