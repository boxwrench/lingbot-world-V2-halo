#!/usr/bin/env python3
"""Deterministic localhost browser-protocol benchmark client."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from pathlib import Path

from aiohttp import ClientSession, WSMsgType
from PIL import Image

from browser_protocol import unpack_frame_envelope


DEFAULT_ACTIONS = [
    "w", "w", "j", "w", "l", "l", "s", "s", "j", "w",
    "d", "d", "w", "l", "a", "a", "w", "j", "s", "l",
    "w", "w", "d", "j", "a", "s", "l", "w", "d", "d",
    "j", "w", "a", "a", "l", "s", "w", "j", "l", "w",
]


def clock_ms(origin: float) -> float:
    return (time.perf_counter() - origin) * 1000.0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


async def main_async(args: argparse.Namespace) -> int:
    origin = time.perf_counter()
    wall_start = time.time()
    received: dict[int, float] = {}
    decoded: dict[int, float] = {}
    presented: dict[int, float] = {}
    keydowns: dict[int, float] = {}
    records: dict[int, dict] = {}
    startup: dict = {}
    frames: list[dict] = []
    tail_frames_received = 0
    async with ClientSession() as session:
        async with session.ws_connect(args.url, heartbeat=30.0, max_msg_size=2 * 1024 * 1024) as ws:
            ready = False
            deadline = time.monotonic() + args.ready_timeout
            while not ready:
                if time.monotonic() >= deadline:
                    raise TimeoutError("browser server did not become ready")
                try:
                    message = await ws.receive(timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                if message.type == WSMsgType.TEXT:
                    payload = json.loads(message.data)
                    if payload.get("type") == "status":
                        if payload.get("state") == "INTERACTIVE_READY":
                            ready = True
                            startup = payload.get("startup", {})
                elif message.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                    raise RuntimeError("browser websocket closed before READY")

            for action_id, key in enumerate(args.actions, start=1):
                keydown = clock_ms(origin)
                keydowns[action_id] = keydown
                await ws.send_json({
                    "type": "action",
                    "action_id": action_id,
                    "key": key,
                    "client_keydown_ms": keydown,
                })
                got_frame = False
                got_record = False
                while not (got_frame and got_record):
                    message = await ws.receive(timeout=args.action_timeout)
                    if message.type == WSMsgType.BINARY:
                        header, jpeg = unpack_frame_envelope(bytes(message.data))
                        if int(header.get("action_id", -1)) != action_id:
                            continue
                        frame_index = int(header.get("frame_index", -1))
                        if frame_index != 0:
                            tail_frames_received += 1
                            continue
                        received[action_id] = clock_ms(origin)
                        with Image.open(io.BytesIO(jpeg)) as image:
                            image.load()
                        decoded[action_id] = clock_ms(origin)
                        # This is a client-side paint-cycle proxy, not monitor latency.
                        await asyncio.sleep(0)
                        presented[action_id] = clock_ms(origin)
                        await ws.send_json({
                            "type": "browser_telemetry",
                            "action_id": action_id,
                            "field": "frame_received_ms",
                            "value_ms": received[action_id],
                        })
                        await ws.send_json({
                            "type": "browser_telemetry",
                            "action_id": action_id,
                            "field": "frame_decoded_ms",
                            "value_ms": decoded[action_id],
                        })
                        await ws.send_json({
                            "type": "browser_telemetry",
                            "action_id": action_id,
                            "field": "frame_presented_ms",
                            "value_ms": presented[action_id],
                        })
                        got_frame = True
                        frames.append({"action_id": action_id, "header": header})
                    elif message.type == WSMsgType.TEXT:
                        payload = json.loads(message.data)
                        if payload.get("type") == "action_record":
                            record = payload.get("record", {})
                            if int(record.get("action_id", -1)) == action_id:
                                records[action_id] = record
                                got_record = record.get("status") == "complete"
                        elif payload.get("type") == "status" and payload.get("state") == "ERROR":
                            raise RuntimeError(payload.get("error", payload.get("text", "server error")))
                    elif message.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                        raise RuntimeError("browser websocket closed during benchmark")
            await ws.send_json({"type": "quit"})

    def values(path: tuple[str, ...], selected: dict[int, dict] | None = None) -> list[float]:
        output = []
        source = records if selected is None else selected
        for action_id in source:
            value = source[action_id]
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                output.append(float(value))
        return output

    def rolled(record: dict) -> bool:
        cache = record.get("cache", {})
        return int(cache.get("global_end_index", 0)) > int(cache.get("capacity_tokens", 0))

    rolled_records = {action_id: record for action_id, record in records.items() if rolled(record)}

    def server_summary(selected: dict[int, dict]) -> dict:
        action_to_rgb = [
            float(selected[i]["derived"]["action_to_base_rgb_ms"])
            for i in selected
            if selected[i].get("derived", {}).get("action_to_base_rgb_ms") is not None
        ]
        action_to_ready = [
            float(selected[i]["derived"]["action_to_next_ready_ms"])
            for i in selected
            if selected[i].get("derived", {}).get("action_to_next_ready_ms") is not None
        ]
        return {
            "actions": len(selected),
            "action_to_rgb_p50_ms": percentile(action_to_rgb, 50),
            "action_to_rgb_p95_ms": percentile(action_to_rgb, 95),
            "next_ready_p50_ms": percentile(action_to_ready, 50),
            "next_ready_p95_ms": percentile(action_to_ready, 95),
            "denoise_p50_ms": percentile(values(("denoise_ms",), selected), 50),
            "clean_kv_p50_ms": percentile(values(("clean_kv_ms",), selected), 50),
        }
    browser_received = [received[i] - keydowns[i] for i in received if i in keydowns]
    browser_decoded = [decoded[i] - keydowns[i] for i in decoded if i in keydowns]
    browser_presented = [presented[i] - keydowns[i] for i in presented if i in keydowns]
    encode_times = [
        float(frame["header"]["jpeg_encode_ms"])
        for frame in frames
        if frame["header"].get("jpeg_encode_ms") is not None
    ]
    jpeg_bytes = [
        int(frame["header"]["jpeg_bytes"])
        for frame in frames
        if frame["header"].get("jpeg_bytes") is not None
    ]
    output = Path(args.output).resolve()
    server_report_path = output.parent / "browser_metrics.json"
    server_report = None
    for _ in range(40):
        if server_report_path.is_file() and server_report_path.stat().st_mtime >= wall_start:
            try:
                server_report = json.loads(server_report_path.read_text())
                break
            except json.JSONDecodeError:
                pass
        await asyncio.sleep(0.05)
    report = {
        "url": args.url,
        "actions": args.actions,
        "startup": startup,
        "records": list(records.values()),
        "frames": frames,
        "server_metrics": server_summary(records),
        "rolled_server_metrics": server_summary(rolled_records),
        "browser_metrics": {
            "keydown_to_frame_received_p50_ms": percentile(browser_received, 50),
            "keydown_to_frame_received_p95_ms": percentile(browser_received, 95),
            "keydown_to_frame_decoded_p50_ms": percentile(browser_decoded, 50),
            "keydown_to_frame_decoded_p95_ms": percentile(browser_decoded, 95),
            "keydown_to_frame_presented_p50_ms": percentile(browser_presented, 50),
            "keydown_to_frame_presented_p95_ms": percentile(browser_presented, 95),
        },
        "presentation": {
            "jpeg_encode_p50_ms": percentile(encode_times, 50),
            "jpeg_encode_p95_ms": percentile(encode_times, 95),
            "jpeg_bytes_p50": percentile([float(value) for value in jpeg_bytes], 50),
            "jpeg_bytes_p95": percentile([float(value) for value in jpeg_bytes], 95),
            "tail_frames_received": tail_frames_received,
            "tail_frames_dropped": (
                server_report.get("presentation", {}).get("dropped_tail_frames")
                if server_report else None
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "server_metrics": report["server_metrics"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/ws")
    parser.add_argument("--output", default="results/raw/browser-serving-20260910/benchmark.json")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    parser.add_argument("--action-timeout", type=float, default=120.0)
    parser.add_argument("--actions", nargs="*", default=DEFAULT_ACTIONS)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
