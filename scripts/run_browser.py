#!/usr/bin/env python3
"""Local browser frontend for the frozen LingBot RC1 inference path.

The model owner is one worker thread.  It owns the model, TAEHV decoder, KV
state, camera state, and exact clean-latent transaction.  The aiohttp loop,
WebSocket handlers, and JPEG worker never execute GPU work.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aiohttp import WSMsgType, web
from PIL import Image

from browser_protocol import (
    ActionRequest,
    BoundedPresentationQueue,
    PendingActionMailbox,
    PresentationFrame,
    TelemetryLedger,
    is_action_key,
    is_quit_key,
    monotonic_ms,
    normalize_key,
    pack_frame_envelope,
)
from run_interactive import (
    StreamingTAEHVDecoder,
    build_parser as build_interactive_parser,
    commit_clean_kv,
    generate_chunk,
    prepare_session,
    sync,
)
from run_live import action_from_key, make_plucker


STATE_TEXT = {
    "SERVER_STARTING": "Starting local server",
    "MODEL_LOADING": "Loading model",
    "SESSION_PREPARING": "Preparing world",
    "RUNTIME_PREWARM": "Loading tuned kernels and warming runtime",
    "BOOTSTRAP_GENERATING": "Generating initial world",
    "INTERACTIVE_READY": "Ready",
    "ACTION_RUNNING": "Generating next world step",
    "STOPPING": "Stopping safely",
    "ERROR": "Runtime error",
}
BROWSER_ROOT_KEY = web.AppKey("browser_root", Path)


class EventBus:
    """Bounded cross-thread event buffer consumed by the aiohttp loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, max_events: int = 128) -> None:
        self.loop = loop
        self.max_events = max_events
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque()
        self._wake = asyncio.Event()
        self.latest_status: dict[str, Any] = {
            "type": "status",
            "state": "SERVER_STARTING",
            "text": STATE_TEXT["SERVER_STARTING"],
        }
        self.latest_frame: dict[str, Any] | None = None

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            if event.get("type") == "status":
                self.latest_status = dict(event)
            elif event.get("type") == "frame" and event.get("frame_index") == 0:
                self.latest_frame = dict(event)
            if len(self._events) >= self.max_events:
                if event.get("type") == "frame":
                    for index, old in enumerate(self._events):
                        if old.get("type") == "frame":
                            del self._events[index]
                            break
                    else:
                        return
                else:
                    self._events.popleft()
            self._events.append(event)
        self.loop.call_soon_threadsafe(self._wake.set)

    async def take(self) -> dict[str, Any]:
        while True:
            with self._lock:
                if self._events:
                    return self._events.popleft()
            await self._wake.wait()
            self._wake.clear()


class ClientOutbound:
    """Per-client bounded output queue; stale frame tails may be dropped."""

    def __init__(self, max_items: int = 16) -> None:
        self.max_items = max_items
        self.items: deque[dict[str, Any]] = deque()
        self.wake = asyncio.Event()
        self.dropped_frames = 0

    def put(self, event: dict[str, Any]) -> None:
        if len(self.items) >= self.max_items:
            removed = False
            for index, old in enumerate(self.items):
                if old.get("type") == "frame":
                    del self.items[index]
                    self.dropped_frames += 1
                    removed = True
                    break
            if not removed:
                self.items.popleft()
        self.items.append(event)
        self.wake.set()

    async def take(self) -> dict[str, Any]:
        while not self.items:
            await self.wake.wait()
            self.wake.clear()
        return self.items.popleft()


class PresentationWorker:
    """CPU-only RGB-to-JPEG worker."""

    def __init__(self, bus: EventBus, quality: int) -> None:
        self.bus = bus
        self.quality = quality
        self.queue = BoundedPresentationQueue(max_frames=8)
        self._stop = threading.Event()
        self._encoded: set[tuple[int, int]] = set()
        self._encoded_condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="lingbot-jpeg", daemon=True)
        self._thread.start()

    @staticmethod
    def _rgb_array(frame: torch.Tensor) -> np.ndarray:
        array = frame.detach().float().clamp(0.0, 1.0)
        if array.ndim == 4:
            array = array[0]
        if array.shape[0] == 3:
            array = array.permute(1, 2, 0)
        return np.ascontiguousarray((array.numpy() * 255.0).round().astype(np.uint8))

    def submit(self, item: PresentationFrame) -> int:
        return self.queue.submit(item)

    def wait_encoded(self, action_id: int, frame_index: int, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._encoded_condition:
            while (action_id, frame_index) not in self._encoded:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._encoded_condition.wait(remaining)
            return True

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self.queue.take(timeout=0.1)
            if item is None:
                continue
            encode_start = time.perf_counter()
            array = self._rgb_array(item.frame)
            image = Image.fromarray(array, mode="RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.quality, optimize=False)
            jpeg = buffer.getvalue()
            encode_ms = (time.perf_counter() - encode_start) * 1000.0
            header = {
                "type": "frame",
                "action_id": item.action_id,
                "frame_index": item.frame_index,
                "frame_count": item.frame_count,
                "width": int(array.shape[1]),
                "height": int(array.shape[0]),
                "bootstrap": bool(item.bootstrap),
                "server": dict(item.server),
                "jpeg_encode_ms": encode_ms,
                "jpeg_bytes": len(jpeg),
            }
            self.bus.publish({
                "type": "frame",
                "action_id": item.action_id,
                "frame_index": item.frame_index,
                "packet": pack_frame_envelope(header, jpeg),
            })
            with self._encoded_condition:
                self._encoded.add((item.action_id, item.frame_index))
                self._encoded_condition.notify_all()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class ModelOwner:
    """The single context permitted to execute LingBot inference."""

    def __init__(self, args: argparse.Namespace, bus: EventBus, process_start: float) -> None:
        self.args = args
        self.bus = bus
        self.process_start = process_start
        self.mailbox = PendingActionMailbox()
        self.ledger = TelemetryLedger()
        self.presentation = PresentationWorker(bus, args.jpeg_quality)
        self.thread = threading.Thread(target=self._run, name="lingbot-model-owner", daemon=True)
        self.stop_event = threading.Event()
        self.finished = threading.Event()
        self.ready = False
        self.status = "SERVER_STARTING"
        self.status_lock = threading.Lock()
        self.pipe = None
        self.state: dict[str, Any] | None = None
        self.decoder: StreamingTAEHVDecoder | None = None
        self.device = torch.device("cuda:0")
        self.installation: dict[str, Any] | None = None
        self.original_init = None
        self.startup: dict[str, float] = {"process_start_ms": 0.0}
        self.action_count = 0
        self.error: str | None = None

    def now_ms(self) -> float:
        return monotonic_ms(self.process_start)

    def start(self) -> None:
        self.thread.start()

    def set_status(self, state: str, **extra: Any) -> None:
        with self.status_lock:
            self.status = state
        event = {
            "type": "status",
            "state": state,
            "text": STATE_TEXT.get(state, state),
            "server_ms": self.now_ms(),
            "config": {
                "width": 672,
                "height": 384,
                "local_attn_size": self.args.local_attn_size,
                "sink_size": self.args.sink_size,
                "model": "LingBot World v2 1.3B",
            },
        }
        event.update(extra)
        self.bus.publish(event)

    def status_snapshot(self) -> dict[str, Any]:
        with self.status_lock:
            state = self.status
        return {
            "type": "status",
            "state": state,
            "text": STATE_TEXT.get(state, state),
            "server_ms": self.now_ms(),
            "config": {
                "width": 672,
                "height": 384,
                "local_attn_size": self.args.local_attn_size,
                "sink_size": self.args.sink_size,
                "model": "LingBot World v2 1.3B",
            },
            "startup": dict(self.startup),
        }

    def submit_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_id = payload.get("action_id")
        raw_key = payload.get("key")
        try:
            action_id = int(raw_id)
        except (TypeError, ValueError):
            return {"status": "invalid", "reason": "action_id must be an integer"}
        if action_id < 1:
            return {"status": "invalid", "reason": "action_id must be positive"}
        key = normalize_key(raw_key)
        if not is_action_key(key) and not is_quit_key(key):
            return {"status": "ignored", "action_id": action_id}
        if not is_quit_key(key) and not self.ready:
            return {"status": "not_ready", "action_id": action_id}
        existing = self.ledger.get(action_id)
        # Browser timing can legally arrive immediately before the action
        # frame on a busy websocket.  Such a record is only telemetry-only
        # scaffolding; it is not an accepted action and must be completed by
        # the authoritative action message.  A real duplicate action still
        # has a non-unknown action label and is rejected.
        if (
            not is_quit_key(key)
            and existing is not None
            and existing.get("action") not in (None, "unknown")
        ):
            return {"status": "duplicate", "action_id": action_id}
        received_ms = self.now_ms()
        client_keydown = payload.get("client_keydown_ms")
        try:
            client_keydown = float(client_keydown) if client_keydown is not None else None
        except (TypeError, ValueError):
            client_keydown = None
        if not is_quit_key(key):
            self.ledger.start(action_id, key)
            self.ledger.mark_server(action_id, "input_received_ms", received_ms)
            if client_keydown is not None:
                self.ledger.mark_browser(action_id, "keydown_ms", client_keydown)
        result = self.mailbox.submit(ActionRequest(action_id, key, client_keydown, received_ms))
        if result.get("status") == "replaced":
            replaced_id = int(result["replaced_action_id"])
            replaced = self.ledger.replace(replaced_id, action_id)
            self.bus.publish({"type": "action_record", "record": replaced})
        self.bus.publish({
            "type": "server_event",
            "event": "input_received",
            "action_id": action_id,
            "server_ms": received_ms,
        })
        return result

    def mark_browser(self, payload: dict[str, Any]) -> None:
        try:
            action_id = int(payload["action_id"])
            field = str(payload["field"])
            value = float(payload["value_ms"])
        except (KeyError, TypeError, ValueError):
            return
        if field in {"keydown_ms", "frame_received_ms", "frame_decoded_ms", "frame_presented_ms"}:
            self.ledger.mark_browser(action_id, field, value)

    def request_quit(self) -> None:
        self.mailbox.request_quit()

    def _mark(self, name: str) -> float:
        value = self.now_ms()
        self.startup[name] = value
        return value

    def _configure_optimized_model(self):
        import torch.cuda.tunable as tunable
        from pure_compile_helpers import install_on_blocks, prewarm_pure_tensor_islands
        from wan.image2video import WanI2VCausal

        result_path = Path(self.args.tunableop_results).resolve()
        tunable.set_filename(str(result_path))
        tunable.enable(True)
        tunable.tuning_enable(False)
        tunable.record_untuned_enable(False)
        if not tunable.read_file(str(result_path)):
            raise RuntimeError(f"TunableOp rejected result file: {result_path}")
        self._mark("tunableop_loaded_ms")
        self.tunable_info = {
            "enabled": tunable.is_enabled(),
            "tuning_enabled": tunable.tuning_is_enabled(),
            "record_untuned": tunable.record_untuned_is_enabled(),
            "loaded_results": len(tunable.get_results()),
            "filename": str(result_path),
        }
        self.original_init = WanI2VCausal.__init__

        def patched_init(model_self, *init_args, **init_kwargs):
            self._mark("model_initialization_start_ms")
            self.original_init(model_self, *init_args, **init_kwargs)
            self._mark("model_initialization_end_ms")
            self.set_status("RUNTIME_PREWARM")
            self.installation = install_on_blocks(
                model_self.model, self.args.pure_compile_block_count
            )
            if self.args.prewarm_pure_compile:
                self._mark("prewarm_start_ms")
                prewarm_pure_tensor_islands(self.installation["islands"], model_self.device)
                self._mark("prewarm_end_ms")

        WanI2VCausal.__init__ = patched_init
        return WanI2VCausal

    def _restore_optimized_model(self) -> None:
        if self.original_init is not None:
            from wan.image2video import WanI2VCausal
            WanI2VCausal.__init__ = self.original_init

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.device)
            self.set_status("MODEL_LOADING")
            WanI2VCausal = self._configure_optimized_model()
            self.pipe = WanI2VCausal(
                config=__import__("wan.configs", fromlist=["WAN_CONFIGS"]).WAN_CONFIGS["i2v-A14B"],
                checkpoint_dir=self.args.model_dir,
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,
                convert_model_dtype=False,
                local_attn_size=self.args.local_attn_size,
                sink_size=self.args.sink_size,
                infer_mode="causal_fast",
                metrics=None,
            )
            sync(self.device)
            self.set_status("SESSION_PREPARING")
            self._mark("session_preparation_start_ms")
            self.state = prepare_session(self.pipe, self.args, self.device)
            self._mark("session_preparation_end_ms")
            self.decoder = StreamingTAEHVDecoder(self.args.taehv_dir, self.device, None)
            self._mark("runtime_ready_ms")
            self.set_status("BOOTSTRAP_GENERATING")
            self._generate(
                ActionRequest(0, "bootstrap", None, self.now_ms()),
                self.state["plucker_chunks"][0],
                bootstrap=True,
            )
            self.presentation.wait_encoded(0, 0, timeout=20.0)
            self.ready = True
            self._mark("interactive_ready_ms")
            self.set_status("INTERACTIVE_READY", startup=dict(self.startup))
            while not self.stop_event.is_set():
                request = self.mailbox.take(timeout=0.25)
                if request is None:
                    if self.mailbox.quit_requested:
                        break
                    continue
                self.ledger.mark_server(request.action_id, "input_selected_ms", self.now_ms())
                self.bus.publish({
                    "type": "server_event",
                    "event": "input_selected",
                    "action_id": request.action_id,
                    "server_ms": self.now_ms(),
                })
                selected = action_from_key(request.key)
                if selected is None or selected[0] == "ignored":
                    self.mailbox.complete()
                    continue
                _, pose = selected
                self.set_status("ACTION_RUNNING", action_id=request.action_id)
                plucker = make_plucker(
                    pose, self.args, self.state, self.device, self.pipe.param_dtype
                )
                self._generate(request, plucker, bootstrap=False)
                self.mailbox.complete()
                if not self.mailbox.quit_requested:
                    self.set_status("INTERACTIVE_READY")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.set_status("ERROR", error=self.error)
        finally:
            self.ready = False
            if self.error is None:
                self.set_status("STOPPING")
            if self.decoder is not None:
                self.decoder.clear()
            self.presentation.stop()
            self._restore_optimized_model()
            self.finished.set()

    def _generate(
        self,
        request: ActionRequest,
        plucker: torch.Tensor,
        *,
        bootstrap: bool,
    ) -> None:
        assert self.pipe is not None and self.state is not None and self.decoder is not None
        action_id = request.action_id
        action_label = "bootstrap" if bootstrap else request.key
        if bootstrap:
            self.ledger.start(action_id, action_label, bootstrap=True)
        generation_start = self.now_ms()
        self.ledger.mark_server(action_id, "generation_start_ms", generation_start)
        self.bus.publish({
            "type": "server_event",
            "event": "generation_start",
            "action_id": action_id,
            "server_ms": generation_start,
        })
        chunk_id = 0 if bootstrap else self.action_count + 1
        if not bootstrap:
            self.action_count += 1
            if chunk_id >= len(self.state["noise_chunks"]):
                raise RuntimeError("maximum configured browser actions reached")
        with torch.amp.autocast("cuda", dtype=self.pipe.param_dtype), torch.no_grad():
            generated = generate_chunk(
                self.pipe,
                self.state,
                chunk_id,
                self.state["noise_chunks"][chunk_id],
                self.state["condition_chunks"][chunk_id],
                plucker,
                self.device,
                lambda: sync(self.device),
                defer_clean_kv=True,
            )
            accepted_x0 = self.now_ms()
            self.ledger.mark_server(action_id, "accepted_x0_ms", accepted_x0)
            self.bus.publish({
                "type": "server_event",
                "event": "accepted_x0",
                "action_id": action_id,
                "server_ms": accepted_x0,
            })
            first_pending, tae_first_ms, _ = self.decoder.begin_latent(
                generated["x0"], self.device
            )
            first_host = first_pending[0, 0].float().clamp(0, 1).cpu()
            base_ready = self.now_ms()
            self.ledger.mark_server(action_id, "base_rgb_ready_ms", base_ready)
            self.bus.publish({
                "type": "server_event",
                "event": "base_rgb_ready",
                "action_id": action_id,
                "server_ms": base_ready,
            })
            frame_server = dict(self.ledger.get(action_id)["server"])
            self.presentation.submit(PresentationFrame(
                action_id=action_id,
                frame_index=0,
                frame_count=4,
                frame=first_host,
                bootstrap=bootstrap,
                server=frame_server,
            ))
            clean_start = self.now_ms()
            self.ledger.mark_server(action_id, "clean_kv_start_ms", clean_start)
            self.bus.publish({
                "type": "server_event",
                "event": "clean_kv_start",
                "action_id": action_id,
                "server_ms": clean_start,
            })
            commit_clean_kv(self.pipe, self.state, generated, self.device, lambda: sync(self.device))
            clean_end = self.now_ms()
            self.ledger.mark_server(action_id, "clean_kv_complete_ms", clean_end)
            self.bus.publish({
                "type": "server_event",
                "event": "clean_kv_complete",
                "action_id": action_id,
                "server_ms": clean_end,
            })
            decoded_gpu, tae_remaining_ms = self.decoder.drain_latent(first_pending, self.device)
            frame_count = int(decoded_gpu.shape[1])
            completed_server = dict(self.ledger.get(action_id)["server"])
            for frame_index in range(1, frame_count):
                tail_host = decoded_gpu[:, frame_index].float().clamp(0, 1).cpu()
                self.presentation.submit(PresentationFrame(
                    action_id=action_id,
                    frame_index=frame_index,
                    frame_count=frame_count,
                    frame=tail_host,
                    bootstrap=bootstrap,
                    server=completed_server,
                ))
            all_ready = self.now_ms()
            self.ledger.mark_server(action_id, "all_rgb_ready_ms", all_ready)
            self.ledger.mark_server(action_id, "next_ready_ms", all_ready)
            self.bus.publish({
                "type": "server_event",
                "event": "all_rgb_ready",
                "action_id": action_id,
                "server_ms": all_ready,
            })
            self.bus.publish({
                "type": "server_event",
                "event": "next_ready",
                "action_id": action_id,
                "server_ms": all_ready,
            })
            denoise_ms = sum(
                float(row["elapsed_ms"])
                for row in generated["forward_records"]
                if row.get("kind") == "denoise"
            )
            server = self.ledger.get(action_id)["server"]
            derived = {
                "action_to_base_rgb_ms": (
                    server["base_rgb_ready_ms"] - server["input_received_ms"]
                    if server["input_received_ms"] is not None
                    else None
                ),
                "action_to_next_ready_ms": (
                    server["next_ready_ms"] - server["input_received_ms"]
                    if server["input_received_ms"] is not None
                    else None
                ),
                "generation_to_x0_ms": server["accepted_x0_ms"] - server["generation_start_ms"],
                "generation_to_clean_kv_ms": (
                    server["clean_kv_complete_ms"] - server["generation_start_ms"]
                ),
            }
            finished = self.ledger.finish(
                action_id,
                action=action_label,
                bootstrap=bootstrap,
                status="complete",
                denoise_forward_count=len(state_timesteps(self.state)),
                denoise_ms=denoise_ms,
                transformer_ms=float(generated["transformer_ms"]),
                forward_records=generated["forward_records"],
                action_prepare_ms=float(generated["action_prepare_ms"]),
                latent_postprocess_ms=float(generated["latent_postprocess_ms"]),
                clean_kv_ms=float(generated["clean_kv_ms"]),
                tae_first_rgb_gpu_ms=tae_first_ms,
                tae_remaining_rgb_gpu_ms=tae_remaining_ms,
                derived=derived,
                cache={
                    "global_end_index": int(generated["cache"]["global_end_index"]),
                    "local_end_index": int(generated["cache"]["local_end_index"]),
                    "capacity_tokens": int(self.state["kv_size"]),
                },
                finite=all(
                    bool(row.get("finite", True))
                    for row in generated["forward_records"]
                    if row.get("kind") == "denoise"
                ),
            )
            self.bus.publish({"type": "action_record", "record": finished})

    def report(self) -> dict[str, Any]:
        actions = self.ledger.all()
        for record in actions:
            browser = record.get("browser", {})
            keydown = browser.get("keydown_ms")
            presented = browser.get("frame_presented_ms")
            if keydown is not None and presented is not None:
                browser["keydown_to_presented_ms"] = presented - keydown
        return {
            "status": self.status,
            "startup": dict(self.startup),
            "tunableop": getattr(self, "tunable_info", None),
            "actions": actions,
            "presentation": {
                "dropped_tail_frames": self.presentation.queue.dropped_tail_frames,
                "dropped_by_action": dict(self.presentation.queue.dropped_by_action),
                "jpeg_quality": self.args.jpeg_quality,
            },
            "error": self.error,
        }


def state_timesteps(state: dict[str, Any]) -> list[Any]:
    return list(state["timesteps"])


class BrowserServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process_start = time.perf_counter()
        self.loop = asyncio.get_running_loop()
        self.bus = EventBus(self.loop)
        self.owner = ModelOwner(args, self.bus, self.process_start)
        self.clients: set[ClientOutbound] = set()
        self.stop_requested = asyncio.Event()
        self.app = web.Application()
        self.app.router.add_get("/", self.index)
        self.app.router.add_get("/app.js", self.asset)
        self.app.router.add_get("/style.css", self.asset)
        self.app.router.add_get("/ws", self.websocket)
        self.app.router.add_get("/health", self.health)
        self.app[BROWSER_ROOT_KEY] = Path(args.browser_root).resolve()

    async def index(self, request: web.Request) -> web.Response:
        response = web.FileResponse(self.app[BROWSER_ROOT_KEY] / "index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    async def asset(self, request: web.Request) -> web.Response:
        name = request.match_info.get("filename") or request.path.rsplit("/", 1)[-1]
        if name not in {"app.js", "style.css"}:
            raise web.HTTPNotFound()
        response = web.FileResponse(self.app[BROWSER_ROOT_KEY] / name)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response(self.owner.status_snapshot())

    async def _client_sender(self, ws: web.WebSocketResponse, outgoing: ClientOutbound) -> None:
        while not ws.closed:
            event = await outgoing.take()
            if event.get("type") == "frame":
                await ws.send_bytes(event["packet"])
            else:
                await ws.send_json(event)

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=2 * 1024 * 1024)
        await ws.prepare(request)
        outgoing = ClientOutbound(max_items=16)
        self.clients.add(outgoing)
        await ws.send_json(self.owner.status_snapshot())
        if self.bus.latest_frame is not None:
            await ws.send_bytes(self.bus.latest_frame["packet"])
        sender = asyncio.create_task(self._client_sender(ws, outgoing))
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "error": "invalid JSON"})
                        continue
                    if not isinstance(payload, dict):
                        continue
                    kind = payload.get("type")
                    if kind == "action":
                        result = self.owner.submit_action(payload)
                        await ws.send_json({"type": "input_ack", **result})
                    elif kind == "quit":
                        self.owner.request_quit()
                    elif kind == "browser_telemetry":
                        self.owner.mark_browser(payload)
                elif message.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            self.clients.discard(outgoing)
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
        return ws

    async def broadcast_loop(self) -> None:
        while not self.stop_requested.is_set():
            event = await self.bus.take()
            for outgoing in tuple(self.clients):
                outgoing.put(event)

    async def run(self) -> None:
        self.bus.publish({
            "type": "status",
            "state": "SERVER_STARTING",
            "text": STATE_TEXT["SERVER_STARTING"],
            "server_ms": 0.0,
        })
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.args.host, self.args.port)
        await site.start()
        self.owner.start()
        broadcast_task = asyncio.create_task(self.broadcast_loop())
        url = f"http://{self.args.host}:{self.args.port}/"
        print(f"LingBot browser server: {url}", flush=True)
        if self.args.open_browser:
            webbrowser.open(url)
        try:
            while not self.stop_requested.is_set():
                if self.owner.finished.is_set() and (
                    self.owner.mailbox.quit_requested or self.owner.error is not None
                ):
                    break
                await asyncio.sleep(0.1)
        finally:
            self.owner.request_quit()
            await asyncio.to_thread(self.owner.finished.wait, 30.0)
            self.stop_requested.set()
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass
            report_path = Path(self.args.output_dir) / "browser_metrics.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(self.owner.report(), indent=2, default=str) + "\n")
            await runner.cleanup()


def build_browser_parser() -> argparse.ArgumentParser:
    parser = build_interactive_parser()
    output_dir_action = parser._option_string_actions["--output-dir"]
    output_dir_action.required = False
    output_dir_action.default = "results/raw/browser-serving-20260910"
    parser.set_defaults(
        size="480*832",
        max_area_pixels=264192,
        frames=165,
        denoise_schedule="3-drop-957",
        local_attn_size=12,
        sink_size=6,
        vae_attention_backend="math",
        vae_dtype="fp16",
        display_decoder="taehv",
        taehv_dir=".upstream/taehv",
        defer_clean_kv=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--browser-root", default=str(Path(__file__).resolve().parent.parent / "web"))
    parser.add_argument(
        "--tunableop-results",
        default="docs/artifacts/tunable-op-20260910/tunableop_results.csv",
    )
    parser.add_argument("--pure-compile-block-count", type=int, default=30)
    parser.add_argument(
        "--prewarm-pure-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-open-browser", dest="open_browser", action="store_false", default=True)
    return parser


def main() -> int:
    args = build_browser_parser().parse_args()
    if args.display_decoder != "taehv":
        raise SystemExit("browser RC1 is pinned to the TAEHV presentation path")
    if args.overlap or args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("RC1 browser server is localhost-only and serial")
    if args.max_actions < 1:
        raise SystemExit("--max-actions must be positive")
    args.frames = 4 * (args.max_actions + 1) + 1
    async def run_server() -> None:
        server = BrowserServer(args)
        await server.run()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
