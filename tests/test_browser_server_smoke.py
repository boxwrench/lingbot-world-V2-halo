from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from aiohttp import ClientSession
from aiohttp.test_utils import TestServer

import run_browser
from browser_protocol import PendingActionMailbox, TelemetryLedger


class _FakeOwner:
    def __init__(self, args, bus, process_start):
        self.status = "INTERACTIVE_READY"
        self.error = None
        self.finished = asyncio.Event()
        self.mailbox = SimpleNamespace(quit_requested=False)

    def status_snapshot(self):
        return {"type": "status", "state": self.status, "text": "Ready", "server_ms": 0.0}


def test_browser_http_smoke(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ok")
    (tmp_path / "app.js").write_text("ok")
    (tmp_path / "style.css").write_text("ok")
    monkeypatch.setattr(run_browser, "ModelOwner", _FakeOwner)

    args = SimpleNamespace(
        browser_root=str(tmp_path),
        host="127.0.0.1",
        port=0,
        jpeg_quality=90,
        local_attn_size=12,
        sink_size=6,
    )

    async def exercise() -> None:
        browser = run_browser.BrowserServer(args)
        server = TestServer(browser.app)
        await server.start_server()
        try:
            async with ClientSession() as client:
                health = await client.get(server.make_url("/health"))
                assert health.status == 200
                payload = await health.json()
                assert payload["state"] == "INTERACTIVE_READY"
                index = await client.get(server.make_url("/"))
                assert await index.text() == "ok"
        finally:
            await server.close()

    asyncio.run(exercise())


def test_action_completes_when_browser_telemetry_arrives_first() -> None:
    owner = object.__new__(run_browser.ModelOwner)
    owner.ready = True
    owner.ledger = TelemetryLedger()
    owner.mailbox = PendingActionMailbox()
    owner.bus = SimpleNamespace(publish=lambda _event: None)
    owner.now_ms = lambda: 10.0

    # WebSocket ordering normally sends the action first, but the server must
    # tolerate an auxiliary browser telemetry message arriving first.
    owner.mark_browser({
        "action_id": 7,
        "field": "keydown_ms",
        "value_ms": 4.0,
    })
    assert owner.submit_action({
        "action_id": 7,
        "key": "w",
        "client_keydown_ms": 4.0,
    })["status"] == "queued"
    record = owner.ledger.get(7)
    assert record is not None
    assert record["action"] == "w"
    assert record["server"]["input_received_ms"] == 10.0
