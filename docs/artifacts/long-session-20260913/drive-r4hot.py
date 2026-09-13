"""R4-hot: 12x 'w' on ex-03 in current (warm) machine state. Compares fresh-
process ramp against cold R2a/R2c values. Saves f0 frames."""
import aiohttp, asyncio, json, struct, sys, time
from pathlib import Path
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
OUT = Path("/tmp/r4-hot"); FR = OUT / "frames"; OUT.mkdir(parents=True, exist_ok=True); FR.mkdir(parents=True, exist_ok=True)

def unpack(p):
    import struct as s, json as j
    h = s.unpack(">I", p[:4])[0]
    return j.loads(p[4:4+h].decode()), p[4+h:]

async def main():
    async with aiohttp.ClientSession() as sess:
        for _ in range(400):
            try:
                async with sess.get(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if (await r.json()).get("state") == "INTERACTIVE_READY":
                        break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("never ready")
        async with sess.ws_connect(f"http://127.0.0.1:{PORT}/ws", max_msg_size=8*1024*1024) as ws:
            await asyncio.sleep(1)
            while True:
                try:
                    m = await asyncio.wait_for(ws.receive(), timeout=0.2)
                    if m.type == aiohttp.WSMsgType.BINARY:
                        h, jpg = unpack(m.data)
                        (FR / f"a0000_f{h.get('frame_index',0)}.jpg").write_bytes(jpg)
                    if m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                except asyncio.TimeoutError:
                    break
            recs = []
            for idx in range(1, 13):
                await ws.send_json({"type": "action", "action_id": idx, "key": "w",
                                    "client_keydown_ms": time.time()*1000.0})
                got = None; t0 = time.monotonic()
                while True:
                    assert time.monotonic()-t0 < 180, f"timeout {idx}"
                    m = await ws.receive()
                    if m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        if d.get("type") == "action_record" and d["record"]["action_id"] == idx:
                            got = d["record"]; break
                    elif m.type == aiohttp.WSMsgType.BINARY:
                        h, jpg = unpack(m.data)
                        if h.get("action_id") == idx:
                            (FR / f"a{idx:04d}_f{h.get('frame_index',0)}.jpg").write_bytes(jpg)
                    else:
                        raise RuntimeError(f"ws closed at {idx}")
                der = got["derived"]
                print(f"action {idx:2d} base={der['action_to_base_rgb_ms']:.1f} "
                      f"next={der['action_to_next_ready_ms']:.1f} den={got['denoise_ms']:.1f}", flush=True)
                recs.append(got)
            json.dump(recs, open(OUT/"records.json","w"))
    print("HOT_DONE", flush=True)
asyncio.run(main())
