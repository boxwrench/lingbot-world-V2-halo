"""R2/R3 driver: scripted websocket actions against production browser server.
Uses only existing telemetry (action_record events, /health) + external host
sampling. No intrusive module profiling.
Sequence: 200x 'w' (maturity curve) then 16x R3 pattern [w,w,j,j,w,k,w,l]x2.
Saves per-action records + frame JPEGs for R6 clips.
"""
import aiohttp
import asyncio
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8767
SRV_PID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = Path("/tmp/r2-long")
FRAMES = OUT / "frames"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)

SEQ_MAIN = ["w"] * 200
SEQ_R3 = ["w", "w", "j", "j", "w", "k", "w", "l"] * 2
SEQ = SEQ_MAIN + SEQ_R3
CHECKPOINTS = {1, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200,
               201, 202, 203, 204, 205, 206, 207, 208, 216}


def host_sample():
    s = {"t": time.time()}
    try:
        s["loadavg"] = open("/proc/loadavg").read().strip()
    except Exception:
        pass
    if SRV_PID:
        try:
            for line in open(f"/proc/{SRV_PID}/status"):
                if line.startswith("VmRSS:"):
                    s["server_rss"] = line.strip()
                    break
        except Exception as e:
            s["server_rss_err"] = str(e)
    try:
        p = subprocess.run(["rocm-smi", "--showuse", "--showtemp", "--showpower"],
                           capture_output=True, text=True, timeout=10)
        txt = p.stdout
        import re
        m = re.search(r"GPU use.*?:\s*(\d+)", txt)
        if m:
            s["gpu_use"] = int(m.group(1))
        m = re.search(r"Sensor edge.*?([\d.]+)", txt)
        if m:
            s["gpu_temp_c"] = float(m.group(1))
        m = re.search(r"Package Power.*?([\d.]+)", txt)
        if m:
            s["gpu_power_w"] = float(m.group(1))
    except Exception as e:
        s["rocm_err"] = str(e)[:120]
    return s


def unpack(packet: bytes):
    hlen = struct.unpack(">I", packet[:4])[0]
    hdr = json.loads(packet[4:4 + hlen].decode())
    return hdr, packet[4 + hlen:]


async def wait_ready():
    url = f"http://127.0.0.1:{PORT}/health"
    async with aiohttp.ClientSession() as sess:
        for _ in range(300):
            try:
                async with sess.get(url, timeout=5) as r:
                    d = await r.json()
                    if d.get("state") == "INTERACTIVE_READY":
                        return d
            except Exception:
                pass
            await asyncio.sleep(2)
    raise RuntimeError("server never reached INTERACTIVE_READY")


async def main():
    ready = await wait_ready()
    print(f"READY startup_keys={sorted(ready.get('startup', {}).keys())}", flush=True)
    rec_f = open(OUT / "records.jsonl", "a")
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(f"http://127.0.0.1:{PORT}/ws",
                                   max_msg_size=8 * 1024 * 1024) as ws:
            # drain initial status + bootstrap frame
            await asyncio.sleep(1)
            while True:
                try:
                    m = await asyncio.wait_for(ws.receive(), timeout=0.2)
                    if m.type == aiohttp.WSMsgType.BINARY:
                        hdr, jpg = unpack(m.data)
                        (FRAMES / f"a0000_f{hdr.get('frame_index', 0)}.jpg").write_bytes(jpg)
                    if m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                except asyncio.TimeoutError:
                    break
            for idx, key in enumerate(SEQ, start=1):
                t_send = time.time()
                await ws.send_json({"type": "action", "action_id": idx,
                                    "key": key, "client_keydown_ms": t_send * 1000.0})
                got_record = None
                frames = 0
                t0 = time.monotonic()
                while True:
                    dt = time.monotonic() - t0
                    if dt > 180:
                        raise RuntimeError(f"action {idx} timed out after 180s")
                    m = await ws.receive()
                    if m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        t = d.get("type")
                        if t == "action_record" and d.get("record", {}).get("action_id") == idx:
                            got_record = d["record"]
                            break
                    elif m.type == aiohttp.WSMsgType.BINARY:
                        hdr, jpg = unpack(m.data)
                        if hdr.get("action_id") == idx:
                            frames += 1
                            (FRAMES / f"a{idx:04d}_f{hdr.get('frame_index', 0)}.jpg").write_bytes(jpg)
                    elif m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise RuntimeError(f"ws closed during action {idx}")
                wall = time.time()
                hs = host_sample() if idx in CHECKPOINTS else {"t": wall}
                rec = {"wall": wall, "idx": idx, "key": key,
                       "frames_seen": frames, "host": hs, "record": got_record}
                rec_f.write(json.dumps(rec) + "\n")
                rec_f.flush()
                if idx in CHECKPOINTS or idx % 20 == 0:
                    der = (got_record or {}).get("derived", {})
                    print(f"action {idx:3d} key={key} "
                          f"base_rgb={der.get('action_to_base_rgb_ms')} "
                          f"next_ready={der.get('action_to_next_ready_ms')} "
                          f"denoise={got_record.get('denoise_ms') if got_record else None} "
                          f"g={(got_record.get('cache', {}) or {}).get('global_end_index') if got_record else None}"
                          f"/{(got_record.get('cache', {}) or {}).get('local_end_index') if got_record else None} "
                          f"rss={hs.get('server_rss', '?')} "
                          f"gpu={hs.get('gpu_use', '?')}%/{hs.get('gpu_temp_c', '?')}C",
                          flush=True)
    rec_f.close()
    print("DRIVE_DONE", flush=True)


asyncio.run(main())
