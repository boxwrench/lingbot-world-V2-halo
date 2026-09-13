"""R2/R3 extended driver: scripted websocket actions, examples/00 path.
Labeled diffs vs human play (no compute effect): scripted keys via same
submit path; camera trajectory from examples/00 (384x672 shapes identical);
output under /tmp.
Sequence: 150x w, R3 block [w,w,j,j,w,k,w,l], 70x w, R3 block, 10x w
=> 236 actions max (ceiling 240 on ex-00). Checkpoints throughout.
"""
import aiohttp
import asyncio
import glob
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8768
OUT = Path("/tmp/r2b-long")
FRAMES = OUT / "frames"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)

R3 = ["w", "w", "j", "j", "w", "k", "w", "l"]
SEQ = ([("w", "maturity")] * 150
       + [(k, "r3a") for k in R3]
       + [("w", "maturity")] * 70
       + [(k, "r3b") for k in R3]
       + [("w", "maturity")] * 8)
CHECKPOINTS = {1, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 151, 152, 153,
               154, 155, 156, 157, 158, 180, 200, 220, 228, 229, 230, 231,
               232, 233, 234, 235, 236}


def server_py_pid():
    for cmdf in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            raw = open(cmdf, "rb").read().replace(b"\0", b" ").decode()
            if "run_browser.py" in raw and str(PORT) in raw:
                return int(cmdf.split("/")[2])
        except Exception:
            pass
    return None


def host_sample():
    s = {"t": time.time()}
    try:
        s["loadavg"] = open("/proc/loadavg").read().strip()
    except Exception:
        pass
    pid = server_py_pid()
    s["srv_pid"] = pid
    if pid:
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith(("VmRSS:", "VmHWM:")):
                    s[line.split(":")[0]] = line.strip().split()[-2] + " " + line.strip().split()[-1]
        except Exception as e:
            s["rss_err"] = str(e)[:100]
    try:
        p = subprocess.run(["rocm-smi", "--showuse", "--showtemp", "--showpower"],
                           capture_output=True, text=True, timeout=10)
        import re
        m = re.search(r"GPU use.*?:\s*(\d+)", p.stdout)
        if m:
            s["gpu_use"] = int(m.group(1))
        m = re.search(r"Sensor edge.*?([\d.]+)", p.stdout)
        if m:
            s["gpu_temp_c"] = float(m.group(1))
        m = re.search(r"Package Power.*?([\d.]+)", p.stdout)
        if m:
            s["gpu_power_w"] = float(m.group(1))
    except Exception as e:
        s["rocm_err"] = str(e)[:120]
    return s


def unpack(packet: bytes):
    hlen = struct.unpack(">I", packet[:4])[0]
    return json.loads(packet[4:4 + hlen].decode()), packet[4 + hlen:]


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
    print(f"READY startup session_prep_s="
          f"{(ready['startup']['session_preparation_end_ms'] - ready['startup']['session_preparation_start_ms']) / 1000:.1f}",
          flush=True)
    rec_f = open(OUT / "records.jsonl", "a")
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.ws_connect(f"http://127.0.0.1:{PORT}/ws",
                                       max_msg_size=8 * 1024 * 1024) as ws:
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
                for idx, (key, phase) in enumerate(SEQ, start=1):
                    t_send = time.time()
                    await ws.send_json({"type": "action", "action_id": idx,
                                        "key": key, "client_keydown_ms": t_send * 1000.0})
                    got_record = None
                    frames = 0
                    t0 = time.monotonic()
                    while True:
                        if time.monotonic() - t0 > 180:
                            raise RuntimeError(f"action {idx} timed out after 180s")
                        m = await ws.receive()
                        if m.type == aiohttp.WSMsgType.TEXT:
                            d = json.loads(m.data)
                            t = d.get("type")
                            if t == "action_record" and d.get("record", {}).get("action_id") == idx:
                                got_record = d["record"]
                                break
                            if t == "status" and d.get("state") == "ERROR":
                                raise RuntimeError(f"server ERROR at action {idx}: {d}")
                        elif m.type == aiohttp.WSMsgType.BINARY:
                            hdr, jpg = unpack(m.data)
                            if hdr.get("action_id") == idx:
                                frames += 1
                                (FRAMES / f"a{idx:04d}_f{hdr.get('frame_index', 0)}.jpg").write_bytes(jpg)
                        elif m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            try:
                                async with sess.get(f"http://127.0.0.1:{PORT}/health",
                                                    timeout=5) as r:
                                    print("HEALTH_AFTER_CLOSE:", (await r.text())[:500], flush=True)
                            except Exception as e:
                                print("HEALTH_AFTER_CLOSE failed:", e, flush=True)
                            raise RuntimeError(f"ws closed during action {idx}")
                    hs = host_sample() if idx in CHECKPOINTS else {"t": time.time()}
                    rec_f.write(json.dumps({"wall": time.time(), "idx": idx, "key": key,
                                            "phase": phase, "frames_seen": frames,
                                            "host": hs, "record": got_record}) + "\n")
                    rec_f.flush()
                    if idx in CHECKPOINTS or idx % 20 == 0:
                        der = (got_record or {}).get("derived", {})
                        print(f"action {idx:3d} {phase:>8} key={key} "
                              f"base={der.get('action_to_base_rgb_ms')} "
                              f"next={der.get('action_to_next_ready_ms')} "
                              f"den={got_record.get('denoise_ms') if got_record else None} "
                              f"g/l={(got_record.get('cache', {}) or {}).get('global_end_index') if got_record else None}"
                              f"/{(got_record.get('cache', {}) or {}).get('local_end_index') if got_record else None} "
                              f"rss={hs.get('VmRSS', '?')} gpu={hs.get('gpu_use', '?')}%/"
                              f"{hs.get('gpu_temp_c', '?')}C", flush=True)
    finally:
        rec_f.close()
    print("DRIVE_DONE", flush=True)


asyncio.run(main())
