"""Queue experiment v2 (hardened): actions 1-4 serial, 5-12 at 700ms cadence.
Every receive has a timeout; every stage prints with flush; no swallowed errors."""
import aiohttp, asyncio, json, struct, sys, time
from pathlib import Path
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8771
OUT = Path("/tmp/r5q"); OUT.mkdir(parents=True, exist_ok=True)

def unpack(p):
    h = struct.unpack(">I", p[:4])[0]
    return json.loads(p[4:4+h].decode()), p[4+h:]

async def get_health(sess):
    async with sess.get(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
        return await r.json()

async def main():
    print(f"driver start port={PORT}", flush=True)
    async with aiohttp.ClientSession() as sess:
        st = None
        for i in range(400):
            try:
                st = await get_health(sess)
                if st.get("state") == "INTERACTIVE_READY":
                    print(f"ready after {i+1} polls", flush=True)
                    break
            except Exception as e:
                if i % 30 == 0:
                    print(f"health poll {i}: {type(e).__name__} {e}", flush=True)
            await asyncio.sleep(2)
        else:
            raise RuntimeError("never ready")
        print(f"ws connecting (server state={st.get('state')})", flush=True)
        async with sess.ws_connect(f"http://127.0.0.1:{PORT}/ws",
                                   max_msg_size=8*1024*1024) as ws:
            print("ws connected, draining 1s", flush=True)
            await asyncio.sleep(1)
            while True:
                try:
                    m = await asyncio.wait_for(ws.receive(), timeout=0.2)
                    if m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        print("ws closed during drain", flush=True)
                        break
                except asyncio.TimeoutError:
                    break
            print("drain done", flush=True)
            out = []

            async def do_serial(idx):
                print(f"serial {idx}: sending", flush=True)
                await ws.send_json({"type": "action", "action_id": idx, "key": "w",
                                    "client_keydown_ms": time.time()*1000.0})
                t0 = time.monotonic()
                while True:
                    el = time.monotonic() - t0
                    assert el < 180, f"serial {idx} no record after {el:.0f}s"
                    m = await asyncio.wait_for(ws.receive(), timeout=10)
                    if m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        if d.get("type") == "input_ack":
                            print(f"serial {idx}: ack={d.get('status')}", flush=True)
                        if d.get("type") == "action_record" and d["record"]["action_id"] == idx:
                            return d["record"]
                    elif m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise RuntimeError(f"serial {idx}: ws closed")

            for idx in (1, 2, 3, 4):
                out.append(("serial", idx, await do_serial(idx)))

            async def reader(store, acks, done_after_s=30):
                # Mailbox depth is 1 with latest-wins: fast keys REPLACE rather
                # than queue, so fewer than 8 records may ever arrive. Stop
                # after done_after_s idle seconds past the last send.
                while True:
                    try:
                        m = await asyncio.wait_for(ws.receive(), timeout=5)
                    except asyncio.TimeoutError:
                        if time.monotonic() - sends_done_at[0] > done_after_s:
                            print("pipelined: idle timeout, stopping reader", flush=True)
                            break
                        continue
                    if m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        if d.get("type") == "action_record":
                            store.append(d["record"])
                            print(f"pipelined: record {d['record']['action_id']} "
                                  f"status={d['record'].get('status')}", flush=True)
                        elif d.get("type") == "input_ack":
                            acks.append((d.get("action_id"), d.get("status"),
                                         d.get("replaced_action_id")))
                            print(f"pipelined: ack id={d.get('action_id')} status={d.get('status')} "
                                  f"replaced={d.get('replaced_action_id')}", flush=True)
                    elif m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        print("pipelined: ws closed", flush=True)
                        break

            store = []
            acks = []
            sends_done_at = [float("inf")]
            rt = asyncio.create_task(reader(store, acks))
            t_start = time.monotonic()
            for k, idx in enumerate(range(5, 13)):
                await ws.send_json({"type": "action", "action_id": idx, "key": "w",
                                    "client_keydown_ms": time.time()*1000.0})
                print(f"pipelined {idx}: sent at +{time.monotonic()-t_start:.2f}s", flush=True)
                delay = t_start + (k+1)*0.7 - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            sends_done_at[0] = time.monotonic()
            await asyncio.wait_for(rt, timeout=180)
            print(f"pipelined: acks={acks}", flush=True)
            by_id = {r["action_id"]: r for r in store if r.get("status") == "complete"}
            for idx in sorted(by_id):
                out.append(("pipelined", idx, by_id[idx]))
            for phase, idx, got in out:
                sv = got["server"]; d = got["derived"]
                q = (sv["input_selected_ms"] - sv["input_received_ms"]) if sv["input_selected_ms"] else None
                print(f"{phase:>9} action {idx:2d} queue={q if q is None else round(q,1)} "
                      f"base={d['action_to_base_rgb_ms']:.1f} next={d['action_to_next_ready_ms']:.1f} "
                      f"den={got['denoise_ms']:.1f}", flush=True)
            json.dump([{"phase": p, "idx": i, "r": r} for p, i, r in out], open(OUT/"records.json", "w"))
    print("QUEUE_DONE", flush=True)

asyncio.run(main())
