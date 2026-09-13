#!/usr/bin/env python3
"""P2.6 paired per-action ON-vs-OFF analysis (A4 closure).

Same 15-action string in both runs, so pair by action index (within-run
estimator): delta_i = ON_i - OFF_i for action_to_base_rgb, action_to_next_ready,
denoise. Reports mean delta, sd, and paired t. Written to paired-analysis.json.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def series(doc, key):
    if key == "denoise_ms":
        return [r["denoise_ms"] for r in doc["records"]]
    return [r["derived"][key] for r in doc["records"]]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1))
    return {"n": n, "mean_delta_ms": mean, "sd_delta_ms": sd,
            "t": mean / (sd / math.sqrt(n)), "deltas_ms": d}


def main():
    on = load("on-benchmark.json")
    off = load("off-benchmark.json")
    assert on["actions"] == off["actions"], "action strings must match for pairing"
    out = {
        "method": "paired per-action (ON_i - OFF_i), same action order",
        "actions": on["actions"],
        "base_rgb": paired(series(on, "action_to_base_rgb_ms"),
                           series(off, "action_to_base_rgb_ms")),
        "next_ready": paired(series(on, "action_to_next_ready_ms"),
                             series(off, "action_to_next_ready_ms")),
        "denoise": paired(series(on, "denoise_ms"), series(off, "denoise_ms")),
        "stop_rule_ms": 10,
    }
    with open(os.path.join(HERE, "paired-analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k in ("base_rgb", "next_ready", "denoise"):
        r = out[k]
        print(f"{k}: mean {r['mean_delta_ms']:+.2f} sd {r['sd_delta_ms']:.2f} "
              f"t {r['t']:+.2f} n={r['n']}")


if __name__ == "__main__":
    main()
