#!/usr/bin/env python3
"""P2.0 pooled baseline analysis: 3x15 rolled actions on tagged main."""
import json
import statistics

RUNS = ['waterfall', 'aa-run-1', 'aa-run-2']
BASE = '/tmp/p20-20260913'


def pct(data, q):
    s = sorted(data)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def load(run):
    return json.load(open(f'{BASE}/{run}/benchmark.json'))


pooled = {'base_rgb': [], 'next_ready': [], 'denoise': [], 'clean_kv': [],
          'fwd0': [], 'fwd1': [], 'fwd2': [], 'clean_fwd': [],
          'clean_vs_d3': [], 'decode_present': []}
per_run = {}
for run in RUNS:
    b = load(run)
    recs = b['records']
    assert len(recs) == 15, (run, len(recs))
    m = {'base_rgb': [], 'next_ready': [], 'denoise': [], 'clean_kv': []}
    for r in recs:
        s = r['server']
        m['base_rgb'].append(s['base_rgb_ready_ms'] - s['input_received_ms'])
        m['next_ready'].append(s['next_ready_ms'] - s['input_received_ms'])
        m['denoise'].append(r['denoise_ms'])
        m['clean_kv'].append(s['clean_kv_complete_ms'] - s['clean_kv_start_ms'])
        fr = r['forward_records']
        dn = [f['elapsed_ms'] for f in fr if f['kind'] == 'denoise']
        cl = [f['elapsed_ms'] for f in fr if f['kind'] == 'cache_update'][0]
        pooled['fwd0'].append(dn[0])
        pooled['fwd1'].append(dn[1])
        pooled['fwd2'].append(dn[2])
        pooled['clean_fwd'].append(cl)
        pooled['clean_vs_d3'].append(cl - dn[2])
        if 'browser' in r and r['browser'].get('frame_presented_ms'):
            pooled['decode_present'].append(
                r['browser']['frame_presented_ms'] - r['browser']['frame_received_ms'])
    for k in m:
        pooled[k].extend(m[k])
    per_run[run] = {k: {'p50': statistics.median(v), 'p95': pct(v, 0.95),
                        'mean': statistics.mean(v), 'n': len(v)} for k, v in m.items()}

summary = {
    'runs': RUNS, 'actions_per_run': 15, 'pooled_n': len(pooled['base_rgb']),
    'pooled': {k: {'p50': statistics.median(v), 'p95': pct(v, 0.95),
                   'mean': statistics.mean(v),
                   'stdev': statistics.stdev(v) if len(v) > 1 else 0.0}
               for k, v in pooled.items() if v},
    'per_run': per_run,
    'clean_vs_d3_wins': sum(1 for d in pooled['clean_vs_d3'] if d < 0),
    'clean_vs_d3_total': len(pooled['clean_vs_d3']),
}
json.dump(summary, open(f'{BASE}/summary.json', 'w'), indent=2)
for k, v in summary['pooled'].items():
    print(f'{k:14s} p50={v["p50"]:8.1f} p95={v["p95"]:8.1f} mean={v["mean"]:8.1f} sd={v["stdev"]:6.1f}')
print('clean-vs-d3 wins:', summary['clean_vs_d3_wins'], '/', summary['clean_vs_d3_total'])
print('per-run baseRGB p50:', {r: round(per_run[r]['base_rgb']['p50'], 1) for r in RUNS})
print('per-run nextReady p50:', {r: round(per_run[r]['next_ready']['p50'], 1) for r in RUNS})
print('per-run denoise p50:', {r: round(per_run[r]['denoise']['p50'], 1) for r in RUNS})
