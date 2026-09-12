# C1 LingRuntime state/cache validation

Classification: **PASS**

The current LingBot recurrence/cache path matched an independent logical-list
reference at startup, partial fill, full/pre-eviction, first eviction, first
rolled state, repeated roll, and a later rolled state.  The real 1.3B model ran
the same boundaries on the Radeon 8060S/gfx1151.  A fresh session after the
later rolled state reproduced both the bootstrap accepted latent and layer-zero
clean KV cache bit-for-bit.  Two intentionally invalid controls were detected.

This is a correctness campaign, not a timing result.  No production semantic
or optimization code was changed.

## Semantic contract

Status vocabulary follows the campaign request.

| Item | Contract | Evidence class |
|---|---|---|
| Physical self-KV layout | One contiguous per-layer `[batch, capacity_tokens, heads, head_dim]` K tensor and one V tensor, plus global and local end cursors. It is not a circular ring. | SOURCE-PROVEN |
| Reported local capacity | `local_attn_size * frame_seqlen`; this is the total physical capacity and **includes** the sink budget. With local=4 and sink=1, the rolled membership is one sink frame plus three recent frames. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| Sink | `sink_size * frame_seqlen` tokens at the physical prefix. Before overflow this is only a retention policy; on overflow the prefix is left unchanged. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| Recent/local tokens | On overflow, the oldest non-sink tokens are removed, the remaining non-sink tokens are cloned left, and the new tokens are appended. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| Logical order | Sink tokens retain their original absolute identity, followed by retained recent tokens in chronological order, followed by the current chunk. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| “Wrap”/roll | There is no modulo slot cursor. The first rolled state is the first state with `global_end_index > capacity`; each later advancing append compacts again. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| Append versus overwrite | The first denoise forward at a new `current_start` advances/evicts. Later denoise forwards for that same start and the clean transaction overwrite the same physical tail without advancing either logical position. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| Absolute positions and RoPE | `current_start` is an absolute token offset. Temporal RoPE starts at `current_start // frame_seqlen`; cached K is already rotated at that absolute frame and is not rebased after compaction. Spatial RoPE uses the current latent grid. | SOURCE-PROVEN, GPU-VALIDATED |
| Cache consumer | The current query consumes the occupied contiguous K/V slice, bounded by `max_attention_size`. The campaign uses the full configured capacity. | SOURCE-PROVEN, GPU-VALIDATED |
| Action/camera alignment | Chunk `i` uses latent/noise, image condition, and Plücker condition chunk `i`. In the live path bootstrap is chunk 0 and user action index `i-1` is chunk `i`. | SOURCE-PROVEN; batch pose hashes RUNTIME-OBSERVABLE |
| Latent/frame alignment | The interactive path uses one latent frame per chunk/action. Wan's temporal VAE stride maps later latent frames to four RGB frames; bootstrap has the initial-frame special case. | SOURCE-PROVEN |
| Denoise state | The released four-step sequence is 999, 957, 899, 702. Each step overwrites the current logical cache tail with the current noisy latent's K/V while x0 is updated; only the final clean write persists. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| Deferred clean transaction | The accepted x0 is forwarded once at exact `t=0`, its output is discarded, and its K/V overwrites the current tail before the next action is permitted. | SOURCE-PROVEN, GPU-VALIDATED |
| Cross attention | Text K/V is populated on the first DiT call of a generation/session and reused thereafter. The per-pipe Python gate mirrors the per-layer `is_init` state. | SOURCE-PROVEN, RUNTIME-OBSERVABLE |
| Fresh-rollout reset | New self-KV and cross-KV objects/cursors, cross-attention gate false, newly seeded noise generator, chunk index/current_start, condition/control sequence, and decoder feature cache are required. Scheduler timesteps are reset. T5 prompt embeddings may intentionally persist because they are keyed by prompt hash and are not recurrent world state. | SOURCE-PROVEN, GPU-VALIDATED |
| Remaining unknown | No correctness-relevant unknown remains for the tested single-device causal-fast, chunk-size-1 path. Sequence-parallel and causal-pretrain caches were not in campaign scope. | UNKNOWN (out of scope) |

## Source locations

- `.upstream/lingbot-world-v2/wan/modules/model_fast.py:22-50`: absolute-frame 3-D RoPE.
- `.upstream/lingbot-world-v2/wan/modules/model_fast.py:114-196`: append,
  overwrite, sink preservation, compaction, selection, and cursors.
- `.upstream/lingbot-world-v2/wan/modules/model_fast.py:223-236`: cross-KV first-call behavior.
- `.upstream/lingbot-world-v2/wan/modules/model_fast.py:580-656`: tokenization,
  camera-token alignment, and per-layer cache dispatch.
- `.upstream/lingbot-world-v2/wan/image2video.py:520-530`: per-generation cross gate and seeded noise reset.
- `.upstream/lingbot-world-v2/wan/image2video.py:645-764`: capacity,
  per-chunk `current_start`, denoise loop, and exact clean update.
- `.upstream/lingbot-world-v2/wan/image2video.py:1062-1090`: fresh self/cross cache allocation.
- `scripts/run_interactive.py:604-738`: chunk-size-1 fixture preparation.
- `scripts/run_interactive.py:741-880`: denoise and deferred exact clean-KV transaction.
- `scripts/run_live.py:639-716,785-816`: bootstrap/action association and the clean barrier.
- `.upstream/lingbot-world-v2/wan/modules/vae2_1.py:515-565,580-587`: VAE cache clearing and temporal iteration.
- `patches/0004-opt-in-host-kv-cursor.patch`: accepted optional host-cursor mirror path.

The upstream checkout is pinned at `45fa40673607c9acba6cf96a1f9396c95bcef25f`;
the four repository patches were applied by `scripts/prepare_upstream.sh`.

## Fixtures and checks

The deterministic fixture uses one token per frame, total capacity 4, sink 1,
positions 0 through 8, and three writes per position (two discriminating noisy
writes and one clean write).  It runs both the device-tensor cursor and optional
host-cursor variants.  The reference keeps a plain logical-token list and
reconstructs a contiguous cache.  RoPE and attention are independently
calculated.  Exact cache tolerance is 0; FP32 consumer output tolerance is
`1e-6`.  Both variants passed every write and boundary exactly.

Real-model validation used 9 chunk-size-1 latent actions, seed 42, four denoise
steps, local capacity 4 frames, sink 1 frame, 384x672 output geometry
(`max_area_pixels=264192`), 1008 tokens/frame, and BF16 model/cache state.  The
independent layer-zero mirror had exact K/V agreement and exact consumer-output
agreement at all selected clean boundaries (allowed BF16 output tolerance
`0.02`).  All layer cursors agreed.

| Boundary | Global/local tokens after clean | Logical frame membership | Result |
|---|---:|---|---|
| Startup | 1008 / 1008 | `[0]` | PASS |
| Partial fill | 2016 / 2016 | `[0,1]` | PASS |
| Before first eviction | 4032 / 4032 | `[0,1,2,3]` | PASS |
| First eviction / first rolled | 5040 / 4032 | `[0,2,3,4]`; frame 1 removed, sink 0 unchanged | PASS |
| Repeated roll | 6048 / 4032 | `[0,3,4,5]` | PASS |
| Later rolled | 9072 / 4032 | `[0,6,7,8]` | PASS |

At every boundary the final denoise call and clean `t=0` call reported identical
cursors, proving that clean KV overwrote the current association instead of
advancing it.

## Reset and negative controls

After chunk 8, `prepare_session` allocated fresh recurrent state on the same
loaded pipe.  Replaying bootstrap reproduced x0 SHA-256
`5e66f83a768b74454ed1ae3b0504dcca1163bdc1c946148bd0876aa1452ef640`
and layer-zero occupied clean-K SHA-256
`33ddfef8f2046e4e86ad27b0ae8ee686d4cddf4e915212eda743786925f3ae6e`
exactly, with cursors reset to 1008/1008 after its first clean write.

The controlled reuse of the later rolled self/cross cache at `current_start=0`
was detected immediately by production slice arithmetic (zero-length target for
1008 incoming tokens).  Separately, reversing non-sink keys while retaining
their original V order changed the minimal-reference consumer output by
`7.6324234`, so the validator distinguishes correct state from an invalid
key/value association.

## Evidence files

- `fixtures.json`: both minimal cursor variants and broken controls.
- `gfx1151-validation.json`: real-model boundary records, hashes, reference
  comparisons, reset replay, and stale negative control.
- `summary.json`: machine-readable verdict and required result fields.
- `environment.json`: repository, input, patch, and machine provenance.
- `commands.md`: exact commands.

Phase 0 environment/performance evidence is referenced at
`docs/artifacts/phase0-20260912/`; it is not duplicated here.

**C2 quality-causality testing is unblocked.**
