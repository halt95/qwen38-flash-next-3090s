# v2 qualification card — 2026-09-17 (G6 run 6, two arms, five boots each)

> **Provenance.** This card is the release gate of Flash-Next v2, not a throughput sweep: the same frozen
> ladder was run on **v2** (tag `v2` = fork `d8ea70ae0e`, the entry `scripts/serve-v2.sh` transcribes) and on
> the **v1 TP4 lane** (the 2026-09-08 card's entry at its 2026-09-10 KV pin 2.9e9), alternating boots
> A1,B1,…,A5,B5 on the same box under one frozen manifest (`FREEZE v11`, sha256 `e7b89822e5db…`), with the
> judge, ladder, prompts and reference files pinned by hash. Per boot: three sends per cell, streamed,
> client-timed, T=0 with `seed`, thinking on ("decode") and off ("decode_off"). Rows are judged as
> **median over the five v2 boots vs median over the five v1 boots** (rule: ≥ 0.97 and every boot ≥ 0.90).
> Full judge output: `records/flashnext-v2-rc7-g6-run6-close-2026-09-17.md`. Instrument: `fnbench.py` ladder
> + `g6_judge.py` (maintainer scripts, not published). The judge's machine verdict for the run is FAIL, driven by the one
> thinking-off 4K cell below; every other row passes.

Host: 4× NVIDIA GeForce RTX 3090, driver 610.43.02, 220 W | v2: vLLM fork `d8ea70ae0e` on torch 2.13.0+cu130 |
v1: vLLM 0.28.0 fn28 tree, TP4.

## Summary (the same table as the model card and the README; medians over the five boots of each arm)

| | **v2** | v1 TP4 build (same gate, same box) |
|---|---|---|
| shape | TP2 × PP2 + expert parallel, MTP K=3 | TP4 + EP, MTP K=3 |
| KV pool, FP8 tokens, whole box | **806,792** (3.08 × a full-context request; three 262K sessions resident, measured) | 342,912 (1.31 ×) |
| context per request | 262,144 | 262,144 |
| concurrent sequences admitted | **8** (3 × 262K, 1 × 262K + 3 × 131K, 5 × 131K, 8 × 65K, 8 × 32K all resident) | 2 |
| decode, single stream, thinking on, 4K / 32K / 131K / 261K prompt | **162 / 166 / 169 / 176** tok/s (medians of 5 boots) | 163 / 165 / 172 / — (261K preregistered for the v2 arm only; v1 serves it, see its 2026-09-08 card) |
| decode, thinking off, same depths | 118 / 123 / 123 / 125 | 124 / 122 / 120 / — |
| median event interval, thinking on, same depths | 18.72 / 18.73 / 19.10 / 19.33 ms | 18.61 / 18.73 / 19.17 / — ms |
| MTP tokens per step, thinking on, same depths | 3.02 / 3.07 / 3.13 / 3.28 | 3.00 / 3.04 / 3.17 / — |
| prefill, 10K / 100K / 261K prompt | **4,944 / 5,282 / 5,122** tok/s | 4,498 / 4,194 / — |
| time to first token, cold, 4K / 32K / 131K / 261K | 0.91 / 6.19 / 24.77 / 50.68 s | 0.97 / 7.16 / 31.94 / — s |
| quality vs the BF16 teacher (24 held-out prompts, 5 boots; lower is closer) | 0.0365–0.0378 | 0.0378–0.0385 |
| tool-call structure (150 cases) / exact recall (160 cases) | 130/150 / 160/160 | reference quant 125/150 / 160/160 (parity band, ≥ ref − 2) |
| vision | on, 2 images per request | on |

The per-boot values behind every cell follow.

## Decode, thinking on (`decode`: needle prompt at depth, ≤ 256 tokens, 3 sends per boot)

| depth | v1 median tok/s | v2 boots tok/s | v2 median | ratio | floor (min/v1) | event interval ms v1 / v2 | tokens per step v1 / v2 |
|---|---|---|---|---|---|---|---|
| 4K | 162.6 | 162.2 162.5 160.6 162.6 161.8 | 162.2 | **0.997** | 0.988 | 18.61 / 18.72 | 3.00 / 3.02 |
| 32K | 164.9 | 166.6 166.8 165.8 165.6 166.3 | 166.3 | **1.009** | 1.004 | 18.73 / 18.73 | 3.04 / 3.07 |
| 131K | 171.5 | 173.9 169.4 165.7 166.5 172.9 | 169.4 | **0.988** | 0.966 | 19.17 / 19.10 | 3.17 / 3.13 |
| 261K (v2 arm only, per the preregistration; not run on v1 in this gate) | not run | 175.0 178.6 171.1 176.2 182.4 | 176.2 | no v1 cell | no v1 cell | not run / 19.33 | not run / 3.28 |

## Decode, thinking off (`decode_off`, same prompts)

| depth | v1 median tok/s | v2 boots tok/s | v2 median | ratio | floor | event interval ms v1 / v2 | tokens per step v1 / v2 |
|---|---|---|---|---|---|---|---|
| 4K | 124.1 | 122.7 118.2 119.3 115.4 114.4 | 118.2 | **0.952 (FAIL vs the 0.97 rule)** | 0.922 | 18.64 / 18.66 | 2.31 / 2.21 |
| 32K | 122.0 | 123.0 124.8 122.0 120.6 123.5 | 123.0 | 1.008 | 0.988 | 18.77 / 18.81 | 2.28 / 2.31 |
| 131K | 120.4 | 123.3 123.0 123.1 124.8 127.3 | 123.3 | 1.024 | 1.022 | 19.17 / 18.97 | 2.29 / 2.35 |
| 261K (v2 arm only, per the preregistration) | not run | 125.9 122.7 125.2 126.3 125.0 | 125.2 | no v1 cell | no v1 cell | not run / 19.15 | not run / 2.37 |

The 261K depth was preregistered for the v2 arm only (4.1 posted-decode rule against 173 tok/s; 4.3 prefill rule), so the v1
columns at that depth are empty by design, not because v1 cannot serve it: the v1 lane holds one 262,144-token request
(pool 342,912) and its own 2026-09-08 card measured 163 tok/s at a 260K prompt under a different protocol (image-bearing
prompts, served defaults), which is not comparable to this ladder. The 261K event interval and tokens per step (added 09-17 evening from the same boots' streams: thinking on 19.14–19.38 ms and 3.24–3.40 per boot; thinking off 19.13–19.21 ms and 2.33–2.41) follow the 4K–131K trend.

The one failing cell is a documented shortfall (README, "Known behaviours"): the event interval is identical,
the whole gap is tokens per step (v2 boots 2.31 / 2.21 / 2.23 / 2.15 / 2.14 vs v1 2.31), consistent with a
draft-acceptance difference on short no-think greedy answers, clustered per boot; the same shape appeared on
the previous candidate, so it is not the v2 environment. Cause not established.

## Prefill, TTFT, pool, quality, tools

| row | v1 | v2 (per boot) | rule / verdict |
|---|---|---|---|
| prefill 10K prompt, tok/s | 4498 | 4944 4950 4940 4949 4944 | v2 ≥ 0.90 × v1 every boot: PASS (+10 %) |
| prefill 100K prompt, tok/s | 4194 | 5281 5285 5282 5281 5285 | PASS (+26 %) |
| prefill 261K prompt, tok/s (v2 only, median of 3 salted sends) | — | 5124 5122 5124 5119 5120 | ≥ 0.90 × 4,900: PASS |
| TTFT at 4K, s | 0.967 0.965 0.966 0.974 0.976 | 0.911 0.910 0.911 0.912 0.910 | reported |
| TTFT by depth, cold salted prompts, s (medians of 5 boots; per-boot spread ≤ 0.05 s): 4K / 32K / 131K / 261K | 0.97 / 7.16 / 31.94 / — | 0.91 / 6.19 / 24.77 / 50.68 | reported (the 261K prompt is 259,587 tokens; that depth was preregistered for the v2 arm only) |
| KV pool, tokens | 342,912 ×5 | 806,792 ×5 | ≥ 804,247: PASS |
| quality, v-vs-teacher divergence (24 held-out prompts, 4 captures per boot; lower = closer to the BF16 teacher) | a = 0.0378–0.0385 | b = 0.0365–0.0378 | delta −0.0009, U −0.0004 < 0.0015: PASS |
| tool-call structure, 150 cases (V.8 battery) | reference quant 125/150 | 130/150 | ≥ ref − 2 per family: PASS (parity band) |
| exact recall, 160 cases (V.8 battery) | 160/160 | 160/160 | PASS |
| three 262K sessions resident (P3F) | — | interval 8.41 s, peak 302 / 317 blocks, 0 faults | PASS |
| no-think single-stream (r9 protocol) | — | decode ratio 1.015, interval ratio 1.000 | PASS |
| faults, replacements, empty warm completions over 10 boots | 0 | 0 | PASS |

Fingerprint of the v2 arm (each boot; `${PORT}` per arm):

```
vllm serve Qwen3.8-Flash-Next-W4A16-Merlin(-e1 config) --served-model-name v7final --port 8170 --host 127.0.0.1
  --tensor-parallel-size 2 --pipeline-parallel-size 2 --enable-expert-parallel
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.96
  --kv-cache-memory 4100000000 --kv-cache-dtype fp8_e4m3 --limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-cache-gb 0
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,24,32]}'
  --enable-prefix-caching --enable-prompt-tokens-details --enable-log-requests --enable-request-id-headers
  --no-enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --shutdown-timeout 60
env: PYTHONPATH=<v2 tree> CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
     VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
     VLLM_PLE_OFFLOAD_HOME_DEVICE=0 VLLM_PP_LAYER_PARTITION=25,23 VLLM_HOST_EMBED_TABLE=1 VLLM_QSA_KV_SCALES=<sidecar> HF_HUB_OFFLINE=1
```

The v1 arm ran the 2026-09-08 card's command at `--kv-cache-memory 2900000000` with the same sidecar. Both arms ran with
`--enable-log-requests --enable-request-id-headers` (the gate's request logging); the shipped entry has them off.
