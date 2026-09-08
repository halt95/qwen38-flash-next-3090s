# Production reference bench — 2026-09-08

> **Provenance.** Same protocol and instrument as `benchmarks/2026-09-05/`, re-run after the KV-scale
> sidecar shipped in this repository (Merlin-calibrated, margin 1.10) was promoted on the `flash-next-mtp`
> entry; this is the entry `scripts/serve.sh` transcribes. The no-MTP fallback entry still serves the
> previous sidecar and has no `VLLM_QSA_KV_CLIP_COUNT`, so its `clip_warnings: 0` means "not instrumented".
> The "inter-chunk ms" column is the mean gap between streamed content events, which may carry several
> tokens; it is not a per-token latency.

Host: NVIDIA GeForce RTX 3090, 610.43.02, 220.00 W | 0.28.1.dev9+g414c3c643 2.13.0+cu130

Protocol: as served through the front door — no per-request sampling or thinking overrides (entry defaults apply: thinking on at low effort, server-side sampling). Single rows: image-bearing nonce prompt filled to the target depth, 256 max tokens, 1 warmup + 3 repeats, streamed, client-timed. decode = (tokens−1)/(last−first token) [decode-window kind]; wall = tokens/e2e. Aggregate: 1K prompt, 512 tokens, ignore_eos, distinct prompts, rate over the window in which every stream is decoding. Instrument: fnbench.py / fnconc.py (port of alesha-pro tests/bench.py metric definitions).

## flash-next-mtp

Fingerprint (live `/running` cmd):

```
vllm serve Qwen3.8-Flash-Next-W4A16-Merlin --served-model-name flash-next-mtp flash-next flash-mtp --port 8102 --host 127.0.0.1 --tensor-parallel-size 4 --enable-expert-parallel --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' --max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.96 --kv-cache-memory 2600000000 --kv-cache-dtype fp8_e4m3 --limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-cache-gb 0 --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,4,8]}' --enable-prefix-caching --no-enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}'
```

Live process env (`/proc/<pid>/environ`):

```
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_HOME=/usr/local/cuda
CUDA_VISIBLE_DEVICES=0,1,2,3
NCCL_P2P_LEVEL=SYS
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
VLLM_PLE_CPU_OFFLOAD=1
VLLM_PLE_FP8_GLOBAL_SCALE=1
VLLM_PLE_OFFLOAD_HOME_DEVICE=3
VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
VLLM_QSA_KV_CLIP_COUNT=300
VLLM_QSA_KV_SCALES=<scales sidecar>
VLLM_SKIP_P2P_CHECK=1
```

concurrency_limit=2

| prompt tokens | out tokens (finish) | TTFT s | prefill t/s | decode-window t/s | wall t/s | inter-chunk ms (mean gap between streamed content events, which may carry several tokens; not per-token) |
|---|---|---|---|---|---|---|
| 3967 | 118–161 (stop) | 0.9 (CV 0.8%, n=3) | 4238.6 (CV 0.8%, n=3) | **165.9 (CV 2.6%, n=3)** | 79.6 (CV 7.2%, n=3) | 19.1 (CV 0.3%, n=3) |
| 32632 | 105–199 (stop) | 7.0 (CV 0.2%, n=3) | 4636.9 (CV 0.2%, n=3) | **173.9 (CV 9.8%, n=3)** | 18.0 (CV 23.8%, n=3) | 19.3 (CV 0.2%, n=3) |
| 130933 | 131–256 (length,stop) | 32.0 (CV 0.2%, n=3) | 4093.4 (CV 0.2%, n=3) | **171.7 (CV 2.9%, n=3)** | 5.9 (CV 25.3%, n=3) | 19.6 (CV 0.6%, n=3) |
| 260566 | 199–256 (length,stop) | 75.7 (CV 0.1%, n=3) | 3440.6 (CV 0.1%, n=3) | **162.5 (CV 5.5%, n=3)** | 3.0 (CV 10.4%, n=3) | 20.0 (CV 0.5%, n=3) |

| N | ok/err | aggregate all-decoding t/s | per-stream decode median | TTFT median s |
|---|---|---|---|---|
| 1 | 3/0 | **122.7 (CV 2.6%, n=3)** | 122.4 (CV 2.6%, n=3) | 0.3 (CV 0.4%, n=3) |
| 2 | 6/0 | **232.5 (CV 2.9%, n=3)** | 113.7 (CV 2.1%, n=3) | 0.5 (CV 0.9%, n=3) |

Integrity panel:

- errors/stderr lines: 0
- clip_warnings: 2
- acceptance: INFO 09-08 14:16:14 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.43, Accepted throughput: 119.10 tokens/s, Drafted throughput: 249.89
- gpu after: 0, 23809 MiB, 195.97 W, 52, 1965 MHz / 1, 23809 MiB, 179.41 W, 42, 1980 MHz / 2, 23809 MiB, 192.27 W, 48, 1980 MHz / 3, 23855 MiB, 206.21 W, 47, 1950 MHz

## flash-next-w4a16-fp8kv

Fingerprint (live `/running` cmd):

```
vllm serve Qwen3.8-Flash-Next-W4A16-vnim-plefp8 --served-model-name flash-next-w4a16-fp8kv flash-next flash-base --port 8103 --host 127.0.0.1 --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.96 --kv-cache-memory 2086941635 --kv-cache-dtype fp8_e4m3 --limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-cache-gb 0 --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4]}' --enable-prefix-caching --no-enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}'
```

Live process env (`/proc/<pid>/environ`):

```
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_HOME=/usr/local/cuda
CUDA_VISIBLE_DEVICES=0,1,2,3
NCCL_P2P_LEVEL=SYS
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
VLLM_PLE_CPU_OFFLOAD=1
VLLM_PLE_FP8_GLOBAL_SCALE=1
VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
VLLM_QSA_KV_SCALES=<scales sidecar>
VLLM_SKIP_P2P_CHECK=1
```

concurrency_limit=4

| prompt tokens | out tokens (finish) | TTFT s | prefill t/s | decode-window t/s | wall t/s | inter-chunk ms (mean gap between streamed content events, which may carry several tokens; not per-token) |
|---|---|---|---|---|---|---|
| 3967 | 105–256 (length,stop) | 0.8 (CV 0.9%, n=3) | 5009.3 (CV 0.9%, n=3) | **78.1 (CV 1.0%, n=3)** | 58.1 (CV 9.9%, n=3) | 13.0 (CV 0.8%, n=3) |
| 32632 | 122–242 (stop) | 6.5 (CV 0.1%, n=3) | 5036.3 (CV 0.1%, n=3) | **77.5 (CV 0.9%, n=3)** | 20.6 (CV 20.0%, n=3) | 13.1 (CV 0.6%, n=3) |
| 130933 | 120–128 (stop) | 30.5 (CV 0.3%, n=3) | 4296.2 (CV 0.3%, n=3) | **77.1 (CV 3.4%, n=3)** | 3.9 (CV 2.8%, n=3) | 13.5 (CV 3.4%, n=3) |
| 260566 | 160–195 (stop) | 72.7 (CV 0.1%, n=3) | 3584.4 (CV 0.1%, n=3) | **77.7 (CV 0.1%, n=3)** | 2.3 (CV 8.8%, n=3) | 13.5 (CV 0.2%, n=3) |

| N | ok/err | aggregate all-decoding t/s | per-stream decode median | TTFT median s |
|---|---|---|---|---|
| 1 | 3/0 | **74.7 (CV 1.2%, n=3)** | 74.6 (CV 1.2%, n=3) | 0.2 (CV 0.0%, n=3) |
| 4 | 12/0 | **256.1 (CV 1.4%, n=3)** | 64.1 (CV 1.4%, n=3) | 0.8 (CV 0.3%, n=3) |

Integrity panel:

- errors/stderr lines: 0
- clip_warnings: 0
- acceptance: 0
- gpu after: 0, 23801 MiB, 203.71 W, 53, 1965 MHz / 1, 23755 MiB, 187.25 W, 42, 1965 MHz / 2, 23755 MiB, 206.47 W, 49, 1980 MHz / 3, 23755 MiB, 219.15 W, 48, 1950 MHz

