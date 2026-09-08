# Production reference bench — 2026-09-05

> **Provenance.** Both entries were served with the *previous* KV-scale sidecar (calibrated on the
> VnimanieAI lineage). The sidecar shipped in this repository was calibrated on the Merlin checkpoint on
> 2026-09-08 and has served the `flash-next-mtp` entry since; see `benchmarks/2026-09-08/` for the same
> protocol on the current entry. The no-MTP entry's recorded environment has no `VLLM_QSA_KV_CLIP_COUNT`,
> so its `clip_warnings: 0` below means "not instrumented", not "no clipping". The "inter-chunk ms"
> column is the mean gap between streamed content events, which may carry several tokens; it is not a
> per-token latency.

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
| 3967 | 133–167 (stop) | 0.9 (CV 0.4%, n=3) | 4249.9 (CV 0.4%, n=3) | **162.9 (CV 1.9%, n=3)** | 81.9 (CV 5.1%, n=3) | 19.2 (CV 0.2%, n=3) |
| 32632 | 120–146 (stop) | 7.0 (CV 0.1%, n=3) | 4654.3 (CV 0.1%, n=3) | **162.8 (CV 4.4%, n=3)** | 17.0 (CV 7.5%, n=3) | 19.3 (CV 0.1%, n=3) |
| 130933 | 143–180 (stop) | 31.9 (CV 0.3%, n=3) | 4100.9 (CV 0.3%, n=3) | **169.2 (CV 6.1%, n=3)** | 5.1 (CV 9.9%, n=3) | 19.6 (CV 0.1%, n=3) |
| 260566 | 163–256 (stop) | 75.6 (CV 0.1%, n=3) | 3447.3 (CV 0.1%, n=3) | **174.4 (CV 4.1%, n=3)** | 2.9 (CV 18.6%, n=3) | 20.0 (CV 0.0%, n=3) |

| N | ok/err | aggregate all-decoding t/s | per-stream decode median | TTFT median s |
|---|---|---|---|---|
| 1 | 3/0 | **125.9 (CV 4.9%, n=3)** | 125.6 (CV 4.9%, n=3) | 0.3 (CV 1.7%, n=3) |
| 2 | 6/0 | **227.5 (CV 1.2%, n=3)** | 111.3 (CV 3.0%, n=3) | 0.5 (CV 3.8%, n=3) |

Integrity panel:

- errors/stderr lines: 0
- clip_warnings: 38
- acceptance: INFO 09-05 22:58:28 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 2.37, Accepted throughput: 110.99 tokens/s, Drafted throughput: 242.98
- gpu after: 0, 23753 MiB, 219.28 W, 53, 1965 MHz / 1, 23753 MiB, 219.12 W, 45, 1965 MHz / 2, 23753 MiB, 219.53 W, 50, 1980 MHz / 3, 23799 MiB, 219.40 W, 48, 1950 MHz

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
| 3967 | 137–256 (length,stop) | 0.8 (CV 0.6%, n=3) | 4985.9 (CV 0.6%, n=3) | **78.7 (CV 0.3%, n=3)** | 58.6 (CV 6.2%, n=3) | 12.9 (CV 0.2%, n=3) |
| 32632 | 139–158 (stop) | 6.5 (CV 0.1%, n=3) | 5047.0 (CV 0.1%, n=3) | **77.4 (CV 0.4%, n=3)** | 17.6 (CV 4.1%, n=3) | 13.2 (CV 0.3%, n=3) |
| 130933 | 104–132 (stop) | 30.4 (CV 0.2%, n=3) | 4303.4 (CV 0.2%, n=3) | **77.8 (CV 3.1%, n=3)** | 3.7 (CV 9.5%, n=3) | 13.4 (CV 3.5%, n=3) |
| 260566 | 169–256 (length,stop) | 72.9 (CV 0.6%, n=3) | 3574.4 (CV 0.6%, n=3) | **76.5 (CV 2.3%, n=3)** | 2.8 (CV 16.5%, n=3) | 13.6 (CV 2.4%, n=3) |

| N | ok/err | aggregate all-decoding t/s | per-stream decode median | TTFT median s |
|---|---|---|---|---|
| 1 | 3/0 | **76.5 (CV 1.8%, n=3)** | 76.3 (CV 1.8%, n=3) | 0.2 (CV 3.2%, n=3) |
| 4 | 12/0 | **255.5 (CV 0.8%, n=3)** | 63.9 (CV 0.8%, n=3) | 0.8 (CV 1.3%, n=3) |

Integrity panel:

- errors/stderr lines: 0
- clip_warnings: 0
- acceptance: 0
- gpu after: 0, 23801 MiB, 219.32 W, 53, 1965 MHz / 1, 23755 MiB, 210.05 W, 43, 1965 MHz / 2, 23755 MiB, 217.14 W, 49, 1980 MHz / 3, 23755 MiB, 204.28 W, 47, 1950 MHz

