#!/usr/bin/env bash
# Serve Flash-Next v2.0.1: Qwen3.8-Flash-Next-W4A16-Merlin at 262K per request with an 806,792-token KV pool
# on 4x RTX 3090 -- TP2 x PP2 + expert parallel, MTP K=3, host-mapped PLE pull transport, host-resident
# embeddings, prefix caching. This is the production entry of the reference host (the v2 shape since 2026-09-17,
# the v2.0.1 tree since 2026-09-18), transcribed from its llama-swap config; every variable exported.
#
#   TREE=./vllm-v2 VENV=./venv-v2 scripts/serve-v2.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin [extra vllm args]
#
# The checkpoint needs one config key (`ple_embedding_dtype: float8_e4m3fn`) that the published HF
# checkpoint does not carry: scripts/make-e1-config.py adds it (weights untouched; see README, "Checkpoint").
#
# Env knobs: PORT (8000), HOST (127.0.0.1), MODEL_NAME (names requests use in "model"; default = the six names the
# production entry serves), PLE_HOME (0: the
# card that hosts the PLE offload home; stage-0 card on the reference host), SCALES (KV scale sidecar,
# defaults to scales/qsa_kv_scales_262k.json = the Merlin margin-1.10 file the v1 release shipped; v2 serves
# the same bytes), CACHE_ROOT (torch compile / cudagraph cache dir; first boot compiles, ~10 min),
# GUARD=warn to turn the bounds-guard telemetry reader on (off by default since v2; README, "Known behaviours").
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:?usage: serve-v2.sh <checkpoint-dir> [extra vllm args]}"; shift || true
TREE="${TREE:-$HERE/vllm-v2}"; VENV="${VENV:-$HERE/venv-v2}"
[ -f "$TREE/vllm/__init__.py" ] || { echo "TREE=$TREE is not the built v2 tree (scripts/build-v2.sh)"; exit 1; }
# the key must sit INSIDE text_config (the engine reads its text config; a top-level copy does nothing), so parse, do not grep
"$VENV/bin/python" -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if c.get("text_config",{}).get("ple_embedding_dtype")=="float8_e4m3fn" else 1)' "$CKPT/config.json" \
  || { echo "config.json lacks text_config.ple_embedding_dtype = float8_e4m3fn: run scripts/make-e1-config.py $CKPT"; exit 1; }

# A `vllm/` directory in the LAUNCH directory silently wins over $TREE. vLLM inspects the model registry in a child
# process started with `python -m ...`, and that puts the current directory first on sys.path -- ahead of PYTHONPATH.
# Anyone who already has a vLLM checkout and starts the server from beside it therefore serves that copy, and the
# failure surfaces far from its cause (typically `AttributeError: '_ModelInfo' object has no attribute ...`).
# Reproduced byte-for-byte on the reference host; fail closed instead.
if [ -d "$PWD/vllm" ] && [ "$(cd "$PWD/vllm" 2>/dev/null && pwd -P)" != "$(cd "$TREE/vllm" && pwd -P)" ]; then
  echo "refusing to start: $PWD/vllm would shadow $TREE/vllm."
  echo "  The registry-inspection child puts the current directory before PYTHONPATH on sys.path, so that copy"
  echo "  would be imported instead of the v2 tree. Start the server from a directory without a vllm/ beside it"
  echo "  (e.g. cd elsewhere), or pass TREE=<that checkout> if it really is the built v2 tree."
  exit 1
fi

export PYTHONPATH="$TREE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
[ -n "${CUDA_HOME:-}" ] && export CUDA_HOME   # not needed to serve (CUDA runtime comes from the venv wheels); exported only if you set it
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# consumer cards on a P2P-enabled driver: skip vLLM's P2P probe, tell NCCL the topology; LL protocol for
# the small pipeline relay / all-gather messages (measured: decode per-step parity with the TP4 lane, prefill unchanged)
export VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL
export VLLM_PP_LAYER_PARTITION=25,23
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
export VLLM_PLE_OFFLOAD_HOME_DEVICE="${PLE_HOME:-0}"
export VLLM_HOST_EMBED_TABLE=1
export VLLM_QSA_KV_SCALES="${SCALES:-$HERE/scales/qsa_kv_scales_262k.json}"
export VLLM_CACHE_ROOT="${CACHE_ROOT:-$HERE/.vllm-cache-v2}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# the reference host also exports LD_LIBRARY_PATH=/usr/local/cuda-13.3/...; it is inert (every CUDA library the
# processes map comes from the venv wheels, /proc/<pid>/maps checked), so it is not set here.
# instrumentation is OFF in the served profile (two 50 ms reader threads cost ~0.3-0.5 ms per decode step and
# once deadlocked a worker, README, "Benchmarks"); opt in with GUARD=warn. Never inherit the calibration collector.
[ -n "${GUARD:-}" ] && export VLLM_E2_GUARD_MODE="$GUARD"
unset PYTORCH_CUDA_ALLOC_CONF VLLM_QSA_KV_COLLECT VLLM_QSA_KV_CLIP_COUNT VLLM_PLE_TEST_FAULT || true
[ -f "$VLLM_QSA_KV_SCALES" ] || { echo "scales sidecar missing: $VLLM_QSA_KV_SCALES (without it FP8 KV runs at scale 1.0)"; exit 1; }

read -r -a NAMES <<< "${MODEL_NAME:-flash-next-v2 flash-next flash-mtp flash-next-mtp flash-next-e2 flash-next-v2-sub}"
exec "$VENV/bin/vllm" serve "$CKPT" \
  --served-model-name "${NAMES[@]}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 --enable-expert-parallel \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.96 --kv-cache-memory 4100000000 --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-cache-gb 0 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,24,32]}' \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}' \
  --shutdown-timeout 60 \
  "$@"
