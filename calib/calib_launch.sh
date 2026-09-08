#!/usr/bin/env bash
# Serve for K/V absmax calibration (patch 0002): the served entry's environment,
# plus VLLM_QSA_KV_COLLECT, with --enforce-eager (the collector mutates state per
# step; graphs/compile would freeze or fold it), no speculative config (the
# draft head's QSA layer must not land in the dump), no prefix caching (every
# request must run its full prefill through the collector).
#
#   mkdir -p /some/dir
#   VENV=./venv COLLECT=/some/dir calib/calib_launch.sh /path/to/checkpoint
#   ./venv/bin/python calib/qsa_calib_traffic.py 8140 flash-next-calib   # in another shell; the venv python has Pillow
#   ./venv/bin/python calib/qsa_calib_merge.py /some/dir scales/my_scales.json --margin 1.10
#
# This is the reference host's calibration launch with paths made relative. The
# shipped sidecar was collected this way on Qwen3.8-Flash-Next-W4A16-Merlin on
# 2026-09-08 (dumps in calib/dumps-2026-09-08-merlin/). Run the traffic script
# under the vLLM venv's python: it refuses to start without Pillow.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:?usage: calib_launch.sh <checkpoint-dir>}"
VENV="${VENV:-$HERE/venv}"
COLLECT="${COLLECT:?set COLLECT=<existing dump dir>}"
[ -d "$COLLECT" ] || { echo "COLLECT dir must exist: $COLLECT"; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_FP8_GLOBAL_SCALE=1 VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
export VLLM_PLE_OFFLOAD_HOME_DEVICE="${PLE_HOME:-3}"
export VLLM_QSA_KV_COLLECT="$COLLECT"
unset VLLM_QSA_KV_SCALES VLLM_QSA_KV_CLIP_COUNT PYTORCH_CUDA_ALLOC_CONF || true

exec "$VENV/bin/vllm" serve "$CKPT" \
  --served-model-name flash-next-calib \
  --host 127.0.0.1 --port "${PORT:-8140}" \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.96 --kv-cache-memory 2086941635 --kv-cache-dtype fp8_e4m3 \
  --limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-cache-gb 0 \
  --enforce-eager \
  --no-enable-prefix-caching --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}'
