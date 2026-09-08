#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next-W4A16-Merlin at 262K on 4x RTX 3090: the served entry
# (MTP K=3, calibrated FP8 KV, FULL graphs). The 2026-09-05 bench card was measured
# on this entry with the previous scale sidecar; the sidecar this script defaults to
# (scales/qsa_kv_scales_262k.json) was calibrated on this checkpoint on 2026-09-08 and
# gated separately at this shape (README, "The scale sidecar").
#
#   VENV=./venv scripts/serve.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin [extra vllm args]
#
# Env knobs: PORT (8000); MODEL_NAME ("flash-next-mtp flash-next flash-mtp": the names requests
# use in the "model" field); PLE_HOME (3 = index of the card that hosts the PLE
# offload worker's context, pick the least-loaded one); SCALES (defaults to
# scales/qsa_kv_scales_262k.json); CLIP_COUNT (300 s report interval);
# NO_MTP=1 for a no-speculation entry (4 seqs, 4096-token prefill batch, 2.09 GB
# KV pin, as the fallback entry on the reference host). The bench card's no-MTP
# numbers were measured on a different checkpoint (the VnimanieAI lineage), so
# do not expect them from this switch with the Merlin checkpoint.
#
# This preserves the principal served settings of the reference host's entry
# (2026-09-07), including the explicit Qwen thinking-mode sampling override the
# bench card was measured with; port, host and model aliases are configurable and
# the sidecar is the newly calibrated one. HOST defaults to 127.0.0.1 as there;
# set HOST=0.0.0.0 to expose it on a LAN.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:?usage: serve.sh <checkpoint-dir> [extra vllm args]}"; shift || true
VENV="${VENV:-$HERE/venv}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# the reference host runs a P2P-enabled driver on consumer cards; vLLM's own P2P
# probe is skipped and NCCL is told the topology explicitly. Keep both.
export VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_PLE_FP8_GLOBAL_SCALE=1 VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
export VLLM_PLE_OFFLOAD_HOME_DEVICE="${PLE_HOME:-3}"
export VLLM_QSA_KV_SCALES="${SCALES:-$HERE/scales/qsa_kv_scales_262k.json}"
export VLLM_QSA_KV_CLIP_COUNT="${CLIP_COUNT:-300}"
# never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here: it cost ~27 % of four-stream aggregate throughput;
# and never inherit the calibration collector (eager-only) into a graph-capturing server
unset PYTORCH_CUDA_ALLOC_CONF VLLM_QSA_KV_COLLECT || true
[ -f "$VLLM_QSA_KV_SCALES" ] || { echo "scales sidecar missing: $VLLM_QSA_KV_SCALES (without it FP8 KV runs at scale 1.0)"; exit 1; }

if [ "${NO_MTP:-0}" = 1 ]; then
  SPEC=()
  GRAPH='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4]}'
  SEQS=4; MNBT=4096; KVPIN=2086941635
else
  SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}')
  # captures must cover max-num-seqs x (K+1) = 2 x 4 = 8
  GRAPH='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,4,8]}'
  # 2048 was selected at the 2.6 GB KV pin; 4096 passed a gate only with the pin lowered to 2.35 GB and no margin left at the deep-prefill peak
  SEQS=2; MNBT=2048; KVPIN=2600000000
fi

# the model is addressed by these names in API requests (default matches the reference host's entry)
read -r -a NAMES <<< "${MODEL_NAME:-flash-next-mtp flash-next flash-mtp}"
exec "$VENV/bin/vllm" serve "$CKPT" \
  --served-model-name "${NAMES[@]}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 262144 --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" \
  --gpu-memory-utilization 0.96 --kv-cache-memory "$KVPIN" --kv-cache-dtype fp8_e4m3 \
  "${SPEC[@]}" \
  --compilation-config "$GRAPH" \
  --enable-prefix-caching --no-enable-flashinfer-autotune \
  --limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-cache-gb 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}' \
  "$@"
