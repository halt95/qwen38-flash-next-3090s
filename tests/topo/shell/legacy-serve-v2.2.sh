#!/usr/bin/env bash
# Serve Flash-Next v2.2.0: Qwen3.8-Flash-Next-W4A16-Merlin at 262K per request with an 806,792-token KV pool on
# 4x RTX 3090 -- TP2 x PP2 + expert parallel, MTP K=3, host-mapped PLE pull transport, host-resident embeddings,
# prefix caching, the greedy-repeatability switches on. It follows the reference host's serving entry (the v2.2.0 tree
# since 2026-09-24) with every variable exported; two defaults differ from it: the logging-only counters are off
# (COUNTERS=1) and CUDA_HOME is the venv's CUDA 13.0 toolkit (see below).
#
#   TREE=./vllm-v2.2 VENV=./venv-v2.2 scripts/serve-v2.2.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin [extra vllm args]
#
# The checkpoint needs one config key (`ple_embedding_dtype: float8_e4m3fn` inside text_config). The Hugging Face
# checkpoint carries it since 2026-09-18; for an older download scripts/make-e1-config.py adds it (weights untouched).
#
# Env knobs: PORT (8000), HOST (127.0.0.1), MODEL_NAME (names requests use in "model"; default = the four public names
# below), PLE_HOME (0: the card that hosts the PLE offload home; stage-0 card on the reference host), SCALES (KV scale
# sidecar, defaults to scales/qsa_kv_scales_262k.json, the same bytes v2.0.1 serves), CACHE_ROOT (torch compile /
# cudagraph cache dir; use a NEW one for v2.2 -- 0.28-era artefacts do not carry over; first boot compiles, ~6 min),
# GUARD (bounds-guard telemetry mode; default warn, as qualified -- the code default is off), COUNTERS=1 to add the two
# logging-only counters (exact-restart counters every 30 s, FP8 KV clip counter every 300 s; see below), CUDA_HOME (the
# toolkit FlashInfer and Triton compile with on the first serve; default: the venv's CUDA 13.0 wheels).
#
# Changed from serve-v2.sh (v2.0.1):
#   VLLM_PLE_CPU_OFFLOAD=1 -> VLLM_E2_PLE_PULL_TRANSPORT=1. vLLM 0.30 reuses the old name for its own UVA PLE backend
#     with the opposite default; this tree refuses to boot with both on, so the old name is explicitly unset here.
#   VLLM_USE_BREAKABLE_CUDAGRAPH=0 (exactly "0"): the tree's profile guard accepts only the qualified shape (compiled,
#     FULL_AND_PIECEWISE, breakable graphs pinned off) and refuses to boot otherwise.
#   the four determinism switches + in-kernel sorted emission, and guard telemetry in warn mode (README, "What changed").
#   The switches, the sorted-emit knob and the clip counter are compile factors: changing one costs a fresh compile.
#   a first-serve toolchain check: FlashInfer compiles its prefill/sampling/top-k kernels on the first request and
#     Triton compiles its launchers, so a C compiler, ninja, the Python headers and an nvcc no newer than the driver
#     must be present; the build products are re-hashed on every start.
#   warnings (not failures) when host RAM or /dev/shm is below the qualified sizes: a short host does not fail fast.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:?usage: serve-v2.2.sh <checkpoint-dir> [extra vllm args]}"; shift || true
TREE="${TREE:-$HERE/vllm-v2.2}"; VENV="${VENV:-$HERE/venv-v2.2}"
case "$TREE" in /*) ;; *) TREE="$PWD/$TREE" ;; esac
case "$VENV" in /*) ;; *) VENV="$PWD/$VENV" ;; esac
[ -f "$TREE/vllm/__init__.py" ] || { echo "TREE=$TREE is not the built v2.2 tree (scripts/build-v2.2.sh)"; exit 1; }
[ -f "$TREE/.v2.2-build-products" ] && [ -s "$TREE/.v2.2-build-products.sha256" ] \
  || { echo "TREE=$TREE has no verified v2.2 build products (scripts/build-v2.2.sh)"; exit 1; }
(cd "$TREE" && sha256sum -c --quiet --strict .v2.2-build-products.sha256) \
  || { echo "TREE=$TREE: build products changed since scripts/build-v2.2.sh verified them"; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "VENV=$VENV has no bin/python (scripts/build-v2.2.sh creates it)"; exit 1; }
# the key must sit INSIDE text_config (the engine reads its text config; a top-level copy does nothing), so parse, do not grep
"$VENV/bin/python" -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if c.get("text_config",{}).get("ple_embedding_dtype")=="float8_e4m3fn" else 1)' "$CKPT/config.json" \
  || { echo "config.json lacks text_config.ple_embedding_dtype = float8_e4m3fn: run scripts/make-e1-config.py $CKPT"; exit 1; }

# A `vllm/` directory in the LAUNCH directory silently wins over $TREE: vLLM inspects the model registry in a child
# process started with `python -m ...`, which puts the current directory first on sys.path, ahead of PYTHONPATH.
if [ -d "$PWD/vllm" ] && [ "$(cd "$PWD/vllm" 2>/dev/null && pwd -P)" != "$(cd "$TREE/vllm" && pwd -P)" ]; then
  echo "refusing to start: $PWD/vllm would shadow $TREE/vllm."
  echo "  The registry-inspection child puts the current directory before PYTHONPATH on sys.path, so that copy"
  echo "  would be imported instead of the v2.2 tree. Start the server from a directory without a vllm/ beside it,"
  echo "  or pass TREE=<that checkout> if it really is the built v2.2 tree."
  exit 1
fi

# First-serve toolchain. FlashInfer resolves nvcc from CUDA_HOME (then PATH, then /usr/local/cuda) and links
# -lcudart from <CUDA_HOME>/lib64; Triton compiles its launchers with a C compiler against Python.h. PTX from an nvcc
# newer than the driver is rejected ("Unsupported .version"), so the nvcc must not be newer than the driver's CUDA
# version. The reference host compiles these with its system CUDA 13.3 toolkit (CUDA_HOME=/usr/local/cuda); the default
# here is the venv's CUDA 13.0 wheels, which any CUDA 13 driver accepts. CUDA_HOME=<a toolkit no newer than the
# driver> overrides it.
if [ -z "${CUDA_HOME:-}" ]; then
  CUDA_HOME="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
for t in cc c++ ninja; do command -v "$t" >/dev/null || { echo "first serve needs $t (Debian/Ubuntu: apt install gcc g++ ninja-build)"; exit 1; }; done
"$VENV/bin/python" -c 'import os,sys,sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"],"Python.h")) else 1)' \
  || { echo "first serve needs the Python 3.13 headers: Triton compiles a C module against Python.h (Debian 13, or Ubuntu with the deadsnakes PPA: apt install python3.13-dev; uv and pyenv Pythons ship Python.h; the venv must come from that interpreter)"; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "no nvcc at CUDA_HOME=$CUDA_HOME (scripts/build-v2.2.sh installs the CUDA 13.0 wheels)"; exit 1; }
[ -e "$CUDA_HOME/lib64/libcudart.so" ] || { echo "$CUDA_HOME/lib64/libcudart.so missing (scripts/build-v2.2.sh creates the lib64 links)"; exit 1; }
nv_rel="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' || true)"
[ -n "$nv_rel" ] || { echo "could not read a CUDA release from \`$CUDA_HOME/bin/nvcc --version\` (see its error above)"; exit 1; }
# (|| true: under pipefail a missing or failing nvidia-smi must skip this check, not end the script silently)
drv_rel="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\2/p' | head -1 || true)"
[ -n "$drv_rel" ] || echo "note: could not read the driver's CUDA version from nvidia-smi; the nvcc-versus-driver check is skipped"
if [ -n "$drv_rel" ] && [ -n "$nv_rel" ] && [ "$(printf '%s\n%s\n' "$nv_rel" "$drv_rel" | sort -V | tail -1)" != "$drv_rel" ]; then
  echo "nvcc at $CUDA_HOME is CUDA $nv_rel but the driver supports CUDA $drv_rel: kernels compiled on the first serve"
  echo "  would be rejected. Upgrade the driver or set CUDA_HOME to a toolkit no newer than $drv_rel."; exit 1
fi

export PYTHONPATH="$TREE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# consumer cards on a P2P-enabled driver: skip vLLM's P2P probe, tell NCCL the topology; LL protocol for the small
# pipeline relay / all-gather messages (README, "Hardware": the release is qualified with peer-to-peer working).
# Without P2P these settings are harmless: vLLM turns its custom all-reduce off and NCCL goes through host memory.
export VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL
export VLLM_PP_LAYER_PARTITION=25,23
# the qualified profile sets every fork knob explicitly: drop anything inherited first (test-only fault injectors,
# diagnostics, the calibration collector, the breakable-graph opt-out, a correctness switch turned off by hand)
for v in $(env | sed -n 's/^\(MERLIN_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset VLLM_PLE_CPU_OFFLOAD VLLM_PLE_FP8_GLOBAL_SCALE VLLM_PLE_TEST_FAULT VLLM_PLE_TEST_FAULT_GATE VLLM_PLE_TEST_FAULT_LOG \
      VLLM_QSA_KV_COLLECT VLLM_QSA_KV_CLIP_COUNT VLLM_E2_ALLOW_BREAKABLE VLLM_E2_EXPECTED_CUDAGRAPH_MODE \
      VLLM_E2_MAMBA_SLOT_OWNERSHIP VLLM_E2_MEMSTATS VLLM_E2_MEMSTATS_DIR VLLM_E2_PLE_FLAG_CHECK VLLM_E2_SCHED_LOG \
      VLLM_E2_ZERO_OWNERSHIP_CHECK VLLM_E2_GUARD_INTERVAL VLLM_PLE_OFFLOAD_WAIT_TIMEOUT VLLM_PLE_PROGRESS_DEADLINE_S \
      VLLM_PLE_STARTUP_DEADLINE_S PYTORCH_CUDA_ALLOC_CONF || true
export VLLM_E2_PLE_PULL_TRANSPORT=1 VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
export VLLM_PLE_OFFLOAD_HOME_DEVICE="${PLE_HOME:-0}"
export VLLM_HOST_EMBED_TABLE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_QSA_KV_SCALES="${SCALES:-$HERE/scales/qsa_kv_scales_262k.json}"
export VLLM_CACHE_ROOT="${CACHE_ROOT:-$HERE/.vllm-cache-v2.2}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# greedy repeatability within one compile cache (README, "What changed in v2.2.0") + in-kernel sorted emission
export MERLIN_FULL_K=1 MERLIN_QSA_SORT=1 MERLIN_TIE_DET=1 MERLIN_TIE_RECENT=1 MERLIN_TOPK_SORTED_EMIT=1
export VLLM_E2_GUARD_MODE="${GUARD:-warn}"
# Logging-only counters, OFF by default. The reference host ran them (the published numbers were measured with them on),
# but the FP8 KV clip counter's reader starts on the first eager forward and registers its buffers from its own thread,
# which can overlap cudagraph capture on a fresh-compile boot. COUNTERS=1 opts in.
if [ "${COUNTERS:-0}" = 1 ]; then export MERLIN_TOPK_EXACT_COUNTERS_LOG=30 VLLM_QSA_KV_CLIP_COUNT=300; fi
# the reference host also exports LD_LIBRARY_PATH=<its CUDA 13.3 toolkit lib dir>; it is inert (every CUDA library the
# processes map comes from the venv wheels), so it is not set here.
[ -f "$VLLM_QSA_KV_SCALES" ] || { echo "scales sidecar missing: $VLLM_QSA_KV_SCALES (without it FP8 KV runs at scale 1.0)"; exit 1; }

# Host memory. The ~48 GiB n-gram table (ordinary memory of the offload process) and ~4.2 GiB of pinned embedding
# tables live in host RAM, so a short host does not fail fast: it OOMs well into the boot with a kernel message that
# says nothing about why. 96 GB is the qualified allocation; the measured resident floor is ~69 GiB and the boot peak
# is not measured. A container sees the host's MemTotal, so a finite cgroup memory limit counts when it is smaller.
mem_kb="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || true)"
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  lim="$(cat "$f" 2>/dev/null || true)"
  case "$lim" in ''|*[!0-9]*) continue ;; esac   # absent, or "max" (no limit)
  lim_kb=$((lim / 1024))
  if [ -z "${mem_kb:-}" ] || [ "$lim_kb" -lt "$mem_kb" ]; then mem_kb="$lim_kb"; fi
done
if [ -n "${mem_kb:-}" ] && [ "$mem_kb" -lt 100663296 ]; then
  echo "WARNING: host RAM available here is $((mem_kb / 1048576)) GiB; the qualified configuration is 96 GB"
  echo "         (measured resident floor ~69 GiB: ~48 GiB n-gram table + ~4.2 GiB pinned embeddings + workers)."
  if [ "$mem_kb" -lt 72351744 ]; then echo "         Below the ~69 GiB resident floor the model load is expected to OOM."; fi
fi
shm_kb="$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $2}' || true)"
case "$shm_kb" in *[!0-9]*) shm_kb= ;; esac   # not a plain size (no /dev/shm, or an unexpected df layout): skip
if [ -n "${shm_kb:-}" ] && [ "$shm_kb" -lt 1048576 ]; then
  echo "WARNING: /dev/shm is $((shm_kb / 1024)) MB; the PLE transport needs >= 1 GB (docker run --shm-size=8g, compose shm_size: 8g)"
fi

read -r -a NAMES <<< "${MODEL_NAME:-flash-next-v2 flash-next flash-mtp flash-next-mtp}"
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
