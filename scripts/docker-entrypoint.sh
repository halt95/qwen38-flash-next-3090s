#!/usr/bin/env bash
# Container entrypoint for Flash-Next v2.0.1: checks the mounted checkpoint, adds the one config key if it is
# missing (and the mount is writable), warns about the two host settings people forget, then execs
# scripts/serve-v2.sh with every variable the production entry uses. Extra arguments go to `vllm serve`.
#
#   docker run ... qwen38-flash-next-3090s:v2.0.1 [/path/inside/container/to/checkpoint] [extra vllm args]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:-/models/Qwen3.8-Flash-Next-W4A16-Merlin}"; shift || true
if [ ! -f "$CKPT/config.json" ]; then
  echo "checkpoint not found at $CKPT"
  echo "mount it: -v /path/to/Qwen3.8-Flash-Next-W4A16-Merlin:/models/Qwen3.8-Flash-Next-W4A16-Merlin"
  exit 1
fi
if ! python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if c.get("text_config",{}).get("ple_embedding_dtype")=="float8_e4m3fn" else 1)' "$CKPT/config.json"; then
  if [ -w "$CKPT" ] && [ -w "$CKPT/config.json" ]; then
    echo "config.json lacks text_config.ple_embedding_dtype; adding it (weights untouched, backup kept)"
    python3 "$HERE/scripts/make-e1-config.py" "$CKPT"
  else
    echo "config.json lacks text_config.ple_embedding_dtype and the mount is read-only."
    echo "Run once on the host copy: python3 scripts/make-e1-config.py /path/to/Qwen3.8-Flash-Next-W4A16-Merlin"
    exit 1
  fi
fi
shm_kb="$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $2}')"
if [ -n "${shm_kb:-}" ] && [ "$shm_kb" -lt 1048576 ]; then
  echo "WARNING: /dev/shm is $((shm_kb / 1024)) MB; the PLE transport needs >= 1 GB (docker run --shm-size=8g, compose shm_size: 8g)"
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "WARNING: no NVIDIA GPU visible in the container (docker run --gpus all; nvidia-container-toolkit on the host)"
fi

# Host RAM: the n-gram table and the embedding tables are pinned in host memory, not VRAM, so a short host does
# not fail fast -- it OOMs well into the boot with a kernel message that says nothing about why. 96 GB is the
# qualified allocation; the measured resident floor is ~69 GiB and the boot peak is not measured.
mem_kb="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)"
if [ -n "${mem_kb:-}" ] && [ "$mem_kb" -lt 100663296 ]; then
  echo "WARNING: host RAM visible here is $((mem_kb / 1048576)) GB; the qualified configuration is 96 GB"
  echo "         (measured resident floor ~69 GiB: ~48 GiB n-gram table + ~4.2 GiB pinned embeddings + workers)."
  [ "$mem_kb" -lt 72351744 ] && echo "         Below the ~69 GiB resident floor the model load is expected to OOM."
fi
exec "$HERE/scripts/serve-v2.sh" "$CKPT" "$@"
