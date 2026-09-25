#!/usr/bin/env bash
# Container entrypoint for Flash-Next v2.2.0: checks the mounted checkpoint, adds the one config key if it is
# missing (and the mount is writable), warns when no GPU is visible, then execs scripts/serve-v2.2.sh (which also
# warns about host RAM and /dev/shm). Extra arguments go to `vllm serve`.
#
#   docker run ... qwen38-flash-next-3090s:v2.2.0 [/path/inside/container/to/checkpoint] [extra vllm args]
#
# A first argument that starts with `-` is a vllm argument: the default checkpoint path is used.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CKPT=/models/Qwen3.8-Flash-Next-W4A16-Merlin
if [ $# -gt 0 ]; then
  case "$1" in -*) ;; "") shift ;; *) CKPT="$1"; shift ;; esac
fi
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
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "WARNING: no NVIDIA GPU visible in the container (docker run --gpus all; nvidia-container-toolkit on the host,"
  echo "         registered with: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker)"
fi
exec "$HERE/scripts/serve-v2.2.sh" "$CKPT" "$@"
