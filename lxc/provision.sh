#!/usr/bin/env bash
# Provision a fresh Debian 13 LXC (from lxc/pve-create.sh) into a Flash-Next v2.2.0 server. Run inside the container as root.
# Idempotent: re-running skips what is done. Needs network (apt, GitHub, PyPI).
#
#   NVIDIA_RUN=/models/NVIDIA-Linux-x86_64-<host driver version>.run bash /root/provision.sh
#
# The one input the recipe cannot ship is the NVIDIA userspace: it must match the kernel module the host runs
# (`nvidia-smi` on the host prints the version). Put the matching NVIDIA-Linux-x86_64-<version>.run where the
# container can read it (e.g. the /models mount) and pass NVIDIA_RUN=; it is installed with --no-kernel-module.
# On the reference host the driver is 610.43.02 with the consumer-Ampere peer-to-peer patch; a stock driver of the
# same version works for the userspace side, P2P itself is a host property.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/halt95/qwen38-flash-next-3090s.git}"
REPO_REF="${REPO_REF:-v2.2.0}"
BUNDLE_URL="${BUNDLE_URL:-https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.2.0/v2.2.0-from-upstream-v0.30.0.bundle.gz}"
ARTIFACTS_URL="${ARTIFACTS_URL:-https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.2.0/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz}"
APP=/opt/flash-next
export DEBIAN_FRONTEND=noninteractive
# right after `pct start` eth0 may not have a DHCP lease yet: wait up to 60 s for name resolution
for _ in $(seq 60); do getent hosts deb.debian.org >/dev/null 2>&1 && break; sleep 1; done
apt-get update -q
apt-get install -y -q --no-install-recommends git curl ca-certificates tar gzip python3.13 python3.13-venv python3.13-dev python3-pip \
  kmod pciutils procps gcc g++ build-essential ninja-build
# NVIDIA userspace matching the host module
if ! command -v nvidia-smi >/dev/null 2>&1; then
  : "${NVIDIA_RUN:?nvidia-smi not present: set NVIDIA_RUN=/path/to/NVIDIA-Linux-x86_64-<host version>.run}"
  [ -f "$NVIDIA_RUN" ] || { echo "NVIDIA_RUN=$NVIDIA_RUN not found"; exit 1; }
  sh "$NVIDIA_RUN" --silent --no-kernel-module --no-drm --no-x-check --no-nouveau-check --no-backup
fi
nvidia-smi -L || { echo "nvidia-smi cannot see the cards: check the dev0..dev6 lines of the CT config and the host module"; exit 1; }
# the host driver must support CUDA 13.0 or newer (the 580 series or later); checked here, before the long build
cuda_major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
if [ -n "$cuda_major" ] && [ "$cuda_major" -lt 13 ]; then
  echo "the host driver supports CUDA $cuda_major.x; v2.2.0 needs CUDA 13.0 or newer (driver 580 or later)"; exit 1
fi
# the checkpoint must be mounted before the build starts (it is not downloaded here)
CKPT="${CKPT:-/models/Qwen3.8-Flash-Next-W4A16-Merlin}"
[ -f "$CKPT/config.json" ] || { echo "checkpoint not found at $CKPT: download it into the host directory given as MODELS= to pve-create.sh (hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --local-dir <MODELS>/Qwen3.8-Flash-Next-W4A16-Merlin)"; exit 1; }
# the repository at the release ref, the v2.2.0 bundle + build-artifacts asset, the pinned build (plus the
# first-serve toolchain: gcc/ninja/python3.13-dev above, the CUDA 13.0 wheels and lib64 links from build-v2.2.sh)
if [ ! -d "$APP/.git" ]; then git clone -q --branch "$REPO_REF" --depth 1 "$REPO_URL" "$APP"; fi
cd "$APP"
# a re-run on an existing clone must be at the same ref, not silently whatever was cloned before
have="$(git describe --tags --exact-match 2>/dev/null || git rev-parse --abbrev-ref HEAD)"
[ "$have" = "$REPO_REF" ] || { echo "$APP is checked out at $have, not $REPO_REF; remove it or set REPO_REF=$have"; exit 1; }
# downloads go to .part and are renamed on success, so an interrupted one is fetched again on the next run
# (build-v2.2.sh hash-checks both files either way)
BUNDLE_FILE="$(basename "$BUNDLE_URL" .gz)"
[ -f "$BUNDLE_FILE" ] || { curl -fsSL "$BUNDLE_URL" | gunzip > "$BUNDLE_FILE.part" && mv "$BUNDLE_FILE.part" "$BUNDLE_FILE"; } \
  || { rm -f "$BUNDLE_FILE.part"; echo "could not fetch $BUNDLE_URL"; exit 1; }
ARTIFACTS_FILE="$(basename "$ARTIFACTS_URL")"
[ -f "$ARTIFACTS_FILE" ] || { curl -fsSL -o "$ARTIFACTS_FILE.part" "$ARTIFACTS_URL" && mv "$ARTIFACTS_FILE.part" "$ARTIFACTS_FILE"; } \
  || { rm -f "$ARTIFACTS_FILE.part"; echo "could not fetch $ARTIFACTS_URL"; exit 1; }
BUNDLE="$APP/$BUNDLE_FILE" ARTIFACTS="$APP/$ARTIFACTS_FILE" bash scripts/build-v2.2.sh /opt/vllm-v2.2 /opt/venv-v2.2
# the checkpoint: one config key, added in place if missing (weights untouched, backup kept)
python3.13 scripts/make-e1-config.py "$CKPT"
# systemd unit: serve-v2.2.sh with the production environment; edit /etc/default/flash-next to change host/port/names
install -m 0644 lxc/flash-next.service /etc/systemd/system/flash-next.service
[ -f /etc/default/flash-next ] || cat > /etc/default/flash-next <<EOF
CKPT=$CKPT
HOST=0.0.0.0
PORT=8000
CACHE_ROOT=/cache/v2.2
# HOST=0.0.0.0 serves every interface without an API key: set VLLM_API_KEY=<key> (clients send
# Authorization: Bearer <key>), or HOST=127.0.0.1 behind a proxy, if the container is reachable from other machines
# VLLM_API_KEY=
# MODEL_NAME="flash-next"
# PLE_HOME=0
# CUDA_VISIBLE_DEVICES=0,1,2,3
EOF
systemctl daemon-reload
systemctl enable --now flash-next.service
echo "provisioned. The first start compiles the cudagraphs (~6 min) and the first request FlashInfer's kernels: journalctl -fu flash-next"
echo "the log should show \"GPU KV cache size: 806,792 tokens\" and then \"Application startup complete\""
echo "ready check (a real one-token generation; /v1/models answers even while the engine is dead):"
echo "  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"flash-next\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
