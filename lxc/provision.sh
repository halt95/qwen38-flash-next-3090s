#!/usr/bin/env bash
# Provision a fresh Debian 13 LXC (from lxc/pve-create.sh) into a Flash-Next v2.0.1 server. Run inside the container as root.
# Idempotent: re-running skips what is done. Needs network (apt, GitHub, wheels.vllm.ai, the PyTorch cu130 index).
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
REPO_REF="${REPO_REF:-v2.0.1}"
BUNDLE_URL="${BUNDLE_URL:-https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.0.1/v2.0.1-from-upstream-e962733e08.bundle.gz}"
APP=/opt/flash-next
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends git curl ca-certificates tar gzip python3.13 python3.13-venv python3-pip \
  kmod pciutils procps
# NVIDIA userspace matching the host module
if ! command -v nvidia-smi >/dev/null 2>&1; then
  : "${NVIDIA_RUN:?nvidia-smi not present: set NVIDIA_RUN=/path/to/NVIDIA-Linux-x86_64-<host version>.run}"
  [ -f "$NVIDIA_RUN" ] || { echo "NVIDIA_RUN=$NVIDIA_RUN not found"; exit 1; }
  sh "$NVIDIA_RUN" --silent --no-kernel-module --no-drm --no-x-check --no-nouveau-check --no-backup
fi
nvidia-smi -L || { echo "nvidia-smi cannot see the cards: check the dev0..dev6 lines of the CT config and the host module"; exit 1; }
# the repository at the release ref, the v2.0.1 bundle, the pinned build
if [ ! -d "$APP/.git" ]; then git clone -q --branch "$REPO_REF" --depth 1 "$REPO_URL" "$APP"; fi
cd "$APP"
# a re-run on an existing clone must be at the same ref, not silently whatever was cloned before
have="$(git describe --tags --exact-match 2>/dev/null || git rev-parse --abbrev-ref HEAD)"
[ "$have" = "$REPO_REF" ] || { echo "$APP is checked out at $have, not $REPO_REF; remove it or set REPO_REF=$have"; exit 1; }
BUNDLE_FILE="$(basename "$BUNDLE_URL" .gz)"
[ -f "$BUNDLE_FILE" ] || curl -fsSL "$BUNDLE_URL" | gunzip > "$BUNDLE_FILE"
BUNDLE="$APP/$BUNDLE_FILE" scripts/build-v2.sh /opt/vllm-v2 /opt/venv-v2
# the checkpoint: one config key, added in place if missing (weights untouched, backup kept)
CKPT="${CKPT:-/models/Qwen3.8-Flash-Next-W4A16-Merlin}"
[ -f "$CKPT/config.json" ] || { echo "checkpoint not found at $CKPT (mount it into the container, see pve-create.sh MODELS=)"; exit 1; }
python3.13 scripts/make-e1-config.py "$CKPT"
# systemd unit: serve-v2.sh with the production environment; edit /etc/default/flash-next to change host/port/names
install -m 0644 lxc/flash-next.service /etc/systemd/system/flash-next.service
[ -f /etc/default/flash-next ] || cat > /etc/default/flash-next <<EOF
CKPT=$CKPT
HOST=0.0.0.0
PORT=8000
CACHE_ROOT=/cache
# MODEL_NAME="flash-next"
# PLE_HOME=0
# CUDA_VISIBLE_DEVICES=0,1,2,3
EOF
systemctl daemon-reload
systemctl enable --now flash-next.service
echo "provisioned. First start compiles the cudagraphs (~10 min): journalctl -fu flash-next; then curl -s localhost:8000/v1/models"
