#!/usr/bin/env bash
# Build the Flash-Next v2.0.1 tree from the pinned base + the delta bundle, exactly as the clean-room
# reproduction that qualified the release (commit AND tree hash verified against upstream/PIN-v2).
#
#   scripts/build-v2.sh [SRC_DIR] [VENV_DIR]
#
# Inputs (next to this repo or given by env):
#   BUNDLE     path to v2.0.1-from-upstream-e962733e08.bundle (release asset; sha256 in PIN-v2)
#   ARTIFACTS  path to build-artifacts-sm86-py313-cu130.tar.gz (optional; sha256 in PIN-v2). If absent,
#              the compiled ops come from the upstream precompiled wheel named in PIN-v2
#              (set WHEEL=/path/to/that.whl, or the script curls it from wheels.vllm.ai by the pinned URL).
#
# 1. git init, fetch the three public prerequisite commits from GitHub at depth 1, fetch the bundle,
#    check out the tag named in PIN-v2 (v2.0.1), assert commit == PIN commit and tree == PIN tree (fails closed).
# 2. Fresh venv on python 3.13 from release/v2/requirements-pinned.txt (torch 2.13.0+cu130 from the
#    PyTorch cu130 index). Never reuse another venv.
# 3. Compiled ops: extract the artefact tarball into the tree (hash-checked), or unpack the .so files
#    from the precompiled wheel. Then a metadata-only editable install (VLLM_TARGET_DEVICE=empty) so the
#    `vllm` console script and importlib metadata exist without compiling anything.
# Reference: clean-room run v10-v2-20260917-1422 (records/flashnext-v2-rc7-qualification-close-2026-09-17.md).
set -euo pipefail
for t in git curl tar sha256sum python3.13; do command -v "$t" >/dev/null || { echo "missing tool: $t"; exit 1; }; done
# Debian and Ubuntu ship the venv machinery in a SEPARATE package, so `python3.13` can be present and working while
# `python3.13 -m venv` still fails with "ensurepip is not available". Probe the real operation now: without this the
# build dies ~100 lines later, after it has already fetched from GitHub and verified the bundle. Observed on a stock
# Debian 13 host with python 3.13.5 installed.
_vprobe="$(mktemp -d)"; trap 'rm -rf "$_vprobe"' EXIT
python3.13 -m venv "$_vprobe/probe" >/dev/null 2>&1 || {
  echo "python3.13 is installed but cannot create a virtual environment."
  echo "  On Debian/Ubuntu install the separate venv package:  sudo apt install python3.13-venv"
  echo "  (build-v2.sh needs a fresh venv; it never reuses an existing one.)"
  rm -rf "$_vprobe"; exit 1; }
rm -rf "$_vprobe"; trap - EXIT
HERE="$(cd "$(dirname "$0")/.." && pwd)"
sha_ok(){ [ "$(sha256sum "$2" | cut -d" " -f1)" = "$1" ]; }
SRC="${1:-$PWD/vllm-v2}"; VENV="${2:-$PWD/venv-v2}"
# shellcheck disable=SC1091
. "$HERE/upstream/PIN-v2"   # repo base_commit prereqs bundle bundle_sha256 tag commit tree ...
BUNDLE="${BUNDLE:-$HERE/$bundle}"; [ -f "$BUNDLE" ] || BUNDLE="$HERE/release/v2/$bundle"
[ -f "$BUNDLE" ] || { echo "bundle not found: set BUNDLE=/path/to/$bundle (release asset)"; exit 1; }
sha_ok "$bundle_sha256" "$BUNDLE" || { echo "bundle sha256 mismatch"; exit 1; }

if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$SRC"; git -C "$SRC" init -q; git -C "$SRC" remote add upstream "$repo"
  for c in $prereqs; do git -C "$SRC" fetch -q --depth 1 upstream "$c"; done
  git -C "$SRC" fetch -q "$BUNDLE" "refs/tags/$tag:refs/tags/$tag"
  git -C "$SRC" -c advice.detachedHead=false checkout -q "$tag"
fi
got="$(git -C "$SRC" rev-parse HEAD)"; got_tree="$(git -C "$SRC" rev-parse 'HEAD^{tree}')"
[ "$got" = "$commit" ] || { echo "checkout is $got, expected $commit"; exit 1; }
[ "$got_tree" = "$tree" ] || { echo "tree $got_tree != $tree"; exit 1; }
# build products are untracked files, so the tracked-file check holds on a re-run too
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "$SRC has modified tracked files"; exit 1; }
echo "source: tag $tag = $commit, tree $tree verified"

# never reuse a venv this script did not create (README: "Never reuse an existing venv for this tree")
if [ -e "$VENV" ] && [ ! -f "$VENV/.v2-venv" ]; then echo "$VENV exists and was not created by build-v2.sh; pick a fresh VENV path"; exit 1; fi
[ -x "$VENV/bin/python" ] || { python3.13 -m venv "$VENV" && echo "created by build-v2.sh for tag $tag" > "$VENV/.v2-venv"; }
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cu130 -r "$HERE/release/v2/requirements-pinned.txt"
"$VENV/bin/pip" install -q "setuptools-rust>=1.9" "setuptools-scm>=8"

if [ -f "$SRC/.v2-build-products" ]; then
  echo "build products: already present ($(cat "$SRC/.v2-build-products"))"
elif [ -n "${ARTIFACTS:-}" ] && [ -f "$ARTIFACTS" ]; then
  sha_ok "$artifacts_sha256" "$ARTIFACTS" || { echo "artefact tarball sha256 mismatch"; exit 1; }
  tar -xzf "$ARTIFACTS" -C "$SRC"; echo "tarball $artifacts_sha256" > "$SRC/.v2-build-products"; echo "build products: extracted from $ARTIFACTS"
else
  WHEEL="${WHEEL:-}"
  if [ -z "$WHEEL" ]; then
    mkdir -p "$SRC/.wheel"; WHEEL="$SRC/.wheel/vllm-${precompiled_wheel_version}-cp38-abi3-manylinux_2_28_x86_64.whl"
    [ -f "$WHEEL" ] || curl -fsSL -o "$WHEEL" "$precompiled_wheel_url" || { echo "could not fetch $precompiled_wheel_url; download the wheel and set WHEEL="; exit 1; }
  fi
  sha_ok "$precompiled_wheel_sha256" "$WHEEL" || { echo "precompiled wheel sha256 mismatch"; exit 1; }
  "$VENV/bin/python" - "$WHEEL" "$SRC" "$HERE/release/v2/build-artifacts.list" <<'EOF'
import sys, zipfile, os
whl, src, lst = sys.argv[1:4]
want = [l.strip().rstrip('/') for l in open(lst) if l.strip()]
z = zipfile.ZipFile(whl); n = 0
for m in z.namelist():
    if any(m == w or m.startswith(w + '/') for w in want) and not m.endswith('/'):
        z.extract(m, src); n += 1
print(f"build products: {n} files from {os.path.basename(whl)}")
EOF
  echo "wheel $precompiled_wheel_sha256" > "$SRC/.v2-build-products"
fi
(cd "$SRC" && VLLM_TARGET_DEVICE=empty "$VENV/bin/pip" install -q -e . --no-deps --no-build-isolation)
PYTHONPATH="$SRC" "$VENV/bin/python" -c "import vllm, sys; print('vllm', vllm.__version__, vllm.__file__)"
echo "done: TREE=$SRC VENV=$VENV scripts/serve-v2.sh /path/to/checkpoint"
