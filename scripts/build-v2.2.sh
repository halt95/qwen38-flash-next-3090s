#!/usr/bin/env bash
# Build the Flash-Next v2.2.0 tree from stock vLLM v0.30.0 + the delta bundle (commit AND tree hash verified against
# upstream/PIN-v2.2), with a fresh python 3.13 venv, the compiled ops the release pins, and the toolchain that
# FlashInfer and Triton need to compile their kernels on the FIRST serve.
#
#   scripts/build-v2.2.sh [SRC_DIR] [VENV_DIR]          (relative paths are fine; they are made absolute first)
#
# Inputs (next to this repo or given by env):
#   BUNDLE     path to v2.2.0-from-upstream-v0.30.0.bundle (release asset, shipped gzipped; sha256 of the UNCOMPRESSED
#              bundle in PIN-v2.2). A .bundle.gz next to it is decompressed for you.
#   ARTIFACTS  path to build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz (release asset; sha256 in PIN-v2.2). This is the
#              default route: the stock 0.30.0 wheel's build products plus the two v2.2 extensions built from this
#              tree in an Ubuntu 22.04 root (not the reference host's serving builds; PIN-v2.2).
#   BUILD_OWN=1  the alternative when you do not want the tarball: the stock products come from the vLLM 0.30.0 wheel
#              (PyPI, hash-pinned) and the two extensions v2.2 changes (_C_stable_libtorch, _moe_C_stable_libtorch) are
#              compiled from this tree for sm_86 with cmake + ninja. Needs cmake >= 3.26, ninja, a C/C++ compiler and
#              nvcc >= 13.0 (default: the venv's CUDA 13.0 toolkit; NVCC=... to use another). Takes 15-30 min. Your
#              hashes will differ from PIN-v2.2's; the script prints and records them.
#
# v2.2 cannot reuse the stock wheel's _C / _moe_C: the delta changes csrc (top-k kernels, Marlin MoE split-K switch),
# so a tree without its own two extensions fails at the first top-k call. The script refuses that combination.
#
# 1. git init, fetch the public v0.30.0 commit at depth 1, fetch the bundle's branch, check out, assert commit == PIN
#    commit and tree == PIN tree (fails closed). A re-run completes a checkout this script started and a failure
#    interrupted; it never fetches into or checks out a git tree it did not create.
# 2. Fresh venv on python 3.13 from release/v2.2/requirements-pinned.txt (torch 2.13.0 from PyPI = the CUDA 13.0
#    build; the CUDA 13.0 nvcc/crt/nvvm/cccl wheels for runtime kernel compilation), plus pinned build tools.
#    Never reuses another venv.
# 3. Runtime-JIT layout inside the venv's CUDA wheel directory: lib64 -> lib and an unversioned link for each
#    versioned library (libcudart.so -> libcudart.so.13, libnvrtc.so -> libnvrtc.so.13, ...): FlashInfer links
#    -lcudart from <cuda>/lib64, and BUILD_OWN's cmake looks for the unversioned names. serve-v2.2.sh points
#    CUDA_HOME at this directory.
# 4. Compiled ops (tarball, or wheel + own build). Every product file is then hashed into .v2.2-build-products.sha256
#    (except vllm/_version.py, which step 5 regenerates; its version is asserted instead); a re-run and every
#    serve-v2.2.sh start re-check the whole set, never trusting the marker.
# 5. A metadata-only editable install (VLLM_TARGET_DEVICE=empty) so the `vllm` console script and importlib metadata
#    exist without compiling the rest.
set -euo pipefail
for t in git curl tar gzip sha256sum python3.13; do command -v "$t" >/dev/null || {
  echo "missing tool: $t"
  [ "$t" != python3.13 ] || echo "  Debian 12 and Ubuntu 22.04/24.04 do not package Python 3.13: use the deadsnakes PPA (Ubuntu), \`uv python install 3.13\` or pyenv (README, Requirements)"
  exit 1; }; done
# Debian and Ubuntu ship the venv machinery in a SEPARATE package: probe the real operation before fetching anything.
_vprobe="$(mktemp -d)"; trap 'rm -rf "$_vprobe"' EXIT
python3.13 -m venv "$_vprobe/probe" >/dev/null 2>&1 || {
  echo "python3.13 is installed but cannot create a virtual environment."
  echo "  Debian 13, or Ubuntu with the deadsnakes PPA, ship it separately:  sudo apt install python3.13-venv"
  echo "  (uv and pyenv builds include venv; with another Python 3.13, install its venv/ensurepip support)"
  echo "  (build-v2.2.sh needs a fresh venv; it never reuses an existing one.)"
  rm -rf "$_vprobe"; exit 1; }
rm -rf "$_vprobe"; trap - EXIT
HERE="$(cd "$(dirname "$0")/.." && pwd)"
sha_ok(){ [ "$(sha256sum "$2" | cut -d" " -f1)" = "$1" ]; }
abspath(){ case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
SRC="$(abspath "${1:-vllm-v2.2}")"; VENV="$(abspath "${2:-venv-v2.2}")"
# shellcheck disable=SC1091
. "$HERE/upstream/PIN-v2.2"   # repo base_commit prereqs bundle bundle_sha256 ref commit tree own_* stock_wheel_* artifacts_* build_tools
REL="$HERE/release/v2.2"
MARK="$SRC/.v2.2-build-products"

# the tarball route is the default; BUILD_OWN=1 is the explicit alternative. Decide before any network access.
ARTIFACTS="$(abspath "${ARTIFACTS:-$HERE/$artifacts_tar}")"
if [ "${BUILD_OWN:-0}" != 1 ] && [ ! -f "$MARK" ] && [ ! -f "$ARTIFACTS" ]; then
  echo "compiled ops: neither ARTIFACTS=$ARTIFACTS (release asset) nor BUILD_OWN=1 given."
  echo "  v2.2 needs its own _C_stable_libtorch and _moe_C_stable_libtorch; the stock 0.30.0 ones do not carry its kernels."
  exit 1
fi
# (only when there is something to build: a re-run over existing build products re-verifies them, it compiles nothing)
if [ "${BUILD_OWN:-0}" = 1 ] && [ ! -f "$MARK" ]; then
  for t in cmake ninja cc c++; do command -v "$t" >/dev/null || { echo "BUILD_OWN=1 needs $t"; exit 1; }; done
  # (|| true: under pipefail a cmake that fails to start must reach the message below, not end the script silently)
  cm="$(cmake --version | sed -n 's/^cmake version \([0-9][0-9.]*\).*/\1/p' || true)"
  [ -n "$cm" ] || { echo "BUILD_OWN=1: could not read a version from \`cmake --version\`"; exit 1; }
  [ "$(printf '%s\n3.26\n' "$cm" | sort -V | head -1)" = 3.26 ] || {
    echo "BUILD_OWN=1 needs cmake >= 3.26 (found $cm; Debian 12 ships 3.25): pip install 'cmake>=3.26,<4' and put it first on PATH"; exit 1; }
  [ "$(printf '%s\n4\n' "$cm" | sort -V | head -1)" = "$cm" ] \
    || echo "note: cmake $cm is untested for this build (it was built with 3.31; pip install 'cmake>=3.26,<4' if it fails)"
fi

BUNDLE="$(abspath "${BUNDLE:-$HERE/$bundle}")"; [ -f "$BUNDLE" ] || [ ! -f "$BUNDLE.gz" ] || gzip -dk "$BUNDLE.gz"
[ -f "$BUNDLE" ] || { echo "bundle not found: set BUNDLE=/path/to/$bundle (release asset $bundle.gz)"; exit 1; }
sha_ok "$bundle_sha256" "$BUNDLE" || { echo "bundle sha256 mismatch (the pin hashes the UNCOMPRESSED bundle)"; exit 1; }

# Resumable: the checkout this script creates carries a marker in .git, so a re-run after an interrupted fetch
# completes it. A git tree without the marker (not created here) is only verified, never fetched into or checked out.
INIT_MARK="$SRC/.git/v2.2-build-init"
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$SRC"; git -C "$SRC" init -q; : > "$INIT_MARK"
fi
if [ -f "$INIT_MARK" ]; then
  if ! git -C "$SRC" rev-parse -q --verify "$commit^{commit}" >/dev/null 2>&1; then
    git -C "$SRC" remote get-url upstream >/dev/null 2>&1 || git -C "$SRC" remote add upstream "$repo"
    for c in $prereqs; do git -C "$SRC" fetch -q --depth 1 upstream "$c"; done
    git -C "$SRC" fetch -q "$BUNDLE" "+$ref:refs/heads/v2.2.0"
  fi
  [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" = "$commit" ] \
    || git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$commit"
fi
got="$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)"
[ "$got" = "$commit" ] || { echo "$SRC: checkout is ${got:-empty (no commit checked out)}, expected $commit (a tree this script did not create is not modified; pick a fresh SRC_DIR)"; exit 1; }
got_tree="$(git -C "$SRC" rev-parse 'HEAD^{tree}')"
[ "$got_tree" = "$tree" ] || { echo "tree $got_tree != $tree"; exit 1; }
# build products are untracked files, so the tracked-file check holds on a re-run too
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "$SRC has modified tracked files"; exit 1; }
echo "source: $tag = $commit, tree $tree verified"

if [ -e "$VENV" ] && [ ! -f "$VENV/.v2.2-venv" ]; then echo "$VENV exists and was not created by build-v2.2.sh; pick a fresh VENV path"; exit 1; fi
[ -x "$VENV/bin/python" ] || { python3.13 -m venv "$VENV" && echo "created by build-v2.2.sh for $tag" > "$VENV/.v2.2-venv"; }
# shellcheck disable=SC2086
"$VENV/bin/python" -m pip install -q $build_tools_pip
"$VENV/bin/pip" install -q -r "$REL/requirements-pinned.txt"
# shellcheck disable=SC2086
"$VENV/bin/pip" install -q $build_tools
"$VENV/bin/python" -c "import torch; assert torch.__version__.split('+')[0] == '$torch' and torch.version.cuda == '$torch_cuda', (torch.__version__, torch.version.cuda)"

# runtime-JIT layout: the NVIDIA CUDA wheels install bin/, include/ and lib/ with versioned libraries only
# (libcudart.so.13, libnvrtc.so.13, ...); FlashInfer builds with -L<cuda>/lib64 -lcudart and cmake's FindCUDAToolkit
# (BUILD_OWN=1) looks for the unversioned names. All links are idempotent.
CU="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
[ -x "$CU/bin/nvcc" ] || { echo "the venv has no CUDA toolkit wheel at $CU (nvidia-cuda-nvcc from requirements-pinned.txt)"; exit 1; }
[ -e "$CU/lib64" ] || ln -s lib "$CU/lib64"
for so in "$CU"/lib/lib*.so.[0-9]*; do
  [ -e "$so" ] || continue
  base="${so%%.so.*}.so"; [ -e "$base" ] || ln -s "$(basename "$so")" "$base"
done
[ -e "$CU/lib/libcudart.so" ] || { echo "$CU/lib has no libcudart.so.13"; exit 1; }
nv_rel="$("$CU/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' || true)"
[ "$nv_rel" = "$jit_cuda" ] || { echo "venv nvcc is CUDA ${nv_rel:-unknown (\`$CU/bin/nvcc --version\` failed)}, expected $jit_cuda (requirements-pinned.txt)"; exit 1; }
echo "runtime JIT toolchain: nvcc $nv_rel at $CU (serve-v2.2.sh sets CUDA_HOME to it)"

check_own(){  # $1 = expected _C sha, $2 = expected _moe_C sha
  sha_ok "$1" "$SRC/vllm/_C_stable_libtorch.abi3.so" && sha_ok "$2" "$SRC/vllm/_moe_C_stable_libtorch.abi3.so"; }
check_list(){  # every path of build-artifacts.list is present
  local miss=0 p
  while read -r p; do [ -n "$p" ] || continue; [ -e "$SRC/${p%/}" ] || { echo "missing build product: $p"; miss=1; }; done < "$REL/build-artifacts.list"
  return $miss; }
# every file of every listed product, hashed once they are verified; re-runs and serve-v2.2.sh check the whole set.
# vllm/_version.py is left out: the editable install below regenerates it (the version is asserted separately).
SUMS="$SRC/.v2.2-build-products.sha256"
write_sums(){
  (cd "$SRC" && while read -r p; do [ -n "$p" ] || continue; find "${p%/}" -type f; done < "$REL/build-artifacts.list" \
    | grep -vx 'vllm/_version.py' | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$SUMS.tmp" && mv "$SUMS.tmp" "$SUMS"; }
check_sums(){ [ -s "$SUMS" ] && (cd "$SRC" && sha256sum -c --quiet --strict "$SUMS"); }

if [ -f "$MARK" ]; then
  # re-run: re-verify what the marker records instead of trusting it
  read -r route c_sha moe_sha < "$MARK"
  want_route=tarball; [ "${BUILD_OWN:-0}" != 1 ] || want_route=own
  [ "$route" = "$want_route" ] || echo "note: $SRC already holds the $route build products; they are re-verified and kept (delete $MARK to switch to the $want_route route)"
  if [ "$route" = tarball ]; then check_own "$own_C_sha256" "$own_moe_C_sha256" || { echo "$SRC: extensions no longer match PIN-v2.2; delete $MARK and rebuild"; exit 1; }
  else check_own "$c_sha" "$moe_sha" || { echo "$SRC: extensions changed since they were built; delete $MARK and rebuild"; exit 1; }; fi
  check_list || { echo "$SRC: build products incomplete; delete $MARK and rebuild"; exit 1; }
  check_sums || { echo "$SRC: build products changed since they were verified ($SUMS); delete $MARK and rebuild"; exit 1; }
  echo "build products: already present ($route), all $(wc -l < "$SUMS") product hashes re-verified"
elif [ "${BUILD_OWN:-0}" != 1 ]; then
  sha_ok "$artifacts_sha256" "$ARTIFACTS" || { echo "artefact tarball sha256 mismatch"; exit 1; }
  tar -xzf "$ARTIFACTS" -C "$SRC"
  check_own "$own_C_sha256" "$own_moe_C_sha256" || { echo "tarball extracted but the two v2.2 extensions do not match PIN-v2.2"; exit 1; }
  check_list || exit 1
  write_sums
  echo "tarball $own_C_sha256 $own_moe_C_sha256" > "$MARK"; echo "build products: extracted from $ARTIFACTS (own _C/_moe_C hashes verified)"
else
  # the venv's CUDA 13.0 nvcc by default (the tested route); a distribution nvcc on PATH is often 11.x/12.x, which
  # cannot build against torch's CUDA 13.0 headers, so NVCC= must name an nvcc 13.0 or newer
  NVCC="$(abspath "${NVCC:-$CU/bin/nvcc}")"
  [ -x "$NVCC" ] || { echo "BUILD_OWN=1 needs nvcc; set NVCC=/path/to/nvcc (13.0 or newer)"; exit 1; }
  own_nv="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' || true)"
  { [ -n "$own_nv" ] && [ "$(printf '%s\n13.0\n' "$own_nv" | sort -V | head -1)" = 13.0 ]; } || {
    echo "BUILD_OWN=1 needs nvcc 13.0 or newer (torch $torch is built for CUDA $torch_cuda); $NVCC reports ${own_nv:-no release}."
    echo "  Unset NVCC to use the venv's CUDA $jit_cuda nvcc at $CU/bin/nvcc."; exit 1; }
  WHEEL="${WHEEL:-}"; wheel_ours=0
  if [ -z "$WHEEL" ]; then
    mkdir -p "$SRC/.wheel"; WHEEL="$SRC/.wheel/vllm-${stock_wheel_version}-cp38-abi3-manylinux_2_28_x86_64.whl"; wheel_ours=1
    # download to .part and rename on success, so an interrupted download is not mistaken for the wheel on a re-run
    if [ ! -f "$WHEEL" ]; then
      { curl -fsSL -o "$WHEEL.part" "$stock_wheel_url" && mv "$WHEEL.part" "$WHEEL"; } \
        || { rm -f "$WHEEL.part"; echo "could not fetch $stock_wheel_url; download the wheel and set WHEEL="; exit 1; }
    fi
  fi
  WHEEL="$(abspath "$WHEEL")"
  sha_ok "$stock_wheel_sha256" "$WHEEL" || {
    if [ "$wheel_ours" = 1 ]; then rm -f "$WHEEL"; echo "stock wheel sha256 mismatch: removed $WHEEL; re-run to fetch it again"
    else echo "stock wheel sha256 mismatch: WHEEL=$WHEEL is not the pinned vllm $stock_wheel_version wheel"; fi
    exit 1; }
  # stock products first (the two own extensions are skipped: they are built next); keep the executable bits the
  # wheel records, and fail if any listed product is absent from the wheel
  "$VENV/bin/python" - "$WHEEL" "$SRC" "$REL/build-artifacts.list" "$own_extensions" <<'EOF'
import os, sys, zipfile
whl, src, lst, own = sys.argv[1:5]
own = set(own.split())
want = [l.strip().rstrip('/') for l in open(lst) if l.strip() and l.strip() not in own]
z = zipfile.ZipFile(whl); n = 0; found = set()
for info in z.infolist():
    m = info.filename
    hit = [w for w in want if m == w or m.startswith(w + '/')]
    if hit and not m.endswith('/'):
        path = z.extract(info, src); n += 1; found.update(hit)
        mode = (info.external_attr >> 16) & 0o777
        if mode:
            os.chmod(path, mode)
missing = sorted(set(want) - found)
if missing:
    sys.exit(f"stock wheel lacks listed build products: {missing}")
print(f"build products: {n} stock files from {os.path.basename(whl)}")
EOF
  BD="$SRC/.build-v2.2"; B0=$(date +%s)
  CUDA_BIN="$(dirname "$NVCC")"
  env PATH="$CUDA_BIN:$PATH" TORCH_CUDA_ARCH_LIST="$build_arch" MAX_JOBS="${MAX_JOBS:-8}" \
    cmake -S "$SRC" -B "$BD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_CUDA_COMPILER="$NVCC" > "$BD.configure.log" 2>&1 \
    || { echo "cmake configure failed; see $BD.configure.log"; exit 1; }
  for tgt in _C_stable_libtorch _moe_C_stable_libtorch; do
    env PATH="$CUDA_BIN:$PATH" TORCH_CUDA_ARCH_LIST="$build_arch" \
      cmake --build "$BD" --target "$tgt" -j "${MAX_JOBS:-8}" > "$BD.$tgt.log" 2>&1 || { echo "build $tgt failed; see $BD.$tgt.log"; exit 1; }
    so="$(find "$BD" -name "$tgt.abi3.so" | head -1 || true)"; [ -n "$so" ] || { echo "$tgt.abi3.so not produced"; exit 1; }
    cp "$so" "$SRC/vllm/$tgt.abi3.so"
    echo "built $tgt sha256=$(sha256sum "$SRC/vllm/$tgt.abi3.so" | cut -d' ' -f1) (reference build: see PIN-v2.2 own_*_sha256)"
  done
  echo "built in $(( $(date +%s) - B0 )) s"
  check_list || exit 1
  write_sums
  echo "own $(sha256sum "$SRC/vllm/_C_stable_libtorch.abi3.so" | cut -d' ' -f1) $(sha256sum "$SRC/vllm/_moe_C_stable_libtorch.abi3.so" | cut -d' ' -f1)" > "$MARK"
fi
# The shallow fetch carries no release tag, so setuptools-scm would name the tree 0.1.devN; pin the version the tree is
# (the stock release it is based on, as the reference host reports it) so version checks and cache keys see 0.30.0.
(cd "$SRC" && VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE="$stock_wheel_version" "$VENV/bin/pip" install -q -e . --no-deps --no-build-isolation)
got_ver="$(cd / && PYTHONPATH="$SRC" "$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
[ "$got_ver" = "$stock_wheel_version" ] || { echo "vllm reports version $got_ver, expected $stock_wheel_version"; exit 1; }
(cd / && PYTHONPATH="$SRC" "$VENV/bin/python" -c "import vllm, sys; print('vllm', vllm.__version__, vllm.__file__)")
echo "done: TREE=$SRC VENV=$VENV scripts/serve-v2.2.sh /path/to/checkpoint"
