#!/usr/bin/env bash
# Build the patched vLLM this repo serves with, from a pinned, hash-checked base.
#
#   scripts/build.sh [SRC_DIR] [VENV_DIR] [--no-csrc]
#
# 1. clones vllm-project/vllm at the release tag in upstream/PIN (v0.28.0, a
#    permanent ref), applies patches/upstream/*.patch (the eight commits of the
#    peakcrosser7 Flash-Next branch, as cherry-picked onto that tag on the
#    reference host) and then patches/*.patch (ours), all with `git am`, which
#    fails closed on any reject; checks the vendored upstream/pinned/ files match
#    the tree after the upstream patches,
# 2. downloads the exact precompiled compiled-ops wheel named in upstream/PIN
#    (the 0.28.0 cu130 release wheel from wheels.vllm.ai, verified by sha256) and
#    installs the tree editable against it: python from the tree, compiled ops
#    from that wheel, no floating nightly,
# 3. rebuilds only the _C_stable_libtorch target for sm_86 and swaps it into the
#    tree (skip with --no-csrc). The branch changed the fused GDN decode kernel
#    (15-argument fused_gdn_decode_post_conv_mtp) and the 0.28.0 wheel carries the
#    14-argument one; the python-side hasattr guard cannot see that and the call
#    fails at runtime, so the served entry needs the rebuilt kernel. Patch 0003
#    is what makes an Ampere-only csrc build compile at all. Needs CUDA toolkit
#    >= 13.0 (the GDN decode sources are gated on it upstream), cmake >= 3.26,
#    ninja and gcc >= 11.3; cmake and ninja are installed into the venv here.
#
# Reference build resolved torch 2.13.0, triton 3.7.1 (pip resolution, not locked); nvcc 13.3.
set -euo pipefail
SRC="$PWD/vllm-src"; VENV="$PWD/venv"; CSRC=1; n=0
for a in "$@"; do
  case "$a" in
    --csrc) CSRC=1 ;;
    --no-csrc) CSRC=0 ;;
    *) n=$((n+1)); [ $n = 1 ] && SRC="$a"; [ $n = 2 ] && VENV="$a" ;;
  esac
done
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# absolute paths: pip runs the build backend from inside SRC, so a relative wheel path would resolve wrongly
mkdir -p "$SRC" "$VENV"; SRC="$(cd "$SRC" && pwd)"; VENV="$(cd "$VENV" && pwd)"
# shellcheck disable=SC1091
. "$HERE/upstream/PIN"   # sets repo= tag= commit= tree= wheel_url= wheel_sha256=
NUP="$(ls "$HERE"/patches/upstream/*.patch | wc -l)"
NPATCH="$(ls "$HERE"/patches/*.patch | wc -l)"
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"

if [ ! -d "$SRC/.git" ]; then
  rmdir "$SRC" 2>/dev/null || true
  git clone -q --depth 1 --branch "$tag" "$repo" "$SRC"
  got="$(git -C "$SRC" rev-parse HEAD)"
  [ "$got" = "$commit" ] || { echo "tag $tag resolved to $got, expected $commit"; exit 1; }
  git -C "$SRC" config core.autocrlf false
  git -C "$SRC" -c user.name=patches -c user.email=patches@localhost am -q "$HERE"/patches/upstream/*.patch
  # the vendored copies are the tree our patches were written against: tag + upstream patches
  while IFS= read -r f; do
    cmp -s "$HERE/upstream/pinned/$f" "$SRC/$f" || { echo "upstream/pinned/$f differs from tag+upstream patches: wrong base"; exit 1; }
  done < <(cd "$HERE/upstream/pinned" && find . -type f | sed 's|^\./||')
  git -C "$SRC" -c user.name=patches -c user.email=patches@localhost am -q "$HERE"/patches/*.patch
fi
have="$(git -C "$SRC" rev-list --count HEAD)"
[ "$have" = $((NUP + NPATCH + 1)) ] || { echo "$SRC has $have commits, expected $((NUP + NPATCH + 1)) (tag + $NUP upstream + $NPATCH ours); remove it and rerun"; exit 1; }
i=0
for pf in "$HERE"/patches/upstream/*.patch "$HERE"/patches/*.patch; do
  i=$((i + 1))
  # git am strips leading [tags] such as [BugFix] from subjects; compare after the same strip
  want="$(sed -n 's/^Subject: \[PATCH[^]]*\] //p' "$pf" | head -1 | sed -E 's/^(\[[^]]*\] *)+//' | cut -c1-40)"
  got="$(git -C "$SRC" log --reverse --format=%s | sed -n "$((i + 1))p" | cut -c1-40)"
  [ "$want" = "$got" ] || { echo "patch $i mismatch: expected '$want' got '$got'"; exit 1; }
done
# content check that survives reuse: the patched tree hash is fixed by the tag + patches, and nothing may be edited
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "$SRC has modified tracked files; reset it or remove it"; exit 1; }
got_tree="$(git -C "$SRC" rev-parse "HEAD^{tree}")"
[ "$got_tree" = "$tree" ] || { echo "$SRC tree $got_tree != expected $tree (upstream/PIN)"; exit 1; }
echo "source: $tag + $NUP upstream-branch patches + $NPATCH patches, tree $tree verified"

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q "cmake>=3.26" ninja
export PATH="$VENV/bin:$PATH"
if [ "$CSRC" = 1 ]; then
  gcc_ver="$(gcc -dumpfullversion 2>/dev/null || echo 0)"
  [ "$(printf '%s\n11.3\n' "$gcc_ver" | sort -V | head -1)" = "11.3" ] || { echo "gcc >= 11.3 required for the csrc build (found: $gcc_ver)"; exit 1; }
  nvcc_major="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
  [ "${nvcc_major:-0}" -ge 13 ] || { echo "nvcc >= 13.0 required for the fused GDN decode kernel (found: ${nvcc_major:-none}; is /usr/local/cuda/bin on PATH?)"; exit 1; }
fi

WHEEL="$SRC/../$(basename "$wheel_url")"
if [ ! -f "$WHEEL" ] || ! echo "$wheel_sha256  $WHEEL" | sha256sum -c --quiet - 2>/dev/null; then
  curl -fL -o "$WHEEL" "$wheel_url"
  echo "$wheel_sha256  $WHEEL" | sha256sum -c --quiet - || { echo "wheel sha256 mismatch: $WHEEL"; exit 1; }
fi
echo "compiled-ops wheel: $(basename "$WHEEL") sha256 ok"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" "$VENV/bin/pip" install -e "$SRC"
"$VENV/bin/python" -c "import vllm, torch, triton; print('vllm', vllm.__version__, '| torch', torch.__version__, '| triton', triton.__version__)"

if [ "$CSRC" = 1 ]; then
  # The reference host's build (Ninja, nvcc 13.3, MAX_JOBS=12, only the _C_stable_libtorch target, arch 8.6).
  NVCC="$(command -v nvcc)"
  B="$SRC/build-stable86"; mkdir -p "$B"
  TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS="${MAX_JOBS:-$(nproc)}" cmake -S "$SRC" -B "$B" -G Ninja \
     -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
     -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_CUDA_COMPILER="$NVCC"
  cmake --build "$B" --target _C_stable_libtorch -j"${MAX_JOBS:-$(nproc)}"
  so="$(find "$B" -name '_C_stable_libtorch*.so' | head -1)"
  wheel_so="$(find "$SRC/vllm" -maxdepth 1 -name '_C_stable_libtorch*.so' | head -1)"
  [ -n "$so" ] && [ -n "$wheel_so" ] || { echo "could not locate built or wheel _C_stable_libtorch .so"; exit 1; }
  cp "$wheel_so" "$wheel_so.wheel14" && cp "$so" "$wheel_so"
  echo "swapped $wheel_so (backup: .wheel14)"
  "$VENV/bin/python" - <<'EOF'
import torch, vllm._custom_ops  # noqa: F401  (registers the ops)
op = torch.ops._C.fused_gdn_decode_post_conv_mtp
n = len(op.default._schema.arguments)
assert n == 15, f"fused_gdn_decode_post_conv_mtp has {n} arguments, expected 15: the rebuilt kernel was not picked up"
print("fused_gdn_decode_post_conv_mtp: 15-argument schema OK")
EOF
else
  echo "NOTE: --no-csrc: the served MTP entry needs the rebuilt GDN decode kernel; without it the branch's 15-argument call fails at runtime against the wheel's 14-argument op"
fi
echo "done. serve with: VENV=$VENV $HERE/scripts/serve.sh /path/to/checkpoint"
