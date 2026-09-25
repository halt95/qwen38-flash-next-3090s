#!/usr/bin/env bash
# Shell-path tests for the AUTO_TOPO integration in scripts/serve-v2.2.sh. Each case runs the real serve-v2.2.sh end
# to end -- snapshot, selector call, env-file validation, legacy fallback, exec-prefix wiring -- in a throwaway
# sandbox of stub binaries, usually through the SERVE_PRINT_ENV_AND_EXIT=1 test hook, so no GPU, driver or real vllm
# install is needed. Runs under plain bash on Linux and under Git Bash on Windows; only GNU coreutils (mktemp,
# sha256sum, timeout) are assumed, all of which this repo's build already requires.
#
# Most cases replace scripts/topo_select.py with a stub (a bash script, not Python) whose behaviour is chosen by
# STUB_TOPO_MODE, so they test serve's handling of every selector outcome independently of the selector. The cases
# marked "real selector" run the actual scripts/topo_select.py on a fixture with the python3 found on PATH.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REAL_SERVE="$REPO_ROOT/scripts/serve-v2.2.sh"
REAL_PY="$(command -v python3 || command -v python || true)"

PASS=0
FAIL=0
FAILED_CASES=()

note() { printf '%s\n' "$*" >&2; }
has_line() { printf '%s\n' "$1" | grep -qx -- "$2"; }      # exact whole-line match (regex)
has_fixed() { printf '%s\n' "$1" | grep -qF -- "$2"; }     # fixed substring

# --- sandbox construction ----------------------------------------------------------------------------------------

# Everything serve-v2.2.sh's pre-topo checks require, plus a copy of the script itself (so $0 -- and therefore its
# own $HERE -- resolves inside the sandbox; the real script computes HERE from its own path, so running the repo's
# copy in place would make it look for scripts/topo_select.py in the real repo instead of the sandbox).
make_sandbox() {
  local d
  d="$(mktemp -d)" || { note "mktemp -d failed"; return 1; }
  mkdir -p "$d/scripts" "$d/vllm-v2.2/vllm" "$d/venv-v2.2/bin" "$d/venv-v2.2/nvidia/cu13/bin" \
           "$d/venv-v2.2/nvidia/cu13/lib64" "$d/scales" "$d/bin" "$d/ckpt"

  cp "$REAL_SERVE" "$d/scripts/serve-v2.2.sh"

  # TREE: the build-product markers the top-of-script checks require, with a checksum that actually verifies.
  : > "$d/vllm-v2.2/vllm/__init__.py"
  : > "$d/vllm-v2.2/.v2.2-build-products"
  ( cd "$d/vllm-v2.2" && sha256sum .v2.2-build-products > .v2.2-build-products.sha256 )

  # VENV: bin/python is a tiny dispatcher, not real Python. Every "-c ..." call in serve-v2.2.sh uses one of three
  # fixed code strings; the script-path form is the selector call, which runs the stub selector with bash, or the
  # real selector with the real interpreter when STUB_REAL_PY is set. There is deliberately no bin/python3: serve
  # must run the selector with bin/python.
  cat > "$d/venv-v2.2/bin/python" <<'PYEOF'
#!/usr/bin/env bash
set -u
if [ "$1" = "-c" ]; then
  code="$2"
  case "$code" in
    *ple_embedding_dtype*) grep -q '"ple_embedding_dtype": *"float8_e4m3fn"' "$3" && exit 0 || exit 1 ;;
    *purelib*) printf '%s\n' "${STUB_PURELIB:?STUB_PURELIB not set}"; exit 0 ;;
    *Python.h*) exit 0 ;;
    *) exit 0 ;;
  esac
fi
script="$1"; shift
if [ -n "${STUB_REAL_PY:-}" ]; then exec "$STUB_REAL_PY" "$script" "$@"; fi
exec bash "$script" "$@"
PYEOF
  chmod +x "$d/venv-v2.2/bin/python"

  # stub vllm: prints its argv instead of starting a real server.
  cat > "$d/venv-v2.2/bin/vllm" <<'VLLMEOF'
#!/usr/bin/env bash
printf 'STUB_VLLM_ARGV:'
for a in "$@"; do printf ' [%s]' "$a"; done
printf '\n'
VLLMEOF
  chmod +x "$d/venv-v2.2/bin/vllm"

  # stub CUDA toolkit (a self-consistent nvcc/driver pair so the version-skew check passes) and nvidia-smi.
  cat > "$d/venv-v2.2/nvidia/cu13/bin/nvcc" <<'NVCCEOF'
#!/usr/bin/env bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 13.0, V13.0.88"
NVCCEOF
  chmod +x "$d/venv-v2.2/nvidia/cu13/bin/nvcc"
  : > "$d/venv-v2.2/nvidia/cu13/lib64/libcudart.so"

  cat > "$d/bin/nvidia-smi" <<'SMIEOF'
#!/usr/bin/env bash
echo "CUDA Version: 13.0"
SMIEOF
  chmod +x "$d/bin/nvidia-smi"
  for t in cc c++ ninja; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$d/bin/$t"
    chmod +x "$d/bin/$t"
  done

  printf '{"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}\n' > "$d/ckpt/config.json"
  : > "$d/scales/qsa_kv_scales_262k.json"

  printf '%s' "$d"
}

# Stub selector. Modes (STUB_TOPO_MODE):
#   marker            touch $STUB_TOPO_MARKER and exit 0 (proves the selector was never invoked)
#   ok_reorder        a valid status=ok env with reordered UUIDs + a numactl prefix, exit 0
#   echo_timeout      as ok_reorder, and write the TOPO_TIMEOUT it was given to $STUB_TOPO_LOG
#   echo_caller       write the TOPO_CALLER_* vars it was given to $STUB_TOPO_LOG, a valid legacy env, exit 0
#   exit5             exit 5 with no env file (any nonzero other than the strict-mode 3)
#   exit3             exit 3 with a stderr message, as strict mode does
#   garbage_env       exit 0, but the "env file" isn't one
#   unterminated      exit 0, valid header, then a line with an unterminated quote
#   header_only       exit 0, the header line and nothing else
#   extra_line        exit 0, the 8 valid lines plus a 9th that would touch $STUB_TOPO_MARKER if sourced
#   bad_prefix        exit 0, 8 lines in order but TOPO_LAUNCH_PREFIX is not a numactl prefix
#   hang              sleep past any sane timeout
#   hang_ignore_term  ignore SIGTERM, then sleep 60 s (only SIGKILL stops it sooner)
write_stub_topo_select() {
  local d="$1"
  cat > "$d/scripts/topo_select.py" <<'TOPOEOF'
#!/usr/bin/env bash
# NOT Python: the sandbox's bin/python runs this file with bash.
set -u
env_out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --env-out) env_out="$2"; shift 2 ;;
    *) shift ;;
  esac
done

ok_env() {
  echo "# topo_select v1 status=ok reason=reordered"
  echo "export CUDA_VISIBLE_DEVICES='GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222,GPU-33333333-3333-3333-3333-333333333333,GPU-44444444-4444-4444-4444-444444444444'"
  echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID"
  echo "export VLLM_SKIP_P2P_CHECK=0"
  echo "export NCCL_P2P_LEVEL=SYS"
  echo "export NCCL_PROTO=LL"
  echo "export TOPO_LAUNCH_PREFIX='numactl --physcpubind=24-47 --preferred=1 --'"
  echo "export TOPO_DECISION_JSON='/tmp/topo-decision.json'"
}

case "${STUB_TOPO_MODE:-ok_reorder}" in
  marker) : > "${STUB_TOPO_MARKER:?}"; exit 0 ;;
  ok_reorder) ok_env > "$env_out"; exit 0 ;;
  echo_timeout) echo "TOPO_TIMEOUT=${TOPO_TIMEOUT-<unset>}" > "${STUB_TOPO_LOG:?}"; ok_env > "$env_out"; exit 0 ;;
  echo_caller)
    {
      echo "TOPO_CALLER_SET_NCCL_PROTO=${TOPO_CALLER_SET_NCCL_PROTO:-}"
      echo "TOPO_CALLER_NCCL_PROTO=${TOPO_CALLER_NCCL_PROTO:-}"
      echo "TOPO_CALLER_SET_CUDA_VISIBLE_DEVICES=${TOPO_CALLER_SET_CUDA_VISIBLE_DEVICES:-}"
      echo "TOPO_CALLER_CUDA_VISIBLE_DEVICES=${TOPO_CALLER_CUDA_VISIBLE_DEVICES:-}"
    } > "${STUB_TOPO_LOG:?}"
    {
      echo "# topo_select v1 status=legacy reason=stub_echo_caller"
      echo "export CUDA_VISIBLE_DEVICES='0,1,2,3'"
      echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID"
      echo "export VLLM_SKIP_P2P_CHECK=1"
      echo "export NCCL_P2P_LEVEL=SYS"
      echo "export NCCL_PROTO=LL"
      echo "export TOPO_LAUNCH_PREFIX=''"
      echo "export TOPO_DECISION_JSON=''"
    } > "$env_out"
    exit 0 ;;
  exit5) exit 5 ;;
  exit3) echo "topo_select: strict mode refuses to guess -- more than 4 GPUs visible and no lane chosen" >&2; exit 3 ;;
  garbage_env) printf 'not a topo env file at all\nrandom garbage\n' > "$env_out"; exit 0 ;;
  unterminated) { echo "# topo_select v1 status=ok reason=kept"; echo "export CUDA_VISIBLE_DEVICES='0,1,2,3"; } > "$env_out"; exit 0 ;;
  header_only) echo "# topo_select v1 status=ok reason=kept" > "$env_out"; exit 0 ;;
  extra_line) { ok_env; echo "touch '${STUB_TOPO_MARKER:?}'"; } > "$env_out"; exit 0 ;;
  bad_prefix) ok_env | sed "s|^export TOPO_LAUNCH_PREFIX=.*|export TOPO_LAUNCH_PREFIX='sh -c true'|" > "$env_out"; exit 0 ;;
  hang) sleep 3600 ;;
  hang_ignore_term) trap '' TERM; sleep 60 ;;   # bounded: a regression then fails in ~60 s instead of hanging
  *) exit 5 ;;
esac
TOPOEOF
  chmod +x "$d/scripts/topo_select.py"
}

# Real selector on a fixture: copy the real scripts/topo_select.py into the sandbox and let bin/python run it.
use_real_selector() {
  local sandbox="$1" fixture="$2"
  cp "$REPO_ROOT/scripts/topo_select.py" "$sandbox/scripts/topo_select.py"
  export STUB_REAL_PY="$REAL_PY"
  export TOPO_FIXTURE="$REPO_ROOT/tests/topo/fixtures/$fixture"
  export AUTO_TOPO=1
}

# --- case runner -------------------------------------------------------------------------------------------------

# Runs one case in a fresh sandbox + subshell (so a case's env exports never leak into the next one), captures
# combined stdout+stderr and the exit code, hands both to the case's check function, then tears the sandbox down.
# The outer `timeout 40` is a safety net only: it stops a hung case (or a serve bug) from hanging the whole suite.
run_case() {
  local name="$1" setup_fn="$2" check_fn="$3"
  local sandbox out rc

  sandbox="$(make_sandbox)" || { printf 'FAIL  %s (sandbox setup failed)\n' "$name"; FAIL=$((FAIL+1)); FAILED_CASES+=("$name"); return; }
  write_stub_topo_select "$sandbox"

  out="$(
    set -u
    cd "$sandbox" || exit 99
    unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER VLLM_SKIP_P2P_CHECK NCCL_P2P_LEVEL NCCL_PROTO PLE_HOME CACHE_ROOT \
          AUTO_TOPO AUTO_TOPO_NCCL AUTO_TOPO_BIND LANE_GPUS LANE_CPUS TOPO_TIMEOUT TOPO_FIXTURE VLLM_CACHE_ROOT
    export PATH="$sandbox/bin:$PATH"
    export TREE="$sandbox/vllm-v2.2"
    export VENV="$sandbox/venv-v2.2"
    export STUB_PURELIB="$sandbox/venv-v2.2"
    export SCALES="$sandbox/scales/qsa_kv_scales_262k.json"
    export SERVE_PRINT_ENV_AND_EXIT=1
    "$setup_fn" "$sandbox"
    timeout 40 bash "$sandbox/scripts/serve-v2.2.sh" "$sandbox/ckpt" 2>&1
    printf 'CASE_EXIT_CODE:%d\n' "$?"
  )"
  rc="$(printf '%s\n' "$out" | sed -n 's/^CASE_EXIT_CODE:\([0-9-]*\)$/\1/p' | tail -1)"

  if "$check_fn" "$sandbox" "$out" "$rc"; then
    printf 'PASS  %s\n' "$name"
    PASS=$((PASS+1))
  else
    printf 'FAIL  %s\n' "$name"
    {
      printf -- '  ------ case %s output (exit=%s) ------\n' "$name" "$rc"
      printf '%s\n' "$out" | sed 's/^/  | /'
      printf -- '  ---------------------------------------\n'
    } >&2
    FAIL=$((FAIL+1))
    FAILED_CASES+=("$name")
  fi
  rm -rf "$sandbox"
}

# Shared checks.
legacy_values() {   # $1 = output, $2 = expected CUDA_VISIBLE_DEVICES
  local out="$1" v
  for v in "CUDA_VISIBLE_DEVICES=$2" CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS \
           NCCL_PROTO=LL TOPO_LAUNCH_PREFIX= 'EXEC_ARGV_PREFIX:'; do
    has_line "$out" "$v" || { note "expected line: $v"; return 1; }
  done
}
fell_back() {       # $1 = output, $2 = expected message (regex, whole line)
  has_line "$1" "$2" || { note "missing fallback message: $2"; return 1; }
}

# --- cases ---------------------------------------------------------------------------------------------------
# Each case is a (setup, check) pair. setup runs INSIDE the sandboxed subshell before serve-v2.2.sh and only exports
# env vars / writes files under the sandbox it is given. check runs OUTSIDE that subshell, against the captured
# output, exit code and (while the sandbox still exists) any files the run left behind.

# (a) AUTO_TOPO=0: legacy values exactly, selector never invoked, hook prints an empty exec prefix exactly.
setup_a() { export AUTO_TOPO=0 STUB_TOPO_MODE=marker STUB_TOPO_MARKER="$1/marker"; }
check_a() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  [ ! -e "$1/marker" ] || { note "selector was invoked under AUTO_TOPO=0"; return 1; }
  legacy_values "$2" 0,1,2,3
}

# (b) a valid reordered env + numactl prefix is applied, and the hook splits the prefix as the exec will.
setup_b() { export AUTO_TOPO=1 STUB_TOPO_MODE=ok_reorder; }
check_b() {
  local out="$2"
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  has_line "$out" 'CUDA_VISIBLE_DEVICES=GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222,GPU-33333333-3333-3333-3333-333333333333,GPU-44444444-4444-4444-4444-444444444444' \
    || { note "reordered CVD not applied"; return 1; }
  has_line "$out" 'VLLM_SKIP_P2P_CHECK=0' || { note "VLLM_SKIP_P2P_CHECK not from selector"; return 1; }
  has_line "$out" 'TOPO_LAUNCH_PREFIX=numactl --physcpubind=24-47 --preferred=1 --' || { note "prefix not applied"; return 1; }
  has_line "$out" 'EXEC_ARGV_PREFIX: \[numactl\] \[--physcpubind=24-47\] \[--preferred=1\] \[--\]' \
    || { note "hook split of the prefix is wrong"; return 1; }
}

# (c) selector exits 5: one message, then legacy.
setup_c() { export AUTO_TOPO=1 STUB_TOPO_MODE=exit5; }
check_c() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  fell_back "$2" 'topo: selector unavailable (rc=5), using legacy settings' && legacy_values "$2" 0,1,2,3
}

# (d)-(g2) selector exits 0 but the env file is not exactly the 8-line format: never sourced, legacy.
setup_d() { export AUTO_TOPO=1 STUB_TOPO_MODE=garbage_env; }
setup_d2() { export AUTO_TOPO=1 STUB_TOPO_MODE=unterminated; }
setup_d3() { export AUTO_TOPO=1 STUB_TOPO_MODE=header_only; }
setup_d4() { export AUTO_TOPO=1 STUB_TOPO_MODE=extra_line STUB_TOPO_MARKER="$1/marker"; }
setup_d5() { export AUTO_TOPO=1 STUB_TOPO_MODE=bad_prefix; }
check_d() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  [ ! -e "$1/marker" ] || { note "the env file was sourced"; return 1; }
  fell_back "$2" 'topo: selector unavailable (rc=0, unusable env file), using legacy settings' && legacy_values "$2" 0,1,2,3
}

# (e) selector hangs: serve's own watchdog (TOPO_TIMEOUT + 5 s) fires, then legacy. TOPO_TIMEOUT=1 keeps it ~6 s.
setup_e() { export AUTO_TOPO=1 STUB_TOPO_MODE=hang TOPO_TIMEOUT=1; }
check_e() {
  [ "$3" = 0 ] || { note "expected exit 0 after timeout+fallback, got $3"; return 1; }
  fell_back "$2" 'topo: selector unavailable (rc=124), using legacy settings' && legacy_values "$2" 0,1,2,3
}

# (e2) selector hangs and ignores SIGTERM: --kill-after must end it (~11 s); without it the case hits the 40 s net.
setup_e2() { export AUTO_TOPO=1 STUB_TOPO_MODE=hang_ignore_term TOPO_TIMEOUT=1; }
check_e2() {
  [ "$3" = 0 ] || { note "expected exit 0 after kill+fallback, got $3"; return 1; }
  fell_back "$2" 'topo: selector unavailable (rc=\(124\|137\)), using legacy settings' && legacy_values "$2" 0,1,2,3
}

# (f) selector exits 3 (strict-mode refusal): serve exits 1 with the message, hook never reached.
setup_f() { export AUTO_TOPO=1 STUB_TOPO_MODE=exit3; }
check_f() {
  [ "$3" = 1 ] || { note "expected exit 1, got $3"; return 1; }
  has_fixed "$2" 'strict mode refuses to guess' || { note "strict-mode message missing"; return 1; }
  ! has_line "$2" 'SERVE_PRINT_ENV_AND_EXIT:' || { note "hook ran after a strict-mode refusal"; return 1; }
}

# (f2)/(f3) AUTO_TOPO=strict and the selector cannot be used (fails / unusable file): exit 1, no legacy.
setup_f2() { export AUTO_TOPO=strict STUB_TOPO_MODE=exit5; }
setup_f3() { export AUTO_TOPO=strict STUB_TOPO_MODE=garbage_env; }
check_f2() {
  [ "$3" = 1 ] || { note "expected exit 1, got $3"; return 1; }
  has_fixed "$2" 'AUTO_TOPO=strict refuses to start without a selector decision' || { note "strict message missing"; return 1; }
  ! has_line "$2" 'SERVE_PRINT_ENV_AND_EXIT:' || { note "hook ran after a strict-mode refusal"; return 1; }
}

# (g) caller-set NCCL_PROTO and CUDA_VISIBLE_DEVICES reach the selector via the TOPO_CALLER_* snapshot.
setup_g() {
  export AUTO_TOPO=1 STUB_TOPO_MODE=echo_caller NCCL_PROTO=Simple CUDA_VISIBLE_DEVICES=1,0,3,2 STUB_TOPO_LOG="$1/log"
}
check_g() {
  local log="$1/log"
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  [ -f "$log" ] || { note "stub never wrote the caller log"; return 1; }
  grep -qx 'TOPO_CALLER_SET_NCCL_PROTO=1' "$log" || { note "TOPO_CALLER_SET_NCCL_PROTO"; return 1; }
  grep -qx 'TOPO_CALLER_NCCL_PROTO=Simple' "$log" || { note "TOPO_CALLER_NCCL_PROTO"; return 1; }
  grep -qx 'TOPO_CALLER_SET_CUDA_VISIBLE_DEVICES=1' "$log" || { note "TOPO_CALLER_SET_CUDA_VISIBLE_DEVICES"; return 1; }
  grep -qx 'TOPO_CALLER_CUDA_VISIBLE_DEVICES=1,0,3,2' "$log" || { note "TOPO_CALLER_CUDA_VISIBLE_DEVICES"; return 1; }
}

# (g2) the fallback keeps the caller's own CUDA_VISIBLE_DEVICES (from the snapshot).
setup_g2() { export AUTO_TOPO=1 STUB_TOPO_MODE=exit5 CUDA_VISIBLE_DEVICES=1,0,3,2; }
check_g2() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  fell_back "$2" 'topo: selector unavailable (rc=5), using legacy settings' && legacy_values "$2" 1,0,3,2
}

# (h) the selector runs with $VENV/bin/python, never with a python3 found on PATH.
setup_h() {
  printf '#!/usr/bin/env bash\n: > "%s/marker"\nexit 5\n' "$1" > "$1/bin/python3"
  chmod +x "$1/bin/python3"
  export AUTO_TOPO=1 STUB_TOPO_MODE=ok_reorder
}
check_h() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  [ ! -e "$1/marker" ] || { note "a python3 from PATH was run"; return 1; }
  has_line "$2" 'VLLM_SKIP_P2P_CHECK=0' || { note "selector result not applied"; return 1; }
}

# (i) SERVE_PRINT_ENV_AND_EXIT unset: the hook is inert, the real exec runs with no prefix and no empty word.
setup_i() { export AUTO_TOPO=0; unset SERVE_PRINT_ENV_AND_EXIT; }
check_i() {
  [ "$3" = 0 ] || { note "expected exit 0 (stub vllm exits 0), got $3"; return 1; }
  ! has_line "$2" 'SERVE_PRINT_ENV_AND_EXIT:' || { note "hook fired despite being unset"; return 1; }
  has_fixed "$2" "STUB_VLLM_ARGV: [serve] [$1/ckpt] [--served-model-name]" || { note "stub vllm argv wrong"; return 1; }
}

# (u) AUTO_TOPO unset -- the v2.2.0 default: the selector is off, never invoked, and the hook shows the legacy values
# with an empty exec prefix. (The golden cases (g0-*) below check the full env + argv against the base script.)
setup_u() { export STUB_TOPO_MODE=marker STUB_TOPO_MARKER="$1/marker"; }
check_u() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  [ ! -e "$1/marker" ] || { note "selector was invoked with AUTO_TOPO unset"; return 1; }
  ! has_fixed "$2" 'topo:' || { note "a topo: line was printed with the selector off"; return 1; }
  legacy_values "$2" 0,1,2,3
}

# (u2) AUTO_TOPO=yes (not 0, 1 or strict): a note, the selector stays off, legacy values.
setup_u2() { export AUTO_TOPO=yes STUB_TOPO_MODE=marker STUB_TOPO_MARKER="$1/marker"; }
check_u2() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  [ ! -e "$1/marker" ] || { note "selector was invoked with AUTO_TOPO=yes"; return 1; }
  has_fixed "$2" "topo: note: AUTO_TOPO='yes' is not 0, 1 or strict; the selector stays off" || { note "no note"; return 1; }
  legacy_values "$2" 0,1,2,3
}

# (i2) a non-empty prefix reaches the REAL exec: a stub numactl prints its argv and execs the rest.
setup_i2() {
  cat > "$1/bin/numactl" <<'EOF'
#!/usr/bin/env bash
printf 'STUB_NUMACTL_ARGV:'; for a in "$@"; do printf ' [%s]' "$a"; done; printf '\n'
while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done
shift
exec "$@"
EOF
  chmod +x "$1/bin/numactl"
  export AUTO_TOPO=1 STUB_TOPO_MODE=ok_reorder
  unset SERVE_PRINT_ENV_AND_EXIT
}
check_i2() {
  local sb="$1" out="$2"
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  has_fixed "$out" "STUB_NUMACTL_ARGV: [--physcpubind=24-47] [--preferred=1] [--] [$sb/venv-v2.2/bin/vllm] [serve] [$sb/ckpt] [--served-model-name]" \
    || { note "numactl did not receive the prefix + the vllm command"; return 1; }
  has_fixed "$out" "STUB_VLLM_ARGV: [serve] [$sb/ckpt] [--served-model-name]" || { note "vllm not exec'd by numactl"; return 1; }
}

# (t1)-(t3) TOPO_TIMEOUT is validated in the shell; the selector gets the same validated value.
setup_t1() { export AUTO_TOPO=1 STUB_TOPO_MODE=echo_timeout STUB_TOPO_LOG="$1/log" TOPO_TIMEOUT=2.5; }
setup_t2() { export AUTO_TOPO=1 STUB_TOPO_MODE=echo_timeout STUB_TOPO_LOG="$1/log" TOPO_TIMEOUT=08; }
setup_t3() { export AUTO_TOPO=1 STUB_TOPO_MODE=echo_timeout STUB_TOPO_LOG="$1/log" TOPO_TIMEOUT=7; }
check_t_default() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  has_fixed "$2" "topo: note: TOPO_TIMEOUT=" || { note "no note about the invalid TOPO_TIMEOUT"; return 1; }
  grep -qx 'TOPO_TIMEOUT=20' "$1/log" || { note "selector did not get TOPO_TIMEOUT=20"; return 1; }
  has_line "$2" 'VLLM_SKIP_P2P_CHECK=0' || { note "selector result not applied"; return 1; }
}
check_t3() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  ! has_fixed "$2" "topo: note: TOPO_TIMEOUT=" || { note "valid TOPO_TIMEOUT flagged"; return 1; }
  grep -qx 'TOPO_TIMEOUT=7' "$1/log" || { note "selector did not get TOPO_TIMEOUT=7"; return 1; }
}

# (m1)/(m2) mktemp fails (TMPDIR does not exist): legacy with a message; under strict, exit 1.
setup_m1() { export AUTO_TOPO=1 STUB_TOPO_MODE=ok_reorder TMPDIR="$1/no/such/dir"; }
setup_m2() { export AUTO_TOPO=strict STUB_TOPO_MODE=ok_reorder TMPDIR="$1/no/such/dir"; }
check_m1() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  fell_back "$2" 'topo: could not create a temporary file for the selector, using legacy settings' && legacy_values "$2" 0,1,2,3
}
check_m2() {
  [ "$3" = 1 ] || { note "expected exit 1, got $3"; return 1; }
  has_fixed "$2" 'AUTO_TOPO=strict refuses to start without a selector decision' || { note "strict message missing"; return 1; }
}

# (v1) caller exports CUDA_VISIBLE_DEVICES= (set but empty), real selector: behaves as unset -> today's values.
setup_v1() { use_real_selector "$1" reference.json; export CUDA_VISIBLE_DEVICES=; }
check_v1() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  has_fixed "$2" 'reason=reference' || { note "selector did not report reason=reference"; return 1; }
  ! has_fixed "$2" 'selector unavailable' || { note "fell back"; return 1; }
  legacy_values "$2" 0,1,2,3
}

# (v2) whitespace-only CUDA_VISIBLE_DEVICES and a failing selector: the fallback treats it as unset too.
setup_v2() { export AUTO_TOPO=1 STUB_TOPO_MODE=exit5 CUDA_VISIBLE_DEVICES='  '; }
check_v2() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  fell_back "$2" 'topo: selector unavailable (rc=5), using legacy settings' && legacy_values "$2" 0,1,2,3
}

# (j) real selector, reference fixture: today's values, and diagnostics under $CACHE_ROOT/topology.
setup_j() { use_real_selector "$1" reference.json; export CACHE_ROOT="$1/cache"; }
check_j() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  has_fixed "$2" 'reason=reference' || { note "selector did not report reason=reference"; return 1; }
  legacy_values "$2" 0,1,2,3 || return 1
  has_line "$2" 'TOPO_DECISION_JSON=.*cache[/\\]topology[/\\][0-9a-f]\{12\}-[0-9TZ]*\.json' \
    || { note "diagnostics not under \$CACHE_ROOT/topology"; return 1; }
}

# (j2) real selector, CACHE_ROOT unset: diagnostics under serve's default VLLM_CACHE_ROOT ($HERE/.vllm-cache-v2.2).
setup_j2() { use_real_selector "$1" reference.json; }
check_j2() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  has_line "$2" 'TOPO_DECISION_JSON=.*\.vllm-cache-v2\.2[/\\]topology[/\\][0-9a-f]\{12\}-[0-9TZ]*\.json' \
    || { note "diagnostics not under the default cache root"; return 1; }
}

# (k) real selector, NVLink fixture: reordered UUID lane applied.
setup_k() { use_real_selector "$1" nvlink_pairs.json; }
check_k() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  has_line "$2" 'CUDA_VISIBLE_DEVICES=GPU-.*' || { note "reordered UUID lane not applied"; return 1; }
  ! has_fixed "$2" 'selector unavailable' || { note "fell back to legacy"; return 1; }
}

# (n) real selector, no-P2P fixture with AUTO_TOPO_NCCL=default: the env file's `unset NCCL_PROTO` is applied.
setup_n() { use_real_selector "$1" bulgaria.json; export AUTO_TOPO_NCCL=default; }
check_n() {
  [ "$3" = 0 ] || { note "expected exit 0, got $3"; return 1; }
  ! has_fixed "$2" 'selector unavailable' || { note "fell back to legacy"; return 1; }
  has_line "$2" 'NCCL_PROTO=<unset>' || { note "NCCL_PROTO was not unset"; return 1; }
  has_line "$2" 'VLLM_SKIP_P2P_CHECK=1' || { note "VLLM_SKIP_P2P_CHECK"; return 1; }
}

# --- golden: selector off == the base release script -----------------------------------------------------------
# With the selector off (AUTO_TOPO unset or 0) the environment and argv serve hands to vllm must equal those of the
# serve script as released without the selector (frozen copy: tests/topo/shell/legacy-serve-v2.2.sh), which has no test hook.
# Both scripts run in ONE sandbox, as the same path ($0, and so $HERE, identical), with identical caller environments,
# up to the real exec; the stub vllm prints its argv and `env | sort`. Everything the two runs print (the scripts' own
# lines, the argv and every environment line) must match, except environment lines for the variable names listed in
# GOLDEN_INERT_EXTRA, which the new script alone may add. The list is empty: with the selector off, serve exports
# nothing the base script does not.
GOLDEN_BASE="$REPO_ROOT/tests/topo/shell/legacy-serve-v2.2.sh"
GOLDEN_INERT_EXTRA=()
SKIP=0

write_golden_vllm() {
  cat > "$1/venv-v2.2/bin/vllm" <<'VLLMEOF'
#!/usr/bin/env bash
printf 'GOLDEN_ARGV:'; for a in "$@"; do printf ' [%s]' "$a"; done; printf '\n'
env | sort | sed 's/^/GOLDEN_ENV: /'
VLLMEOF
  chmod +x "$1/venv-v2.2/bin/vllm"
}

# $1 = sandbox, $2 = script to run as $sandbox/scripts/serve-v2.2.sh, $3 = setup fn; prints everything the run prints.
golden_run() {
  local sandbox="$1" script="$2" setup_fn="$3"
  cp "$script" "$sandbox/scripts/serve-v2.2.sh"
  (
    set -u
    cd "$sandbox" || exit 99
    unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER VLLM_SKIP_P2P_CHECK NCCL_P2P_LEVEL NCCL_PROTO PLE_HOME CACHE_ROOT \
          AUTO_TOPO AUTO_TOPO_NCCL AUTO_TOPO_BIND LANE_GPUS LANE_CPUS TOPO_TIMEOUT TOPO_FIXTURE VLLM_CACHE_ROOT \
          SERVE_PRINT_ENV_AND_EXIT TOPO_LAUNCH_PREFIX TOPO_DECISION_JSON STUB_REAL_PY
    export PATH="$sandbox/bin:$PATH"
    export TREE="$sandbox/vllm-v2.2" VENV="$sandbox/venv-v2.2" STUB_PURELIB="$sandbox/venv-v2.2"
    export SCALES="$sandbox/scales/qsa_kv_scales_262k.json"
    export STUB_TOPO_MODE=marker STUB_TOPO_MARKER="$sandbox/marker"
    "$setup_fn" "$sandbox"
    timeout 40 bash "$sandbox/scripts/serve-v2.2.sh" "$sandbox/ckpt" --extra-arg 'two words' 2>&1
    printf 'GOLDEN_EXIT:%d\n' "$?"
  )
}

# drop the GOLDEN_ENV lines of the allowed inert extras from stdin
golden_filter() {
  local n pat=''
  for n in "${GOLDEN_INERT_EXTRA[@]}"; do pat="$pat|$n"; done
  if [ -n "$pat" ]; then grep -Ev "^GOLDEN_ENV: (${pat#|})=" || true; else cat; fi
}

run_golden() {
  local name="$1" setup_fn="$2" sandbox base_out new_out
  if [ ! -f "$GOLDEN_BASE" ]; then
    printf 'SKIP  %s (no %s)\n' "$name" "$GOLDEN_BASE"
    SKIP=$((SKIP+1))
    return
  fi
  sandbox="$(make_sandbox)" || { printf 'FAIL  %s (sandbox setup failed)\n' "$name"; FAIL=$((FAIL+1)); FAILED_CASES+=("$name"); return; }
  write_stub_topo_select "$sandbox"
  write_golden_vllm "$sandbox"
  cp "$GOLDEN_BASE" "$sandbox/base-serve.sh"
  cp "$REAL_SERVE" "$sandbox/new-serve.sh"

  base_out="$(golden_run "$sandbox" "$sandbox/base-serve.sh" "$setup_fn")"
  new_out="$(golden_run "$sandbox" "$sandbox/new-serve.sh" "$setup_fn")"

  local ok=1 why=""
  if [ -e "$sandbox/marker" ]; then ok=0; why="the selector was invoked"; fi
  if ! has_line "$new_out" 'GOLDEN_EXIT:0' || ! has_line "$base_out" 'GOLDEN_EXIT:0'; then ok=0; why="a run did not exit 0"; fi
  if ! has_fixed "$new_out" 'GOLDEN_ARGV: [serve]' || ! has_fixed "$new_out" 'GOLDEN_ENV: PATH='; then
    ok=0; why="the stub vllm never ran (nothing to compare)"
  fi
  # (the diff below catches any extra variable; this one just gets a clearer message)
  if has_fixed "$new_out" 'GOLDEN_ENV: TOPO_CALLER_'; then ok=0; why="TOPO_CALLER_* snapshot exported with the selector off"; fi
  local d
  d="$(diff <(printf '%s\n' "$base_out") <(printf '%s\n' "$new_out" | golden_filter))" || { ok=0; why="${why:-output differs from the base script}"; }

  if [ "$ok" = 1 ]; then
    printf 'PASS  %s\n' "$name"
    PASS=$((PASS+1))
  else
    printf 'FAIL  %s\n' "$name"
    {
      printf -- '  ------ case %s: %s ------\n' "$name" "$why"
      printf '%s\n' "$d" | sed 's/^/  | /'
      printf -- '  ---------------------------------------\n'
    } >&2
    FAIL=$((FAIL+1))
    FAILED_CASES+=("$name")
  fi
  rm -rf "$sandbox"
}

golden_setup_unset() { :; }
golden_setup_zero() { export AUTO_TOPO=0; }
golden_setup_caller() {
  # caller values the base script keeps (CUDA_VISIBLE_DEVICES, PLE_HOME) or overwrites (the NCCL / P2P ones), plus a
  # stray TOPO_LAUNCH_PREFIX: the base passes it through to vllm untouched and never runs it; so must serve.
  export CUDA_VISIBLE_DEVICES=1,0,3,2 NCCL_PROTO=Simple NCCL_P2P_LEVEL=PIX VLLM_SKIP_P2P_CHECK=0 \
         CUDA_DEVICE_ORDER=FASTEST_FIRST PLE_HOME=1 TOPO_LAUNCH_PREFIX='touch should-not-run --' TOPO_DECISION_JSON=x
}
golden_setup_empty_cvd() { export CUDA_VISIBLE_DEVICES=; }

# --- run -----------------------------------------------------------------------------------------------------

command -v timeout >/dev/null || { note "run.sh needs GNU coreutils' timeout on PATH"; exit 2; }

run_case "(u) AUTO_TOPO unset (default) -> selector off, legacy, never invoked"   setup_u check_u
run_case "(u2) AUTO_TOPO=yes -> note, selector off, legacy"                        setup_u2 check_u2
run_golden "(g0) golden: AUTO_TOPO unset, env + argv == base script"             golden_setup_unset
run_golden "(g0-zero) golden: AUTO_TOPO=0, env + argv == base script"            golden_setup_zero
run_golden "(g0-caller) golden: caller topology vars set, == base script"        golden_setup_caller
run_golden "(g0-empty) golden: CUDA_VISIBLE_DEVICES= (empty), == base script"    golden_setup_empty_cvd
run_case "(a) AUTO_TOPO=0 keeps legacy, selector not invoked, empty hook prefix"  setup_a check_a
run_case "(b) selector reorder + numactl prefix applied"                          setup_b check_b
run_case "(c) selector exit 5 -> legacy + message"                                setup_c check_c
run_case "(d) env file is garbage -> not sourced, legacy"                         setup_d check_d
run_case "(d2) env file has an unterminated quote -> not sourced, legacy"         setup_d2 check_d
run_case "(d3) env file is the header only -> legacy"                             setup_d3 check_d
run_case "(d4) env file has an extra line -> not sourced, legacy"                 setup_d4 check_d
run_case "(d5) env file prefix is not a numactl prefix -> legacy"                 setup_d5 check_d
run_case "(e) selector hangs -> watchdog -> legacy + message"                     setup_e check_e
run_case "(e2) selector hangs ignoring SIGTERM -> killed -> legacy"               setup_e2 check_e2
run_case "(f) selector exit 3 -> serve exits 1 with message"                      setup_f check_f
run_case "(f2) AUTO_TOPO=strict + selector exit 5 -> serve exits 1"               setup_f2 check_f2
run_case "(f3) AUTO_TOPO=strict + unusable env file -> serve exits 1"             setup_f3 check_f2
run_case "(g) caller NCCL_PROTO/CUDA_VISIBLE_DEVICES snapshot reaches selector"   setup_g check_g
run_case "(g2) fallback keeps the caller's CUDA_VISIBLE_DEVICES"                  setup_g2 check_g2
run_case "(h) selector runs with \$VENV/bin/python, not PATH python3"             setup_h check_h
run_case "(i) hook unset -> real exec, no prefix"                                 setup_i check_i
run_case "(i2) hook unset -> real exec through the numactl prefix"                setup_i2 check_i2
run_case "(t1) TOPO_TIMEOUT=2.5 -> note, selector gets 20"                        setup_t1 check_t_default
run_case "(t2) TOPO_TIMEOUT=08 -> note, selector gets 20"                         setup_t2 check_t_default
run_case "(t3) TOPO_TIMEOUT=7 -> passed through"                                  setup_t3 check_t3
run_case "(m1) mktemp fails -> legacy + message"                                  setup_m1 check_m1
run_case "(m2) mktemp fails under AUTO_TOPO=strict -> serve exits 1"              setup_m2 check_m2
run_case "(v2) whitespace CUDA_VISIBLE_DEVICES + selector failure -> 0,1,2,3"     setup_v2 check_v2
if [ -n "$REAL_PY" ]; then
  run_case "(v1) real selector, CUDA_VISIBLE_DEVICES= (empty) -> today's values"  setup_v1 check_v1
  run_case "(j) real selector, reference fixture -> today's values exactly"       setup_j check_j
  run_case "(j2) real selector, CACHE_ROOT unset -> diagnostics in default cache" setup_j2 check_j2
  run_case "(k) real selector, NVLink fixture -> reordered UUID lane"             setup_k check_k
  run_case "(n) real selector, no P2P + AUTO_TOPO_NCCL=default -> NCCL_PROTO unset" setup_n check_n
else
  note "note: no python3 on PATH, skipping the real-selector cases (v1, j, j2, k, n)"
fi

echo "----------------------------------------------------------------"
echo "summary: $PASS passed, $FAIL failed, $SKIP skipped (of $((PASS + FAIL + SKIP)))"
if [ "$FAIL" -gt 0 ]; then
  printf 'failed: %s\n' "${FAILED_CASES[*]}"
  exit 1
fi
exit 0
