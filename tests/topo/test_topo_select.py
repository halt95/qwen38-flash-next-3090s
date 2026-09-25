"""
Unit tests for scripts/topo_select.py, written against the documented behaviour
(docs/topology-selector.md) rather than the implementation. Most cases run the script as a
subprocess with TOPO_FIXTURE + TOPO_CALLER_*/control env and parse the emitted env file and the
diagnostics JSON. The NVML cases run the real-hardware adapter instead, against the fake pynvml
module in fake_pynvml/ and a fake sysfs tree (TOPO_SYSFS_ROOT).

Works two ways:
  - pytest:      python3 -m pytest tests/topo -q
  - plain script: python3 tests/topo/test_topo_select.py   (pytest need not be installed)

If scripts/topo_select.py does not exist yet, every test SKIPs (not fails) and both
runners exit 0.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
FAKE_PYNVML = HERE / "fake_pynvml"
REPO_ROOT = HERE.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "topo_select.py"

CALLER_VARS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SKIP_P2P_CHECK",
    "NCCL_P2P_LEVEL", "NCCL_PROTO", "PLE_HOME",
)

# Everything topo_select.py itself might read, stripped from the child's environment
# before each run so no ambient value leaks in; the test then adds back exactly what
# the case wants via TOPO_CALLER_* / control vars.
_STRIP_PREFIXES = ("CUDA_", "NCCL_", "VLLM_", "AUTO_TOPO", "LANE_", "TOPO_")
_STRIP_EXACT = {"PLE_HOME", "CACHE_ROOT"}


class SkipTest(Exception):
    """Raised to mark a test as skipped (script not present yet). Recognized by both
    the plain runner (by class name, see below) and, when pytest is installed, is
    converted to a real pytest skip via _raise_skip()."""


try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None


def _raise_skip(msg: str):
    # Only use pytest.skip() (raises a BaseException subclass) when actually running
    # under pytest -- PYTEST_CURRENT_TEST is set by pytest for the duration of each
    # test. Otherwise raise our own Exception subclass so the plain runner's
    # `except Exception` below can catch it.
    if pytest is not None and "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(msg)
    raise SkipTest(msg)


def _clean_env() -> dict:
    env = {}
    for k, v in os.environ.items():
        if k in _STRIP_EXACT or any(k.startswith(p) for p in _STRIP_PREFIXES):
            continue
        env[k] = v
    return env


def caller_env(cvd=None, **overrides) -> dict:
    """Build TOPO_CALLER_SET_V / TOPO_CALLER_V lines (the snapshot serve-v2.2.sh takes).
    cvd=<str> sets the caller's CUDA_VISIBLE_DEVICES; omit/None = caller left it unset.
    Same for any of the other CALLER_VARS passed as kwargs (e.g. NCCL_PROTO="Simple")."""
    values = {"CUDA_VISIBLE_DEVICES": cvd}
    values.update(overrides)
    env = {}
    for v in CALLER_VARS:
        val = values.get(v)
        if val is not None:
            env[f"TOPO_CALLER_SET_{v}"] = "1"
            env[f"TOPO_CALLER_{v}"] = val
        else:
            env[f"TOPO_CALLER_SET_{v}"] = "0"
    return env


_HEADER_RE = re.compile(r"^# topo_select v1 status=(\S+) reason=(\S+)\s*$")


def _parse_env_file(lines):
    status = reason = None
    if lines:
        m = _HEADER_RE.match(lines[0])
        if m:
            status, reason = m.group(1), m.group(2)
    env = {}
    for line in lines[1:]:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("unset "):
            env[line[len("unset "):].strip()] = None
        elif line.startswith("export "):
            k, _, v = line[len("export "):].partition("=")
            toks = shlex.split(v)
            env[k] = toks[0] if toks else ""
    return status, reason, env


def run_topo(fixture, caller=None, control=None, extra_args=None, timeout=15, pass_json_dir=True):
    """Run scripts/topo_select.py against a fixture (fixture=None: no TOPO_FIXTURE, so the NVML
    adapter runs; the caller supplies the fake pynvml/sysfs env through `control`). Returns a dict
    with returncode, stdout, stderr, status, reason, env (parsed dict), json (diagnostics dict or
    None), env_out/json_dir (paths, for tests that check file existence directly)."""
    if not SCRIPT.exists():
        _raise_skip(f"scripts/topo_select.py not present yet at {SCRIPT}")

    caller = caller or {}
    control = dict(control or {})

    with tempfile.TemporaryDirectory(prefix="topo_select_test_") as td:
        env_out = os.path.join(td, "env.sh")
        json_dir = control.pop("_json_dir", None) or os.path.join(td, "json")
        try:
            os.makedirs(json_dir, exist_ok=True)
        except OSError:
            pass  # e.g. json_dir deliberately points at an existing file (unwritable-dir test)

        env = _clean_env()
        if fixture is not None:
            env["TOPO_FIXTURE"] = fixture if os.path.isabs(fixture) else str(FIXTURES / fixture)
        env.update(caller_env(**caller))
        env.update(control)

        argv = [sys.executable, str(SCRIPT), "--env-out", env_out]
        if pass_json_dir:
            argv += ["--json-dir", json_dir]
        if extra_args:
            argv += list(extra_args)

        proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)

        lines = None
        status = reason = None
        parsed_env = {}
        if os.path.exists(env_out):
            lines = Path(env_out).read_text(encoding="utf-8").splitlines()
            status, reason, parsed_env = _parse_env_file(lines)

        json_files = sorted(Path(json_dir).glob("*.json")) if os.path.isdir(json_dir) else []
        diag = None
        json_path = None
        if json_files:
            json_path = str(json_files[-1])
            diag = json.loads(json_files[-1].read_text(encoding="utf-8"))

        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "env_out": env_out,
            "env_out_exists": os.path.exists(env_out),
            "env_lines": lines,
            "status": status,
            "reason": reason,
            "env": parsed_env,
            "json_dir": json_dir,
            "json_path": json_path,
            "json": diag,
        }


def uuids_of(fixture_name):
    data = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    return {g["bus_id"]: g["uuid"] for g in data["gpus"]}, [g["uuid"] for g in data["gpus"]]


# Fixture-derived constants used by several tests -----------------------------------
_ref_uuids = json.loads((FIXTURES / "reference.json").read_text())["gpus"]
REF_UUIDS = [g["uuid"] for g in _ref_uuids]

_nvl = json.loads((FIXTURES / "nvlink_pairs.json").read_text())["gpus"]
NVL_UUIDS = [g["uuid"] for g in _nvl]

_del = json.loads((FIXTURES / "delaware.json").read_text())["gpus"]
DEL_UUIDS = [g["uuid"] for g in _del]  # index i == GPU i (bus-id ascending == index order)


# ------------------------------------------------------------------------------------
# 1-2: reference fingerprint
# ------------------------------------------------------------------------------------

def test_reference_cvd_unset_is_byte_identical_legacy():
    r = run_topo("reference.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok" and r["reason"] == "reference", r["env_lines"]
    e = r["env"]
    assert e["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert e["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert e["VLLM_SKIP_P2P_CHECK"] == "1"
    assert e["NCCL_P2P_LEVEL"] == "SYS"
    assert e["NCCL_PROTO"] == "LL"
    assert e["TOPO_LAUNCH_PREFIX"] == ""


def test_reference_env_file_is_exactly_the_8_line_format():
    r = run_topo("reference.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["env_lines"] == [
        "# topo_select v1 status=ok reason=reference",
        "export CUDA_VISIBLE_DEVICES=0,1,2,3",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "export VLLM_SKIP_P2P_CHECK=1",
        "export NCCL_P2P_LEVEL=SYS",
        "export NCCL_PROTO=LL",
        "export TOPO_LAUNCH_PREFIX=''",
        "export TOPO_DECISION_JSON=" + shlex.quote(r["json_path"] and os.path.abspath(r["json_path"])),
    ], r["env_lines"]


def test_reference_numa_minus1_single_node_resolves_to_the_only_node():
    # numa=-1 on every GPU, but cpu.nodes has exactly one key (the reference host reports NUMA -1
    # from sysfs while having only one system node) -> treated as node 0, so the reference
    # fingerprint still fires.
    r = run_topo("reference.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok" and r["reason"] == "reference"
    assert [g["numa"] for g in r["json"]["lane"]] == [0, 0, 0, 0], r["json"]["lane"]


def test_reference_numa_minus1_two_nodes_is_not_reference_fingerprint():
    # Companion case: same GPUs/pairs, but cpu.nodes has TWO keys while every GPU still reports
    # numa=-1. The single-node exception does not apply here, so NUMA stays unknown and
    # this is NOT the reference fingerprint -- even though every pair has P2P OK, so the
    # non-reference P2P-present path fires (VLLM_SKIP_P2P_CHECK=0).
    r = run_topo("reference_numa_unknown_two_nodes.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["reason"] != "reference", r["env_lines"]
    e = r["env"]
    assert e["VLLM_SKIP_P2P_CHECK"] == "0"
    assert e["NCCL_P2P_LEVEL"] == "SYS"
    assert e["NCCL_PROTO"] == "LL"


def test_reference_caller_cvd_numeric_0123_same_as_unset():
    r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok" and r["reason"] == "reference"
    e = r["env"]
    assert e["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert e["VLLM_SKIP_P2P_CHECK"] == "1"
    assert e["NCCL_PROTO"] == "LL"
    assert e["TOPO_LAUNCH_PREFIX"] == ""


# ------------------------------------------------------------------------------------
# 3-4: NVLink reordering
# ------------------------------------------------------------------------------------

def test_nvlink_pairs_cvd_unset_reorders_to_bus_order_0_2_1_3():
    r = run_topo("nvlink_pairs.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["reason"] == "reordered", r["env_lines"]
    expected = ",".join([NVL_UUIDS[0], NVL_UUIDS[2], NVL_UUIDS[1], NVL_UUIDS[3]])
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == expected


def test_nvlink_pairs_caller_cvd_0213_keeps_order_numeric():
    r = run_topo("nvlink_pairs.json", caller={"CUDA_VISIBLE_DEVICES": "0,2,1,3"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok"
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == "0,2,1,3"


# ------------------------------------------------------------------------------------
# 5: >4 visible, unset CVD
# ------------------------------------------------------------------------------------

def test_delaware_cvd_unset_more_than_4_visible_falls_back_to_legacy_with_warning():
    r = run_topo("delaware.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "more_than_4_visible"
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert r["env"]["VLLM_SKIP_P2P_CHECK"] == "1"
    warned_stderr = "more_than_4" in r["stderr"] or "CUDA_VISIBLE_DEVICES" in r["stderr"]
    warned_json = bool(r["json"] and r["json"].get("warnings"))
    assert warned_stderr or warned_json, (r["stderr"], r["json"])


def test_delaware_cvd_unset_strict_mode_refuses_exit_3():
    r = run_topo("delaware.json", control={"AUTO_TOPO": "strict"})
    assert r["returncode"] == 3, (r["returncode"], r["stdout"], r["stderr"])
    assert r["stderr"].strip() != ""


# ------------------------------------------------------------------------------------
# 6-7: Delaware caller-supplied lane
# ------------------------------------------------------------------------------------

def test_delaware_caller_cvd_4567_kept_pairs_45_67_best_score():
    r = run_topo("delaware.json", caller={"CUDA_VISIBLE_DEVICES": "4,5,6,7"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok"
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    js = r["json"]
    assert js is not None
    assert js["chosen_score"] == js["input_score"], js
    assert r["reason"] == "kept", r["env_lines"]


def test_delaware_caller_cvd_4657_reorders_to_pairs_45_67():
    r = run_topo("delaware.json", caller={"CUDA_VISIBLE_DEVICES": "4,6,5,7"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["reason"] == "reordered", r["env_lines"]
    expected = ",".join([DEL_UUIDS[4], DEL_UUIDS[5], DEL_UUIDS[6], DEL_UUIDS[7]])
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == expected


# ------------------------------------------------------------------------------------
# 8: LANE_GPUS=auto
# ------------------------------------------------------------------------------------

def test_delaware_lane_gpus_auto_picks_node1_idle_quad_when_all_idle():
    r = run_topo("delaware.json", control={"LANE_GPUS": "auto"})
    assert r["returncode"] == 0, r["stderr"]
    got = set(r["env"]["CUDA_VISIBLE_DEVICES"].split(","))
    expected = {DEL_UUIDS[4], DEL_UUIDS[5], DEL_UUIDS[6], DEL_UUIDS[7]}
    assert got == expected, got


def test_delaware_lane_gpus_auto_prefers_single_node_when_gpu6_busy():
    # With GPU6 busy, node 1 still has four idle GPUs {3,4,5,7}. A single-NUMA-node lane ranks
    # first, so the pick stays on node 1 even though {0,1,4,5}
    # would score higher on link class alone (two PIX pairs, but spanning both sockets).
    r = run_topo("delaware_busy.json", control={"LANE_GPUS": "auto"})
    assert r["returncode"] == 0, r["stderr"]
    got = set(r["env"]["CUDA_VISIBLE_DEVICES"].split(","))
    expected = {DEL_UUIDS[3], DEL_UUIDS[4], DEL_UUIDS[5], DEL_UUIDS[7]}
    assert got == expected, got
    assert DEL_UUIDS[6] not in got


# ------------------------------------------------------------------------------------
# 9-10: binding
# ------------------------------------------------------------------------------------

def test_oman_bind_skips_because_lane_spans_nodes():
    r = run_topo("oman.json", control={"AUTO_TOPO_BIND": "1"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["env"]["TOPO_LAUNCH_PREFIX"] == ""
    js = r["json"]
    assert js is not None
    assert js["binding"]["enabled"] is False
    assert js["binding"]["skipped_reason"] == "lane_spans_nodes"


def test_delaware_bind_skipped_for_node1_memory():
    r = run_topo(
        "delaware.json",
        caller={"CUDA_VISIBLE_DEVICES": "4,5,6,7"},
        control={"AUTO_TOPO_BIND": "1"},
    )
    assert r["returncode"] == 0, r["stderr"]
    assert r["env"]["TOPO_LAUNCH_PREFIX"] == ""
    js = r["json"]
    assert js is not None
    assert js["binding"]["enabled"] is False
    assert "mem" in js["binding"]["skipped_reason"].lower(), js["binding"]


def test_delaware_bind_ok_emits_numactl_prefix():
    r = run_topo(
        "delaware_bind_ok.json",
        caller={"CUDA_VISIBLE_DEVICES": "4,5,6,7"},
        control={"AUTO_TOPO_BIND": "1"},
    )
    assert r["returncode"] == 0, r["stderr"]
    prefix = r["env"]["TOPO_LAUNCH_PREFIX"]
    assert prefix == "numactl --physcpubind=18-35,54-71 --preferred=1 --", prefix
    js = r["json"]
    assert js["binding"]["enabled"] is True


def test_delaware_bind_ok_lane_cpus_narrows_the_cpulist():
    r = run_topo(
        "delaware_bind_ok.json",
        caller={"CUDA_VISIBLE_DEVICES": "4,5,6,7"},
        control={"AUTO_TOPO_BIND": "1", "LANE_CPUS": "18-25"},
    )
    assert r["returncode"] == 0, r["stderr"]
    prefix = r["env"]["TOPO_LAUNCH_PREFIX"]
    assert prefix == "numactl --physcpubind=18-25 --preferred=1 --", prefix


def test_delaware_bind_narrow_affinity_restricts_the_cpulist():
    r = run_topo(
        "delaware_bind_narrow_affinity.json",
        caller={"CUDA_VISIBLE_DEVICES": "4,5,6,7"},
        control={"AUTO_TOPO_BIND": "1"},
    )
    assert r["returncode"] == 0, r["stderr"]
    prefix = r["env"]["TOPO_LAUNCH_PREFIX"]
    assert prefix == "numactl --physcpubind=18-35 --preferred=1 --", prefix


# ------------------------------------------------------------------------------------
# 11: Bulgaria, no P2P anywhere
# ------------------------------------------------------------------------------------

def test_bulgaria_no_tp_pair_p2p_defaults_to_skip1_ll():
    r = run_topo("bulgaria.json")
    assert r["returncode"] == 0, r["stderr"]
    e = r["env"]
    assert e["VLLM_SKIP_P2P_CHECK"] == "1"
    assert e["NCCL_P2P_LEVEL"] == "SYS"
    assert e["NCCL_PROTO"] == "LL"


def test_bulgaria_auto_topo_nccl_default_unsets_nccl_proto():
    r = run_topo("bulgaria.json", control={"AUTO_TOPO_NCCL": "default"})
    assert r["returncode"] == 0, r["stderr"]
    e = r["env"]
    assert e["VLLM_SKIP_P2P_CHECK"] == "1"
    assert "NCCL_PROTO" in e and e["NCCL_PROTO"] is None, e  # "unset NCCL_PROTO" line


# ------------------------------------------------------------------------------------
# 12: reference-like but p2p OK on a non-reference (two-node) topology
# ------------------------------------------------------------------------------------

def test_split_2_2_p2p_ok_non_reference_topology_unsets_skip_check():
    r = run_topo("split_2_2.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["reason"] != "reference"
    e = r["env"]
    assert e["VLLM_SKIP_P2P_CHECK"] == "0"
    assert e["NCCL_P2P_LEVEL"] == "SYS"
    assert e["NCCL_PROTO"] == "LL"


# ------------------------------------------------------------------------------------
# 13: caller-set NCCL_PROTO wins and is reported with caller provenance
# ------------------------------------------------------------------------------------

def test_caller_nccl_proto_simple_is_emitted_verbatim_with_caller_provenance():
    r = run_topo("reference.json", caller={"NCCL_PROTO": "Simple"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["env"]["NCCL_PROTO"] == "Simple"
    js = r["json"]
    assert js is not None, "no diagnostics JSON written"
    assert js["env"]["NCCL_PROTO"] == {"value": "Simple", "provenance": "caller"}, js["env"]


# ------------------------------------------------------------------------------------
# 14: CVD resolution edge cases
# ------------------------------------------------------------------------------------

def test_cvd_duplicate_tokens_falls_back_to_legacy():
    r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": "0,0,1,2"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "cvd_duplicate"


def test_cvd_invalid_token_falls_back_to_legacy():
    r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": "0,1,2,notagpu"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "cvd_unresolved"


def test_cvd_mig_token_falls_back_to_legacy():
    r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": "MIG-11111111-1111-1111-1111-111111111111,0,1,2"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "cvd_unresolved"


def test_cvd_three_devices_falls_back_to_legacy():
    r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": "0,1,2"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "lane_not_4"


def test_cvd_unique_uuid_prefix_resolves_mixed_with_numeric_tokens():
    prefix = REF_UUIDS[0][: len("GPU-aaaaaaaa")]
    r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": f"{prefix},1,2,3"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok", (r["status"], r["reason"], r["env_lines"])
    got = set(r["env"]["CUDA_VISIBLE_DEVICES"].split(","))
    assert got == set(REF_UUIDS), got


# ------------------------------------------------------------------------------------
# 15: unknown NUMA, order still chosen from link classes
# ------------------------------------------------------------------------------------

def test_numa_unknown_orders_by_link_class_not_numa():
    # 0-2 and 1-3 share a PCIe switch (class 4), every other pair only a host bridge (class 3):
    # the selector must pair them as TP pairs although no GPU's NUMA node is known.
    uuids = [g["uuid"] for g in json.loads((FIXTURES / "numa_unknown.json").read_text())["gpus"]]
    r = run_topo("numa_unknown.json", control={"AUTO_TOPO_BIND": "1"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok" and r["reason"] == "reordered", r["env_lines"]
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == ",".join([uuids[0], uuids[2], uuids[1], uuids[3]])
    assert r["env"]["VLLM_SKIP_P2P_CHECK"] == "0"
    assert r["json"]["binding"]["skipped_reason"] == "numa_unknown", r["json"]["binding"]
    assert r["env"]["TOPO_LAUNCH_PREFIX"] == ""


# ------------------------------------------------------------------------------------
# 16: NVML faults
# ------------------------------------------------------------------------------------

def test_faults_nvml_error_falls_back_to_legacy_exit_0():
    r = run_topo("faults_nvml_error.json")
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "nvml_unavailable"


def test_faults_nvml_timeout_exits_4_within_timeout_budget():
    t0 = time.monotonic()
    r = run_topo("faults_nvml_timeout.json", control={"TOPO_TIMEOUT": "2"}, timeout=10)
    elapsed = time.monotonic() - t0
    assert r["returncode"] == 4, (r["returncode"], r["stdout"], r["stderr"])
    assert elapsed < 8, elapsed


# ------------------------------------------------------------------------------------
# 17: --dry-run writes nothing
# ------------------------------------------------------------------------------------

def test_dry_run_writes_no_files():
    r = run_topo("reference.json", extra_args=["--dry-run"])
    assert r["returncode"] == 0, r["stderr"]
    assert r["env_out_exists"] is False
    assert r["json_path"] is None
    assert r["stdout"].strip() != ""  # dry-run prints the summary + env content to stdout


# ------------------------------------------------------------------------------------
# 18: diagnostics JSON shape
# ------------------------------------------------------------------------------------

def test_diagnostics_json_has_documented_keys_and_correct_ple_home_uuid():
    r = run_topo("reference.json")
    assert r["returncode"] == 0, r["stderr"]
    js = r["json"]
    assert js is not None, "no diagnostics JSON written"
    required = {
        "version", "status", "reason", "lane", "input_order", "link_matrix", "p2p_matrix",
        "candidates", "chosen_score", "input_score", "env", "ple_home_uuid",
        "p2p_probe_cache_hint", "binding", "warnings", "facts_source",
    }
    missing = required - set(js.keys())
    assert not missing, missing
    assert js["facts_source"] == "fixture"
    assert js["ple_home_uuid"] == js["lane"][0]["uuid"]


# ------------------------------------------------------------------------------------
# 19: unwritable json-dir still exits 0 with the env file written
# ------------------------------------------------------------------------------------

def test_unwritable_json_dir_still_exits_0_and_writes_env_file():
    with tempfile.TemporaryDirectory(prefix="topo_select_test_") as td:
        blocked = os.path.join(td, "not_a_dir")
        with open(blocked, "w", encoding="utf-8") as f:
            f.write("occupied\n")  # a file, not a directory: json-dir can't be used
        r = run_topo("reference.json", control={"_json_dir": blocked})
        assert r["returncode"] == 0, (r["returncode"], r["stderr"])
        assert r["env_out_exists"] is True
        assert r["status"] == "ok" and r["reason"] == "reference"


# ------------------------------------------------------------------------------------
# helpers: fixture variants and the module itself
# ------------------------------------------------------------------------------------

def run_variant(base, mutate, **kw):
    """Run against a copy of fixtures/<base> changed by mutate(data) (one tweak per test)."""
    data = copy.deepcopy(json.loads((FIXTURES / base).read_text(encoding="utf-8")))
    mutate(data)
    with tempfile.TemporaryDirectory(prefix="topo_select_fixture_") as td:
        path = os.path.join(td, base)
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        return run_topo(path, **kw)


def load_module():
    if not SCRIPT.exists():
        _raise_skip(f"scripts/topo_select.py not present yet at {SCRIPT}")
    spec = importlib.util.spec_from_file_location("topo_select_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclasses look their module up while the class is created
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------
# 20: set-but-empty / whitespace-only CUDA_VISIBLE_DEVICES behaves as unset
# ------------------------------------------------------------------------------------

def test_cvd_empty_string_behaves_as_unset():
    for value in ("", "   "):
        r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": value})
        assert r["returncode"] == 0, r["stderr"]
        assert r["status"] == "ok" and r["reason"] == "reference", (value, r["env_lines"])
        assert r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3", (value, r["env"])
        assert r["json"]["env"]["CUDA_VISIBLE_DEVICES"]["provenance"] == "legacy"


def test_cvd_empty_string_legacy_fallback_is_0123():
    # legacy paths decided before any lane resolution must also treat '' as unset
    r = run_topo("reference.json", caller={"CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "device_order_not_pci_bus_id"
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3", r["env"]
    r = run_topo("delaware.json", caller={"CUDA_VISIBLE_DEVICES": " "})
    assert r["reason"] == "more_than_4_visible" and r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3", r["env"]


# ------------------------------------------------------------------------------------
# 21: strict mode is checked before LANE_GPUS=auto
# ------------------------------------------------------------------------------------

def test_strict_with_lane_gpus_auto_still_refuses_exit_3():
    r = run_topo("delaware.json", control={"AUTO_TOPO": "strict", "LANE_GPUS": "auto"})
    assert r["returncode"] == 3, (r["returncode"], r["stderr"])
    assert "strict" in r["stderr"]


# ------------------------------------------------------------------------------------
# 22: one P2P direction unknown -> the pair is unknown, not "not supported"
# ------------------------------------------------------------------------------------

def test_p2p_one_direction_missing_is_unknown():
    mod = load_module()
    assert mod.PairFacts("NODE", "CNS", None, False).p2p_ok is None
    assert mod.PairFacts("NODE", None, "OK", False).p2p_ok is None
    assert mod.PairFacts("NODE", "CNS", "OK", False).p2p_ok is False
    assert mod.PairFacts("NODE", "OK", "OK", False).p2p_ok is True

    def drop_write(d):
        d["pairs"]["0-1"] = {"ancestor": "NODE", "p2p_read": "CNS", "nvlink": False}
    r = run_variant("reference.json", drop_write)
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "facts_unknown", r["env_lines"]
    assert r["env"]["VLLM_SKIP_P2P_CHECK"] == "1" and r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


# ------------------------------------------------------------------------------------
# 23: TOPO_TIMEOUT validation and thread failures
# ------------------------------------------------------------------------------------

def test_topo_timeout_rejects_non_finite_and_out_of_range_values():
    for value in ("inf", "nan", "-1", "0", "abc", "1e308"):
        r = run_topo("reference.json", control={"TOPO_TIMEOUT": value})
        assert r["returncode"] == 2, (value, r["returncode"], r["stderr"])
        assert "TOPO_TIMEOUT" in r["stderr"], (value, r["stderr"])
    r = run_topo("reference.json", control={"TOPO_TIMEOUT": "2.5"})
    assert r["returncode"] == 0, r["stderr"]


def _main_with_thread(thread_cls):
    """Run main() in-process with a replacement threading.Thread; returns the exit code."""
    mod = load_module()
    mod.threading = types.SimpleNamespace(Thread=thread_cls, TIMEOUT_MAX=threading.TIMEOUT_MAX)
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(_clean_env())
        os.environ["TOPO_FIXTURE"] = str(FIXTURES / "reference.json")
        with tempfile.TemporaryDirectory(prefix="topo_select_test_") as td:
            return mod.main(["--env-out", os.path.join(td, "env.sh"), "--json-dir", td, "--quiet"])
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_thread_start_failure_exits_5():
    class NoStart(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")
    assert _main_with_thread(NoStart) == 5


def test_thread_join_failure_exits_5():
    class BadJoin(threading.Thread):
        def join(self, timeout=None):
            raise OverflowError("timeout value is too large")
    assert _main_with_thread(BadJoin) == 5


# ------------------------------------------------------------------------------------
# 24: binding preconditions (one fixture tweak each)
# ------------------------------------------------------------------------------------

def _bind(mutate=None, **control):
    ctl = {"AUTO_TOPO_BIND": "1"}
    ctl.update(control)
    r = run_variant("delaware_bind_ok.json", mutate or (lambda d: None),
                    caller={"CUDA_VISIBLE_DEVICES": "4,5,6,7"}, control=ctl)
    assert r["returncode"] == 0, r["stderr"]
    return r


def test_bind_lane_cpus_set_but_empty_skips_binding():
    r = _bind(LANE_CPUS="")
    assert r["env"]["TOPO_LAUNCH_PREFIX"] == ""
    assert r["json"]["binding"]["skipped_reason"] == "too_few_cpus(0)", r["json"]["binding"]


def test_bind_too_few_cpus():
    r = _bind(LANE_CPUS="18-20")
    assert r["json"]["binding"]["skipped_reason"] == "too_few_cpus(3)", r["json"]["binding"]


def test_bind_lane_cpus_invalid():
    r = _bind(LANE_CPUS="eighteen")
    assert r["json"]["binding"]["skipped_reason"] == "lane_cpus_invalid", r["json"]["binding"]


def test_bind_numactl_missing():
    r = _bind(lambda d: d["numactl"].update(present=False))
    assert r["json"]["binding"]["skipped_reason"] == "numactl_missing", r["json"]["binding"]


def test_bind_affinity_unknown():
    r = _bind(lambda d: d["cpu"].update(affinity=None))
    assert r["json"]["binding"]["skipped_reason"] == "affinity_unknown", r["json"]["binding"]


def test_bind_node_not_in_mems_allowed():
    r = _bind(lambda d: d["cpu"].update(mems_allowed="0"))
    assert r["json"]["binding"]["skipped_reason"] == "node_not_in_mems_allowed", r["json"]["binding"]


def test_bind_numactl_dry_run_failed():
    r = _bind(lambda d: d["numactl"].update(dry_run_ok=False))
    assert r["env"]["TOPO_LAUNCH_PREFIX"] == ""
    assert r["json"]["binding"]["skipped_reason"] == "numactl_dry_run_failed", r["json"]["binding"]


# ------------------------------------------------------------------------------------
# 25: remaining legacy reasons
# ------------------------------------------------------------------------------------

def test_facts_unknown_when_a_pair_is_missing():
    r = run_variant("reference.json", lambda d: d["pairs"].pop("2-3"))
    assert r["status"] == "legacy" and r["reason"] == "facts_unknown", r["env_lines"]
    assert r["env"]["VLLM_SKIP_P2P_CHECK"] == "1"


def test_facts_duplicate_bus_id():
    r = run_variant("reference.json", lambda d: d["gpus"][1].update(bus_id=d["gpus"][0]["bus_id"]))
    assert r["status"] == "legacy" and r["reason"] == "facts_duplicate", r["env_lines"]


def test_fewer_than_4_visible():
    def three(d):
        d["gpus"] = d["gpus"][:3]
        d["pairs"] = {k: v for k, v in d["pairs"].items() if "3" not in k}
    r = run_variant("reference.json", three)
    assert r["status"] == "legacy" and r["reason"] == "fewer_than_4_visible", r["env_lines"]


def test_device_order_not_pci_bus_id():
    r = run_topo("nvlink_pairs.json", caller={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"})
    assert r["status"] == "legacy" and r["reason"] == "device_order_not_pci_bus_id", r["env_lines"]
    assert r["env"]["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"   # caller value wins
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_auto_insufficient_idle():
    def busy(d):
        for g in d["gpus"][:5]:
            g.update(procs=1, mem_used_mib=20000)
    r = run_variant("delaware.json", busy, control={"LANE_GPUS": "auto"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "auto_insufficient_idle", r["env_lines"]


def test_auto_facts_unknown():
    r = run_variant("delaware.json", lambda d: d.update(pairs={}), control={"LANE_GPUS": "auto"})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "auto_facts_unknown", r["env_lines"]


# ------------------------------------------------------------------------------------
# 26: CLI details
# ------------------------------------------------------------------------------------

def test_usage_errors_exit_2():
    r = run_topo("reference.json", control={"AUTO_TOPO": "maybe"})
    assert r["returncode"] == 2 and "AUTO_TOPO" in r["stderr"], r["stderr"]
    env = _clean_env()
    env["TOPO_FIXTURE"] = str(FIXTURES / "reference.json")
    p = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=15)
    assert p.returncode == 2 and "--env-out" in p.stderr, p.stderr


def test_summary_line_is_ascii_and_quiet_suppresses_it():
    r = run_topo("nvlink_pairs.json")
    line = [ln for ln in r["stderr"].splitlines() if ln.startswith("topo: lane")]
    assert len(line) == 1, r["stderr"]
    assert line[0].isascii() and " | " in line[0], line[0]
    r = run_topo("nvlink_pairs.json", extra_args=["--quiet"])
    assert r["returncode"] == 0 and "topo: lane" not in r["stderr"], r["stderr"]


def test_diagnostics_default_dir_prefers_vllm_cache_root():
    with tempfile.TemporaryDirectory(prefix="topo_select_cache_") as td:
        vllm_root, cache_root = os.path.join(td, "vllm"), os.path.join(td, "cache")
        r = run_topo("reference.json", control={"VLLM_CACHE_ROOT": vllm_root, "CACHE_ROOT": cache_root},
                     pass_json_dir=False)
        assert r["returncode"] == 0, r["stderr"]
        assert list(Path(vllm_root, "topology").glob("*.json")), os.listdir(td)
        assert not os.path.exists(cache_root)
        assert Path(r["env"]["TOPO_DECISION_JSON"]).parent == Path(vllm_root, "topology").resolve()
        r = run_topo("reference.json", control={"CACHE_ROOT": cache_root}, pass_json_dir=False)
        assert list(Path(cache_root, "topology").glob("*.json")), os.listdir(td)


# ------------------------------------------------------------------------------------
# 27: the real-hardware adapter (NvmlTopology) against a fake pynvml + fake sysfs
# ------------------------------------------------------------------------------------

REF_BUS = ["00000000:01:00.0", "00000000:41:00.0", "00000000:81:00.0", "00000000:C4:00.0"]


def _nvml_spec(**over):
    spec = {
        "gpus": [{"uuid": REF_UUIDS[i], "bus_id": REF_BUS[i]} for i in range(4)],
        "ancestor": {},             # default NODE everywhere
        "p2p": {},                  # default OK everywhere
        "nvlink_links": 12,         # 12 links answer "inactive" (0), link 12 raises NotSupported
        "nvlink": {},
    }
    spec.update(over)
    return spec


def run_nvml(spec, nodes=None, numa=-1, caller=None, control=None):
    """Run the selector with no fixture: fake pynvml on PYTHONPATH, fake sysfs under TOPO_SYSFS_ROOT."""
    nodes = nodes if nodes is not None else {0: ("0-23,32-55", 250_000_000)}
    with tempfile.TemporaryDirectory(prefix="topo_select_nvml_") as root:
        spec_path = os.path.join(root, "spec.json")
        Path(spec_path).write_text(json.dumps(spec), encoding="utf-8")
        for g in spec["gpus"]:
            dom, rest = g["bus_id"].lower().split(":", 1)
            name = f"{int(dom, 16):04x}:{rest}"
            dev = Path(root, "sys", "bus", "pci", "devices", name.replace(":", "_") if os.name == "nt" else name)
            dev.mkdir(parents=True)
            (dev / "numa_node").write_text(f"{numa}\n", encoding="ascii")
        for n, (cpus, free) in nodes.items():
            nd = Path(root, "sys", "devices", "system", "node", f"node{n}")
            nd.mkdir(parents=True)
            (nd / "cpulist").write_text(cpus + "\n", encoding="ascii")
            (nd / "meminfo").write_text(f"Node {n} MemTotal:       263000000 kB\n"
                                        f"Node {n} MemFree:        {free} kB\n", encoding="ascii")
        Path(root, "proc", "self").mkdir(parents=True)
        Path(root, "proc", "self", "status").write_text(
            "Name:\tpython\nMems_allowed_list:\t" + ",".join(str(n) for n in nodes) + "\n", encoding="ascii")
        ctl = {"PYTHONPATH": str(FAKE_PYNVML), "FAKE_PYNVML_SPEC": spec_path, "TOPO_SYSFS_ROOT": root}
        ctl.update(control or {})
        return run_topo(None, caller=caller, control=ctl)


def test_nvml_reference_host_is_the_reference_fingerprint():
    # tuple P2P constants unwrapped, NVLink state 0 on every link then NotSupported, bytes bus ids,
    # sysfs numa -1 on a single-node system -> node 0: exactly today's settings.
    r = run_nvml(_nvml_spec())
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok" and r["reason"] == "reference", (r["env_lines"], r["stderr"])
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3" and r["env"]["VLLM_SKIP_P2P_CHECK"] == "1"
    js = r["json"]
    assert js["facts_source"] == "nvml"
    assert [g["numa"] for g in js["lane"]] == [0, 0, 0, 0]
    assert [g["bus_id"] for g in js["lane"]] == [b.lower() for b in REF_BUS]
    assert all(v["read"] == "OK" and v["write"] == "OK" and v["ancestor"] == "NODE" and not v["nvlink"]
               for v in js["p2p_matrix"].values()), js["p2p_matrix"]
    assert js["warnings"] == [], js["warnings"]      # NotSupported past the last link is not a warning


def test_nvml_numa_minus1_two_nodes_stays_unknown():
    r = run_nvml(_nvml_spec(), nodes={0: ("0-23", 250_000_000), 1: ("24-47", 250_000_000)})
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "ok" and r["reason"] == "kept", r["env_lines"]
    assert [g["numa"] for g in r["json"]["lane"]] == [None] * 4
    assert r["env"]["VLLM_SKIP_P2P_CHECK"] == "0"


def test_nvml_p2p_query_raising_is_unknown_not_unsupported():
    r = run_nvml(_nvml_spec(p2p={"1-2": "raise"}))
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "facts_unknown", r["env_lines"]
    assert r["json"]["p2p_matrix"]["1-2"]["read"] is None, r["json"]["p2p_matrix"]


def test_nvml_p2p_status_codes_map_to_names():
    r = run_nvml(_nvml_spec(p2p={"0-1": {"read": 1, "write": 1}}))
    assert r["returncode"] == 0, r["stderr"]
    m = r["json"]["p2p_matrix"]              # keyed over the FINAL order, so find the pair by value
    assert sorted(v["read"] for v in m.values()) == ["CNS"] + ["OK"] * 5, m
    assert r["reason"] == "reordered", r["env_lines"]   # the non-P2P pair is moved off the TP pairs


def test_nvml_active_nvlink_pairs_reorder():
    spec = _nvml_spec(nvlink={"0": {"0": REF_BUS[2]}, "2": {"0": REF_BUS[0]},
                              "1": {"0": REF_BUS[3]}, "3": {"0": REF_BUS[1]}})
    r = run_nvml(spec)
    assert r["returncode"] == 0, r["stderr"]
    assert r["reason"] == "reordered", r["env_lines"]
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == ",".join([REF_UUIDS[0], REF_UUIDS[2], REF_UUIDS[1], REF_UUIDS[3]])
    assert r["json"]["p2p_matrix"]["0-1"]["nvlink"] is True


def test_nvml_init_failure_is_legacy_nvml_unavailable():
    r = run_nvml(_nvml_spec(init_error=True))
    assert r["returncode"] == 0, r["stderr"]
    assert r["status"] == "legacy" and r["reason"] == "nvml_unavailable", r["env_lines"]
    assert r["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


# ------------------------------------------------------------------------------------
# Plain-script runner (pytest is optional; this mirrors `pytest -q` output/exit code)
# ------------------------------------------------------------------------------------

def _all_tests():
    return [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]


def _main():
    tests = _all_tests()
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("Skipped", "SkipTest"):
                skipped += 1
                print(f"SKIP {name}: {e}")
            else:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped out of {len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
