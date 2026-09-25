# `tests/topo/` -- topology selector tests

Tests for `scripts/topo_select.py` and its integration in `scripts/serve-v2.2.sh`. What the selector does and how
to use it is in [docs/topology-selector.md](../../docs/topology-selector.md). None of the tests needs a GPU.

- **`test_topo_select.py`**: runs the selector once per case, reads the env file and the diagnostics JSON it
  writes, and checks them. Most cases use the recorded or synthetic host descriptions in `fixtures/`
  (`TOPO_FIXTURE=...`). The NVML cases run the real-hardware adapter instead, against the fake NVML module in
  `fake_pynvml/` and a fake `/sys` + `/proc` tree (`TOPO_SYSFS_ROOT`, a test-only input).
- **`shell/run.sh`**: runs the real `serve-v2.2.sh` end to end in a sandbox of stub binaries. Stub selectors cover
  every outcome serve has to handle: a valid decision, a non-zero exit, a hang (including one that ignores SIGTERM),
  a strict-mode refusal, and env files that are garbage, truncated, header-only, carry an extra line or a non-numactl
  prefix. Further cases cover `AUTO_TOPO=strict` when the selector cannot run, a failing `mktemp`, invalid
  `TOPO_TIMEOUT` values, an empty or whitespace-only `CUDA_VISIBLE_DEVICES`, the caller snapshot, and the real `exec`
  with and without a `numactl` prefix. A few cases run the real selector on fixtures. Two cases pin the v2.2.0
  default (`AUTO_TOPO` unset = selector off): the selector is never invoked, and the environment and argv the stub
  `vllm` receives equal those from the serve script as released without the selector (a frozen copy in `tests/topo/shell/legacy-serve-v2.2.sh`).

## Running the Python suite

With pytest:

```
python3 -m pytest tests/topo -q
```

Without pytest, run the file as a plain script. It finds every `test_*` function, prints `PASS`/`FAIL`/`SKIP` per
case and a summary line, and exits non-zero on any failure:

```
python3 tests/topo/test_topo_select.py
```

## Running the shell suite

```
bash tests/topo/shell/run.sh
```

It needs GNU coreutils (`mktemp`, `sha256sum`, `timeout`). The real-selector cases also need `python3` on `PATH`.

## Running everything

```
python3 -m pytest tests/topo -q || python3 tests/topo/test_topo_select.py
bash tests/topo/shell/run.sh
```
