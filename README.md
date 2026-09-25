# Qwen3.8-Flash-Next at 806K KV on 4× RTX 3090

*v2.2.0: an 806,792-token KV pool with 262,144-token requests — three full-context sessions resident on four consumer cards — now on stock vLLM 0.30, with greedy output that repeats within one compile cache and fail-closed faults detected on every GPU rank.*

Qwen3.8-Flash-Next (125B MoE, ~6B activated per token, plus a 51B n-gram embedding table and a 4B MTP
head; Gated DeltaNet linear attention, 512 experts, vision tower) served with vLLM at its **full
262,144-token context** on four consumer Ampere cards, as the daily model behind a
[hermes](https://github.com/NousResearch/hermes-agent) agent.

**v2.2.0** (2026-09-24) moves the same shape onto stock vLLM 0.30.0 and is a reliability release, at parity with v2:
[greedy repeatability, any-rank fault detection and faster shutdown](#what-changed-in-v220). **v2.0.1** (2026-09-18) is
the v2 shape plus one engine fix, [structured output under concurrency](docs/history.md#what-changed-in-v201). **v2**
(2026-09-17) keeps everything the [v1 release](docs/history.md#v1-the-tp4-lane) established (calibrated FP8 KV for the
sparse-attention layers, full cudagraphs, three-token speculative decoding) and changes the shape: **tensor
parallel 2 × pipeline parallel 2** with expert parallel, the FP8 n-gram table served to the GPUs over a
**host-mapped, fail-closed pull transport**, and, the piece that is entirely ours, **the token-embedding tables
of both the target model and the MTP drafter moved out of VRAM into pinned host memory**, read by a device-mapped
lookup that runs inside the captured cudagraph. The KV pool goes from 342,912 tokens (1.31 full-context requests) to
**806,792 tokens: three 262K sessions resident at once**, thinking-on single-stream decode at parity with the TP4 lane
(thinking-off at 4K measured 5 % below it in the v2 gate; a later re-test put it at parity, see Known behaviours),
prefill 10–26 % faster. The v2-versus-v1 table below comes from the pre-registered two-arm gate that qualified v2
([`benchmarks/2026-09-17/BENCH-CARD.md`](benchmarks/2026-09-17/BENCH-CARD.md)); v2.0.1 was requalified against those
medians (maintainer-reported); capacity, memory, fault and prefix-cache figures are maintainer-reported from the
campaign's close records, which are not published.

[![GPU](https://img.shields.io/badge/GPU-4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](#hardware)
[![Context](https://img.shields.io/badge/context-262K_per_request-ffb000)](#the-kv-budget)
[![KV pool](https://img.shields.io/badge/KV_pool-806%2C792_tokens-0969da)](#the-kv-budget)
[![Shape](https://img.shields.io/badge/TP2_x_PP2-%2B_EP-6f42c1)](#stack)
[![MTP](https://img.shields.io/badge/MTP-K%3D3-6f42c1)](#stack)
[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97_checkpoint-Qwen3.8--Flash--Next--W4A16--Merlin-ffd21e)](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)

> **Evidence boundary.** The v2.2.0 card ([`benchmarks/2026-09-24/BENCH-CARD.md`](benchmarks/2026-09-24/BENCH-CARD.md)) is a
> maintainer measurement like the others; its qualification and fault records are not published. The v2 card is a
> maintainer measurement on the reference host with a frozen, hash-pinned manifest; the bench card is published, the
> underlying close records, raw streams and boot logs are not (as for v1). The v2.0.1 requalification figures are
> maintainer-reported and their record is not published; the two upstream pull requests the fix ports are public and
> can be read independently. What you can reproduce independently: the source tree (commit **and** tree hash, from the
> public upstream commit plus the bundle shipped as a release asset), the environment (196 pinned packages for v2.2.0,
> 199 for v2.0.1), the sidecar, and the serve command. Everything else is maintainer-reported. Of the behaviours
> documented for v2, the illegal-address engine death is **fixed in v2.2.0**; the empty warm completion and the
> prefix-cache loss remain **open** — read
> [Known behaviours](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm) before serving.

Earlier release notes (v2.0.1, how v2 got there, the v1 TP4 lane) are in [docs/history.md](docs/history.md).

## At a glance

![Flash-Next v2.2.0 layout on four RTX 3090s: pipeline stage 0 on GPUs 0 and 1 (25 of 48 layers, vision tower, target embedding, PLE home GPU 0) hands off over PCIe P2P to stage 1 on GPUs 2 and 3 (23 of 48 layers, MTP drafter, drafter embedding, LM head); each pair all-reduces over PCIe P2P; the FP8 n-gram table (~48 GiB) and the pinned embedding tables (~4.2 GiB) sit in host RAM and are pulled over PCIe; the whole-box FP8 KV pool is 806,792 tokens](docs/images/flashnext-v2.2-layout.png)

| | **v2** | v1 TP4 build (same gate, same box; † see the note under the table) |
|---|---|---|
| shape | TP2 × PP2 + expert parallel, MTP K=3 | TP4 + EP, MTP K=3 |
| KV pool, FP8 tokens, whole box | **806,792** (3.08 × a full-context request; three 262K sessions resident, measured) | 342,912 (1.31 ×) |
| context per request | 262,144 | 262,144 |
| concurrent sequences admitted (`--max-num-seqs`, configured; the shapes listed were measured resident) | **8** (3 × 262K, 1 × 262K + 3 × 131K, 5 × 131K, 8 × 65K, 8 × 32K) | 2 |
| decode, single stream, thinking on, 4K / 32K / 131K / 261K prompt | **162 / 166 / 169 / 176** tok/s (medians of 5 boots) | 163 / 165 / 172 / 170† |
| decode, thinking off, same depths | 118 / 123 / 123 / 125 | 124 / 122 / 120 / 125† |
| median event interval, thinking on, same depths | 18.72 / 18.73 / 19.10 / 19.33 ms | 18.61 / 18.73 / 19.17 / 19.37† ms |
| MTP tokens per step, thinking on, same depths | 3.02 / 3.07 / 3.13 / 3.28 | 3.00 / 3.04 / 3.17 / 3.24† |
| prefill, 10K / 100K / 261K prompt | **4,944 / 5,282 / 5,122** tok/s | 4,498 / 4,194 / — |
| time to first token, cold, 4K / 32K / 131K / 261K | 0.91 / 6.19 / 24.77 / 50.68 s | 0.97 / 7.16 / 31.94 / — s |
| quality vs the BF16 teacher (24 held-out prompts, 5 boots; lower is closer) | 0.0365–0.0378 | 0.0378–0.0385 |
| tool-call structure (150 cases) / exact recall (160 cases) | 130/150 / 160/160 | reference quant 125/150 / 160/160 (parity band, ≥ ref − 2) |
| vision | on, 2 images per request | on |

† v1 at 261K was measured on 2026-09-18, after the gate, in one boot with three sends (170.1 / 124.6 tok/s; median event
interval 19.37 / 19.40 ms; 3.24 / 2.39 tokens per step, thinking on / off). Every other v1 cell is a median of five boots;
the 261K depth was preregistered for the v2 arm only, so v1 prefill and time to first token at that depth are not measured.

Decode is the streamed event rate over the answer, client-timed at T=0 with a fixed seed (the gate's protocol; the
served entry defaults to T=1.0 / top-p 0.95 / top-k 20 and the numbers were not re-measured under those defaults,
nor under the served entry's request-logging-off flags). With MTP the honest pair is the **median event interval**
(v2 18.7 / 18.7 / 19.1 ms vs v1 18.6 / 18.7 / 19.2 at 4K / 32K / 131K) and **tokens per step** (3.02 / 3.07 / 3.13
vs 3.00 / 3.04 / 3.17), both in the card.

## Quick start (container)

The container recipe is one image that builds the pinned v2.2.0 tree and serves it (for v2.0.1, check out the `v2.0.1`
tag). The image has been built and its entrypoint checks run on a host without GPUs; GPU serving was verified on the
bare-metal routes ([Build and serve](#build-and-serve)), not yet inside the container. Host requirements:

- four 24 GB Ampere cards (qualified with peer-to-peer working on the driver; it also runs without, with lower
  prefill, see [Build and serve](#build-and-serve));
- `nvidia-container-toolkit` registered with Docker
  (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`);
- Docker Compose v2 (`docker compose`, not the 1.x `docker-compose`);
- host RAM (96 GB is the qualified allocation; the measured resident floor is about 69 GiB, see [Hardware](#hardware));
- the checkpoint [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
  on disk (115 GiB).

```bash
git clone https://github.com/halt95/qwen38-flash-next-3090s.git && cd qwen38-flash-next-3090s
docker build -t qwen38-flash-next-3090s:v2.2.0 .      # fetches the v2.2.0 bundle + build-artifacts assets; installs the first-serve toolchain
hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --local-dir /path/to/Qwen3.8-Flash-Next-W4A16-Merlin   # 115 GiB; `hf` comes with `pip install -U huggingface_hub`
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-W4A16-Merlin docker compose up -d
docker compose logs -f flash-next      # wait for "Application startup complete" (first start ~6 min)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json'   -d '{"model":"flash-next","messages":[{"role":"user","content":"hello"}],"max_tokens":512}'
```

The first start compiles the cudagraphs (about 6 minutes) into the `flash-next-cache-v2.2` volume, and the first request
compiles FlashInfer's kernels into the same volume (the image sets `FLASHINFER_WORKSPACE_BASE` and `TRITON_CACHE_DIR`
under `/cache`); later starts take about three minutes. Thinking is on by default, so a short `max_tokens` can end
inside the reasoning with empty `content`; send `"chat_template_kwargs":{"enable_thinking":false}` to turn it off per
request. The endpoint has no API key and the compose file publishes port 8000 on every interface: if the host is
reachable from other machines, set `VLLM_API_KEY` in its `environment` (clients then send
`Authorization: Bearer <key>`), or publish `127.0.0.1:8000:8000` behind a proxy. The host driver must support CUDA 13.0
or newer. The entrypoint adds the one config key the checkpoint needs if it is missing (see [Checkpoint](#checkpoint)).
Everything the container does is in `Dockerfile`, `docker-compose.yml` and `scripts/docker-entrypoint.sh`; the build
and serve scripts run without Docker ([Build and serve](#build-and-serve)).

**Proxmox instead of Docker.** The reference host runs this as a **privileged** Debian 13 LXC with the four cards passed
through as device nodes; `lxc/` reproduces that: `lxc/pve-create.sh` (run on the Proxmox host: template, cores, 96 GB,
the seven `/dev/nvidia*` nodes, checkpoint and cache mounts), `lxc/provision.sh` (run inside: NVIDIA userspace matching the
host module, the pinned build, the config key, a systemd unit) and `lxc/flash-next.service`.

Two things about the LXC route are easy to get wrong. The container is privileged on purpose: in an unprivileged one
root is mapped to a high uid and `/dev/nvidia-uvm-tools` is `0660 root:root`, so the container cannot open it even
though the cgroup rule allows the device. And the `/dev/nvidia*` nodes must already exist **on the Proxmox host**
before the container starts, because the kernel module lives there and only its userspace goes inside; running
`nvidia-smi -L` once as root creates the full set, while `nvidia-modprobe -u -c0` alone leaves out the uvm nodes.
Neither survives a host reboot on its own, so the reference host drives both from a Proxmox pre-start hookscript
(which also refuses to start the container while a GPU VM holds the cards).

```bash
git clone https://github.com/halt95/qwen38-flash-next-3090s.git && cd qwen38-flash-next-3090s   # on the Proxmox host
CTID=201 MODELS=/tank/models lxc/pve-create.sh
pct start 201 && pct push 201 lxc/provision.sh /root/provision.sh
pct exec 201 -- env NVIDIA_RUN=/models/NVIDIA-Linux-x86_64-<host version>.run bash /root/provision.sh
```

The one input the recipe cannot ship is the NVIDIA userspace, which must match the kernel module the Proxmox host runs:
`nvidia-smi` on the host prints the version; download that `NVIDIA-Linux-x86_64-<version>.run` from NVIDIA's driver
archive and put it under the `MODELS` directory (mounted at `/models` inside the container) before the provisioning step.
It is installed with `--no-kernel-module`; peer-to-peer itself is a host property.

**If you already have a vLLM checkout, do not start the server next to it.** vLLM inspects the model registry in a
child process started with `python -m`, which puts the current directory ahead of `PYTHONPATH` on `sys.path`. A
`vllm/` directory beside you therefore wins over the tree you built, and the failure surfaces much later as
something unrelated, typically `AttributeError: '_ModelInfo' object has no attribute ...`. `scripts/serve-v2.2.sh`
(and `serve-v2.sh`) refuse to start in that situation and tell you what to do; the container form cannot hit it at all.

Read [Known behaviours](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm) before putting it in front of clients.

## Build and serve

| path | what |
|---|---|
| `upstream/PIN-v2.2` | **v2.2.0:** base commit (`v0.30.0`), bundle hash and branch, release commit and tree hash, the two own extensions' hashes, the stock 0.30.0 wheel (URL and hash), artefact tarball hash, runtime versions |
| `release/v2.2/patches/0001..0058` | the series over `v0.30.0`, for reading; linear, so `git am --keep-non-patch` on a fresh `v0.30.0` checkout (with `core.autocrlf=false`) replays it to the release tree. Only the tip was built and tested; the intermediate commits are a reading aid. Verified against `SHA256SUMS.v2.2.0` |
| `release/v2.2/v2.2.0-combined.diff` | one diff from `v0.30.0` to the release commit (153 files): `git apply --index` on `v0.30.0` gives the release tree |
| `release/v2.2/requirements-pinned.txt`, `build-artifacts.list`, `SHA256SUMS.v2.2.0`, `PACKAGE-MANIFEST.md` | the environment pins (196; the build tools pip / setuptools-rust / setuptools-scm are pinned in `upstream/PIN-v2.2`), the 23 build-product paths, the hashes of the release assets, the combined diff and the 58 patches (`cd release/v2.2 && sha256sum -c --ignore-missing SHA256SUMS.v2.2.0`), the file manifest |
| `scripts/build-v2.2.sh` | fetch `v0.30.0` from GitHub + the bundle (release asset), check out, assert commit and tree, fresh venv from the pins, the first-serve toolchain links (`lib64` and the unversioned library names) in the venv's CUDA 13.0 wheels, then the compiled ops: the artefact tarball (default, both own extensions hash-checked) or `BUILD_OWN=1` (stock wheel products + the two extensions compiled from the tree; needs cmake 3.26 or newer, ninja and a C/C++ compiler, and compiles with the venv's CUDA 13.0 nvcc unless `NVCC=` names another nvcc 13.0 or newer), metadata-only install. Relative paths are fine; a re-run resumes an interrupted source fetch or wheel download; every product file is hashed once verified (except `vllm/_version.py`, which the install regenerates; the build asserts the version instead), and a re-run re-checks the whole set |
| `scripts/serve-v2.2.sh` | the v2.2.0 served entry with every variable exported; same knobs as `serve-v2.sh` plus `GUARD` (default `warn`), `COUNTERS` (default off) and `CUDA_HOME` (default: the venv's CUDA 13.0 wheels); checks compiler, ninja, Python headers and nvcc-versus-driver and re-hashes the build products (as recorded by the build) before starting, warns when host RAM or `/dev/shm` is below the qualified sizes, and clears inherited fork knobs |
| `upstream/PIN-v2` | base commit, bundle prerequisites, tag commit and tree hash, precompiled-wheel identity and hash, artefact tarball hash |
| `release/v2/patches/0001..0076` | the full series over `e2-base`, for reading (patch subjects carry the campaign's internal task and review labels; they are not renamed because the files are hashed); the history has merge commits, so `git am` cannot replay it (it stops at patch 43). These copies are byte-identical to the reference package and verify against `SHA256SUMS.v2.0.1`; contributors whose work upstream authored keep their own attribution. The bundle remains the source of truth |
| `release/v2/v2.0.1-combined.diff` | one `git diff e2-base..v2.0.1` (112 files, text only, 777 KiB): `git apply` it on `e2-base` (the release asset `e2-base-src.tar.gz`, sha256 `fcd14214f64faaa4175d62f6a51ca39c7bc09d15bf75c190d4dfcc88846b927f`) and you have the tagged source; checked to apply cleanly and to give the tag's tree hash. The way to reproduce the tree from the patches directory without the bundle (`build-v2.sh` itself uses the bundle) |
| `release/v2/requirements-pinned.txt`, `build-artifacts.list`, `SHA256SUMS.v2.0.1`, `PACKAGE-MANIFEST.md` | the environment pins, the 22 build products, the hashes of the release assets, the combined diff and the reference patch series (`cd release/v2 && sha256sum -c --ignore-missing SHA256SUMS.v2.0.1` verifies 78 of 81 from a checkout; the other three are the release assets), the file manifest |
| `scripts/build-v2.sh` | fetch the three prerequisite commits from GitHub + the bundle (release asset), check out `v2.0.1`, assert commit and tree, fresh venv from the pins, compiled ops from the wheel or the tarball, metadata-only install. One prerequisite is a PR-branch head (#53899); if it ever disappears upstream the bundle alone cannot be applied — a full bundle is the fallback |
| `scripts/serve-v2.sh` | the served entry with every variable exported; port, host, names (default `flash-next-v2 flash-next flash-mtp flash-next-mtp`), PLE home, sidecar and cache dir configurable |
| `Dockerfile`, `docker-compose.yml`, `scripts/docker-entrypoint.sh` | the v2.2.0 container: `build-v2.2.sh` at image build (bundle and build-artifacts from the release URLs or the build context; gcc, g++ and ninja for the first serve, the base image's Python headers), `serve-v2.2.sh` as the entrypoint, checkpoint and cache as mounts, the config key added on first start if the mount is writable |
| `lxc/pve-create.sh`, `lxc/provision.sh`, `lxc/flash-next.service` | the Proxmox LXC form of the same thing: create the container with the device nodes and mounts, provision it (NVIDIA userspace, gcc/ninja/Python headers, `build-v2.2.sh`, config key, systemd unit), serve on boot |
| `scripts/make-e1-config.py` | adds the one config key to the published checkpoint, inside `text_config`, and re-parses the result to prove it landed where the engine reads it |
| `scales/qsa_kv_scales_262k.json`, `calib/` | the sidecar and how it was made (unchanged from v1) |

```bash
# v2.2.0. Release assets: v2.2.0-from-upstream-v0.30.0.bundle.gz (required),
# build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz (the compiled ops; or BUILD_OWN=1 to compile the two own extensions)
R=https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.2.0
curl -fLO "$R/v2.2.0-from-upstream-v0.30.0.bundle.gz"
curl -fLO "$R/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz"      # 241 MB
hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --local-dir /path/to/Qwen3.8-Flash-Next-W4A16-Merlin   # 115 GiB
gunzip v2.2.0-from-upstream-v0.30.0.bundle.gz
BUNDLE=./v2.2.0-from-upstream-v0.30.0.bundle ARTIFACTS=./build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz \
  scripts/build-v2.2.sh ./vllm-v2.2 ./venv-v2.2
python3 scripts/make-e1-config.py /path/to/Qwen3.8-Flash-Next-W4A16-Merlin   # no-op if the config already carries the key
TREE=./vllm-v2.2 VENV=./venv-v2.2 CACHE_ROOT=./.vllm-cache-v2.2 HOST=0.0.0.0 scripts/serve-v2.2.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin
```

The v2.0.1 route, unchanged:

```bash
# release assets: v2.0.1-from-upstream-e962733e08.bundle.gz (required), build-artifacts-sm86-py313-cu130.tar.gz (optional),
# e2-base-src.tar.gz (only for the combined-diff route), build-artifacts.list (already in the tree)
gunzip v2.0.1-from-upstream-e962733e08.bundle.gz
BUNDLE=./v2.0.1-from-upstream-e962733e08.bundle scripts/build-v2.sh ./vllm-v2 ./venv-v2
python3 scripts/make-e1-config.py /path/to/Qwen3.8-Flash-Next-W4A16-Merlin   # no-op if the config already carries the key
TREE=./vllm-v2 VENV=./venv-v2 HOST=0.0.0.0 scripts/serve-v2.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin
```

Four known deltas between the gate command and the shipped one:

- the gate ran with `--enable-log-requests --enable-request-id-headers` (request logging on; the served entry has it
  off; no throughput effect is claimed either way);
- the served entry adds `--override-generation-config` (T=1.0 / top-p 0.95 / top-k 20) and
  `--default-chat-template-kwargs` (thinking on, low effort), which the gate did not pass (it sent T=0 with a seed per
  request);
- `serve-v2.sh` also exports `HF_DATASETS_OFFLINE=1`;
- the reference host exports `LD_LIBRARY_PATH` pointing at its CUDA 13.3 toolkit, which is inert: every CUDA library
  the processes map comes from the venv wheels (`/proc/<pid>/maps` checked on a qualification boot), so `serve-v2.sh`
  does not set it.

`HOST=0.0.0.0` exposes an endpoint without an API key on every interface: if the host is reachable from other machines,
set `VLLM_API_KEY` (or pass `--api-key` after the checkpoint; extra arguments go to `vllm serve`), or keep the default
`127.0.0.1` behind a proxy.

**Requirements.** Linux x86-64 with glibc 2.34 or newer for the v2.2.0 build products (Ubuntu 22.04, Debian 12, RHEL 9 or
later); an NVIDIA driver with CUDA 13.0 or newer (the 580 series or later); git, curl; Python 3.13 with venv and
headers, as `python3.13` on `PATH`. Debian 13 packages it (`apt install python3.13-venv python3.13-dev`); Ubuntu
22.04 / 24.04 get it from the deadsnakes PPA (same package names); on Debian 12 or RHEL 9 use a standalone build such
as `uv python install 3.13` (it ships venv and headers; check that `python3.13` is on `PATH`), pyenv, or the
container. A C/C++ compiler and ninja (the first serve compiles kernels; `BUILD_OWN=1` also needs cmake 3.26 or newer,
which Debian 12's package is not: `pip install 'cmake>=3.26,<4'`; the tested build used 3.31, and 4.x is untested),
four visible 24 GB NVIDIA GPUs (qualified with peer-to-peer on the driver; it also runs without, see below), host RAM
(96 GB qualified; ~69 GiB measured floor), `/dev/shm` ≥ 1 GB. The first boot compiles the graphs (v2.0.1 ~10 min,
v2.2.0 ~6 min); later boots reuse `CACHE_ROOT`. (The reference host's qualification arms shared one compile cache
across candidates; give production a v2-named cache root.) Reproduction on the reference host from the bundle: commit
and tree equal to the tagged worktree, package hashes verified, fresh venv, one qualification boot 17/17 rows
(maintainer-reported).

Outside the reference host (maintainer-run, 2026-09-24): on a rented 4× RTX 3090 without peer-to-peer (in a Debian 12
container: the Dockerfile's base image and apt line; driver 595.84 / CUDA 13.2), the published source built with `BUILD_OWN=1`
and served by `serve-v2.2.sh` with its defaults came up in 6 min on a fresh compile cache and 2.5 min on a warm one,
compiled FlashInfer's kernels on the first request, answered text, thinking, JSON-schema and a 54,713-token prompt
correctly, and gave the same T=0 output across three requests and two boots. The release tarball route (2026-09-25,
same host, one fresh-cache boot, up in 5.9 min; no warm-cache restart) passed the same request checks and gave the
same T=0 output as the `BUILD_OWN=1` tree.

**Without peer-to-peer** (a stock NVIDIA driver on consumer cards, as on that host) the release runs unchanged: vLLM
turns its custom all-reduce off by itself and NCCL moves the tensor-parallel and pipeline traffic through host memory;
`serve-v2.2.sh` needs no edit. Expect lower prefill: a cold 54,713-token prompt took 13.8 s there (about 4,000 t/s,
against 5,410 t/s at 50K on the reference host), on a host that also had a PCIe x8 link per card and its cards split across
two NUMA nodes, so not all of the gap is peer-to-peer. Three 512-token decodes of a 24-token prompt took 3.6–4.0 s wall
each (127–141 t/s including time to first token; not stream-timed, so not comparable with the bench card).

### Multi-GPU hosts and topology (opt-in)

By default `serve-v2.2.sh` uses the settings the release was qualified with: GPUs 0-3 in PCI bus order, vLLM's P2P
check skipped, `NCCL_P2P_LEVEL=SYS`, `NCCL_PROTO=LL`. On a host that does not look like the reference one, set
`AUTO_TOPO=1`: serve then runs `scripts/topo_select.py`, which reads the GPU topology from NVML and `/sys` and sets the
GPU order and the P2P and NCCL variables from it, falling back to the fixed settings on any doubt.

- **NVLink pairs**: the selector reorders the four GPUs so that each NVLink-bridged pair is a tensor-parallel pair,
  even when the bridged cards are not next to each other in bus order.
- **More than four GPUs**: pick four by UUID (`nvidia-smi -L`), ideally on one NUMA node,
  `CUDA_VISIBLE_DEVICES=GPU-aaaa...,GPU-bbbb...,GPU-cccc...,GPU-dddd...`. UUIDs do not depend on enumeration order.
  This works with the selector off too; with `AUTO_TOPO=1` it also checks the order and the P2P settings.

![Pairing tensor-parallel ranks with NVLink bridges on four GPUs bridged 0-2 and 1-3: in the default order 0,1,2,3 the tensor-parallel pairs are (0,1) and (2,3) and their all-reduce goes over PCIe; with CUDA_VISIBLE_DEVICES=0,2,1,3 or AUTO_TOPO=1 the pairs are (0,2) and (1,3) and each all-reduce goes over its NVLink bridge](docs/images/flashnext-nvlink-pairing.png)

[docs/topology-selector.md](docs/topology-selector.md) has the decision rules, `AUTO_TOPO=strict` and the other controls.

### Running notes

Lines every boot logs that are not failures: `PLE offload CUDA guard: blocked a CUDA initialization attempt` with a
call-site traceback (through FlashInfer's import) and `Failed to get device capability: CUDA is disabled in the PLE
offload process` (the offload process refuses a CUDA context by design), NCCL's `unbatched P2P op` warnings and
Triton's `JIT compilation during inference` warnings.

Never reuse an existing venv for this tree, and do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
(v1 measured it at −27 % four-stream aggregate).

Process model: the workers set `PR_SET_PDEATHSIG(SIGKILL)` so an engine-core death takes them with it, and start
methods other than fork and spawn are refused. The death signal fires when the *thread* that spawned the workers exits, so an embedded
engine must build its executor from a thread that outlives serving (the `vllm serve` path does). The port of the
upstream MTP-under-PP change (#46994) carries the PR's code without its tests.

Allocator warnings you may see: with several long sessions growing at once, the log can show
`CUDACachingAllocator ... memory allocation failed with OOM on device N while trying to allocate <70–270 MB>` lines.
They are warnings, not failures: the upstream sparse-attention indexer's prefill logits buffer (256 rows × context
length × fp32, so 1 KiB per token of context, 268 MB at 262K) asked for a contiguous block the caching allocator
could not find, the allocator flushed its cache and the retry succeeded. Maintainer-observed on the reference host (no shipped record) with three
sessions climbing from 50K to 170K tokens: 22 lines, no request failed, device free dipped to 18–34 MB at the
moment of the warning, and the lines stopped once the cache had been flushed at each size. If a retry ever fails, that is
the non-KV headroom (about 1 GB per card at the shipped pin) being exhausted, and the KV pin is the knob.

## What changed in v2.2.0

**v2.2.0 is rebased on stock vLLM 0.30.0 and needs the new runtime.** v2.0.x ran on a 0.28-era nightly plus a community
branch; v2.2.0 is the public `v0.30.0` tag plus 58 commits. It cannot be dropped into a v2.0.x environment: it needs
the vLLM 0.30.0 runtime (python 3.13, torch 2.13.0 with CUDA 13.0, triton 3.7.1, flashinfer 0.6.18.post1), its own two
compiled extensions and a new compile cache. Unchanged: the checkpoint, `config.json`, the KV-scale sidecar, the serving
shape (TP2 × PP2 + EP, MTP K=3) and the 806,792-token pool. Build it with `scripts/build-v2.2.sh`, serve it with
`scripts/serve-v2.2.sh` ([Build and serve](#build-and-serve)).

**Published source versus the qualified tree.** The published source is the qualified tree with internal working
labels and dates (task, run and design-note references) removed from comments, docstrings, log text, commit messages
and file names. It differs from that tree only in: comments, docstrings, log/error message text, test-only labels in
test data and one reader-thread name; identifier renames, namely the fork's switch prefix, published as `MERLIN_*`
(environment variable names, the C++ `getenv` names, the build defines and helper functions derived from them, and the
test helpers), and a few test and helper names (`medium_*`, `tiedet_*`, `largedet_*`, `REPLAY_SENTINEL_*`, `pre_tag`,
the `isolated`-process tests and their constants, `_gpu_host_tier`, one test's report-directory variable); two
internal design notes that are not shipped; nineteen test and probe files renamed (one test import follows a rename).
With those renames mapped back, Python is AST-identical once docstrings and the changed string constants (log/error
text, the thread name, test labels) are normalised, C++ is token-identical outside comments and string literals, and
every file present in both trees keeps its line count. Both compiled extensions are rebuilt from the published source;
their device code matches the qualified builds instruction for instruction for every kernel the release changes and
every kernel on this model's path, identified by kernel name (`cuobjdump -sass`, after normalising per-build symbol
hashes). 24 upstream kernels on paths this model does not use differ (MiniMax-M3 / Kimi-K3 fused ops, the DeepGEMM fp8
silu-mul quant and the fp8 MoE finalize); the likely cause, not isolated, is the older host compiler of the build root
(Ubuntu 22.04, gcc 11, chosen so that the extensions load on glibc 2.34). The unit tests give the same per-test results
on both trees.

It is a **reliability release, at parity with v2, not faster**: decode lands at 0.973–1.040 of v2 in every cell of
the ladder. Per-step time is 0.9–2.5 % higher in every cell, a combined difference of the determinism path and the
0.30 base (the switches alone cost 1.2–2.2 % at 65K–262K on one tree); tokens per step are equal or higher in six of
the eight cells ([`benchmarks/2026-09-24/BENCH-CARD.md`](benchmarks/2026-09-24/BENCH-CARD.md)).

![v2.2.0 prefill and decode over context depth, with step time](benchmarks/2026-09-24/flashnext-v2.2.0-ctx-pp-tg-itl.png)

Cold prefill runs 5,123–5,410 t/s from 10K to 200K tokens. Decode with thinking on is 153.5–186.7 t/s (median of three
different prompts per depth; one boot). Step time moves little with depth (18.8 → 19.4 ms), and most of the decode
spread tracks how many drafted tokens each text accepts (2.77–3.58 per step) rather than context length.

**Greedy output repeats within one compile cache.** Four switches remove three sources of run-to-run variation in
T=0 decoding: `MERLIN_FULL_K=1` (Marlin MoE split-K reduction order), `MERLIN_QSA_SORT=1` (order of the blocks the
sparse-attention top-k selects), `MERLIN_TIE_DET=1` and `MERLIN_TIE_RECENT=1` (how ties at the top-k cutoff are broken
and which are kept). On the release candidate, with prefix caching off and thinking off, 8 identical T=0 requests gave
**1 distinct output** in every sampled cell from 3,960 to 261,802 prompt tokens; with the switches off, the deep cells
gave 4 to 8 distinct outputs of 8. These are sampled results, not a batch-invariance guarantee, and they hold only
within one compile cache (Known behaviours). The top-k kernel also emits its selection in canonical order itself
(`MERLIN_TOPK_SORTED_EMIT=1`): output-identical to the Python sort it replaces, and 1.9–3.0 % lower median step time
in one boot per arm.

**A fail-closed fault on any GPU rank now stops the engine.** A fail-closed PLE fault on a non-zero tensor-parallel rank
used to hang the engine: that worker's exit pipes had been inherited by a `torch_shm_manager` helper that outlived it,
so the executor never saw the death and blocked in an RPC until a 300 s deadline. The defect is in v2.0.x as well; no
earlier fault campaign exercised that case. v2.2.0 adds a per-worker fault pipe (the worker writes one byte before
`os._exit`), a pidfd exit monitor with the sentinel as fallback, covering the whole initialisation window, and
close-on-exec on the inherited pipes. Measured on the final tree: an injected fail-closed fault on rank 1 is detected at
+0.000 s, workers are gone in 3.0–3.5 s and the client gets HTTP 500 `EngineDeadError`. This covers fail-closed PLE
faults and worker exits; a GPU that hangs without exiting is not covered by these measurements.

Separately, drain-order fixes shorten orderly shutdown from ~9.0 s to 2.65–3.55 s: a late SIGTERM during teardown is
survived, workers exit right after their orderly shutdown, and a failure that races shutdown still exits 1. These
shutdown figures come from release candidates before the final tree, which also changes the fault-detection path and
parts of the same teardown code.

**Other fixes.**

- The rare illegal-address engine death (Known behaviours) is fixed: every sparse-attention raw-key ring row is tagged
  and validated before pooling, and a mismatched row is masked. A seeded replay faults on the old tree and is contained
  on the new one.
- A hang in the PLE fatal path is fixed: the traceback formatter no longer runs exception-defined code before `os._exit`.
- The PLE pull operator resolves live pointers by layer name, so the compile cache no longer bakes addresses in.
- Carried or re-ported onto 0.30: #54793 / #54795 (PP), #54442 / #56802 (structured output under concurrency, the
  v2.0.1 fix), #46994 (MTP under PP), #57050, #48532, #50021, #55506, #55557.

**Serve-command changes against v2.0.1** (all in `scripts/serve-v2.2.sh`):

- `VLLM_PLE_CPU_OFFLOAD=1` becomes **`VLLM_E2_PLE_PULL_TRANSPORT=1`**. vLLM 0.30 reuses the old name for its own PLE
  backend with the opposite default; the tree refuses to boot with both on.
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, exactly `"0"`: 0.30 would otherwise switch this architecture to breakable graphs
  and turn compilation off. The tree's profile guard refuses anything but the qualified shape.
- The four determinism switches and `MERLIN_TOPK_SORTED_EMIT=1`. They are compile factors: changing one costs a fresh
  compile.
- `VLLM_E2_GUARD_MODE=warn`: bounds-guard telemetry on. Hits are reported, and the designated fatal sites (among them
  the ring-row validation site) stop the engine with a named cause instead of masking the row silently. The code
  default is `off` (a comment in the tree says otherwise; the comment is wrong).
- **A first-serve toolchain.** FlashInfer compiles its prefill, sampling and top-k kernels on the first request and
  Triton compiles its launchers, so serving needs a C/C++ compiler, ninja, the Python 3.13 headers and an nvcc no newer
  than the driver. `build-v2.2.sh` installs the CUDA 13.0 nvcc/crt/nvvm/cccl wheels into the venv (PTX any CUDA 13.0+
  driver accepts) and the `lib64` and unversioned library links FlashInfer and cmake look for; `serve-v2.2.sh` points
  `CUDA_HOME` at them and checks compiler, ninja, headers and nvcc-versus-driver before starting. On Debian/Ubuntu:
  `apt install gcc g++ ninja-build`, plus the Python 3.13 headers (`python3.13-dev` on Debian 13, or from the deadsnakes
  PPA on Ubuntu; uv and pyenv Pythons ship them; see Requirements). Two differences from the reference host: it compiles these kernels
  with its system CUDA 13.3 toolkit (`CUDA_HOME=/usr/local/cuda`; set `CUDA_HOME` to do the same), and its venv
  carries CUDA 13.4 toolkit wheels it does not use, which `requirements-pinned.txt` replaces with the 13.0 ones (four
  pins: nvcc, crt, nvvm, cccl). Every other pin is the reference venv's.
- The two logging-only counters (exact-restart counters every 30 s, FP8 KV clip counter every 300 s) are **off by
  default** (`COUNTERS=1` opts in). The reference host ran them, and the published numbers were measured with them on,
  but the clip counter's reader starts before cudagraph capture and can overlap it on a fresh-compile boot. The clip
  counter is a compile factor, so the default is a different compiled variant from the measured one: on the final
  tree it was booted only on the rented host without peer-to-peer (three boots, [Build and serve](#build-and-serve);
  correct answers, no performance comparison).
- Inherited fork knobs (test-only fault injectors, diagnostics, timeouts, a correctness switch turned off by hand) are
  cleared before the qualified values are set.
- A **new** `VLLM_CACHE_ROOT`: 0.28-era compile artefacts do not carry over. First boot ~6 min, later boots ~3 min.
  The kernels FlashInfer compiles on the first request are cached outside it, under `~/.cache/flashinfer`
  (`FLASHINFER_WORKSPACE_BASE` replaces `~` in that path; the container sets it under `/cache`).
- Keep model runner V2 (the 0.30 default).

Rollback is pointing the entry back at the v2.0.x tree, environment and cache; nothing in the checkpoint changed.

## Benchmarks

v2.2.0 against the v2 gate medians, single-stream decode (one boot of the release candidate; per-cell numbers in
[`benchmarks/2026-09-24/BENCH-CARD.md`](benchmarks/2026-09-24/BENCH-CARD.md)):

![Bar chart of single-stream decode, v2.2.0 against v2, at 4,096 / 32,768 / 131,072 / 261,120 prompt tokens: thinking on 160.0 vs 162.2, 165.6 vs 166.3, 168.9 vs 169.4, 175.1 vs 176.2 tokens/s; thinking off 119.7 vs 118.2, 127.9 vs 123.0, 121.3 vs 123.3, 121.8 vs 125.2 tokens/s; 0.973 to 1.040 of v2 in every cell](docs/images/flashnext-v2.2-vs-v2.png)

The v2 gate ran **five boots per arm, alternating v2 and v1 on the same box**, one frozen manifest, three sends
per cell, judged as medians with a 0.97 rule per depth and a 0.90 floor per boot. Full card with every boot's
numbers, the event intervals, tokens per step, the fingerprint and the environment:
[`benchmarks/2026-09-17/BENCH-CARD.md`](benchmarks/2026-09-17/BENCH-CARD.md).

| cell | verdict |
|---|---|
| decode, thinking on, 4K / 32K / 131K | 0.997 / 1.009 / 0.988 of v1 — **PASS** at every depth |
| decode, thinking off, 32K / 131K | 1.008 / 1.024 — PASS |
| decode, thinking off, **4K** | **0.952 — FAIL against the 0.97 rule**, shipped documented (a draft-acceptance effect of the shape, not a v2 patch: it is identical with the readers on or off): event interval identical (18.66 vs 18.64 ms), tokens per step 2.14–2.31 vs 2.31, clustered per boot; cause not established then (a 2026-09-23 re-test attributes it to a content draw, Known behaviours) |
| prefill 10K / 100K | +10 % / +26 % — PASS |
| quality vs teacher | delta −0.0009, well inside the 0.0015 bound — PASS. (An earlier single-capture quality screen recorded **FAIL** and stays cited as such; the two-arm gate replaced it as the instrument) |
| pool, capacity, tools, no-think, faults | PASS (see card) |

What the earlier candidates looked like: with the two instrumentation reader threads on and the default
NCCL protocol, the thinking-on rows sat at 0.949 / 0.962 of v1 (event interval 19.10 vs 18.56 ms at 4K). The
readers (a bounds-guard telemetry reader and the KV clip counter, each polling every 50 ms) cost ~0.3–0.5 ms
per step on the pipeline's critical path; v2 ships them off and `NCCL_PROTO=LL` on, which is the whole
difference (maintainer-measured).

## Known behaviours of the Qwen3.8-Flash-Next architecture in vLLM

These are behaviours of the Qwen3.8-Flash-Next serving path in vLLM (the hybrid Gated-DeltaNet / sparse-attention /
MTP execution path and its hybrid KV manager). None is introduced by this checkpoint's quantisation. The empty warm
completion is an upstream-reported class and was seen on the v1 TP4 lane; the ring-row death is in upstream code. The
prefix-cache loss did not move across the five release candidates (v2.0.1 included) when the v2 deltas were bisected,
but whether it is upstream's or the patches' is **not established**: a run on the unmodified upstream base is still
owed. v2 documents them, ships mitigations where one exists, and tracks them; none affects the correctness of answers
in the gate.

- **Empty warm completion — an occasional empty completion on a warm repeat of a long cached prefix** (upstream class:
  vllm-project/vllm #53912, prefix caching + speculative decoding on hybrid models; seen on the v1 TP4 lane too). HTTP 200,
  `finish_reason: "stop"`, zero tokens: the first sampled token is EOS. Per boot, not per request (roughly one
  boot in three on the no-MTP diagnostic shape; three empty warm completions in 23 gate boots of two earlier release
  candidates, two of them on the v1 TP4 arm; none in the 10 boots of the final run; on the v2.2 release candidate, 0 of
  200 warm requests over 5 boots without preemption, a request-level 95 % upper bound of 1.88 % that says nothing about
  boot-level incidence). The cached bytes are proven identical between a call that flips and
  the calls around it; the race is inside the flipping request's own forward, it needs pipeline parallel plus
  async scheduling, and async scheduling is required with MTP under PP on this fork. Mitigation shipped: **retry
  once**; it returned the correct answer in every observed case (a mitigation, not a guarantee; the rate on the
  served profile is not measured). Maintainer-reported.
- **Prefix-cache loss — prefix-cache blocks of sessions that finish while other long sessions are still decoding are dropped**
  (in the hybrid KV manager path; upstream or patch origin not established). Not preemption (reproduced with zero
  preemptions). The variable that separates the arms is the survivors' remaining decode: with 320-token outputs only
  the first-finished session loses its blocks, with 2,048-token outputs every session does, and a single long session
  plus a burst retains everything. Free-block state during the episode was not sampled, so eviction under pool
  pressure is not excluded. Reproduced on three release candidates (one boot each, one shared compile cache); one boot
  of a fourth kept the last-finished session; re-run on v2.0.1 with an identical failing set, identical cache-hit
  total and 51/51 answers correct. Cost: first-token latency on that session's next turn; the answer is unaffected.
- **Thinking-off decode at 4K** is 0.92–0.99 of the TP4 lane on every one of the five boots, four of them below the
  gate's 0.97 rule (above); it is the one cell that makes the judge's machine verdict for the run FAIL. A later
  interleaved re-test (2026-09-23; 15 boots: the v1 TP4 lane, and the v2.2 release candidate with the determinism
  switches on and off, five each) put the cell at 1.011 of v1 with the switches on (0.987 off), with tokens per step
  equal (1.007 [0.994, 1.021]): the September shortfall was a content draw (at T=0 a near-tie at the first prose
  tokens picks one of a few equally valid phrasings, and they differ in how well the drafter predicts them), not an
  acceptance defect of the engine.
- **A rare illegal-address engine death** (upstream sparse-attention code path; two occurrences over the whole campaign;
  mechanism: a re-claimed, never-zeroed ring row read as a RoPE position). Bounds guards turn a recurrence into a
  named, counted failure when their reader is on; the reader is **off by default**, so a masked fault at an
  always-fatal site is counted but never read or raised (`GUARD=warn` in `serve-v2.sh` turns the reader on at the cost
  above). Not fixed in v2.0.x; no rate is claimed. **Fixed in v2.2.0** (ring rows are tagged and validated before
  pooling; see [What changed in v2.2.0](#what-changed-in-v220)), where the served profile also runs the reader in `warn`.
- **Greedy T=0 is not byte-reproducible** on v2.0.x with this architecture (bf16 near-ties resolved differently by the
  sparse indexer's top-k and the expert permutation); documented, accepted. **v2.2.0** makes it repeatable **within one
  compile cache** (the determinism switches). Two fresh compiles of the identical tree and environment can still give
  different T=0 outputs: the six FLA gated-delta-rule chunk kernels (prefill only) carry their own Triton autotune, and
  their near-tied picks are cached per compile directory (14 to 18 of 24 picks differed between two fresh caches). If
  byte-identical repeats matter, keep one `VLLM_CACHE_ROOT` per deployment. Inductor's own reduction autotuning is a
  second candidate source; which kernels account for the difference is not fully attributed, so a v2.2.x fix will be
  validated by fresh-cache equality, not assumed from pinning one of them. Any numeric change (layout, kernel build,
  version) can also flip near-tied choices: TP2 × PP2 and TP4 gave byte-identical output for 0 of 22 prompts, differing
  in phrasing, not correctness, and MTP acceptance moves with the text by a few percent per prompt.
- Fixed on the way and worth knowing if you run an older candidate: a worker hang at the end of very long
  prefills (the instrumentation readers deadlocking under the CUDA context lock; fixed before v2, readers off
  in v2) and the upstream Mamba admission-estimate regression (#57050, ported before v2).

## Hardware

| resource | reference host |
|---|---|
| GPU | 4× NVIDIA GeForce RTX 3090 (Ampere sm_86, 24 GB each), **220 W** power cap, no NVLink |
| PCIe | Gen4 x16 to every card; P2P over the aikitoria open-kernel-module patch (`VLLM_SKIP_P2P_CHECK=1`, `NCCL_P2P_LEVEL=SYS`) |
| CPU / RAM | AMD EPYC 7532, 192 GB ECC; the serving container is allocated **96 GB**, the qualified figure. Measured resident floor on the v2.0.1 container: about **69 GiB** = the ~48 GiB FP8 n-gram table (anonymous memory in the offload process) + ~4.2 GiB of pinned embedding tables (1,064 MiB per rank) + ~8 GiB of shared segments + ~8 GiB across the four workers, engine core and API server; the remaining ~26 GiB up to the 96 GiB limit (98,304 MiB, what the recipes call "96 GB") is checkpoint page cache and reclaimable. The boot peak was not measured, so treat the 69 GiB floor as a hard floor (below it the load OOMs), 80 GB as the sensible minimum and 64 GB as not enough |
| disk, `/dev/shm` | ≥ 250 GB free (the checkpoint is 115 GiB); `/dev/shm` ≥ 1 GB (maintainer-measured peak +30.6 MB per boot; a `Bus error` at boot means it is undersized) |
| OS / serving | Linux container on Proxmox, vLLM behind llama-swap; the scripts here run the same engine directly |

## The KV budget

**806,792 tokens** at `--kv-cache-memory 4100000000` per rank, measured on every qualification boot. The
per-request limit stays 262,144; the pool is the aggregate over concurrent requests. Where the room came from
relative to TP4 at the same weights (the layout diagram under [At a glance](#at-a-glance) shows where each piece sits):

- **PP2 halves what each card duplicates.** With tensor parallel alone every card holds a slice of every
  layer plus the whole non-sharded state; with two pipeline stages each pair of cards holds 25 / 23 of the 48
  layers. The stage-0 pair keeps the vision tower and the target embedding, the stage-1 pair the MTP drafter and
  the LM head, each built once instead of on all four (0.59 GiB per copy at TP2).
- **The 51B n-gram (PLE) table is host-mapped.** The FP8 table stays in host RAM; a pull transport gives the
  GPUs the rows they need per step through a pinned slot, with a fail-closed ownership protocol (a transport
  fault kills the engine with a named cause rather than serving stale rows). The policy is wider than the transport:
  on a PLE boot, any exception that escapes a worker's model step or sampling hard-kills that worker (`os._exit(1)`
  with the cause on stderr) and its peers stop at the NCCL watchdog, where upstream would surface an engine error
  and keep the process. The offload process keeps itself CUDA-free by patching torch internals
  (`torch.cuda._lazy_init`, `torch.cuda.Stream.__new__`, `torch.Tensor.pin_memory`); a torch upgrade could disable
  that guard silently, which is one more reason the pins are pins. Its cost is decode-neutral in the
  card; its price was most of the engineering of the campaign.
- **Token embeddings in pinned host memory** (`VLLM_HOST_EMBED_TABLE=1`, [our design, below](#host-resident-embeddings-inside-the-cudagraph)):
  608 MiB of device memory freed per rank, the lever that lifted the pin from 3.8e9 (pool 748,255)
  to 4.1e9 (pool 806,792).
- The 4.1e9 pin is the highest that boots with a complete clean row set (measured on the qualification boots); 24/24
  layer partitions and higher pins fail on the tightest card (3.8e9 was the ceiling before the host-resident embeddings).

Three 262K sessions resident with 302 of 317 blocks in use, three-way qualifying interval 8.41 s, is the
capacity row of the gate.

## Host-resident embeddings inside the cudagraph

This is the v2 change that is ours end to end (patches `0037` and `0039` in `release/v2/patches/`, branch
`v2-host-embed`; designed, reviewed and redesigned during the campaign). The model carries two bf16 token-embedding tables of
248,320 × 2,560, one for the target and one for the MTP drafter; sharded over TP2 that is 0.59 GiB per copy per
card, and at TP4 the copies were part of what capped the pool. v2 keeps both tables in **pinned host memory** and
gives each rank a **device-mapped (UVA) lookup** over its shard: the same TP sharding, id masking and all-reduce as
the device path, invalid ids yield zero rows, and the output is **bytewise identical** to the device table, signed
zeros included (tests: real shard loader with no CUDA transient, TP2 bytewise masking + reduction, GPU capture and
replay with three lookups in one graph).

Why the obvious version does not work: the drafter's lookup sits **inside the FULL cudagraph of the draft step**, so a
CPU-side gather (the first draft) cannot run there and was rejected in review; the lookup has to be a device
operation over host-mapped memory that the graph can replay. The cost is a 1,064 MiB host allocation per rank
(maintainer-measured) and a PCIe read per looked-up row; the card shows no decode penalty against the TP4 lane. It is
on by default (`VLLM_HOST_EMBED_TABLE=1`) and can be turned off, which costs the 608 MiB per card back. One consequence
worth knowing if you build from source: the flag is a torch compile-cache factor (the UVA and device drafter graphs
must not share an AOT cache entry; patch `0068` registers it, and a pre-fix tree needs separate `VLLM_CACHE_ROOT`s per
mode).

## Checkpoint

The same checkpoint as v1, [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
(Intel AutoRound INT4 g128 experts, the RadixArk FP8 n-gram table, MTP draft experts INT4 and GDN projections
INT8 packed in place, attention BF16; lineage in its model card), **plus one config key**: v2 keys the FP8
n-gram table on `ple_embedding_dtype: float8_e4m3fn` inside the `text_config` object of `config.json` (the engine reads
its text config, so a top-level key does nothing) instead of the `VLLM_PLE_FP8_GLOBAL_SCALE` environment variable v1
used (v2.0.x still honours that opt-in for a checkpoint without the key; v2.2.0 does not, and `serve-v2.2.sh` refuses
to start without the key). The checkpoint on Hugging Face carries the key since 2026-09-18; for a download that
predates it, `scripts/make-e1-config.py` adds it (and is a no-op otherwise); every weight file stays byte-identical
(the reference host's serving copy is hard links to the published shards, verified 2026-09-17). Shard hashes are not
published here; the Hugging Face checkpoint carries them, and the v2 delta is the one config key above.

The KV scale sidecar is **the file v1 shipped**, `scales/qsa_kv_scales_262k.json` (Merlin-calibrated, margin
1.10, sha256 `5cbe6ae8…`); the reference host serves the same bytes under a different file name. One file ships.

## Serving an agent

| agent need | as served |
|---|---|
| several long sessions at once | 8 sequences admitted; 3 × 262K, 1 × 262K + 3 × 131K, 5 × 131K, 8 × 65K and 8 × 32K all resident (shape ladder) |
| the same context re-sent every turn | prefix caching on: a repeated 30K prompt reports 25,600 cached tokens, a repeated 131K prompt 124,800, salted controls 0; hits are 3,200-token aligned blocks with the last matched block dropped for the drafter, so prompts under two blocks (~6.4K tokens) cannot hit. See [prefix-cache loss](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm) for the case that loses hits |
| tool calls, thinking, images | Qwen3 coder tool parser, Qwen3 reasoning parser with thinking on at low effort, 2 images per request, one OpenAI-compatible front door |
| a client that expects an answer every time | retry once on `finish_reason == "stop"` with 0 completion tokens ([empty warm completion](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm)) |
| not dying at depth | 14-cut fault campaign on the transport (fail-closed on every cut), stress sequence, memory profiles with three 262K sessions plus a burst, prefix-cache correctness under forced preemption (51/51 answers correct), all maintainer-run |

## Stack

| piece | v2.2.0 |
|---|---|
| vLLM | stock tag **`v0.30.0`** (`ced6857afa`) + 58 commits = the v2.2.0 source, release commit **`9c27ca9a06`** (tree `eebcd648…`, the head of the bundle's branch `v2.2.0-public`; `release/v2.2/patches/`, a linear series that replays with `git am`; build from the bundle); `release/v2.2/v2.2.0-combined.diff` is the same delta as one applyable diff |
| compiled ops | two built from this source for sm_86 (`_C_stable_libtorch` `6cb8bc77…`, `_moe_C_stable_libtorch` `66acb123…`: the delta changes the top-k kernels and the Marlin MoE split-K switch; device code identical to the qualified builds on this model's path; built in an Ubuntu 22.04 root, so they need glibc 2.34 or newer); every other build product is the stock 0.30.0 wheel's. The 23 products ship as one tarball; both routes are hash-pinned in `upstream/PIN-v2.2` |
| environment | Python 3.13, torch 2.13.0 (CUDA 13.0), triton 3.7.1, flashinfer 0.6.18.post1, 196 pins in `release/v2.2/requirements-pinned.txt`; CUDA runtime from the venv wheels; the first serve compiles FlashInfer and Triton kernels with the pinned CUDA 13.0 nvcc wheels plus a system C/C++ compiler, ninja and the Python 3.13 headers; driver CUDA ≥ 13.0 |
| shape | as v2.0.1 below, plus `VLLM_USE_BREAKABLE_CUDAGRAPH=0` |
| PLE table | host-mapped pull transport, home GPU 0, fail-closed (`VLLM_E2_PLE_PULL_TRANSPORT=1`, `VLLM_PLE_OFFLOAD_HOME_DEVICE=0`) |
| embeddings | pinned host tables on both stages (`VLLM_HOST_EMBED_TABLE=1`) |
| determinism | `MERLIN_FULL_K`, `MERLIN_QSA_SORT`, `MERLIN_TIE_DET`, `MERLIN_TIE_RECENT`, `MERLIN_TOPK_SORTED_EMIT` = 1 |
| instrumentation | bounds guards on in `warn`; exact-restart and FP8 KV clip counters off by default (logging only; `COUNTERS=1` turns them on, as the reference host runs them) |

| piece | v2.0.1 (previous) |
|---|---|
| vLLM | fork tag **`v2.0.1`** = `ad5c3c223999de577b04cdb9caeab2dcb76b61b9`: public nightly `e962733e08` (2026-09-10) + the peakcrosser7 PLE-offload branch (#53899) + #54793 / #54795 (that base is what the scripts call `e2-base`) + 76 patches over it, 71 ours and 5 carried upstream commits (`release/v2/patches/`, reading aid; build from the bundle); `release/v2/v2.0.1-combined.diff` is the same delta as one applyable diff |
| compiled ops | upstream's at `f2e2936f9` (no C++ change in v2): the precompiled cu130 wheel `0.28.1rc1.dev450+gf2e2936f9` or the reference host's 22 extracted build products, both hash-pinned in `upstream/PIN-v2` |
| environment | Python 3.13, torch 2.13.0+cu130, flashinfer 0.6.18.post1, 199 pins in `release/v2/requirements-pinned.txt`; CUDA runtime from the venv wheels; the first serve also JIT-compiles FlashInfer and Triton kernels, which needs a CUDA toolkit no newer than the driver, a C compiler, ninja and the Python headers (not stated when v2.0.1 shipped; see the v2.2.0 toolchain notes) |
| shape | TP2 × PP2 + EP, `VLLM_PP_LAYER_PARTITION=25,23`, MTP K=3 probabilistic, FULL_AND_PIECEWISE cudagraphs with captures to 32, `max-num-seqs 8`, prefill chunk 1,024, KV pin 4.1e9, fp8_e4m3 KV with the sidecar, prefix caching, `NCCL_PROTO=LL`, `--shutdown-timeout 60` |
| PLE table | host-mapped pull transport, home GPU 0, fail-closed (`VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_OFFLOAD_HOME_DEVICE=0`) |
| embeddings | pinned host tables on both stages (`VLLM_HOST_EMBED_TABLE=1`) |
| instrumentation | bounds guards + KV clip counter compiled in, **off** in the served profile |

## What ships next

v2.2.x: making greedy output repeat across fresh compiles (the FLA chunk kernels' and Inductor's autotuning, validated
by fresh-cache equality), the guard-mode comment and default brought into line (`warn`), and the FP8 KV clip counter's
reader moved after cudagraph capture. Still open from v2: the prefix-cache retention mechanism (the hybrid KV manager's
free path under concurrent decode), the empty-completion racing pair (stream-fence / cloned-relay / all-gather
interventions and a consumption-time generation check, plus upstream #43650 and #53919), and the packers / converter /
deep-context harness that produce the checkpoint.

## Credit

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- vLLM model support and PLE offload: the [`peakcrosser7` Flash-Next branch](https://github.com/peakcrosser7/vllm/commits/release/qwen38next_offload)
  behind vLLM PRs [#53896](https://github.com/vllm-project/vllm/pull/53896) and [#53899](https://github.com/vllm-project/vllm/pull/53899);
  in the base (`e2-base`): PRs #54793 / #54795 (saichowdary007, open upstream); carried ahead of the base as the five
  upstream-authored patches: the prefix-cache chain #53614 #55747 #53945 #54713 #55450 (ZeldaHuang, yewentao256, akshaver,
  tobymao, lucamotz); ported by us: #46994, #55745, #57050 (merged upstream) and the #54442 / #56802 pair
  ([v2.0.1](docs/history.md#what-changed-in-v201)); upstream issue #54709 is the PP>1 refusal for PLE checkpoints that
  this tree works around. v2.2.0 is built on the public vLLM `v0.30.0` release, which carries the Flash-Next model
  support upstream; its delta re-ports #46994, #48532, #50021, #54442 / #56802, #54793 / #54795, #55506, #55557 and
  #57050, credited by PR number in the commit messages
- Quantised experts: [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound);
  FP8 PLE table: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4);
  MTP INT4 packing recipe adapted from [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090);
  P2P on consumer Ampere: aikitoria's open-kernel-module patch
- Independent work on the same hardware class: [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090),
  [noonghunna/club-3090](https://github.com/noonghunna/club-3090), [tfriedel/qwen3.6-rtx3090-lab](https://github.com/tfriedel/qwen3.6-rtx3090-lab)

## License

Code in this repository (patches, scripts, calibration tooling, harness): Apache-2.0 (`LICENSE`); the vLLM
fork is Apache-2.0 with the modified files listed in `NOTICE`. Model weights, including the quantised
derivatives credited above, carry the Qwen Community License 1.0 (`LICENSE.qwen-community-1.0`); they are not
part of this repository.
