# Qwen3.8-Flash-Next at 806K KV on 4× RTX 3090

*v2.0.1: an 806,792-token KV pool with 262,144-token requests — three full-context sessions resident on four consumer cards — and structured output that holds under concurrency.*

Qwen3.8-Flash-Next (125B MoE, ~6B activated per token, plus a 51B n-gram embedding table and a 4B MTP
head; Gated DeltaNet linear attention, 512 experts, vision tower) served with vLLM at its **full
262,144-token context** on four consumer Ampere cards, as the daily model behind a
[hermes](https://github.com/NousResearch/hermes-agent) agent.

**v2.0.1** (2026-09-18) is the v2 shape plus one engine fix, [structured output under concurrency](#what-changed-in-v201). **v2** (2026-09-17) keeps everything the [v1 release](#v1-the-tp4-lane) established (calibrated FP8 KV for the
sparse-attention layers, full cudagraphs, three-token speculative decoding) and changes the shape: **tensor
parallel 2 × pipeline parallel 2** with expert parallel, the FP8 n-gram table served to the GPUs over a
**host-mapped, fail-closed pull transport**, and, the piece that is entirely ours, **the token-embedding tables
of both the target model and the MTP drafter moved out of VRAM into pinned host memory**, read by a device-mapped
lookup that runs inside the captured cudagraph. The KV pool goes from 342,912 tokens (1.31 full-context requests) to **806,792 tokens: three 262K sessions
resident at once**, thinking-on single-stream decode at parity with the TP4 lane (thinking-off at 4K runs 5 % below it, documented), prefill 10–26 %
faster. The v2-versus-v1 table below comes from the pre-registered two-arm gate that qualified v2
([`benchmarks/2026-09-17/BENCH-CARD.md`](benchmarks/2026-09-17/BENCH-CARD.md)); v2.0.1 was requalified against those
medians (maintainer-reported, below); capacity, memory, fault and prefix-cache figures are maintainer-reported from
the campaign's close records, which are not published.

[![GPU](https://img.shields.io/badge/GPU-4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](#hardware)
[![Context](https://img.shields.io/badge/context-262K_per_request-ffb000)](#the-kv-budget)
[![KV pool](https://img.shields.io/badge/KV_pool-806%2C792_tokens-0969da)](#the-kv-budget)
[![Shape](https://img.shields.io/badge/TP2_x_PP2-%2B_EP-6f42c1)](#stack)
[![MTP](https://img.shields.io/badge/MTP-K%3D3-6f42c1)](#stack)
[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97_checkpoint-Qwen3.8--Flash--Next--W4A16--Merlin-ffd21e)](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)

> **Evidence boundary.** The v2 card is a maintainer measurement on the reference host with a frozen,
> hash-pinned manifest; the bench card is published, the underlying close records, raw streams and boot logs are
> not (as for v1). The v2.0.1
> requalification figures are maintainer-reported and their record is not published; the two upstream pull
> requests the fix ports are public and can be read independently. What you can reproduce independently: the
> source tree (commit **and** tree hash, from the public upstream commit plus the bundle shipped as a release
> asset), the environment (199 pinned packages), the sidecar, and the serve command. Everything else is
> maintainer-reported. Five behaviours of the architecture's upstream implementation are **documented, not
> fixed**; the empty warm completion and the prefix-cache loss remain open — read
> [Known behaviours](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm) before serving.

## What changed in v2.0.1

**Structured output now works under concurrency.** In v2, a `response_format` request could fail with HTTP 500 while
other requests were decoding, because the draft hand-off kept a single unidentified slot that the alternating
pipeline-parallel microbatch overwrote; the `-1` placeholders left behind gave the grammar bitmask an all-allowed row
and the unconstrained token then failed the state machine. v2.0.1 ports two upstream pull requests (both open upstream at the time of writing) that fix it at the source:
**PR #54442**, which refuses to leave an unmasked row for a draft slot the scheduler did not schedule, and **PR #56802**,
which keys draft snapshots by scheduler step so a request is verified against the drafts its own step consumed.

Measured on the reference host with the shipped serve command, 80 structured requests across four load cells (one, two
and three concurrent decodes with thinking on, and two with thinking off), 10 idle and 10 under load in each:

| Concurrent decodes | Thinking | v2: schema-valid responses of 10, under load | v2.0.1: same |
|---|---|---|---|
| 1 | on | 8/10 | 10/10 |
| 2 | on | 1/10 | 10/10 |
| 3 | on | 2/10 | 10/10 |
| 2 | off | 6/10 | 10/10 |

Idle was 10/10 in every cell on both. Every v2.0.1 response body validated against its schema; the engine logged no
grammar rejection and no terminated request anywhere in the run. No throughput regression was found: on the
reference host's two-boot ladder across 4K, 32K, 131K and 261K in both thinking modes, v2.0.1 lands between 0.978 and
1.028 of the v2 median, with median inter-token latency within 0.21 ms and tokens per step unchanged. The
prefix-cache equivalence block was re-run too and fails identically on v2 and v2.0.1 (same three cells, same cache-hit
total, 51/51 answers correct on both): that is the open [prefix-cache loss](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm),
unchanged by this fix.

Nothing else about the release changes: same checkpoint, same serve command, same KV budget, same pool of 806,792
tokens. Upstream PR #43650 is deliberately **not** ported, because this tree's cache coordinator already caps a Mamba
cache hit at the attention groups' post-drop length and dropping again would shrink every warm hit for no gain.
The requalification record is not published; the two upstream pull requests
([#54442](https://github.com/vllm-project/vllm/pull/54442), [#56802](https://github.com/vllm-project/vllm/pull/56802),
for [issue #54437](https://github.com/vllm-project/vllm/issues/54437)) are the independently readable evidence.

## Quick start (container)

The release form of v2 is one image that builds the pinned tree and serves it. Host requirements: four 24 GB Ampere cards
with peer-to-peer working on your driver, `nvidia-container-toolkit`, host RAM (96 GB is the qualified allocation; the
measured resident floor is about 69 GiB, see [Hardware](#hardware)), and the checkpoint
[halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) on disk (116 GiB).

```bash
git clone https://github.com/halt95/qwen38-flash-next-3090s.git && cd qwen38-flash-next-3090s
docker build -t qwen38-flash-next-3090s:v2.0.1 .      # fetches the v2.0.1 bundle + the pinned wheel; no compiler, no CUDA toolkit
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-W4A16-Merlin docker compose up -d
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json'   -d '{"model":"flash-next","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

The first start compiles the cudagraphs (about 10 minutes) into the `flash-next-cache` volume; later starts take about
three minutes. The entrypoint adds the one config key the checkpoint needs if it is missing (see
[Checkpoint](#checkpoint)). Everything the container does is in `Dockerfile`, `docker-compose.yml` and
`scripts/docker-entrypoint.sh`; the build and serve scripts run without Docker ([Build and serve](#build-and-serve)).
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
something unrelated, typically `AttributeError: '_ModelInfo' object has no attribute ...`. `scripts/serve-v2.sh`
now refuses to start in that situation and tells you what to do; the container form cannot hit it at all.

Read [Known behaviours](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm) before putting it in front of clients.

## At a glance

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
nor under the served entry's request-logging-off flags). With MTP the honest pair is the **median event interval** (v2 18.7 / 18.7 / 19.1 ms vs v1 18.6 / 18.7 / 19.2 at 4K / 32K /
131K) and **tokens per step** (3.02 / 3.07 / 3.13 vs 3.00 / 3.04 / 3.17), both in the card.

## Hardware

| resource | reference host |
|---|---|
| GPU | 4× NVIDIA GeForce RTX 3090 (Ampere sm_86, 24 GB each), **220 W** power cap, no NVLink |
| PCIe | Gen4 x16 to every card; P2P over the aikitoria open-kernel-module patch (`VLLM_SKIP_P2P_CHECK=1`, `NCCL_P2P_LEVEL=SYS`) |
| CPU / RAM | AMD EPYC 7532, 192 GB ECC; the serving container is allocated **96 GB**, the qualified figure. Measured resident floor on the v2.0.1 container: about **69 GiB** = the ~48 GiB FP8 n-gram table (anonymous memory in the offload process) + ~4.2 GiB of pinned embedding tables (1,064 MiB per rank) + ~8 GiB of shared segments + ~8 GiB across the four workers, engine core and API server; the remaining ~26 GiB up to the 96 GiB limit (98,304 MiB, what the recipes call "96 GB") is checkpoint page cache and reclaimable. The boot peak was not measured, so treat the 69 GiB floor as a hard floor (below it the load OOMs), 80 GB as the sensible minimum and 64 GB as not enough |
| disk, `/dev/shm` | ≥ 250 GB free (the checkpoint is 116 GiB); `/dev/shm` ≥ 1 GB (maintainer-measured peak +30.6 MB per boot; a `Bus error` at boot means it is undersized) |
| OS / serving | Linux container on Proxmox, vLLM behind llama-swap; the scripts here run the same engine directly |

## The KV budget

**806,792 tokens** at `--kv-cache-memory 4100000000` per rank, measured on every qualification boot. The
per-request limit stays 262,144; the pool is the aggregate over concurrent requests. Where the room came from
relative to TP4 at the same weights:

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
- The 4.1e9 pin is the highest that boots with a complete clean row set (measured on the qualification boots); 24/24 layer partitions and higher
  pins fail on the tightest card (3.8e9 was the ceiling before the host-resident embeddings).

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
(maintainer-measured)
and a PCIe read per looked-up row; the card shows no decode penalty against the TP4 lane. It is on by default
(`VLLM_HOST_EMBED_TABLE=1`) and can be turned off, which costs the 608 MiB per card back. One consequence worth
knowing if you build from source: the flag is a torch compile-cache factor (the UVA and device drafter graphs must
not share an AOT cache entry; patch `0068` registers it, and a pre-fix tree needs separate `VLLM_CACHE_ROOT`s per mode).

## Serving an agent

| agent need | as served |
|---|---|
| several long sessions at once | 8 sequences admitted; 3 × 262K, 1 × 262K + 3 × 131K, 5 × 131K, 8 × 65K and 8 × 32K all resident (shape ladder) |
| the same context re-sent every turn | prefix caching on: a repeated 30K prompt reports 25,600 cached tokens, a repeated 131K prompt 124,800, salted controls 0; hits are 3,200-token aligned blocks with the last matched block dropped for the drafter, so prompts under two blocks (~6.4K tokens) cannot hit. See [prefix-cache loss](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm) for the case that loses hits |
| tool calls, thinking, images | Qwen3 coder tool parser, Qwen3 reasoning parser with thinking on at low effort, 2 images per request, one OpenAI-compatible front door |
| a client that expects an answer every time | retry once on `finish_reason == "stop"` with 0 completion tokens ([empty warm completion](#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm)) |
| not dying at depth | 14-cut fault campaign on the transport (fail-closed on every cut), stress sequence, memory profiles with three 262K sessions plus a burst, block-5 prefix-cache correctness under forced preemption (51/51 answers correct), all maintainer-run |

## Benchmarks

The gate ran **five boots per arm, alternating v2 and v1 on the same box**, one frozen manifest, three sends
per cell, judged as medians with a 0.97 rule per depth and a 0.90 floor per boot. Full card with every boot's
numbers, the event intervals, tokens per step, the fingerprint and the environment:
[`benchmarks/2026-09-17/BENCH-CARD.md`](benchmarks/2026-09-17/BENCH-CARD.md).

| cell | verdict |
|---|---|
| decode, thinking on, 4K / 32K / 131K | 0.997 / 1.009 / 0.988 of v1 — **PASS** at every depth |
| decode, thinking off, 32K / 131K | 1.008 / 1.024 — PASS |
| decode, thinking off, **4K** | **0.952 — FAIL against the 0.97 rule**, shipped documented (a draft-acceptance effect of the shape, not a v2 patch: it is identical with the readers on or off): event interval identical (18.66 vs 18.64 ms), tokens per step 2.14–2.31 vs 2.31, clustered per boot; cause not established |
| prefill 10K / 100K | +10 % / +26 % — PASS |
| quality vs teacher | delta −0.0009, well inside the 0.0015 bound — PASS. (An earlier single-capture quality screen recorded **FAIL** and stays cited as such; the two-arm gate replaced it as the instrument) |
| pool, capacity, tools, no-think, faults | PASS (see card) |

What the earlier candidates looked like: with the two instrumentation reader threads on and the default
NCCL protocol, the thinking-on rows sat at 0.949 / 0.962 of v1 (event interval 19.10 vs 18.56 ms at 4K). The
readers (a bounds-guard telemetry reader and the KV clip counter, each polling every 50 ms) cost ~0.3–0.5 ms
per step on the pipeline's critical path; v2 ships them off and `NCCL_PROTO=LL` on, which is the whole
difference (maintainer-measured).

## Known behaviours of the Qwen3.8-Flash-Next architecture in vLLM

These are behaviours of the Qwen3.8-Flash-Next architecture as implemented upstream in vLLM (the hybrid
Gated-DeltaNet / sparse-attention / MTP execution path and its hybrid KV manager). They are **not** introduced by
this checkpoint's quantisation and not by the v2 patches: the empty warm completion is an upstream-reported class and was seen on the v1 TP4
lane; the prefix-cache loss did not move across the five release candidates (v2.0.1 included) when the v2 deltas were bisected (a run on the unmodified
upstream base is still owed); the ring-row death is in upstream code. v2 documents them, ships mitigations where
one exists, and tracks them; none affects the correctness of answers in the gate.

- **Empty warm completion — an occasional empty completion on a warm repeat of a long cached prefix** (upstream class: vllm-project/vllm #53912, prefix caching + speculative decoding on hybrid models; seen on the v1 TP4 lane too). HTTP 200,
  `finish_reason: "stop"`, zero tokens: the first sampled token is EOS. Per boot, not per request (roughly one
  boot in three on the no-MTP diagnostic shape; three empty warm completions in 23 gate boots across gate runs 1–4 on
  the rc3/rc4 candidates, two of them on the v1 TP4 arm; none in the 10 boots of the final run). The cached bytes are proven identical between a call that flips and
  the calls around it; the race is inside the flipping request's own forward, it needs pipeline parallel plus
  async scheduling, and async scheduling is required with MTP under PP on this fork. Mitigation shipped: **retry
  once**; it returned the correct answer in every observed case (a mitigation, not a guarantee; the rate on the
  served profile is not measured). Maintainer-reported.
- **Prefix-cache loss — prefix-cache blocks of sessions that finish while other long sessions are still decoding are dropped**
  (upstream hybrid KV manager path). Not preemption (reproduced with zero preemptions). The variable that separates the
  arms is the survivors' remaining decode: with 320-token outputs only the first-finished session loses its blocks,
  with 2,048-token outputs every session does, and a single long session plus a burst retains everything. Free-block
  state during the episode was not sampled, so eviction under pool pressure is not excluded. Reproduced on rc3, rc4 and
  rc7 (one boot each, one shared compile cache); the one rc5 boot kept the last-finished session; re-run on v2.0.1 with
  an identical failing set, identical cache-hit total and 51/51 answers correct. Cost: first-token latency on that
  session's next turn; the answer is unaffected.
- **Thinking-off decode at 4K** is 0.92–0.99 of the TP4 lane on every one of the five boots, four of them below the gate's 0.97 rule (above); it is the one cell that makes the judge's machine verdict for the run FAIL.
- **A rare illegal-address engine death** (upstream sparse-attention code path; two occurrences over the whole campaign;
  mechanism: a re-claimed, never-zeroed ring row read as a RoPE position). Bounds guards turn a recurrence into a named, counted failure when their reader is on;
  the reader is **off by default**, so a masked fault at an always-fatal site is counted but never read or raised (`GUARD=warn` in `serve-v2.sh` turns the reader on at the cost
  above). Not fixed; no rate is claimed.
- **Greedy T=0 is not byte-reproducible** with this architecture (bf16 near-ties resolved differently by the sparse
  indexer's top-k and the expert permutation); documented, accepted.
- Fixed on the way and worth knowing if you run an older candidate: a worker hang at the end of very long
  prefills (the instrumentation readers deadlocking under the CUDA context lock; fixed in rc6, readers off
  in v2) and the upstream Mamba admission-estimate regression (#57050, ported in rc4).

## Checkpoint

The same checkpoint as v1, [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
(Intel AutoRound INT4 g128 experts, the RadixArk FP8 n-gram table, MTP draft experts INT4 and GDN projections
INT8 packed in place, attention BF16; lineage in its model card), **plus one config key**: v2 keys the FP8
n-gram table on `ple_embedding_dtype: float8_e4m3fn` inside the `text_config` object of `config.json` (the engine reads its text config, so a top-level key does nothing) instead of the `VLLM_PLE_FP8_GLOBAL_SCALE`
environment variable v1 used (the env opt-in is still honoured for a checkpoint without the key). the checkpoint on Hugging Face carries the key since 2026-09-18; for a download that predates it, `scripts/make-e1-config.py` adds it (and is a no-op otherwise); every weight file stays byte-identical
(the reference host's serving copy is hard links to the published shards, verified 2026-09-17). Shard hashes are not
published here; the Hugging Face checkpoint carries them, and the v2 delta is the one config key above.

The KV scale sidecar is **the file v1 shipped**, `scales/qsa_kv_scales_262k.json` (Merlin-calibrated, margin
1.10, sha256 `5cbe6ae8…`); the reference host serves the same bytes under the name
`qsa_kv_scales_262k_merlin_v3_m110.json`. One file ships.

## Stack

| piece | v2 |
|---|---|
| vLLM | fork tag **`v2.0.1`** = `ad5c3c223999de577b04cdb9caeab2dcb76b61b9`: public nightly `e962733e08` (2026-09-10) + the peakcrosser7 PLE-offload branch (#53899) + #54793 / #54795 (that base is what the scripts call `e2-base`) + 76 patches over it, 71 ours and 5 carried upstream commits (`release/v2/patches/`, reading aid; build from the bundle); `release/v2/v2.0.1-combined.diff` is the same delta as one applyable diff |
| compiled ops | upstream's at `f2e2936f9` (no C++ change in v2): the precompiled cu130 wheel `0.28.1rc1.dev450+gf2e2936f9` or the reference host's 22 extracted build products, both hash-pinned in `upstream/PIN-v2` |
| environment | Python 3.13, torch 2.13.0+cu130, flashinfer 0.6.18.post1, 199 pins in `release/v2/requirements-pinned.txt`; CUDA runtime from the venv wheels (no toolkit needed to serve) |
| shape | TP2 × PP2 + EP, `VLLM_PP_LAYER_PARTITION=25,23`, MTP K=3 probabilistic, FULL_AND_PIECEWISE cudagraphs with captures to 32, `max-num-seqs 8`, prefill chunk 1,024, KV pin 4.1e9, fp8_e4m3 KV with the sidecar, prefix caching, `NCCL_PROTO=LL`, `--shutdown-timeout 60` |
| PLE table | host-mapped pull transport, home GPU 0, fail-closed (`VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_OFFLOAD_HOME_DEVICE=0`) |
| embeddings | pinned host tables on both stages (`VLLM_HOST_EMBED_TABLE=1`) |
| instrumentation | bounds guards + KV clip counter compiled in, **off** in the served profile |

## Build and serve

| path | what |
|---|---|
| `upstream/PIN-v2` | base commit, bundle prerequisites, tag commit and tree hash, precompiled-wheel identity and hash, artefact tarball hash |
| `release/v2/patches/0001..0076` | the full series over `e2-base`, for reading (patch subjects carry the campaign's internal task and review labels; they are not renamed because the files are hashed); the history has merge commits, so `git am` cannot replay it (it stops at patch 43). These copies are byte-identical to the reference package and verify against `SHA256SUMS.v2.0.1`; contributors whose work upstream authored keep their own attribution. The bundle remains the source of truth |
| `release/v2/v2.0.1-combined.diff` | one `git diff e2-base..v2.0.1` (112 files, text only, 777 KiB): `git apply` it on `e2-base` (the release asset `e2-base-src.tar.gz`, sha256 `fcd14214f64faaa4175d62f6a51ca39c7bc09d15bf75c190d4dfcc88846b927f`) and you have the tagged source; checked to apply cleanly and to give the tag's tree hash. The way to reproduce the tree from the patches directory without the bundle (`build-v2.sh` itself uses the bundle) |
| `release/v2/requirements-pinned.txt`, `build-artifacts.list`, `SHA256SUMS.v2.0.1`, `PACKAGE-MANIFEST.md` | the environment pins, the 22 build products, the hashes of the release assets, the combined diff and the reference patch series (`cd release/v2 && sha256sum -c --ignore-missing SHA256SUMS.v2.0.1` verifies 78 of 81 from a checkout; the other three are the release assets), the file manifest |
| `scripts/build-v2.sh` | fetch the three prerequisite commits from GitHub + the bundle (release asset), check out `v2.0.1`, assert commit and tree, fresh venv from the pins, compiled ops from the wheel or the tarball, metadata-only install. One prerequisite is a PR-branch head (#53899); if it ever disappears upstream the bundle alone cannot be applied — a full bundle is the fallback |
| `scripts/serve-v2.sh` | the served entry with every variable exported; port, host, names (default `flash-next-v2 flash-next flash-mtp flash-next-mtp`), PLE home, sidecar and cache dir configurable |
| `Dockerfile`, `docker-compose.yml`, `scripts/docker-entrypoint.sh` | the container: `build-v2.sh` at image build (bundle from the release URL or the build context), `serve-v2.sh` as the entrypoint, checkpoint and cache as mounts, the config key added on first start if the mount is writable |
| `lxc/pve-create.sh`, `lxc/provision.sh`, `lxc/flash-next.service` | the Proxmox LXC form of the same thing: create the container with the device nodes and mounts, provision it (NVIDIA userspace, `build-v2.sh`, config key, systemd unit), serve on boot |
| `scripts/make-e1-config.py` | adds the one config key to the published checkpoint, inside `text_config`, and re-parses the result to prove it landed where the engine reads it |
| `scales/qsa_kv_scales_262k.json`, `calib/` | the sidecar and how it was made (unchanged from v1) |

```bash
# release assets: v2.0.1-from-upstream-e962733e08.bundle.gz (required), build-artifacts-sm86-py313-cu130.tar.gz (optional),
# e2-base-src.tar.gz (only for the combined-diff route), build-artifacts.list (already in the tree)
gunzip v2.0.1-from-upstream-e962733e08.bundle.gz
BUNDLE=./v2.0.1-from-upstream-e962733e08.bundle scripts/build-v2.sh ./vllm-v2 ./venv-v2
python scripts/make-e1-config.py /path/to/Qwen3.8-Flash-Next-W4A16-Merlin   # no-op if the config already carries the key
TREE=./vllm-v2 VENV=./venv-v2 HOST=0.0.0.0 scripts/serve-v2.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin
```

Four known deltas between the gate command and the shipped one: the gate ran with `--enable-log-requests
--enable-request-id-headers` (request logging on; the served entry has it off; no throughput effect is claimed either
way); the served entry adds `--override-generation-config` (T=1.0 / top-p 0.95 / top-k 20) and
`--default-chat-template-kwargs` (thinking on, low effort), which the gate did not pass (it sent T=0 with a seed per request);
`serve-v2.sh` also exports `HF_DATASETS_OFFLINE=1`; and the reference host exports `LD_LIBRARY_PATH=/usr/local/cuda-13.3/...`, which is inert: every CUDA library the
processes map comes from the venv wheels (`/proc/<pid>/maps` checked on a qualification boot), so `serve-v2.sh` does
not set it.

Requirements: Linux x86-64, git, curl, Python 3.13 with venv, four visible 24 GB NVIDIA GPUs with working
peer-to-peer on your driver (assumed, not validated), host RAM (96 GB qualified; ~69 GiB measured floor), `/dev/shm` ≥ 1 GB. The first boot
compiles the graphs (~10 min); later boots reuse `CACHE_ROOT`. (The reference host's qualification arms shared one compile cache across candidates; give production a v2-named cache root.) Reproduction on the reference host from the
bundle: commit and tree equal to the tagged worktree, package hashes verified, fresh venv, one qualification
boot 17/17 rows (maintainer-reported).

Never reuse an existing venv for this tree, and do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
(v1 measured it at −27 % four-stream aggregate).

Process model: the workers set `PR_SET_PDEATHSIG(SIGKILL)` so an engine-core death takes them with it, and non-fork
start methods are refused. The death signal fires when the *thread* that spawned the workers exits, so an embedded
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

## How v2 got there

The three steps that account for the result (maintainer-reported):

| step | result | what it turned out to be |
|---|---|---|
| PLE host-mapped **pull** transport | decode +4–10 % vs push on the complete re-run, pool **806,792** at 4.1e9 | the push transport paid a PCIe round trip per step; the pull transport's first cut leaked a 322 MiB CUDA context into the offload process (an aux stream bypassing lazy init), fixed |
| **pinned-host token embeddings (ours)** | 608 MiB freed per rank → pin 3.8e9 → 4.1e9, pool 748K → 806,792 | the drafter lookup sits inside a full cudagraph, so a CPU gather was rejected; a device-mapped UVA lookup, bytewise-identical to the device table, ships instead |
| the decode gap to TP4 (0.95–0.96) | closed: 0.997 / 1.009 / 0.988 | the two 50 ms reader threads on the PP1 critical path + NCCL protocol; the pipeline itself is serial (stage 0 computes ~9.4 ms then waits in the token broadcast; stage 1 waits ~7.8 ms then computes ~13.3 ms) |

## What ships next (v2.1)

The prefix-cache retention mechanism (the hybrid KV manager's free path under concurrent decode), the empty-completion racing pair
(stream-fence / cloned-relay / all-gather interventions and a consumption-time generation check, plus upstream
#43650 and #53919), the 4K thinking-off acceptance deficit, the true fix for the ring-row fault, and the
packers / converter / deep-context harness that produce the checkpoint.

## v1, the TP4 lane

The v1 files stay in this repository as shipped (`scripts/build.sh`, `scripts/serve.sh`, `patches/`, `upstream/PIN`,
`benchmarks/2026-09-05`, `benchmarks/2026-09-08`): vLLM v0.28.0 + the eight community commits + six patches,
TP4 + EP, MTP K=3, KV pin 2.6e9 (2.9e9 = pool 342,912 since 2026-09-10). It remains the fallback entry on the
reference host.

## Credit

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- vLLM model support and PLE offload: the [`peakcrosser7` Flash-Next branch](https://github.com/peakcrosser7/vllm/commits/release/qwen38next_offload)
  behind vLLM PRs [#53896](https://github.com/vllm-project/vllm/pull/53896) and [#53899](https://github.com/vllm-project/vllm/pull/53899);
  in the base (`e2-base`): PRs #54793 / #54795 (saichowdary007, open upstream); carried ahead of the base as the five
  upstream-authored patches: the prefix-cache chain #53614 #55747 #53945 #54713 #55450 (ZeldaHuang, yewentao256, akshaver,
  tobymao, lucamotz); ported by us: #46994, #55745, #57050 (merged upstream) and the #54442 / #56802 pair above; upstream
  issue #54709 is the PP>1 refusal for PLE checkpoints that this tree works around
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
