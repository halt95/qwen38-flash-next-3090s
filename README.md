# Qwen3.8-Flash-Next at 262K on 4× RTX 3090

Qwen3.8-Flash-Next (125B MoE, ~6B activated per token, plus a 51B n-gram embedding table and a 4B MTP
head; Gated DeltaNet linear attention, 512 experts, vision tower) served with vLLM at its **full
262,144-token context** on four consumer Ampere cards, as the daily model behind a
[hermes](https://github.com/NousResearch/hermes-agent) agent: long tool-calling sessions, an orchestrator
plus a subagent in flight, images in the context.

Two things make it work. A calibrated **FP8 E4M3 KV path for the sparse-attention layers that Ampere
can actually run**, holding **307K tokens of KV** next to 18.5 GiB of weights per card; BF16 KV at this pin
was not measured, and the 1.48× ratio measured at 32K puts it short of one 262K request. And a decode step made to carry more tokens: full cudagraph coverage plus
three-token speculative decoding, so single-stream decode is **163–174 tok/s and flat from 4K to 260K
prompt tokens**.

[![GPU](https://img.shields.io/badge/GPU-4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](#hardware)
[![Context](https://img.shields.io/badge/context-262K_served-ffb000)](#the-kv-budget)
[![KV cache](https://img.shields.io/badge/KV-FP8_E4M3_calibrated-0969da)](#the-kv-budget)
[![MTP](https://img.shields.io/badge/MTP-K%3D3-6f42c1)](#benchmarks)
[![Vision](https://img.shields.io/badge/Vision-on-6f42c1)](#stack)
[![Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97_checkpoint-Qwen3.8--Flash--Next--W4A16--Merlin-ffd21e)](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)

> **Evidence boundary.** The bench cards (`benchmarks/2026-09-05/`, `benchmarks/2026-09-08/`) are maintainer
> measurements with live fingerprints; the drivers, raw streams and gate logs are not included. What you can
> reproduce independently is the sidecar (`calib/dumps-…`; the merge reproduces the file) and the build.
> Every other number (prefix-cache timings, needle depths, quality tables, packing steps, gate results,
> build timings) is maintainer-reported from runs whose logs are not in this release. The checkpoint is
> on Hugging Face: [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin);
> the packers and the deep-context harness are not yet published.

## At a glance

Both columns are the reference host's llama-swap entries, measured 2026-09-08 through the front door,
the served entry on the shipped sidecar.

| | served entry: MTP K=3, Merlin checkpoint | no-MTP fallback entry (VnimanieAI-lineage checkpoint) |
|---|---|---|
| context | **262,144** | 262,144 |
| KV pool, FP8 tokens, whole box | **307,487** (1.17× a full-context request) | 284,359 |
| decode, single stream, 4K / 32K / 131K / 260K prompt | **166 / 174 / 172 / 163** tok/s | 78 / 78 / 77 / 78 |
| time to first token, same depths | 0.9 / 7.0 / 32.0 / 75.7 s | 0.8 / 6.5 / 30.5 / 72.7 s |
| same 100K prompt again (prefix cache, 2026-09-02) | 1.6 s instead of 41 s | 0.7 s |
| needle retrieved exactly at | 259K (2026-09-03 K=2 gate); 207,495 tokens in the shipped-sidecar gate | 259K (2026-09-03 K=2 gate) |
| concurrent sequences (`max-num-seqs`; the pool holds ~1.2 full-context requests, not 2 or 4) | 2 | 4 |
| vision | on, 2 images per request | on |
| weights per card | 18.5 GiB | 18.1 GiB (the packed VnimanieAI checkpoint, measured on its MTP launch) |

Decode is the decode-window rate, (tokens − 1) ÷ (last token − first token), client-timed over a streamed
completion of up to 256 tokens, three repeats, a fresh nonce per prompt so nothing is cached, the entry's
own sampling and thinking defaults.

## Hardware

This is a 3090 system. Every number in this repo comes from this box.

| resource | this box |
|---|---|
| GPU | 4× NVIDIA GeForce RTX 3090 (Ampere sm_86, 24 GB each), **220 W** power cap, no NVLink bridge |
| motherboard | HUANANZHI H12D-8D V2.0 (AMD SP3, four PCIe 4.0 x16 slots), AMI BIOS 2.2 (2026-01) |
| CPU / RAM | AMD EPYC 7532 (32-core Rome), 192 GB ECC RDIMM; the serving container is limited to 96 GB, and the ~48 GiB FP8 PLE table lives in that host RAM |
| PCIe | Gen4 x16 to every card (`LnkSta: 16GT/s x16` on all four) |
| driver | NVIDIA 610.43.02 with the open-kernel-module P2P patch, hence `VLLM_SKIP_P2P_CHECK=1` / `NCCL_P2P_LEVEL=SYS` in the launch |
| OS / serving | Linux container on a Proxmox host, vLLM behind llama-swap; the scripts here run the same engine directly |

## The KV budget

One full-context request plus room to breathe. Where each card's ~23.6 GiB goes on the served entry:

| per card | GiB | note |
|---|---:|---|
| weights (experts INT4 g128, attention BF16, draft experts INT4, GDN projections INT8) | 18.5 | TP=4 + EP |
| KV cache, pinned | 2.42 | `--kv-cache-memory 2600000000`; the profiler's own choice OOMs under concurrent deep prompts |
| CUDA graphs | ~0.6 | FULL_AND_PIECEWISE, captures [1, 4, 8] |
| PLE offload worker context | ~0.3 | the offload worker holds a context on every card |
| prefill transients, headroom | ~1.7 | `--max-num-batched-tokens 2048`; 4096 engine-killed at a 207K prefill on an earlier budget and at 262K survived only with the KV pin lowered, peaking at 24.1 of 24 GiB |

FP8 KV versus BF16, measured on the 2026-08-27 lane at 32K with each dtype at its own profiler-chosen pin:
FP8 391,955 tokens versus BF16 264,932, **1.48×**. A BF16 pool at the served pin was not measured; that ratio would
put it near 208K, short of one full-context request. The 1.7× bytes-per-token figure is real but does not
convert into a pool that holds up. Three things make the FP8 path serve rather than merely boot:

1. **A reader Ampere can run.** Triton on sm_86 has no `fp8e4nv` type; the dequant is integer bit
   manipulation in registers plus FP16/BF16 arithmetic for the scale, ~5% decode cost. Against the BF16-KV
   teacher, 1,495 teacher-forced tokens sat inside the BF16-vs-BF16 noise floor, and at 32K the FP8-KV + MTP
   arm matched BF16-KV/no-MTP on HumanEval (156 vs 154 of 164), exact recall (160/160) and tool structure
   (150/150).
2. **Calibrated per-layer scales**, collected at the 262K shape and shipped as a sidecar. Without it the
   FP8 path runs at scale 1.0.
3. **A runtime clip counter inside the captured graph** (a Python-side counter is a silent no-op under
   FULL cudagraph replay). On an earlier sidecar and counter revision, real traffic showed tens of clipped elements out of
   ~10⁸ after a 207K prefill, while a known-bad sidecar fires thousands per layer within seconds; the
   shipped sidecar's counts are in the gate table below.

## Serving an agent

| agent need | as served |
|---|---|
| a long session that keeps growing | 262,144 context; needle exact at 259K (2026-09-03 gate) and at 207,495 tokens in the shipped-sidecar gate; a cold 260K prompt reaches first token in ~76 s (prefill ~3,300–4,000 tok/s at depth) and decodes at the 4K rate |
| the same context re-sent every turn | prefix caching on: a repeated 100K prompt returned in 1.6 s instead of 41 s (2026-09-02 entry). On the sibling Qwen3.8-27B lane, hits land in ~1.2K-token aligned blocks, so short shared prefixes did not hit in that test; expected here, not measured on this model |
| an orchestrator and a subagent at once | 2 sequences admitted (the pool covers both only while their combined context fits ~307K tokens); generation aggregate 211–240 tok/s at 8K prompts, 217–233 at 16K, zero errors |
| four subagents at shorter context | the same MTP K=3 checkpoint relaunched at `--max-num-seqs 4` with captures `[1,4,8,12,16,20]` (not the shipped entry, which admits 2): 386–403 tok/s at 8K, 389–393 at 16K, near-linear, zero errors, acceptance 2.40 on that greedy probe. The no-MTP entry at four streams reads ~255 |
| tool calls, thinking, images | Qwen3 coder tool parser, Qwen3 reasoning parser with thinking on, 2 images per request, one OpenAI-compatible front door |
| not dying at depth | a deep battery (text, 1–2 images, a needle at 200–259K depending on the run, two-stream decode) ran before each documented promotion; three configurations that passed every shallow test died at 200K+ prompt tokens and the battery caught them (maintainer-reported). An earlier K=2 served configuration once stalled for ~44 min on a ~260K request; fresh-process reruns completed normally and the cause is unknown; not reproduced on the shipped K=3 entry |

## Benchmarks

As served on 2026-09-08, through the front door, no per-request overrides (thinking on at low effort, the
entry's own sampling, an image in every depth prompt), the MTP entry on the shipped sidecar, which is the
configuration `scripts/serve.sh` transcribes. Full card with CVs, live fingerprints and the integrity
panel: [`benchmarks/2026-09-08/BENCH-CARD.md`](benchmarks/2026-09-08/BENCH-CARD.md). Zero errors on both
entries. The clip counter logged 2 warning lines on the MTP entry over the whole run and none on the no-MTP
entry, whose recorded environment has no counter enabled, so that zero means "not instrumented". Per-window
acceptance was 2.43 on these image-bearing prompts, against 2.78 lifetime on text.

The same protocol on 2026-09-05, before the sidecar was recalibrated, is in
[`benchmarks/2026-09-05/BENCH-CARD.md`](benchmarks/2026-09-05/BENCH-CARD.md): the previous sidecar logged 38
clip lines on the MTP entry (the later gate showed it clipping on most target layers), acceptance 2.37, and
decode 163–174 tok/s across the depths. The two cards broadly agree; the deepest row moved from 174.4 to
162.5 tok/s, more than either card's CV, and three repeats do not establish equivalence.

### Decode does not fall with depth

| prompt tokens | served MTP K=3: TTFT / decode | no-MTP: TTFT / decode | alesha-pro reference (no MTP): TTFT / decode |
|---:|---|---|---|
| ~4K | 0.9 s / **166** | 0.8 s / 78 | 2.1 s / 66 |
| ~32K | 7.0 s / **174** | 6.5 s / 78 | 16.7 s / 65 |
| ~131K | 32.0 s / **172** | 30.5 s / 77 | 71.3 s / 69 |
| ~260K | 75.7 s / **163** | 72.7 s / 78 | 162 s / 66 |

Reference rows: `alesha-pro/qwen38-flash-next-4x3090`, `benchmarks/2026-08-28/raw/b5-decode`, PCIe Gen3,
FULL_DECODE_ONLY, that project's own instrument (window closed at stream end, temperature 0), so a
cross-configuration comparison.

### Parallel streams

Generation-only aggregate over the window in which every stream is decoding. The 1K column is the bench
card's as-served protocol; the 8K/16K probes set their own sampling (thinking off, fixed 256–512 tokens)
and are maintainer-reported. Rows are not one configuration: 1 and 2 streams are the served MTP K=3 entry;
the 4-stream 8K/16K cells are that checkpoint relaunched at 4 sequences with captures `[1,4,8,12,16,20]`,
which this repo does not ship; the 1K column is the 2026-09-08 card, entry named per cell.

| streams | 8K prompt | 16K prompt | 1K prompt (2026-09-08 card, 512 out) |
|---|---|---|---|
| 1 | 120–128 | 119–130 | 123 |
| 2 | 211–240 | 217–233 | **233** (114 per stream) |
| 4 | **386–403** (4-seq relaunch) | **389–393** | 256 on the no-MTP entry (64 per stream) |

The capture list must cover `seqs × (K + 1)`: on the earlier K=2 entry, captures [1, 3] left the 6-token
two-stream step without a graph and throughput dropped to 18 tok/s per stream until [1, 3, 6] was served.

### Draft length

Swept on the served checkpoint, each arm alone, graph sizes [1, K+1, 2(K+1)] for K=3 and K=4; the K=2 arm
was the then-served [1, 3] configuration. Maintainer-reported; ranges over three runs, not paired.

| draft tokens K | accepted per step | KV pool | decode 4K / 32K / 131K | 2 streams | needle at 250K |
|---|---|---|---|---|---|
| 2 | 2.36 | 314K | 140–147 / 148 / 155–170 | 18 per stream (no 6-token graph) | exact |
| **3 (served)** | 2.78 | 307K | **152–173 / 164–184 / 172–175** | 220 | exact |
| 4 | 3.01 | 298K | 156–181 / 171–176 / 171–182 | 213 | exact |

K=3 over K=2, comparing the ranges (bounds from unpaired ranges, not a confidence interval): about +3–24%
at 4K, +11–24% at 32K, +1–13% at 131K, for 2% of the pool. K=4 raised acceptance with no clear speed
advantage in these samples.

### Quality, paired

The 2026-09-03 gate compared the packed VnimanieAI lineage and the packed Intel lineage at the then-served
shape (MTP K=2, captures [1, 3, 6], FP8 KV), paired per question, thinking off, temperature 0, McNemar on
the disagreements. All three point estimates favoured Intel; none was individually significant
(p = 0.11 / 0.34 / 0.44).

| experts | weights per card | GSM8K-1319 | ARC-Challenge-1172 | MMLU-1000 | needle at 259K |
|---|---|---|---|---|---|
| VnimanieAI W4A16 g128, packed | 18.1 GiB | 95.68% | 96.93% | 86.60% | exact |
| **Intel AutoRound W4A16 g128, packed (served)** | 18.5 GiB | 96.44% (+0.76) | 97.27% (+0.34) | 87.30% (+0.70) | exact |

A cyankiwi AWQ g32 checkpoint was also tried: 20.9 GiB per card packed, out of memory at a 60K prefill,
so it could not be gated at this shape.

## Checkpoint

Published at [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
with the sidecar and its model card. It is assembled, not trained:

1. experts from [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
   (INT4 g128 symmetric), converted losslessly from `auto_round:auto_gptq` to compressed-tensors;
   attention projections kept BF16 as Intel shipped them;
2. the ~48 GiB FP8 n-gram (PLE) embedding table from
   [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), replacing
   Intel's 102 GB BF16 shard, CPU-offloaded;
3. **in-place packing of the BF16 tensors both leave behind**: the MTP draft experts to INT4 g128 and the
   Gated DeltaNet projections (`in_proj_qkv`, `in_proj_z`, `out_proj`) to INT8 per-channel, −1.3 GiB per
   card, packing gate non-inferior on task accuracy and needle (earlier lineage), greedy agreement 98.1 → 97.4%. Each pack has a fail-closed
   audit through the consumer library's own unpacker (a self-consistent round-trip through our own code
   once served noise: compressed-tensors nibbles are offset-unsigned, not two's complement);
4. calibrated FP8 KV scales for the 12 sparse-attention layers, collected at the 262K shape.

No VnimanieAI tensors are in it; that lineage is the baseline it replaced. The 117 self-attention tensors
stay BF16: packing them would buy ~0.37 GiB per card on a lane that already holds its design point, was
not evaluated for speed or quality, and could introduce error on the one path every token takes.

## Stack

| component | setting |
|---|---|
| engine | vLLM v0.28.0 + the eight `peakcrosser7` Flash-Next commits + the six patches here |
| parallelism | tensor parallel 4 + expert parallel |
| KV cache | FP8 E4M3 with calibrated per-layer scales, pinned at 2.6 GB per card |
| speculative decoding | MTP head, 3 draft tokens, probabilistic draft sampling |
| CUDA graphs | `FULL_AND_PIECEWISE`, captures `[1, 4, 8]` (`seqs × (K+1)`) |
| prefill | chunked, 2,048 tokens per step |
| prefix caching | on |
| vision | on, 2 images per request |
| tool calls / reasoning | Qwen3 coder tool parser, Qwen3 reasoning parser, thinking on |
| PLE | FP8 table in host RAM; offload worker homed on GPU 3, off rank 0's card, which was the FULL-capture OOM site at 262K |

## Build and serve

Everything needed to build the engine is here; the packers that produce the checkpoint follow later.

| path | what |
|---|---|
| `upstream/PIN` + `upstream/pinned/` | the base: `vllm-project/vllm` at tag **v0.28.0**, the patched-tree hash, the compiled-ops wheel URL and its sha256; the files our patches touch, vendored as they stand after the upstream patches |
| `patches/upstream/0001..0008` | the community Flash-Next support + PLE-offload commits, unchanged, with their authors |
| `patches/0001..0006` | ours, `git am`-clean on that base: FP8 QSA reader + strict scale-sidecar loader, absmax collector, Ampere csrc declaration guard, clip counter, PLE home-device, routed-experts capturer |
| `scripts/build.sh` | clone the tag, apply 8 + 6 patches, verify the vendored files and tree hash, download the pinned wheel and check its hash, install the tree against it, rebuild the sm_86 GDN decode kernel and assert its schema (`--no-csrc` to skip) |
| `scripts/serve.sh` | the served entry with every variable exported; port, host and model aliases configurable; `NO_MTP=1` for a no-speculation entry |
| `scales/qsa_kv_scales_262k.json` | the calibrated K/V scales for this checkpoint, 12 target-model layers, margin 1.10 |
| `calib/` | how that sidecar was made: calibration launch, traffic script, rank-dump merge, and the four rank dumps |

```bash
scripts/build.sh ./vllm-src ./venv
VENV=./venv HOST=0.0.0.0 scripts/serve.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin
```

Requirements: Linux x86-64, git, curl, Python 3.10–3.14 with venv, CUDA 13 toolkit, gcc ≥ 11.3 (vLLM's
CMake floor), and working GPU peer-to-peer on your driver, which the scripts assume and do not validate.
The source and the compiled-ops wheel are pinned; python packages resolved by pip are not locked. The
reference build resolved torch 2.13.0 and triton 3.7.1; the GDN decode kernel is rebuilt because the branch changed
its signature. Maintainer-reported reproduction on the reference host from a fresh clone of the release tree: `build.sh`
on an empty venv took 1,115 s and the server it produced passed the sidecar gate (4/4, 0 increments); the
final patch revision was then rebuilt and gated again (row two below).

In `serve.sh`, the QSA variables, `VLLM_PLE_FP8_GLOBAL_SCALE` and `VLLM_PLE_OFFLOAD_HOME_DEVICE` come from
the patches, the P2P/NCCL pair from the reference host's driver setup, and the explicit sampling override
is the served entry's; the rest is stock vLLM. Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`:
in a memory-matched A/B on the no-MTP entry (2026-08-28), removing it took single-stream from 38.1 to
43.2 tok/s and four-stream aggregate from 112.7 to 154.7.

### The scale sidecar

`scales/qsa_kv_scales_262k.json` was calibrated on this checkpoint on 2026-09-08 with `calib/`: eager,
no MTP, text at 1,971 / 7,989 / 31,975 / 127,805 / 199,888 / 254,875 prompt tokens plus image prompts
at 2,848 and 8,841 tokens, then a flush phase so the collector's last dump (every 2,000 layer calls, not
a timer) lands after the deepest request. The four rank dumps are in `calib/dumps-2026-09-08-merlin/`;
merged at `--margin 1.10` they reproduce the shipped file byte for byte. A collection without the flush
gave per-layer maxima within −6.0% to +6.3% of these (flush run against the no-flush run).

Gated at the served shape (MTP K=3, FULL graphs, 262K, prefix caching on, served sampling), each arm a
fresh server: single-stream 4K / 32K / 131K / 250K ×3, N=1 and N=2 ladder ×3, and the daily lane check
(text, 1 image, 2 images, needle at 207,495 tokens with thinking on). Three alternatives got the same gate.

| sidecar | daily check | clip-counter increments during the gate | decode 4K / 32K / 131K / 250K, tok/s, 3 runs |
|---|---|---|---|
| **this file** (Merlin, margin 1.10, with flush) | **4/4** | **24**: layer 47 V +20, layer 3 V +2, layer 35 K +2 | 158–171 / 163–172 / 164–169 / 157–180 |
| **this file**, served by a build from this repo (`build.sh` on a clean venv, final patch revision: strict loader, float32 clip ceilings) | **4/4** | **0** | 151–178 / 157–168 / 158–167 / 139–176 |
| Merlin, margin 1.10, no flush | 4/4 | 6: layer 47 V | 157–169 / 160–172 / 160–182 / 164–176 |
| Merlin, margin 1.0 | 4/4 | 210 across 9 of 12 layers | 158–172 / 170–179 / 161–166 / 167–172 |
| previous sidecar (calibrated on the VnimanieAI lineage) | 3/4 once (needle answered empty); three rerun batteries 4/4, but the rerun needles were prefix-cache hits after the first, so one independent prefill per arm; cause undetermined | 496 across 9 layers (47 V +148, 35 K +66, 15 K/V +52/+48) | 163–170 / 162–178 / 158–165 / 156–178 |

Reading it: the counter increments per clipped element over every row the kernel touches, padded rows
included, so a small residual is reported, not explained; hundreds across most layers versus a few dozen on
three is the signal. Raw K/V absmax on this checkpoint differs from the VnimanieAI lineage's by −39% to
+12% per layer, so the old sidecar's ceiling sat below this checkpoint's activations on several layers.
Decode showed no resolved difference between arms at three samples per depth. The old sidecar's one empty
needle answer did not reproduce; cause undetermined. The draft head's own attention layer is not in the
sidecar and runs at scale 1.0.

## How it got there

Author's working record; the per-step numbers come from runs whose logs are not in this release, from
different stages and configurations, some as wall throughput including prefill (earlier rows), some as
decode-window throughput (later rows). They explain the configuration; they are not one controlled
cumulative experiment.

| step | decode | what it turned out to be |
|---|---|---|
| FP8 KV reader for the sparse-attention layers | 43 tok/s | 1.48× more KV per card, ~5% decode cost |
| remove one allocator env var | +13%, +37% at 4 streams | `expandable_segments` had drifted into the config |
| profile the step with nsys | — | read as ~87% GPU idle, over half the busy time NCCL spin-waiting, by kernel-row summation; that instrument undercounts graph replays, and a 2026-09-05 re-measure put the lane at 91–93% GPU-busy |
| vLLM 0.28 + full cudagraphs | **77** (+92%) | graph capture removed the launch jitter |
| MTP head with full graphs | **110–120** (+51%) | first attempt passed every shallow test and died at 207K prompt tokens |
| restore 262K | 112–125 | PLE worker context moved off the tightest card, prefill chunk halved |
| pack draft experts INT4 + GDN INT8 | +4%, pool 262K → 314K | −1.3 GiB/card; two-stream collapse fixed by the capture list |
| Intel AutoRound experts | −5% decode, +0.3–0.8 points paired | first conversion served garbage (nibble offset) |
| three draft tokens per step | **154–169** (record headline +15–20%; ranges give +3–24% at 4K) | acceptance 2.36 → 2.78 |

## What ships next

The in-place packers (MTP experts INT4, GDN projections INT8), the AutoRound-to-compressed-tensors
converter, and the 16-gate deep-context harness.

## Credit

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- vLLM model support: the [`peakcrosser7` Flash-Next branch](https://github.com/peakcrosser7/vllm/commits/release/qwen38next_offload)
  behind vLLM PRs [#53896](https://github.com/vllm-project/vllm/pull/53896) (merged 2026-08-31) and
  [#53899](https://github.com/vllm-project/vllm/pull/53899) (PLE offload, open)
- Quantised experts: [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
  (served), [VnimanieAI/Qwen3.8-Flash-Next-W4A16](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16)
  (no-MTP entry); FP8 PLE table from [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4);
  MTP INT4 packing recipe adapted from [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090)
- Independent work on the same hardware class:
  [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090),
  [noonghunna/club-3090](https://github.com/noonghunna/club-3090),
  [tfriedel/qwen3.6-rtx3090-lab](https://github.com/tfriedel/qwen3.6-rtx3090-lab)

## License

Code in this repository: Apache-2.0. Model weights follow the licences of Qwen and of the quantised
sources credited above.
