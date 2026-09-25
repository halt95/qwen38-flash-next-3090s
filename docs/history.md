# Release history

Earlier release notes, moved here from the [README](../README.md). The current release is v2.2.0; its notes are in the
README's [What changed in v2.2.0](../README.md#what-changed-in-v220).

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
total, 51/51 answers correct on both): that is the open [prefix-cache loss](../README.md#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm),
unchanged by this fix.

Nothing else about the release changes: same checkpoint, same serve command, same KV budget, same pool of 806,792
tokens. Upstream PR #43650 is deliberately **not** ported, because this tree's cache coordinator already caps a Mamba
cache hit at the attention groups' post-drop length and dropping again would shrink every warm hit for no gain.
The requalification record is not published; the two upstream pull requests
([#54442](https://github.com/vllm-project/vllm/pull/54442), [#56802](https://github.com/vllm-project/vllm/pull/56802),
for [issue #54437](https://github.com/vllm-project/vllm/issues/54437)) are the independently readable evidence.

## How v2 got there

The three steps that account for the result (maintainer-reported):

| step | result | what it turned out to be |
|---|---|---|
| PLE host-mapped **pull** transport | decode +4–10 % vs push on the complete re-run, pool **806,792** at 4.1e9 | the push transport paid a PCIe round trip per step; the pull transport's first cut leaked a 322 MiB CUDA context into the offload process (an aux stream bypassing lazy init), fixed |
| **pinned-host token embeddings (ours)** | 608 MiB freed per rank → pin 3.8e9 → 4.1e9, pool 748K → 806,792 | the drafter lookup sits inside a full cudagraph, so a CPU gather was rejected; a device-mapped UVA lookup, bytewise-identical to the device table, ships instead |
| the decode gap to TP4 (0.95–0.96) | closed: 0.997 / 1.009 / 0.988 | the two 50 ms reader threads on the PP1 critical path + NCCL protocol; the pipeline itself is serial (stage 0 computes ~9.4 ms then waits in the token broadcast; stage 1 waits ~7.8 ms then computes ~13.3 ms) |

## v1, the TP4 lane

The v1 files stay in this repository as shipped (`scripts/build.sh`, `scripts/serve.sh`, `patches/`, `upstream/PIN`,
`benchmarks/2026-09-05`, `benchmarks/2026-09-08`): vLLM v0.28.0 + the eight community commits + six patches,
TP4 + EP, MTP K=3, KV pin 2.6e9 (2.9e9 = pool 342,912 since 2026-09-10). It remains the fallback entry on the
reference host.
