# Changelog

Release notes, newest first. To install and run the current release, see the [README](README.md); reference detail
(scripts, environment variables, capacity, known behaviours, measurement method) is in
[docs/reference.md](docs/reference.md).

## v2.5.1

Released 2026-10-06: [v2.5.1 on GitHub](https://github.com/halt95/qwen38-flash-next-3090s/releases/tag/v2.5.1). Bench
card: [`benchmarks/2026-10-06/BENCH-CARD.md`](benchmarks/2026-10-06/BENCH-CARD.md). To run it:
[Quick start](README.md#quick-start), or [Build and serve](docs/reference.md#build-and-serve) for the detail.

v2.5.0 was not published: v2.5.1 is v2.5.0 plus `--prefix-match-unit 64` and the prefix-cache fixes, so these notes
cover everything since v2.2.0, v2.5.0's changes included.

### In brief

- **Agent follow-up turns resume from cache.** A follow-up resumes on a 64-token grid (`--prefix-match-unit 64`) instead of the 4,096-token block, and the cached prefix a conversation needs is held until its next turn. One conversation alone: median time to first token on follow-ups 0.96 s (v2.5.0: 1.97 s; v2.2.0: 1.56 s). Eight deep conversations at once: cache reuse close to v2.5.0's (92.5 % against 90.6 % of follow-up prompt tokens), a median follow-up time to first token about 55 % lower (1.61 s against 3.54 s) and 22 % more requests started within 5 minutes, all completed (107 against 88); 4 of 99 follow-ups still re-prefilled the whole conversation with the pool near full (99.2 %) (v2.2.0 served 7.5 % from cache, with a 52.7 s median, on the same shape of load at temperature 0.6, the load that reproduced the cache issue reported on Hugging Face).
- **More KV capacity.** The pool grows from 806,792 to 924,993 tokens (+15 %). Six 131K sessions and three 262,144-token sessions were each measured resident at once with 0 preemptions (v2.2.0: five 131K).
- **Tokenizer and image fixes.** Transformers 5.18.0 handles combining marks as intended, with parity on 92 items. Prompts can contain up to 42 images, each capped at 4 MP.
- **Defaults match the shipped launcher.** Unset `VLLM_E2_PLE_PULL_TRANSPORT` and `MERLIN_` kernel toggles use the shipped value of 1; `=0` opts out when you run the engine directly (`serve-v2.5.sh` always sets the shipped values). `VLLM_USE_BREAKABLE_CUDAGRAPH=0` remains required.
- **Faster cold prefill.** 5,219–5,533 tok/s from 10K to 261K tokens, 1.0–2.8 % above v2.2.0 at every depth.
- **Stability checks passed.** GSM8K-200 scored 198/200, structured output passed 80/80 cases, the 262K needle was found exactly, cache hits were valid on 20 of 20 prompt pairs, and 0 engine errors occurred over every run; divergence from the BF16 teacher is 0.0334 (v2.2.0: 0.0338).

v2.5.0 was not published: v2.5.1 is v2.5.0 plus `--prefix-match-unit 64` and a five-commit prefix-cache fix in the scheduler and KV-cache manager. Single-stream decode is about 3 % slower than v2.2.0: v2.5.0's release candidate measured −2.81 % (95 % CI −4.60 % to −1.03 %, p = 0.0087), and v2.5.1, which does not change the decode path, reads −2.87 % (95 % CI −6.84 % to +1.09 %, 2 boots against 7) and −0.01 % against v2.5.0 (95 % CI −4.29 % to +4.27 %, 2 boots each). Eight concurrent requests are effectively unchanged: 533.7 against 531.5 tok/s. The trade is faster agent follow-ups, higher cache reuse and a 15 % larger pool. Bench card: [`benchmarks/2026-10-06/BENCH-CARD.md`](benchmarks/2026-10-06/BENCH-CARD.md).

### What changed in v2.5.1

**v2.5.1 is v2.5.0 plus a prefix-cache fix; v2.5.0 was not published.** v2.5.1 is v2.5.0's source plus five commits
that touch only the scheduler and the KV-cache manager (no compiled code), served with one more option,
`--prefix-match-unit 64`. v2.5.0 kept a running request's Gated-DeltaNet (GDN) prefix checkpoint, which ended the full
re-prefill of follow-up turns: on v2.2.0 most follow-ups re-prefilled the whole conversation (with 8 concurrent 80K
sessions, cache reuse 7.5 %; v2.5.0: 82.0 %). v2.5.1 removes most of what remained:

- **A follow-up turn resumes on a 64-token grid** instead of the 4,096-token block. `--prefix-match-unit` is a stock
  vLLM 0.30 option and changes only the matching granularity; the pool is unchanged. In one conversation a follow-up
  re-prefills a median of 3,202 tokens (v2.5.0: 9,326; v2.2.0: 6,931), and its median time to first token is 0.96 s
  (v2.5.0: 1.97 s; v2.2.0: 1.56 s). Hits stay valid: 20 prompt pairs, each sent cold and again warm, gave identical
  T=0 output, 20/20, every warm send hitting at the expected 64-token boundary.
- **The GDN checkpoint for a partial tail is held until the request is freed.** Upstream releases its copy after one
  step, so another request's allocation could evict it while the turn was still decoding.
- **A prompt tail that ends on a block boundary is keyed**, so the next turn can find it.
- **A waiting request's cached prefix is pinned until admission**, so other requests' allocations do not evict it
  while it queues.
- **A finished turn's prefix stays referenced until its next turn is admitted.**

With 8 deep conversations at once (about 76K tokens at the first turn, growing past 120K; temperature 0; 5 minutes),
cache reuse is close to v2.5.0's (92.5 % against 90.6 % of follow-up prompt tokens), the median follow-up waits about
55 % less (1.61 s against 3.54 s), and 22 % more requests start within the 5 minutes, all completed (107 against 88):
v2.5.0 re-prefilled part of every
follow-up, v2.5.1 resumes most of them on the 64-token grid. v2.2.0 was measured on the same shape of load at
temperature 0.6, the load that reproduced the multi-turn cache issue reported on Hugging Face (discussion #2): 7.5 %
reuse and a 52.7 s median follow-up (v2.5.0 on it: 82.0 %, 3.9 s). **What is left:** with the KV pool near full (99.2 %), 4
of 99 follow-ups in the 5 minutes found no cached prefix and re-prefilled the whole conversation (v2.5.0: none), because
the protection yields rather than preempting a running request. Each fix logs a line at startup ([Check it's working](docs/reference.md#check-its-working)).
Figures below marked v2.5.0 were measured on that build and not re-run; v2.5.1 changes neither the model path nor the
pool.

**Capacity increases, with a measured limit.** The KV pool grows from 806,792 to 924,993 tokens (+15 %), with an
attention block size of 4,096 tokens. Six 131K sessions were measured resident at once with 0 preemptions (v2.2.0:
five), and so were three 262,144-token sessions. Under full-pool pressure (the 8-conversation load above, the pool at
99.2 %), 2 preemptions occurred and 107 of 107 requests completed. Session token
totals overstate capacity because each session also holds fixed GDN state blocks.

**Combining marks tokenize as intended.** The transformers pin moves to 5.18.0. On v2.5.0, tokenizer output matched the reference tokenizer on all 92 test strings, and 102 of 102 generation prompts gave
bit-identical output to the previous pin in a fresh compile. The checkpoint's own `tokenizer.json` on Hugging Face, inherited from the Intel AutoRound release, is replaced with the upstream Qwen3.8-Flash-Next file (vocabulary, merges and special tokens identical), so tools that read it directly (the `tokenizers` library, GGUF converters) match upstream too; vLLM serving on 5.18.0 tokenizes identically with either file.

**Image prompts support more images.** Prompts can contain up to 42 images, each capped at 4 MP. On v2.5.0, every card stayed above the 512 MiB safety floor at the limit, and image tile reading was 41 of 42 on the test set, identical to v2.2.0; this is a base-model property.

**Unset environment variables now use the shipped defaults.** `VLLM_E2_PLE_PULL_TRANSPORT` and the five kernel toggles with the `MERLIN_` prefix default to 1; setting one to 0 opts out when the engine is run directly (`serve-v2.5.sh` clears inherited values and
exports the shipped ones). `VLLM_E2_GUARD_MODE` also defaults to `warn` in code now; its reader costs about 0.3 ms per
decode step on TP2 × PP2 (`off` skips it; the published numbers were measured with `warn`). A boot of v2.5.0's release candidate with these variables absent used the same pool, enabled the toggles on every rank, and passed needle, thinking, and no-thinking checks. `VLLM_USE_BREAKABLE_CUDAGRAPH=0` remains required; the engine refuses to start without it.

**Single-stream decode is about 3 % slower than v2.2.0.** v2.5.0's release candidate measured −2.81 % (95 % CI −4.60 % to −1.03 %, p = 0.0087; 4 boots against 7 reference boots: 2 of v2.2.0 and the 5-boot v2 gate that v2.2.0 matched). v2.5.1 changes the prefix cache and the scheduler, not the decode path: it reads −2.87 % against the reference (95 % CI −6.84 % to +1.09 %, p = 0.085, 2 boots against 7) and −0.01 % against v2.5.0 (95 % CI −4.29 % to +4.27 %, 2 boots against 2); a third boot on the release code, with the logging-only exact-counter reader on as on the v2.2.0 and v2.5.0 comparator boots, read −1.12 %. The median event interval is 0.6–0.7 ms longer while MTP tokens per step are unchanged: RecoverSSM's speculative verify runs the Triton GDN decode kernel and replays the state each step. Eight concurrent requests are effectively unchanged at 533.7 against 531.5 tok/s.

**Cold prefill is faster than v2.2.0.** 5,219–5,533 tok/s from 10K to 261K, 1.0–2.8 % above v2.2.0 at every depth. During v2.5 development a release candidate with RecoverSSM and the default 3,392-token block was 6–12 % slower; two changes recovered it. The block is set to 4,096 tokens (`--block-size 4096`, four 1,024-token prefill chunks per block), which costs 0.5 % of the pool, and the fused packed GDN forward stays on under RecoverSSM (it had been switched off for the whole layer, prefill included).

**Stability and quality checks passed.** Structured output 80/80, GSM8K-200 198/200, the 262,144-token needle found exactly, three 262K and six 131K sessions resident with 0 preemptions, the quality measure 0.0334 against a ceiling of 0.0388 (v2.2.0: 0.0338), and 0 engine errors over every run (the ladder, depth, residency and bench boots and the boot of the quality, needle, structured-output, follow-up and cache-validity runs). On v2.5.0's release candidate, not re-run: a 90-minute mixed soak handled 11,002 requests with 0 errors and 0 HTTP 5xx responses, orderly shutdown completed on all 4 ranks on both boots, and injected PLE faults on rank 0 and rank 1 failed closed, with workers gone and GPUs free in 3.1–5.0 s.

**Published source versus the qualified tree.** The published 39 commits are the qualified series with internal
review labels removed from comments, docstrings and messages, identifiers renamed as in v2.2.0 (the `MERLIN_*`
switch prefix and a few test names), four upstream commits that the series added and later
reverted left out (the final tree is identical either way), and authorship set to the release identity with upstream
ports credited by PR number (authors in NOTICE). For the 34 commits carried over from v2.5.0: of the 85 files they touch, 47 are
identical, 35 differ only in comments, docstrings or strings, one keeps a v2.2.0 published wording, one development note
is not published, and one test's assertions follow the reworded comment it checks. The 5 commits added in v2.5.1 need no
such mapping: the follow-up runs measured them as published (`96a0d13b6e`); one decode check ran on `edd20c413f` and
the other gate runs on `eedb6a34b5`, earlier cuts of the same series that differ only in the waiting-request admission
path, which those runs do not reach (the v2.5.1 card states which run used which). The v2.2.0 transformations are
described in [docs/history.md](docs/history.md#what-changed-in-v220).

**Greedy output repeats at full context, with and without MTP** (v2.5.0's release candidate; not re-run, the decode
path is unchanged). Eight identical T=0 requests per depth (prefix cache off, 256 tokens generated) gave one distinct output at 65,536, 130,948, 134,997, 200,000 and 261,800 prompt
tokens, with MTP and without it. The control boot with the five kernel switches set to 0 gave 5 of 8 distinct outputs
at 65K and 8 of 8 at every deeper cell, so the test can fail; the guard counters stayed at zero on every cell. This
holds within one compile cache (see [Known behaviours](docs/reference.md#known-behaviours-of-the-qwen38-flash-next-architecture-in-vllm)).

**Without MTP** (v2.5.0's release candidate; not re-measured). `--use-replayssm` without a speculative config falls
back to the stock GDN path (a warning says so) and the pool is 1,057,314 tokens (measured at the computed attention
block, not v2.5.1's block 4,096 with `--prefix-match-unit 64`). Three 262,144-token sessions and seven
131,072-token sessions were each measured resident at once with 0 preemptions (a fourth and an eighth were not);
single-stream decode is about 82 tok/s at 65K–262K, against about 100–186 tok/s with MTP. `serve-v2.5.sh` always serves with MTP; the no-MTP line is the same
command without `--speculative-config`.

**Known limits.** The host-RAM KV tier is not included in this release.

Earlier releases (v2.2.0, v2.0.1, v2, the v1 TP4 lane): [docs/history.md](docs/history.md).

## v2.2.0

Released 2026-09-25. The same shape moved onto the public vLLM `v0.30.0` tag plus 58 commits; a reliability release at
parity with v2: greedy repeatability, any-rank fault detection and faster shutdown. Full notes:
[docs/history.md](docs/history.md#what-changed-in-v220).

## v2.0.1

Released 2026-09-18. The v2 shape plus one engine fix, structured output under concurrency. Full notes, build and
stack: [docs/history.md](docs/history.md#what-changed-in-v201).

## v2

Released 2026-09-17. Tensor parallel 2 × pipeline parallel 2 with expert parallel, the host-mapped n-gram table and
host-resident token embeddings: the KV pool grows from 342,912 to 806,792 tokens. Full notes:
[docs/history.md](docs/history.md#v2-against-v1-at-a-glance).

## v1

The TP4 lane: [docs/history.md](docs/history.md#v1-the-tp4-lane).
