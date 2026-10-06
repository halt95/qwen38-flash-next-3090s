# Flash-Next v2.5.1 — package manifest (generated 2026-10-06 from the release tree)

Release assets (hashes from `release/v2.5/SHA256SUMS.v2.5.1`):

| asset | sha256 |
|---|---|
| `v2.5.1-from-upstream-v0.30.0.bundle` (uncompressed; the asset is the `.gz`; attached to the `v2.5.1` release) | `d904e4685877e56b57c9a23a621b537239ba6316ad238c3848c5f03437d03d7c` |
| `v2.5.1-from-upstream-v0.30.0.bundle.gz` (as downloaded) | `660b20b07f6f190890e4c1dbfccd4f4282ec336cd9bb0562f8adf01fd81192a3` |
| `build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz` (v2.2.0's asset, reused unchanged; attached to the `v2.2.0` release) | `cdca5ffcb3003d15e00896f0969134e16d7bfbe0a3c6dfe10698bec32573f592` |
| `build-artifacts.list` | `2befac68d349aea26e0b35f59f4d04eda6c652ae7037f48fab8920c33ab5e432` |

The bundle's prerequisite is the public vLLM tag `v0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`); it carries
`refs/heads/v2.5.1-public` = `96a0d13b6e98ad4ef86e66a3f6ddb5dd9199a591`, tree
`0df6d7ef38f8ac927bb8feb7dc9887c2c94e3901`, 97 commits (v2.2.0's 58, unchanged, plus 39 for v2.5.1).
The 39 commits after v2.2.0 change no compiled code: Python, plus comment-only edits in three C/CUDA files (no CMake
or build file), so the compiled ops are v2.2.0's: the tarball
holds the 23 paths of `build-artifacts.list` (2,005 files), the two own extensions (`_C_stable_libtorch`
`6cb8bc7770112a9b…`, `_moe_C_stable_libtorch` `66acb123e71ba0c7…`; glibc 2.34 or newer) and the stock vLLM 0.30.0
wheel's other build products.

Combined diff: `release/v2.5/v2.5.1-combined.diff` = the diff from `v0.30.0` to the release commit (212 files);
`git apply --index` on `v0.30.0` yields the tree `0df6d7ef…`. Reference patch series: 97 files under
`release/v2.5/patches/` (0001..0058 are v2.2.0's content, re-exported), hashed in `SHA256SUMS.v2.5.1`; `git am --keep-non-patch` on `v0.30.0`
(with `core.autocrlf=false`) replays them to the same tree. Environment pins: 196, identical to v2.2.0's except
`transformers==5.18.0` (was 5.17.0).

Files in this tree for v2.5.1 (sha256 of the committed, LF-normalised content — `git show <commit>:<path> | sha256sum`;
a CRLF checkout on Windows will hash differently). `release/v2.5/SHA256SUMS.v2.5.1` is hashed below it; this manifest
cannot hash itself.

| file | sha256 |
|---|---|
| `README.md` | `7e3a0781d143010a52987a09d2dec56ca613b5a292d027f446e992ec500974bf` |
| `NOTICE` | `62ed7fc368cb68b8ca8d79be7a88569629de527b3a771ad9614c58b555c7fa89` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE.qwen-community-1.0` | `465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4` |
| `upstream/PIN-v2.5` | `481103bc567ff167a668fbe0f9642cb5395132fb0bf1ecf543cfef30fbc2df26` |
| `scripts/build-v2.5.sh` | `b83085052194fff881ccf79a16f061b5f20a60beaed1c9b2a069ad5cb1a757ed` |
| `scripts/serve-v2.5.sh` | `22eb5201ec7f2c7f5c549cc6a63777ca4157c0e373bdea1e53c9e26e7314e912` |
| `scripts/topo_select.py` | `5ec2af2f8fdf179e2b7ab2ec8d29a0633e5570fa780c5c51c5bb11b159329ffc` |
| `docs/topology-selector.md` | `b67592ffa0dad04774413ee610acb3a51d7c71741235690ffd7d32dd428965f5` |
| `docs/history.md` | `7028e63f96f627987ce2e26ced5dcad5e1636347857e6841775f728bdbe85e95` |
| `CHANGELOG.md` | `111424a806eb6504d49a8c720df0a6ac37895d239eeb4923f4de52776152e921` |
| `docs/reference.md` | `29b5660931be542f1df129ef2759a4a465d84334ff3f5e63e7c5765d1d134193` |
| `docs/images/flashnext-v2.2-layout.png` | `f76b34f6710c9ab58bce62dbcede8c1cb63e31246060e989369405cd8a6503c3` |
| `docs/images/flashnext-nvlink-pairing.png` | `f0e5c901d853ec2396db29082d6b2d6edaeeac20dd6463e351c7bfb4e9cd13fc` |
| `Dockerfile` | `01c8a4d31d0589ac08cda3ac4690ec505e6e196e4e6b68189c6f55e3bcb6fa7c` |
| `docker-compose.yml` | `d5b9a8e1868b46c35ce1dbd3a6f757b8cb69c4435f37ae11d70f5bd72cb46af3` |
| `scripts/docker-entrypoint.sh` | `7c2af173c059512a2dc5ab507e0c03b64909df2d0b8bb0183f2c8f17780b4a68` |
| `lxc/pve-create.sh` | `bb5c06565d49182797ba6f3be15c19046e367152551b7701a53f74605d80d94f` |
| `lxc/provision.sh` | `6284e2a76f8084d9105c35737789cd6975d3f7930fd5947b89f3fd17b166619f` |
| `lxc/nvidia-prestart.sh` | `9576cce20a6bdb1ba517f4b2473077858e346a1223d8b5f15ea7e03cf19de690` |
| `lxc/flash-next.service` | `8fc62c97408627af550f304f101c8ddec5f33203c933710444fea367191b50fc` |
| `.dockerignore` | `a3980461921d91966e988a8e332840ab7cd68c004d51212ad53f34ed7f1beaa5` |
| `scripts/make-e1-config.py` | `33f6ff5789e4d9ff6649706e22964e551cfed809c00a9d578fcc83a83b9a5646` |
| `scales/qsa_kv_scales_262k.json` | `5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67` |
| `release/v2.5/requirements-pinned.txt` | `3e108cf03a6ef6c8bae7f376f5e2764b8ef7e3c0db6ea8bba164589fdf8dc041` |
| `release/v2.5/build-artifacts.list` | `2befac68d349aea26e0b35f59f4d04eda6c652ae7037f48fab8920c33ab5e432` |
| `benchmarks/2026-09-24/BENCH-CARD.md` | `02971fb8710be0156428d75230f8158ad858bba3ca260c69cf4509e8f1faea00` |
| `benchmarks/2026-09-24/flashnext-v2.2.0-ctx-pp-tg-itl.png` | `a954b5af737868a398162b2890b9bd5f2715cce73939a496c381bc652921c723` |
| `benchmarks/2026-10-05/BENCH-CARD_OLD.md` | `ad6b617beef31c0e459d8a163cf23c493ff1e1914530007ecd13d5dcefe20e56` |
| `benchmarks/2026-10-05/flashnext-v2.5.0-ctx-pp-tg-itl_OLD.png` | `ee90d9efdfc99750a380c5f36c8b2daeaa4bbcfad5872c69d4f28bbb200451d1` |
| `benchmarks/2026-10-06/BENCH-CARD.md` | `cd187ea6fd4f46e5c8c0e9cd384a10806bf79a4a9bf5f2003145db89d941b7da` |
| `benchmarks/2026-10-06/flashnext-v2.5.1-ctx-pp-tg-itl.png` | `4832f93602fc2c16638901fa11aede507009d6299e0c7b920c2d679944b45061` |
| `docs/images/flashnext-v2.5-layout.png` | `99b55769952494bd852b1c19248353a44d30893399a17dbcb0674114f4b5ccbe` |
| `docs/images/flashnext-v2.5-summary.png` | `80dc42e4f15e6014178e036f3f46cd727d5536feba38597fd63c43f294d99545` |
| `release/v2.5/v2.5.1-combined.diff` | `803b1f35f32ed75e3273c8b1628956b4b913fb545c11c02036f5178d8e272793` |
| `release/v2.5/SHA256SUMS.v2.5.1` | `5aa3e1ab156009e99db5f17a36d67a7d3ab81772be477e4ad67136477e4d3334` |

The v2.2.0, v2.0.1 and v1 files stay in the tree unchanged (their manifests are `release/v2.2/PACKAGE-MANIFEST.md` and
`release/v2/PACKAGE-MANIFEST.md`); the container and LXC recipes (`Dockerfile`, `docker-compose.yml`,
`scripts/docker-entrypoint.sh`, `lxc/`) build v2.5.1 (listed above).

Checkpoint: `Qwen3.8-Flash-Next-W4A16-Merlin` from Hugging Face plus the one config key added by
`scripts/make-e1-config.py`, unchanged from v2.0.1; shard hashes are the checkpoint's own (not republished here).
