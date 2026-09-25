# Flash-Next v2.2.0 — package manifest (generated 2026-09-24 from the release tree)

Release assets (attached to the `v2.2.0` release; hashes from `release/v2.2/SHA256SUMS.v2.2.0`):

| asset | sha256 |
|---|---|
| `v2.2.0-from-upstream-v0.30.0.bundle` (uncompressed; the asset is the `.gz`) | `599b77644d661a2aae8c5ac69576deac1bca93d4820b6e08f5b9ca6b5d8e2f16` |
| `v2.2.0-from-upstream-v0.30.0.bundle.gz` (as downloaded) | `4de4ae5e175f58cdde9bfbfa570e813ff9d089b3e6627f4cf22c933a9c9cb206` |
| `build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz` | `cdca5ffcb3003d15e00896f0969134e16d7bfbe0a3c6dfe10698bec32573f592` |
| `build-artifacts.list` | `2befac68d349aea26e0b35f59f4d04eda6c652ae7037f48fab8920c33ab5e432` |

The bundle's prerequisite is the public vLLM tag `v0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`); it carries
`refs/heads/v2.2.0-public` = `9c27ca9a0657338852c7d3e59561d424fc036fde`, tree
`eebcd6480dba463294e9ca34f24eb770a45e11fe`, 58 commits.
The tarball holds the 23 paths of `build-artifacts.list` (2,005 files): the two extensions built from this tree in an
Ubuntu 22.04 root (`_C_stable_libtorch` `6cb8bc7770112a9b…`, `_moe_C_stable_libtorch` `66acb123e71ba0c7…`;
glibc 2.34 or newer) and the stock vLLM 0.30.0 wheel's other build products, byte-identical to that wheel's.

Combined diff: `release/v2.2/v2.2.0-combined.diff` = the diff from `v0.30.0` to the release commit; `git apply --index` on
`v0.30.0` yields the tree `eebcd648…`. Reference patch series: 58 files under `release/v2.2/patches/`, hashed in `SHA256SUMS.v2.2.0`;
`git am --keep-non-patch` on `v0.30.0` (with `core.autocrlf=false`) replays them to the same tree.

Files in this tree for v2.2.0 (sha256 of the committed, LF-normalised content — `git show <commit>:<path> | sha256sum`;
a CRLF checkout on Windows will hash differently). `release/v2.2/SHA256SUMS.v2.2.0` is hashed below it; this manifest
cannot hash itself.

| file | sha256 |
|---|---|
| `README.md` | `de34dd1b85d13a604216d766b22e18e528bf7cadc4c8d89a387621710f76134c` |
| `NOTICE` | `146c7cbe775cbf8c180091bb029b949abf32c909b93a92ae4d58e0a7fd43a162` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE.qwen-community-1.0` | `465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4` |
| `upstream/PIN-v2.2` | `cf3cfe64af32dad2d7ad17ac0307a4b5bc135fd50367641a2b6e857422178080` |
| `scripts/build-v2.2.sh` | `2b0d8f7709312c6cf7973d72533fffeba787b880e9bf266980a8ce74e0c6f8d1` |
| `scripts/serve-v2.2.sh` | `f599dac7a9861ad3358d56c14636a66c7e4bb3ce221d9804612ce41bf2d7009f` |
| `scripts/topo_select.py` | `5ec2af2f8fdf179e2b7ab2ec8d29a0633e5570fa780c5c51c5bb11b159329ffc` |
| `docs/topology-selector.md` | `b67592ffa0dad04774413ee610acb3a51d7c71741235690ffd7d32dd428965f5` |
| `docs/history.md` | `d39adc51198ccf9cf760ea976f673259057060fd73231e948d1c7ec1ee1396ff` |
| `docs/images/flashnext-v2.2-layout.png` | `f76b34f6710c9ab58bce62dbcede8c1cb63e31246060e989369405cd8a6503c3` |
| `docs/images/flashnext-v2.2-summary.png` | `a6c4953e3be71b01d6b2210d80b4753e3064f5da29de443f02618672cdc69514` |
| `docs/images/flashnext-nvlink-pairing.png` | `f0e5c901d853ec2396db29082d6b2d6edaeeac20dd6463e351c7bfb4e9cd13fc` |
| `Dockerfile` | `03abdc1f13935beda4c9020cb7ed1fc9a6a6ee756fdb2e33aa63dc81c4b5b031` |
| `docker-compose.yml` | `19eff348b15197bb285a172929cebf68d25a02a756ca5b36b4aeb736e88e7b22` |
| `scripts/docker-entrypoint.sh` | `2782acb1eaf411a253b18b1d2ac607722aa057b0f0c86ccf3776c7909ab94bec` |
| `lxc/pve-create.sh` | `b01ab83c23b2cc73b24541c5fbce2f311181ceb6527231079d478c9ac83b095c` |
| `lxc/provision.sh` | `7cf5e289496d4ad8562b31caab12df2a9e314e123473b56b4610e8d94875021c` |
| `lxc/nvidia-prestart.sh` | `9576cce20a6bdb1ba517f4b2473077858e346a1223d8b5f15ea7e03cf19de690` |
| `lxc/flash-next.service` | `b107f264b35fa1f3b8a017e74c69d4b49d143e2c33aecdd3779111be8cacca91` |
| `.dockerignore` | `0098a8d01ad9878466b2645f28931469fd44314581a5b15a19056e5234dc0c22` |
| `scripts/make-e1-config.py` | `33f6ff5789e4d9ff6649706e22964e551cfed809c00a9d578fcc83a83b9a5646` |
| `scales/qsa_kv_scales_262k.json` | `5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67` |
| `release/v2.2/requirements-pinned.txt` | `8a3d43ed732494cc6e87912e8e25e57cf3241d4e65c6df9dad9e179f2017e986` |
| `release/v2.2/build-artifacts.list` | `2befac68d349aea26e0b35f59f4d04eda6c652ae7037f48fab8920c33ab5e432` |
| `benchmarks/2026-09-24/BENCH-CARD.md` | `02971fb8710be0156428d75230f8158ad858bba3ca260c69cf4509e8f1faea00` |
| `benchmarks/2026-09-24/flashnext-v2.2.0-ctx-pp-tg-itl.png` | `a954b5af737868a398162b2890b9bd5f2715cce73939a496c381bc652921c723` |
| `release/v2.2/v2.2.0-combined.diff` | `1b8544a001b64a2c22669938cbb98d2b11b6cb6816ea9095926c9735011778c7` |
| `release/v2.2/SHA256SUMS.v2.2.0` | `fda958e5a1318091f8d1e42279a4d5b4f8fb8bfc1f0e09c5a4e41465914eddf3` |

The v2.0.1 and v1 files stay in the tree unchanged (their manifest is `release/v2/PACKAGE-MANIFEST.md`); the container
and LXC recipes (`Dockerfile`, `docker-compose.yml`, `scripts/docker-entrypoint.sh`, `lxc/`) build v2.2.0 (listed above).

Checkpoint: `Qwen3.8-Flash-Next-W4A16-Merlin` from Hugging Face plus the one config key added by
`scripts/make-e1-config.py`, unchanged from v2.0.1; shard hashes are the checkpoint's own (not republished here).
