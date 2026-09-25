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
| `README.md` | `54fabd42caca5a19f908cebedb0168932cded600c9535b5374c3ce788fe7d327` |
| `NOTICE` | `21a08f7f6b053c9caf3385edf94706e782cc36ed9c926468ba393006ba2afc69` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE.qwen-community-1.0` | `465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4` |
| `upstream/PIN-v2.2` | `cf3cfe64af32dad2d7ad17ac0307a4b5bc135fd50367641a2b6e857422178080` |
| `scripts/build-v2.2.sh` | `2b0d8f7709312c6cf7973d72533fffeba787b880e9bf266980a8ce74e0c6f8d1` |
| `scripts/serve-v2.2.sh` | `f599dac7a9861ad3358d56c14636a66c7e4bb3ce221d9804612ce41bf2d7009f` |
| `scripts/topo_select.py` | `5ec2af2f8fdf179e2b7ab2ec8d29a0633e5570fa780c5c51c5bb11b159329ffc` |
| `docs/topology-selector.md` | `dcdd87dac7c458bbdfb2f2472876cdf0654ce268c3f8f54b0c592eb4084c2dad` |
| `docs/history.md` | `e92f17a12752584ab468a82e2a5d43693304319c205c34fc2152120cecd57007` |
| `docs/images/flashnext-v2.2-layout.png` | `f76b34f6710c9ab58bce62dbcede8c1cb63e31246060e989369405cd8a6503c3` |
| `docs/images/flashnext-v2.2-vs-v2.png` | `92dd9032e23a5808df2b0a264aa9a934f7db93bb9f54cd30f6680f35dff7570f` |
| `docs/images/flashnext-nvlink-pairing.png` | `f0e5c901d853ec2396db29082d6b2d6edaeeac20dd6463e351c7bfb4e9cd13fc` |
| `Dockerfile` | `c8085e6bdde56fbc22bef042326ea2cb1fb0d4ef929a04d1ce1d3c875f1e5085` |
| `docker-compose.yml` | `c0da1cee7f8c5c0cec1ac7d9abdb1909c271f2a39ba2d8ba82f7d64a04d19afe` |
| `scripts/docker-entrypoint.sh` | `2782acb1eaf411a253b18b1d2ac607722aa057b0f0c86ccf3776c7909ab94bec` |
| `lxc/pve-create.sh` | `31d1c857081543ad9c279e7e5990ec0b17417c9280b03a9906742b9ef9f43458` |
| `lxc/provision.sh` | `bda91c83d10209900df44dc651ffd08d62d2b5f608c43dd14d63293521370573` |
| `lxc/flash-next.service` | `b107f264b35fa1f3b8a017e74c69d4b49d143e2c33aecdd3779111be8cacca91` |
| `.dockerignore` | `be5926c958f5313d562e04859b7a29ccaed45ecf87b094f11a7d50825c11c7c1` |
| `scripts/make-e1-config.py` | `33f6ff5789e4d9ff6649706e22964e551cfed809c00a9d578fcc83a83b9a5646` |
| `scales/qsa_kv_scales_262k.json` | `5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67` |
| `release/v2.2/requirements-pinned.txt` | `8a3d43ed732494cc6e87912e8e25e57cf3241d4e65c6df9dad9e179f2017e986` |
| `release/v2.2/build-artifacts.list` | `2befac68d349aea26e0b35f59f4d04eda6c652ae7037f48fab8920c33ab5e432` |
| `benchmarks/2026-09-24/BENCH-CARD.md` | `e8ef66a2a3be6c4bba1e2fc4f25136863db3644912b80240ae153d97db853eb4` |
| `benchmarks/2026-09-24/flashnext-v2.2.0-ctx-pp-tg-itl.png` | `a954b5af737868a398162b2890b9bd5f2715cce73939a496c381bc652921c723` |
| `release/v2.2/v2.2.0-combined.diff` | `1b8544a001b64a2c22669938cbb98d2b11b6cb6816ea9095926c9735011778c7` |
| `release/v2.2/SHA256SUMS.v2.2.0` | `fda958e5a1318091f8d1e42279a4d5b4f8fb8bfc1f0e09c5a4e41465914eddf3` |

The v2.0.1 and v1 files stay in the tree unchanged (their manifest is `release/v2/PACKAGE-MANIFEST.md`); the container
and LXC recipes (`Dockerfile`, `docker-compose.yml`, `scripts/docker-entrypoint.sh`, `lxc/`) build v2.2.0 (listed above).

Checkpoint: `Qwen3.8-Flash-Next-W4A16-Merlin` from Hugging Face plus the one config key added by
`scripts/make-e1-config.py`, unchanged from v2.0.1; shard hashes are the checkpoint's own (not republished here).
