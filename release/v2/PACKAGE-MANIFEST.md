# Flash-Next v2.0.1 — package manifest (generated 2026-09-18 from the release tree, final audit)

Release assets (attached to the `v2.0.1` release; hashes from `release/v2/SHA256SUMS.v2.0.1`, which the reference package verified before its clean-room build):

| asset | sha256 |
|---|---|
| `v2.0.1-from-upstream-e962733e08.bundle` (uncompressed; the asset is the `.gz`) | `eb2b17399794512d57156fff1e9b2c72af18c8b4dd6c55abd1dd0ed39097e9e3` |
| `v2.0.1-from-upstream-e962733e08.bundle.gz` (as downloaded) | `7145fd659caea704d2b80c0a40a9bf3c7e5c5b5132de219fc21eea5048f42980` |
| `e2-base-src.tar.gz` | `fcd14214f64faaa4175d62f6a51ca39c7bc09d15bf75c190d4dfcc88846b927f` |
| `build-artifacts-sm86-py313-cu130.tar.gz` | `845987ac2a5b0894a93adf831e8795dc12104bd9addea7f91e1aca8ae975f110` |
| `build-artifacts.list` | `d26029273f3c50b4983b1c20219152de9b1cf3867977c17189184fc3bd85b6f5` |

Combined diff: `release/v2/v2.0.1-combined.diff` = `git diff e2-base..v2.0.1`, applies with `git apply` onto `e2-base` and yields the tag's tree `5426cf19586a66946a1d8f7e68f48bdc7f52d62d` (sha256 in `SHA256SUMS.v2.0.1`).
Reference patch series: 76 files under `release/v2/patches/`, hashed in `SHA256SUMS.v2.0.1`. These copies are byte-identical to the reference package; contributors whose work upstream authored keep their own attribution.

Files in this tree (sha256 of the committed, LF-normalised content — `git show <commit>:<path> | sha256sum`; a CRLF checkout on Windows will hash differently):

| file | sha256 |
|---|---|
| `README.md` | `e3af7de1046fbabbbba78da7627a3279af1d21f3f0873f39d873773712bd8c02` |
| `NOTICE` | `7e8daa7f156ecc0b60d2e5b6b2c9d88ee52bd64da0dd2d00903f21f531dac44e` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE.qwen-community-1.0` | `465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4` |
| `upstream/PIN-v2` | `6d49bdc9947acf24409c8761c94056304b8c2678a0206fe7c80a33c9e44d572b` |
| `scripts/build-v2.sh` | `b48c90306a4db8d61eb54929f661f74c2848439d0d9c918990beb87c50cac85e` |
| `scripts/serve-v2.sh` | `6ca1c400fd6cdacf2bba850124dcbf91d2b9da712d57395c69584704a8cd1edf` |
| `scripts/make-e1-config.py` | `19bb3a3f99e0ba8b450605a4675693118f413f2364af3bc7f59b0f0549ee24fd` |
| `Dockerfile` | `5d91a2b45aba152f19933e355b456c5e97a1d314ba69eac26e91e87ac58eeda2` |
| `docker-compose.yml` | `235d7048b119938dec044a44a833802b80a9585226c91a3809d09d34fee2bcfc` |
| `scripts/docker-entrypoint.sh` | `41f5502f4f20f1a1c0d9be40d7b37d3163a4c8543c85c59cb70f8c47bff58520` |
| `.dockerignore` | `4febdf68980e9b17408de238c8516365079b79a4eabc51129a8a056853be6e93` |
| `lxc/pve-create.sh` | `705043e2dac7083f321b7e0c4a32d8212eaa2a9aa413f620fb711faf584fb60b` |
| `lxc/provision.sh` | `364c3c8066ffe38b3d5712f73e8b33564dfa7ac1b841d5edfe52d2c867a0f83e` |
| `lxc/flash-next.service` | `3eaf49b51d981fd540a905c1112ad6a6ef93d9eb76b922144c9d6fb4d7292b7a` |
| `scales/qsa_kv_scales_262k.json` | `5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67` |
| `release/v2/requirements-pinned.txt` | `8390c8407106f4440359be1966b29cc3aeb3483490f2f059a83e58abfa057f62` |
| `release/v2/build-artifacts.list` | `d26029273f3c50b4983b1c20219152de9b1cf3867977c17189184fc3bd85b6f5` |
| `benchmarks/2026-09-17/BENCH-CARD.md` | `e691672d42e932648c3956ffc68f4c256e274759a70d9607e71ac0600f958294` |
| `release/v2/SHA256SUMS.v2.0.1` | `e651890ffa8dfb4b0bc7b9c4178656de9687adc1f12e5e99759e5167cfe23b8d` |
| `release/v2/v2.0.1-combined.diff` | `f051b2b6952e2c74229f7a43f1bcb95811361f3c984fa4885e93b0f379f3f888` |

Checkpoint: `Qwen3.8-Flash-Next-W4A16-Merlin` from Hugging Face plus the one config key added by `scripts/make-e1-config.py`; shard hashes are the checkpoint's own (not republished here).
