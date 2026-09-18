# Flash-Next v2.0.1 — package manifest (generated 2026-09-18 from the release tree, final audit)

Release assets (attached to the `v2.0.1` release; hashes from `release/v2/SHA256SUMS.v2.0.1`, which the reference package verified before its clean-room build):

| asset | sha256 |
|---|---|
| `v2.0.1-from-upstream-e962733e08.bundle` | `eb2b17399794512d57156fff1e9b2c72af18c8b4dd6c55abd1dd0ed39097e9e3` |
| `e2-base-src.tar.gz` | `fcd14214f64faaa4175d62f6a51ca39c7bc09d15bf75c190d4dfcc88846b927f` |
| `build-artifacts-sm86-py313-cu130.tar.gz` | `845987ac2a5b0894a93adf831e8795dc12104bd9addea7f91e1aca8ae975f110` |
| `build-artifacts.list` | `d26029273f3c50b4983b1c20219152de9b1cf3867977c17189184fc3bd85b6f5` |

Combined diff: `release/v2/v2.0.1-combined.diff` = `git diff e2-base..v2.0.1`, applies with `git apply` onto `e2-base` and yields the tag's tree `5426cf19586a66946a1d8f7e68f48bdc7f52d62d` (sha256 in `SHA256SUMS.v2.0.1`).
Reference patch series: 76 files under `release/v2/patches/`, hashed in `SHA256SUMS.v2.0.1`. Unlike v2, these copies are byte-identical to the reference package: our own commits carry the maintainer's GitHub identity in the tagged history itself, and contributors whose work upstream authored keep their own attribution.

Files in this tree (sha256 of the committed, LF-normalised content — `git show <commit>:<path> | sha256sum`; a CRLF checkout on Windows will hash differently):

| file | sha256 |
|---|---|
| `README.md` | `493ddd95feb4b823642679bc01231758eea6ba58fb091422744901b3a50a5990` |
| `NOTICE` | `ad3a23f32a99e6d5ac404cfdd1100ed180b19a6fb87ab67f91ac0f6e7b489b21` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE.qwen-community-1.0` | `465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4` |
| `upstream/PIN-v2` | `6d49bdc9947acf24409c8761c94056304b8c2678a0206fe7c80a33c9e44d572b` |
| `scripts/build-v2.sh` | `6838c2056cd645ca8c4467e81b074fc3c67ef6c6e15067b5d4486d8b34f375c9` |
| `scripts/serve-v2.sh` | `6a37cde91c416ee4d5ae714d45bdcfdcb9b043f9b51829c3d8b45be0f77ff1b3` |
| `scripts/make-e1-config.py` | `19bb3a3f99e0ba8b450605a4675693118f413f2364af3bc7f59b0f0549ee24fd` |
| `Dockerfile` | `5d91a2b45aba152f19933e355b456c5e97a1d314ba69eac26e91e87ac58eeda2` |
| `docker-compose.yml` | `5700f86292dff154aa31960b87f5dd7b7f92ff11e25a831e00e94459cc4ac315` |
| `scripts/docker-entrypoint.sh` | `01cc19ad8138fb05c9cd62d745194080daaaac45d769e46e281a0cd9f9ae8cea` |
| `.dockerignore` | `4febdf68980e9b17408de238c8516365079b79a4eabc51129a8a056853be6e93` |
| `lxc/pve-create.sh` | `705043e2dac7083f321b7e0c4a32d8212eaa2a9aa413f620fb711faf584fb60b` |
| `lxc/provision.sh` | `e6402c17443d6ebe59fe48228c96daee311d6a7a23bf257509a49aa6d79bdb5d` |
| `lxc/flash-next.service` | `3eaf49b51d981fd540a905c1112ad6a6ef93d9eb76b922144c9d6fb4d7292b7a` |
| `scales/qsa_kv_scales_262k.json` | `5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67` |
| `release/v2/requirements-pinned.txt` | `8390c8407106f4440359be1966b29cc3aeb3483490f2f059a83e58abfa057f62` |
| `release/v2/build-artifacts.list` | `d26029273f3c50b4983b1c20219152de9b1cf3867977c17189184fc3bd85b6f5` |
| `benchmarks/2026-09-17/BENCH-CARD.md` | `1bb3794e1225ce5cc0df3b9b3878ba3140d7c3e13558c611f1a2b803afa2f381` |
| `release/v2/SHA256SUMS.v2.0.1` | `e651890ffa8dfb4b0bc7b9c4178656de9687adc1f12e5e99759e5167cfe23b8d` |
| `release/v2/v2.0.1-combined.diff` | `f051b2b6952e2c74229f7a43f1bcb95811361f3c984fa4885e93b0f379f3f888` |

Checkpoint: `Qwen3.8-Flash-Next-W4A16-Merlin` from Hugging Face plus the one config key added by `scripts/make-e1-config.py`; shard hashes are the checkpoint's own (not republished here).
