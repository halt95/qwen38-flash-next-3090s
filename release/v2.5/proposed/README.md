# Proposed patches (not part of the v2.5.1 release)

Patches here apply on top of the v2.5.1 series (`../patches/0001..0097`) but are **not** in the pinned
release tree, the bundle, or `SHA256SUMS.v2.5.1` — so `../patches/` still replays exactly to the v2.5.1 tree.
They are candidates for the next release; adopting one means moving it into `../patches/` and rebuilding the
bundle and the pins.

| Patch | What | Issue |
|---|---|---|
| `0098-guard_telemetry-skip-reader-polls-while-the-engine-i.patch` | The bounds-guard reader stops issuing its 50 ms device->host copy while the engine is idle (2 s tail), so idle GPUs reach P8. 4x RTX 3090 idle: 450 W -> 85-88 W; full 50 ms cadence while serving. | #2 |

Apply on a v2.5.1 tree (the bundle checkout, i.e. the tip of `../patches`):

```
git am --keep-non-patch release/v2.5/proposed/0098-guard_telemetry-skip-reader-polls-while-the-engine-i.patch
```
