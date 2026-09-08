#!/usr/bin/env python3
"""Max-merge per-rank QSA absmax dumps into the VLLM_QSA_KV_SCALES sidecar.

This is the script that produced scales/qsa_kv_scales_262k.json (reference host,
2026-09-08, --margin 1.10), with the paths and the expected counts as arguments.

Fails closed: requires positive rank and layer counts, every rank file (--ranks),
exactly the expected number of QSA layers in each (--layers; 12 for
Qwen3.8-Flash-Next's target model), positive finite per-rank maxima, a margin in
[1.0, 2.0], and positive finite output scales.
scale = absmax / 448.0 (E4M3 max normal), per the collector design in patch 0002.

    python calib/qsa_calib_merge.py <dump-dir> <out.json> [--ranks 4] [--layers 12] [--margin 1.10]

How the dumps are made: see calib/calib_launch.sh (serve with VLLM_QSA_KV_COLLECT
and --enforce-eager, no speculative config) and calib/qsa_calib_traffic.py (the
traffic the shipped sidecar was calibrated with: text at 2K/8K/32K/128K/200K/255K
plus two image prompts, then a flush phase). Each TP rank writes <dir>/qsa_absmax_rank<r>.json.

Why MTP must be off while collecting: the collector also sees the draft head's
own QSA layer, but the loader in patch 0001 applies scales to the target model
only and rejects unknown layer names, so a dump taken with MTP on yields a
sidecar it will not load. The draft QSA layer runs at scale 1.0 in the shipped
configuration; the clip counter logs by full layer name, so it would show under
its own name (separate draft-layer totals were not retained for the shipped gate).

The collector persists its running maxima every 2,000 layer calls, not on a
timer, and the exit flush can race distributed teardown, so the traffic script
ends with a flush phase of short completions that pushes past the next dump
boundary. Use a fresh, empty dump directory per run and check the rank files'
mtimes are later than your last deep request before merging.
"""
import argparse
import json
import math
import sys
from pathlib import Path

E4M3_MAX = 448.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ranks", type=int, default=4, help="TP size; one dump per rank is required")
    ap.add_argument("--layers", type=int, default=12, help="QSA layers the target model has")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="multiply absmax before dividing. The shipped sidecar uses 1.10: at 1.0 the "
                         "no-flush collection gated 210 clip increments on 9 of 12 layers (6 at 1.10); the shipped flush "
                         "collection at 1.10 gated 24 on 3 layers, 0 with the final counter revision "
                         "(the counter reduces over padded rows too; it cannot attribute a residual)")
    a = ap.parse_args()
    if a.ranks < 1 or a.layers < 1:
        sys.exit(f"FAIL: --ranks and --layers must be positive, got {a.ranks} / {a.layers}")
    if not (math.isfinite(a.margin) and 1.0 <= a.margin <= 2.0):
        sys.exit(f"FAIL: --margin must be a finite value in [1.0, 2.0], got {a.margin}")

    files = [a.dump_dir / f"qsa_absmax_rank{r}.json" for r in range(a.ranks)]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        sys.exit("FAIL: missing rank files (a missing rank under-estimates the global absmax):\n  "
                 + "\n  ".join(missing))

    merged: dict[str, dict[str, float]] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if len(d) != a.layers:
            sys.exit(f"FAIL: {f.name} has {len(d)} layers, expected {a.layers} "
                     "(collected with MTP on? the draft QSA layer must not be in the dump)")
        for name, st in d.items():
            k, v = float(st["k_absmax"]), float(st["v_absmax"])
            if not (math.isfinite(k) and math.isfinite(v) and k > 0 and v > 0):
                sys.exit(f"FAIL: non-finite or non-positive absmax on {name} in {f.name}: k={k} v={v}")
            m = merged.setdefault(name, {"k_absmax": 0.0, "v_absmax": 0.0})
            m["k_absmax"] = max(m["k_absmax"], k)
            m["v_absmax"] = max(m["v_absmax"], v)

    if len(merged) != a.layers:
        sys.exit(f"FAIL: merged {len(merged)} layers, expected {a.layers} (ranks disagree on the layer set)")
    for name, m in merged.items():
        if m["k_absmax"] <= 0 or m["v_absmax"] <= 0:
            sys.exit(f"FAIL: non-positive absmax on {name}: {m}")

    sidecar = {name: {"k_scale": m["k_absmax"] * a.margin / E4M3_MAX,
                      "v_scale": m["v_absmax"] * a.margin / E4M3_MAX}
               for name, m in sorted(merged.items())}
    for name, sc in sidecar.items():
        if not all(math.isfinite(x) and x > 0 for x in sc.values()):
            sys.exit(f"FAIL: non-finite or non-positive scale computed for {name}: {sc}")
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(sidecar, indent=1, allow_nan=False))
    print(f"wrote {a.out} ({len(sidecar)} layers)")
    for name, s in sidecar.items():
        m = merged[name]
        print(f"  {name}: k_absmax={m['k_absmax']:.4g} v_absmax={m['v_absmax']:.4g} "
              f"-> k_scale={s['k_scale']:.6g} v_scale={s['v_scale']:.6g}")


if __name__ == "__main__":
    main()
