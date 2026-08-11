#!/usr/bin/env python3
"""
ODD sweep for the lane_keeping scenario.

Maps the failure rate over a grid of ODDs, varying the two most significant axes -- maximum
speed and maximum curve angle -- and tabulates, per cell, the failure rate (plus rare rate,
median margin, control rate). Use it to locate the informative band (typically ~20-40%) to use
as a reference ODD for rare-failure detection.

Usage:
    python scripts/sweep_lanekeeping.py
    python scripts/sweep_lanekeeping.py --n 20 --workers 4 --speeds 14,20,25,30 --angles 45,60,75,85
    python scripts/sweep_lanekeeping.py --csv sweep.csv

Prerequisites: simulator containers running (as for run_lanekeeping.py).
Time: ~ (cells) x N x time/sample.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Base ODD = "realistic" preset. The sweep overrides only the max angle (cols 0-4) and the
# max speed (col 6 upper).
#   angles    : [0, angle_max] per segment
#   min_speed : [5, 8]  m/s (fixed band, not overlapping max_speed)
#   max_speed : [9, speed_max] m/s
#   seg_len   : [20, 40] m ; map_size: [200, 350] m
BASE_LOWER = [0, 0, 0, 0, 0, 5.0, 9.0, 20.0, 200.0]
BASE_UPPER = [45, 45, 45, 45, 45, 8.0, 14.0, 40.0, 350.0]


def build_bounds(angle_max: float, speed_max: float):
    """Build (lower, upper) for one grid cell."""
    lower = list(map(float, BASE_LOWER))
    upper = list(map(float, BASE_UPPER))
    for k in range(5):                 # angles 1..5
        upper[k] = float(angle_max)
    upper[6] = float(speed_max)        # max_speed upper
    for i in range(len(lower)):        # keep lower < upper (scipy.scale requires it)
        if upper[i] <= lower[i]:
            upper[i] = lower[i] + 1e-6
    return lower, upper


def parse_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="ODD sweep (max speed x max angle) for lane_keeping.")
    ap.add_argument("--n", type=int, default=15, help="LHS samples per cell (default 15)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers/containers (default: NUM_WORKERS env or 4)")
    ap.add_argument("--seed", type=int, default=42, help="seed (default 42)")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--speeds", default="14,22,30",
                    help="comma-separated max speeds (m/s) (default 14,22,30)")
    ap.add_argument("--angles", default="45,65,85",
                    help="comma-separated max angles (deg) (default 45,65,85)")
    ap.add_argument("--rare-fraction", type=float, default=0.05)
    ap.add_argument("--csv", default=None, help="also save results to this CSV")
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)
    n_workers = int(os.environ.get("NUM_WORKERS", "4"))

    import numpy as np
    from pipeline.orchestrator import run

    speeds = parse_list(args.speeds)
    angles = parse_list(args.angles)

    line = "=" * 72
    sub = "-" * 72
    print(line)
    print(" LANE_KEEPING — ODD sweep (failure rate over a speed x angle grid)")
    print(line)
    print(f" Samples/cell : {args.n}    Seed: {args.seed}    Workers: {n_workers}")
    print(f" Max speeds   : {speeds}  m/s")
    print(f" Max angles   : {angles}  deg")
    print(f" Total cells  : {len(speeds) * len(angles)}   "
          f"(~{args.n * len(speeds) * len(angles)} simulations)")
    print(sub)

    rows = []  # one dict per cell
    t0 = time.time()
    for a in angles:
        for s in speeds:
            lower, upper = build_bounds(a, s)
            r = run(args.scenario, n_samples=args.n, seed=args.seed,
                    rare_fraction=args.rare_fraction,
                    param_lower=lower, param_upper=upper, verbose=False)
            m = np.asarray(r.safety_margins, dtype=float)
            valid = (np.asarray(r.valid_mask, dtype=bool)
                     if getattr(r, "valid_mask", None) is not None else ~np.isnan(m))
            mv = m[valid]
            chz = getattr(r, "control_hz", None)
            chz_med = (float(np.nanmedian(np.asarray(chz, float)))
                       if chz is not None and np.isfinite(np.asarray(chz, float)).any() else float("nan"))
            rows.append({
                "angle_max": a, "speed_max": s,
                "fail": r.failure_rate * 100.0,
                "rare": r.rare_failure_rate * 100.0,
                "n_valid": int(valid.sum()),
                "margin_med": float(np.median(mv)) if mv.size else float("nan"),
                "hz_med": chz_med,
            })
            print(f"  angle_max={a:5.0f}  speed_max={s:5.0f}  ->  "
                  f"fail={r.failure_rate*100:5.1f}%  rare={r.rare_failure_rate*100:4.1f}%  "
                  f"median_margin={np.median(mv) if mv.size else float('nan'):+.2f}  "
                  f"median_rate={chz_med:4.1f}Hz  (valid {int(valid.sum())}/{args.n})",
                  flush=True)
    dt = time.time() - t0

    # Failure-rate matrix: rows = angle_max, columns = speed_max.
    print(sub)
    print(" FAILURE RATE MATRIX (%)   rows = max angle | columns = max speed")
    print(sub)
    header = "  ang\\vel |" + "".join(f"{s:8.0f}" for s in speeds)
    print(header)
    print("  " + "-" * (len(header) - 2))
    grid = {(d["angle_max"], d["speed_max"]): d["fail"] for d in rows}
    for a in angles:
        cells = "".join(f"{grid[(a, s)]:8.1f}" for s in speeds)
        print(f"  {a:6.0f}  |{cells}")
    print(sub)

    # Reading: cells nearest the informative band (20-40%).
    target_lo, target_hi = 20.0, 40.0
    in_band = [d for d in rows if target_lo <= d["fail"] <= target_hi]
    print(f" Target informative band: {target_lo:.0f}-{target_hi:.0f}% failures")
    if in_band:
        print(" Cells in band (candidate ODDs for rare-failure search):")
        for d in sorted(in_band, key=lambda x: abs(x["fail"] - 30.0)):
            print(f"   angle_max={d['angle_max']:.0f}  speed_max={d['speed_max']:.0f}"
                  f"  ->  fail={d['fail']:.1f}%  rare={d['rare']:.1f}%")
    else:
        nearest = min(rows, key=lambda x: abs(x["fail"] - 30.0))
        print(" No cell in band. Nearest to 30%:")
        print(f"   angle_max={nearest['angle_max']:.0f}  speed_max={nearest['speed_max']:.0f}"
              f"  ->  fail={nearest['fail']:.1f}%")
        print(" Hint: extend/refine the grid toward that value (--speeds/--angles) to hit the band.")
    print(sub)
    print(f" Total time: {dt:.0f}s")

    if args.csv:
        import csv
        path = args.csv if os.path.isabs(args.csv) else os.path.join(_PROJECT_ROOT, args.csv)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["angle_max", "speed_max", "fail", "rare",
                                              "n_valid", "margin_med", "hz_med"])
            w.writeheader()
            for d in rows:
                w.writerow(d)
        print(f" Results saved to: {path}")
    print(line)


if __name__ == "__main__":
    main()
