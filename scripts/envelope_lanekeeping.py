#!/usr/bin/env python3
"""
Operational envelope of the lane keeper: P(failure) vs METERS-PER-STEER.

Central finding of the analysis: after the unit fix, failures are governed not by nominal
speed but by the SPATIAL CONTROL RESOLUTION -- the metres the car travels between two steering
decisions (= speed / control rate). Speed is only a proxy, confounded by the control rate
(which varies with machine load).

This script measures it cleanly:
  1. run several conditions (different max speeds) to SPAN the meters-per-step range;
  2. POOL every valid run, each with its measured meters-per-step and its outcome;
  3. bin by meters-per-step and compute P(failure) per bin (with Wilson CI);
  4. print the table, save the P-vs-(m/step) curve, and derive the threshold and the CONTROL-RATE
     REQUIREMENT (to drive at a target speed you need rate >= speed / threshold).

The fidelity gate is forced OFF here: we do NOT want to drop under-sampled runs, we want them
as the high-m/step points of the curve.

Usage:
    python scripts/envelope_lanekeeping.py --workers 4
    python scripts/envelope_lanekeeping.py --speeds 4,5,6,8,10,12 --n 40 --min-speed 3
    python scripts/envelope_lanekeeping.py --out envelope.png --csv envelope.csv

Prerequisites: simulator containers running. Tip: keep the machine idle (other apps closed) so
the control rate does not fluctuate during collection.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# "realistic" preset bounds (same as run_lanekeeping.py).
REALISTIC_LOWER = [0,  0,  0,  0,  0,   5.0,  9.0, 20.0, 200.0]
REALISTIC_UPPER = [45, 45, 45, 45, 45,  8.0, 14.0, 40.0, 350.0]


def build_bounds(max_speed: float, min_speed_cap: float):
    """Realistic ODD with max_speed capped and min_speed kept below that band."""
    lo = list(map(float, REALISTIC_LOWER))
    up = list(map(float, REALISTIC_UPPER))
    up[6] = float(max_speed)                                  # max_speed upper
    lo[6] = max(1.0, min(lo[6], max_speed - 1.0))
    cap = min(float(min_speed_cap), lo[6] - 0.5)             # keep min_speed below the max band
    up[5] = max(1.0, cap)
    lo[5] = max(0.5, min(lo[5], up[5] - 1.0))
    for i in range(len(lo)):
        if up[i] <= lo[i]:
            up[i] = lo[i] + 1e-6
    return lo, up


def main() -> None:
    ap = argparse.ArgumentParser(description="Operational envelope: P(failure) vs meters-per-steer.")
    ap.add_argument("--speeds", default="4,5,6,8,10,12",
                    help="max speeds (m/s) used to span m/step (default 4,5,6,8,10,12)")
    ap.add_argument("--n", type=int, default=40, help="samples per condition (default 40)")
    ap.add_argument("--min-speed", type=float, default=3.0,
                    help="min_speed cap (m/s) to avoid overlapping speed bands (default 3)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=8, help="number of m/step bins (default 8)")
    ap.add_argument("--out", default="envelope.png", help="curve PNG path (default envelope.png)")
    ap.add_argument("--csv", default=None, help="also save per-run points to this CSV")
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)
    n_workers = int(os.environ.get("NUM_WORKERS", "4"))
    # Force the fidelity gate OFF: keep under-sampled runs as points of the curve.
    os.environ["LK_MIN_CONTROL_HZ"] = "0"
    os.environ["LK_MAX_METERS_PER_STEP"] = "0"

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pipeline.orchestrator import run, _wilson_ci

    speeds = [float(x) for x in args.speeds.split(",") if x.strip()]

    line = "=" * 70
    print(line)
    print(" LANE_KEEPING — operational envelope: P(failure) vs meters-per-steer")
    print(line)
    print(f" Conditions (max_speed): {speeds}  m/s   |  N/cond: {args.n}  |  workers: {n_workers}")
    print(" Fidelity gate OFF (keep every run as a point of the curve)")
    print("-" * 70)

    # Collect a pool of (m/step, failed) over all conditions.
    mstep_all, fail_all = [], []
    t0 = time.time()
    for X in speeds:
        lo, up = build_bounds(X, args.min_speed)
        r = run("lane_keeping", n_samples=args.n, seed=args.seed,
                sampling="realistic", param_lower=lo, param_upper=up, verbose=False)
        valid = (np.asarray(r.valid_mask, bool) if getattr(r, "valid_mask", None) is not None
                 else ~np.isnan(np.asarray(r.safety_margins, float)))
        mps = np.asarray(getattr(r, "meters_per_step", np.full(args.n, np.nan)), float)
        failed = np.asarray(r.failures, float)
        keep = valid & np.isfinite(mps)
        mstep_all.extend(mps[keep].tolist())
        fail_all.extend(failed[keep].tolist())
        nv = int(keep.sum())
        pf = float(failed[keep].mean()) * 100 if nv else float("nan")
        print(f"  max_speed={X:5.1f}  ->  {nv:2d} valid runs  |  median m/step "
              f"{np.median(mps[keep]) if nv else float('nan'):.2f}  |  P={pf:5.1f}%", flush=True)
    dt = time.time() - t0

    mstep_all = np.asarray(mstep_all)
    fail_all = np.asarray(fail_all)
    if mstep_all.size == 0:
        print(" No valid runs collected."); print(line); return

    # Bin by meters-per-step.
    edges = np.linspace(mstep_all.min(), mstep_all.max(), args.bins + 1)
    print("-" * 70)
    print(" P(failure) vs meters-per-steer:")
    print(f" {'m/step (bin)':>16} | {'n':>4} | {'P(fail)':>8} | {'CI95%':>16}")
    centers, ps, los, his = [], [], [], []
    for b in range(args.bins):
        m = (mstep_all >= edges[b]) & (mstep_all < edges[b + 1] if b < args.bins - 1
                                       else mstep_all <= edges[b + 1])
        n = int(m.sum())
        if n == 0:
            continue
        k = int(fail_all[m].sum())
        p = k / n
        lo_ci, hi_ci = _wilson_ci(k, n)
        c = 0.5 * (edges[b] + edges[b + 1])
        centers.append(c); ps.append(p); los.append(lo_ci); his.append(hi_ci)
        print(f" {edges[b]:6.2f}-{edges[b+1]:5.2f} | {n:4d} | {p*100:7.1f}% | "
              f"[{lo_ci*100:5.1f}, {hi_ci*100:5.1f}]")

    centers = np.asarray(centers); ps = np.asarray(ps)

    # Threshold: m/step at which P crosses 50% (linear interpolation).
    thr = None
    for i in range(1, len(centers)):
        if ps[i - 1] < 0.5 <= ps[i]:
            x0, x1, y0, y1 = centers[i-1], centers[i], ps[i-1], ps[i]
            thr = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else centers[i]
            break
    print("-" * 70)
    if thr is not None:
        print(f" Threshold (P=50%): ~{thr:.2f} m/step")
        print(f" CONTROL-RATE REQUIREMENT for a target speed v:  rate >= v / {thr:.2f}")
        for v in (6.0, 10.0, 14.0):
            print(f"   {v:.0f} m/s safely needs ~{v/thr:4.1f} Hz  (Unity is ~9 Hz today)")
    else:
        print(" P does not cross 50% in the sampled range: extend --speeds.")
    print(f" Total time: {dt:.0f}s   |  total valid runs: {mstep_all.size}")

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = np.vstack([ps - np.asarray(los), np.asarray(his) - ps])
    ax.errorbar(centers, ps * 100, yerr=yerr * 100, fmt="o-", color="#c0392b",
                ecolor="#e59866", capsize=3, label="P(failure) (bin, CI95%)")
    if thr is not None:
        ax.axvline(thr, ls="--", color="#2c3e50", alpha=0.7)
        ax.text(thr, 5, f" threshold ~{thr:.2f} m/step", color="#2c3e50")
    ax.set_xlabel("meters per steer  (speed / control rate)")
    ax.set_ylabel("P(failure)  [%]")
    ax.set_title("Lane keeper operational envelope")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    out_path = args.out if os.path.isabs(args.out) else os.path.join(_PROJECT_ROOT, args.out)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f" Curve saved to: {out_path}")

    # Optional per-run CSV.
    if args.csv:
        import csv
        path = args.csv if os.path.isabs(args.csv) else os.path.join(_PROJECT_ROOT, args.csv)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["meters_per_step", "failed"])
            for mm, ff in zip(mstep_all, fail_all):
                w.writerow([f"{mm:.4f}", int(ff)])
        print(f" Per-run points saved to: {path}")
    print(line)


if __name__ == "__main__":
    main()
