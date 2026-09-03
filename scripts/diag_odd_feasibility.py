#!/usr/bin/env python3
"""
ODD drivability diagnostic: how much of the scenario space is above the grip
limit.

The script draws `--n` LHS samples from the ODD, and for each one computes the
tightest curve radius `R` of the generated road and the target speed `v` at
each `speed_scale` in `--scales`. From those it derives the lateral
acceleration the curve demands,

    a_lat = v^2 / R

and compares it with the grip limit `mu * g` (`--mu`, default 0.8).

Printed output:

  * a header with the sample count, the grip limit, the minimum and median
    curve radius and the target-speed range at `speed_scale = 1`;
  * one row per `speed_scale`: median speed, median and maximum `a_lat` in g,
    and the share of samples whose `a_lat` exceeds the limit ("over");
  * with `--suggest-cap`, a second block computing the per-scenario speed cap
    `v_max = sqrt(0.6 * mu * g * R)`, its range, the ratio `v_max / v_target`
    and the global `speed_scale` equivalent to the smallest of those ratios.

The script runs no simulation: it only evaluates road geometry and speeds.

Usage
-----
    python scripts/diag_odd_feasibility.py
    python scripts/diag_odd_feasibility.py --n 200 --mu 0.7 --suggest-cap
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

G = 9.81


def _build_parser() -> argparse.ArgumentParser:
    """The command line of this script."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="LHS samples")
    ap.add_argument("--seed", type=int, default=42,
                    help="design seed (42 = the calibration one)")
    ap.add_argument("--mu", type=float, default=0.8,
                    help="assumed grip coefficient (0.8 = dry asphalt)")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 0.5, 0.3625, 0.2562, 0.1766, 0.15])
    ap.add_argument("--suggest-cap", action="store_true", dest="cap",
                    help="estimate the per-scenario speed cap that makes the ODD drivable")
    return ap


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()

    from scipy.stats.qmc import LatinHypercube, scale as qmc_scale
    from scenarios.lane_keeping_md import LaneKeepingMetaDriveScenario
    from scenarios.lane_keeping_md.map_builder import build_scenario_spec, target_speed

    bounds = LaneKeepingMetaDriveScenario().param_bounds()
    params = qmc_scale(LatinHypercube(d=len(bounds["lower"]), seed=args.seed)
                       .random(n=args.n), bounds["lower"], bounds["upper"])

    specs = [build_scenario_spec(r, len(bounds["lower"])) for r in params]
    radii = np.array([s.sharpest_curve_radius for s in specs])
    v_full = np.array([target_speed(s) for s in specs])
    limit = args.mu * G

    print("=" * 72)
    print(" PHYSICAL DRIVABILITY OF THE ODD")
    print("=" * 72)
    print(f" Samples       : {args.n} (LHS seed {args.seed})")
    print(f" Grip          : mu={args.mu}  ->  limit {limit:.1f} m/s^2 ({args.mu:.1f}g)")
    print(f" Min radius    : {radii.min():.1f} m   (median {np.median(radii):.1f} m)")
    print(f" Target speed  : {v_full.min():.1f} - {v_full.max():.1f} m/s at scale=1")
    print("-" * 72)
    print(f" {'scale':>8} {'v med':>7} {'a_lat med':>11} {'a_lat max':>11} {'over':>7}")
    print("-" * 72)

    for sc in sorted(args.scales, reverse=True):
        v = v_full * sc
        a = v ** 2 / radii
        over = float((a > limit).mean())
        print(f" {sc:>8.4f} {np.median(v):>6.1f}m/s {np.median(a) / G:>10.3f}g "
              f"{a.max() / G:>10.3f}g {over:>6.0%}")

    print("-" * 72)
    print(" How to read it:")
    print("   high 'over'  -> those scenarios are NOT drivable by any controller:")
    print("                   the failure is physics, not control.")
    print("   'over' = 0%  -> physics does not constrain: every residual failure")
    print("                   is attributable to the controller.")

    if args.cap:
        # Highest drivable speed per scenario, with margin.
        v_max = np.sqrt(0.6 * limit * radii)      # 60% of the limit = manoeuvring margin
        ratio = v_max / v_full
        print()
        print("=" * 72)
        print(" PER-SCENARIO SPEED CAP (alternative to a global speed_scale)")
        print("=" * 72)
        print(" v_max = sqrt(0.6 * mu * g * R): 60% of the limit leaves margin for")
        print(" control instead of putting the vehicle on the edge of grip.")
        print()
        print(f"   v_max            : {v_max.min():.1f} - {v_max.max():.1f} m/s")
        print(f"   v_max / v_target : {ratio.min():.3f} - {ratio.max():.3f}")
        print(f"   global scale equivalent to the worst case: {ratio.min():.4f}")
        print()
        print(" Why it beats a global scale: a per-scenario cap keeps wide curves")
        print(" fast and slows down only where needed. With a global scale, making")
        print(" the worst curve drivable slows EVERYTHING down -- which is how one")
        print(" ends up at 0.04g on average, where no physical challenge is left and")
        print(" only the controller is being measured.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
