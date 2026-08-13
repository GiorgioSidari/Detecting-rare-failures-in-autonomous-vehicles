#!/usr/bin/env python3
"""
Is the scenario physically drivable? -- an ODD diagnostic.

The question it answers
-----------------------
When a backend fails in 100% of cases there are two very different explanations,
with opposite consequences for the report:

  * **the controller is weak**  -> a result about the controller, improved by
                                   tuning the gains or changing the control law;
  * **the ODD is physically impossible** -> no controller could manage, and the
                                   "failure" is not a defect of the system under
                                   test but of the scenario space.

The two are told apart by a calculation, not by conjecture: the lateral
acceleration required to drive the tightest curve at the target speed.

    a_lat = v^2 / R

Above the grip limit (~0.8g on dry asphalt) the vehicle slides regardless, by
definition.

Result measured on this ODD (seed 42, 20 LHS samples)
-----------------------------------------------------
    speed_scale   median a_lat   over the limit   observed failures
      1.0000          1.21 g           70%             100%
      0.1766          0.04 g            0%            15.8%

How to read it, in two parts:

1. **At `speed_scale = 1.0`, 70% of the space is over the grip limit.** The 100%
   failure rate measured at the start on Udacity and MetaDrive is therefore
   largely PHYSICS. Reporting it as "the controller always fails" would be
   misleading: much of that space is not drivable by anyone.

2. **At the calibrated point the maximum acceleration is 0.095 g.** At that
   level physics is no longer a constraint -- it is a parking manoeuvre. So the
   residual 15.8% of failures **cannot** be physics: it is entirely the
   controller's shortcoming.

Methodological consequence
--------------------------
Calibrating with speed alone moves the experiment from one regime to another
without flagging it. At the calibrated point one measures the controller's
weakness in conditions with no physical challenge; at scale=1 one mostly
measures physical impossibility. Neither is the interesting regime, which would
be: drivable but demanding scenarios.

The structural fix is to constrain the speed PER SCENARIO from the curvature
radius (`--suggest-cap`), so every sample sits below the grip limit by
construction and failures are always attributable to control.

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


def main() -> int:
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
