"""
Run the active-learning failure-boundary learner (Step B) on a scenario.

Defaults to the MetaDrive lane-keeping scenario in the discriminative regime
(speed_scale=0.4), where failures depend on scenario difficulty and there is a
real boundary to learn. Prints P(failure) with a credible interval and the ARD
parameter importance (which parameters drive the failure).

Run:
    python scripts/run_active_boundary.py
    python scripts/run_active_boundary.py --n-seed 60 --batch 20 --n-iter 10
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from scenarios import SCENARIOS
from pipeline.active_boundary import run_active_boundary

# A few GP dimensions are genuinely irrelevant (large length-scale) and may still
# nudge a bound; silence that expected noise for a clean report.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lane_keeping_md")
    ap.add_argument("--speed-scale", type=float, default=0.4,
                    help="target-speed scale for the discriminative regime (MetaDrive)")
    ap.add_argument("--n-seed", type=int, default=40, help="initial LHS evaluations")
    ap.add_argument("--batch", type=int, default=16, help="points per active iteration")
    ap.add_argument("--n-iter", type=int, default=8, help="active-learning iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uniform", action="store_true",
                    help="estimate P over a uniform ODD instead of param_distributions")
    ap.add_argument("--learned-model", default=None,
                    help="path to a trained BC model; drive with LearnedDriver "
                         "(genuine generalization failures instead of an analytic driver)")
    ap.add_argument("--max-angle", type=float, default=None,
                    help="narrow the ODD: cap all 5 road angles (deg) — gentler curves")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="narrow the ODD: cap max_speed (m/s) — lower speeds")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="narrow the ODD: raise the lower speed bound (m/s)")
    ap.add_argument("--max-seg", type=float, default=None,
                    help="narrow the ODD: cap segment_length (m)")
    ap.add_argument("--min-seg", type=float, default=None,
                    help="narrow the ODD: raise the segment_length lower bound (m)")
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    if hasattr(sc, "speed_scale"):
        sc.speed_scale = args.speed_scale
        print(f"[scenario] {args.scenario}  speed_scale={args.speed_scale}")

    if args.learned_model:
        from scenarios.lane_keeping_md.bc_controller import LearnedDriver
        sc._driver_factory = lambda: LearnedDriver(model_path=args.learned_model)
        print(f"[driver] LearnedDriver <- {args.learned_model}")

    # Optionally narrow the operational design domain (ODD) to make failures rare.
    b = sc.param_bounds()
    lower, upper = np.array(b["lower"], float), np.array(b["upper"], float)
    narrowed = False
    if args.max_angle is not None:
        for k in range(5):                                 # angles 1..5
            upper[k] = args.max_angle
            lower[k] = min(lower[k], upper[k])
        narrowed = True
    if args.max_speed is not None:                         # keep a non-degenerate band:
        X = args.max_speed                                 # lower the LOWER bound too, else
        upper[6] = X                                       # max_speed [10,30] inverts at X<10
        lower[6] = max(1.0, min(lower[6], X - 1.0))
        upper[5] = min(upper[5], X)                        # min_speed not above max_speed cap
        lower[5] = max(1.0, min(lower[5], upper[5] - 1.0))
        narrowed = True
    if args.min_speed is not None:
        Y = args.min_speed
        upper[5] = Y
        lower[5] = max(0.5, min(lower[5], Y - 1.0))
        narrowed = True
    if args.max_seg is not None:
        upper[7] = args.max_seg
        lower[7] = min(lower[7], upper[7] - 1.0)
        narrowed = True
    if args.min_seg is not None:
        lower[7] = args.min_seg
        upper[7] = max(upper[7], lower[7] + 1.0)
        narrowed = True
    p_lower = lower if narrowed else None
    p_upper = upper if narrowed else None
    if narrowed:
        print(f"[ODD] narrowed: max_angle={args.max_angle} max_speed={args.max_speed} "
              f"min_speed={args.min_speed} max_seg={args.max_seg} min_seg={args.min_seg}")

    res = run_active_boundary(
        sc, seed=args.seed, n_seed=args.n_seed, batch=args.batch,
        n_iter=args.n_iter, weight_by_odd=not args.uniform,
        param_lower=p_lower, param_upper=p_upper, verbose=True,
    )

    s = res.boundary_summary
    lo, hi = res.p_fail_ci
    print("=" * 70)
    print(" ACTIVE BOUNDARY — RESULT")
    print("=" * 70)
    print(f" total evaluations   : {res.n_evaluations}")
    print(f" empirical fail rate : {s['empirical_failure_rate']*100:.1f}%")
    print(f" P(failure) [{s['weighting']}]")
    print(f"                     : {res.p_fail*100:.2f}%   CI95 [{lo*100:.1f}, {hi*100:.1f}]%")
    print(" most influential parameters (ARD importance):")
    for name, imp in s["top_params"]:
        bar = "#" * int(round(imp * 40))
        print(f"    {name:<20} {imp*100:5.1f}%  {bar}")

    # Concrete failing scenarios found (in a rare ODD these ARE the rare failures).
    fails = np.where(res.labels == 1)[0]
    if len(fails):
        order = fails[np.argsort(res.margins[fails])]      # worst (most negative) first
        print(f" failing scenarios found: {len(fails)}/{res.n_evaluations}  (worst first)")
        for i in order[:8]:
            th = res.theta_evaluated[i]
            ang = ",".join(f"{int(round(a)):2d}" for a in th[:5])
            print(f"   margin={res.margins[i]:+.3f}  angles=[{ang}]  "
                  f"minV={th[5]:.1f} maxV={th[6]:.1f} seg={th[7]:.1f} map={th[8]:.0f}")
    print("=" * 70)
    print(" Note: P(failure) is the rarity under the ODD; the importance shows WHICH")
    print(" parameters drive the failure (short ARD length-scale = sensitive).")
    print(" Evaluated points concentrate on the fail/safe boundary by construction.")


if __name__ == "__main__":
    main()
