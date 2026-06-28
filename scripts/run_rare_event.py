#!/usr/bin/env python3
"""
Cross-Entropy rare-event runner (M4) on the real simulator (Docker).

Estimates P(failure) under the operational distribution with the Cross-Entropy method: samples
adaptively toward the failure region instead of relying on blind sampling. Useful when failures
are rare (where plain Monte Carlo is inefficient).

Prerequisites: as for run_lanekeeping.py, the simulator containers must be running.

Usage:
    python scripts/run_rare_event.py
    python scripts/run_rare_event.py --preset full --spi 60 --final 300
    python scripts/run_rare_event.py --workers 4 --seed 1 --quiet

Budget: each iteration runs `--spi` simulations plus `--final` at the end. At ~10 s/run a budget
of ~500-800 runs takes roughly 1-2 hours. Start small.

ODD note: on the 'realistic' preset, at correct control rate, failures are ~absent, so CE would
estimate P~0 (a valid result: "no failures in this ODD"). To see CE work you need an ODD where
failures exist but are rare: the 'full' preset (angles up to 85, speed up to 30) is the default.
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-Entropy (M4) on the lane_keeping simulator.")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--preset", choices=["full", "realistic"], default="full",
                    help="'full' = wide ODD (default: failures exist but are rare); "
                         "'realistic' = narrow ODD (at correct rate ~no failures)")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="cap max_speed (m/s): narrows the ODD toward the rare regime")
    ap.add_argument("--max-angle", type=float, default=None,
                    help="cap angles (deg): narrows the ODD toward the rare regime")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="cap min_speed (m/s): with a low --max-speed avoids overlapping bands")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers/containers (default: NUM_WORKERS env or 4)")
    ap.add_argument("--spi", type=int, default=60, help="CE samples_per_iter (default 60)")
    ap.add_argument("--final", type=int, default=300, help="final-estimate samples (default 300)")
    ap.add_argument("--max-iter", type=int, default=10, help="max CE iterations (default 10)")
    ap.add_argument("--rho", type=float, default=0.2, help="elite fraction (default 0.2)")
    ap.add_argument("--alpha", type=float, default=0.2, help="mixture fraction from f (default 0.2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="do not show the per-iteration gamma descent")
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)
    n_workers = int(os.environ.get("NUM_WORKERS", "4"))

    import numpy as np
    from pipeline.orchestrator import run_rare_event
    from scenarios import SCENARIOS

    lower = REALISTIC_LOWER if args.preset == "realistic" else None
    upper = REALISTIC_UPPER if args.preset == "realistic" else None

    # Explicit overrides to target a narrower ODD (rare regime). They require explicit bounds:
    # for the 'full' preset we start from the scenario defaults.
    if (args.max_speed is not None or args.max_angle is not None
            or args.min_speed is not None):
        if lower is None or upper is None:
            _b = SCENARIOS[args.scenario].param_bounds()
            lower = list(np.asarray(_b["lower"], dtype=float))
            upper = list(np.asarray(_b["upper"], dtype=float))
        else:
            lower = list(map(float, lower)); upper = list(map(float, upper))
        if args.max_angle is not None:
            for k in range(5):                               # angles 1..5
                upper[k] = float(args.max_angle)
                lower[k] = min(lower[k], upper[k])
        if args.max_speed is not None:
            X = float(args.max_speed)
            upper[6] = X                                     # max_speed upper
            lower[6] = max(1.0, min(lower[6], X - 1.0))
            upper[5] = min(upper[5], X)                      # min_speed not above X
            lower[5] = max(1.0, min(lower[5], upper[5] - 1.0))
        if args.min_speed is not None:
            Y = float(args.min_speed)
            upper[5] = Y                                     # min_speed upper
            lower[5] = max(0.5, min(lower[5], Y - 1.0))
        for i in range(len(lower)):                          # keep lower < upper
            if upper[i] <= lower[i]:
                upper[i] = lower[i] + 1e-6

    names = SCENARIOS[args.scenario].param_bounds()["names"]

    line = "=" * 66
    sub = "-" * 66
    print(line)
    print(f" {args.scenario.upper()} — rare-event estimate (Cross-Entropy)")
    print(line)
    budget_max = args.spi * args.max_iter + args.final
    print(f" ODD preset     : {args.preset}")
    if args.max_angle is not None:
        print(f" Max angle (cap): {args.max_angle} deg")
    if args.max_speed is not None:
        print(f" Max speed (cap): {args.max_speed} m/s")
    if args.min_speed is not None:
        print(f" Min speed (cap): {args.min_speed} m/s")
    print(" Sampling       : adaptive Cross-Entropy + importance sampling")
    print(f" CE budget      : {args.spi}/iter x max {args.max_iter} iter + {args.final} final "
          f"(<= {budget_max} runs)")
    print(f" Worker pool    : {n_workers}  (NUM_WORKERS={n_workers})")
    print(sub)
    if not args.quiet:
        print(" Gamma descent (the 'near-failure' threshold toward 0):")

    t0 = time.time()
    res = run_rare_event(
        args.scenario, seed=args.seed,
        param_lower=lower, param_upper=upper,
        samples_per_iter=args.spi, final_samples=args.final,
        rho=args.rho, max_iter=args.max_iter, alpha=args.alpha,
        verbose=not args.quiet,
    )
    dt = time.time() - t0

    print(sub)
    print(" RESULTS")
    print(sub)
    print(f" Total time          : {dt:6.1f}s   ({res.n_evaluations} runs)")
    print(f" CE iterations       : {res.iterations}")
    lo, hi = res.ci
    print(f" P(failure)          : {res.p_fail*100:.4f}%   CI95% [{lo*100:.4f}, {hi*100:.4f}]%")
    print(f" Effective failures (final estimate): {res.n_fail_effective}")
    if res.gamma_history:
        gh = ", ".join(f"{g:+.3f}" for g in res.gamma_history)
        print(f" Gamma per iteration: [{gh}]   (reaches <=0 = failure threshold)")
    print(sub)

    # Failure region: where CE moved the proposal. The parameters that shifted the most from the
    # ODD center are those that drive the failure.
    print(" FAILURE REGION (where the proposal concentrated):")
    bnd = SCENARIOS[args.scenario].param_bounds()
    mid = (np.asarray(bnd["lower"], float) + np.asarray(bnd["upper"], float)) / 2.0
    if lower is not None:
        mid = (np.asarray(lower, float) + np.asarray(upper, float)) / 2.0
    for j, nm in enumerate(names):
        shift = res.q_loc[j] - mid[j]
        arrow = "^" if shift > 0 else ("v" if shift < 0 else ".")
        print(f"   {nm:<20} loc={res.q_loc[j]:7.2f}  (ODD center {mid[j]:7.2f})  {arrow}{abs(shift):6.2f}")
    print(sub)

    if res.p_fail <= 0.0 or res.n_fail_effective == 0:
        print(" Reading: no failure found in this ODD -> P ~ 0. At correct control rate the narrow")
        print("          ODD is very safe; try --preset full or a harder ODD to surface (and")
        print("          estimate) rare failures.")
    else:
        print(" Reading: CE located the failures and estimated P via importance sampling; the")
        print("          'failure region' above tells which parameters drive them. Compare P")
        print("          against brute force only if you can afford it.")
    print(line)


if __name__ == "__main__":
    main()
