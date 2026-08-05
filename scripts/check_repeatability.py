#!/usr/bin/env python3
"""
How deterministic is the simulator? Evaluate the SAME points several times.

Why this exists. Every comparison in this project pairs arms by seed and assumes
that a seed fixes the experiment. It does not: the seed fixes which parameter
points get simulated, but the simulator itself is stochastic — the control loop
is not synchronous, so the same road driven twice does not produce the same
trajectory. Running the identical campaign twice has already produced 4 failures
out of 37 one day and 0 out of 37 the next.

That matters for how the results are read:

  * if the noise is small, seed-to-seed variation really is sampling variation
    and adding seeds buys statistical power as expected;
  * if the noise is large, a good part of what looks like "this design found more
    failures" is the simulator rolling dice, and the fix is not more seeds but
    REPEATING each point and averaging — a different experiment, and one that
    costs R times as much per point.

So measure it before spending hours on a campaign whose noise floor is unknown.

The script draws n points once and simulates them R times. What it reports:

  P(fail) per repeat     how much the headline number moves on identical input
  verdict flips          points that failed in one repeat and passed in another —
                         the honest measure of how binary the noise is
  margin spread          per-point standard deviation and range of the QoI, which
                         says whether the noise is small jitter or a coin flip

Usage:
    python scripts/check_repeatability.py --n 30 --repeats 3 \
        --max-angle 8 --max-speed 9.6 --max-seg 12
    python scripts/check_repeatability.py --n 30 --repeats 3 --workers 1
    python scripts/check_repeatability.py --n 40 --repeats 5 --csv results/repeat.csv

Cost is n x repeats simulations: 30 x 3 is about 5 minutes on 4 workers.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure simulator repeatability on a fixed set of points.")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--n", type=int, default=30, help="points to evaluate (default 30)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent evaluations of every point (default 3)")
    ap.add_argument("--seed", type=int, default=42, help="seed of the point design")
    ap.add_argument("--sampler", choices=["lhs", "random"], default="lhs")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel containers (default: NUM_WORKERS or 4)")
    ap.add_argument("--csv", default=None, help="write the per-point margins here")

    from pipeline.odd_presets import add_odd_args, resolve_bounds
    add_odd_args(ap)
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)

    from pipeline.samplers import get_sampler
    from scenarios import SCENARIOS

    sc = SCENARIOS[args.scenario]
    lower, upper = resolve_bounds(sc, args)
    if lower is None:
        b = sc.param_bounds()
        lower, upper = np.array(b["lower"], float), np.array(b["upper"], float)
    names = list(sc.param_bounds()["names"])
    thr = float(sc.failure_threshold())

    # One design, drawn once. Every repeat sees exactly these points.
    sampler = get_sampler(args.sampler)
    if hasattr(sc, "param_distributions"):
        theta = sampler.from_dists(args.n, sc.param_distributions(lower, upper),
                                   seed=args.seed)
    else:
        theta = sampler.bounded(args.n, lower, upper, seed=args.seed)

    print(f"[scenario] {args.scenario}")
    print(f"[design]   {args.n} points ({sampler.name}, seed={args.seed}), "
          f"{args.repeats} repeats = {args.n * args.repeats} simulations")
    print(f"[ODD]      lower={np.asarray(lower).tolist()}")
    print(f"[ODD]      upper={np.asarray(upper).tolist()}")
    print()

    margins = np.full((args.repeats, args.n), np.nan)
    t0 = time.time()
    for r in range(args.repeats):
        traj = sc.run_simulation(theta)
        m = np.asarray(sc.compute_qoi(traj, theta), float)
        valid = getattr(sc, "_valid_mask", None)
        if valid is None or np.asarray(valid).shape[0] != m.shape[0]:
            valid = np.isfinite(m)
        m = np.where(np.asarray(valid, bool), m, np.nan)
        margins[r] = m
        k, n = int(np.nansum(m < thr)), int(np.isfinite(m).sum())
        print(f"  repeat {r + 1}/{args.repeats}: P(fail) = {k / max(n, 1):6.1%}  "
              f"({k}/{n} valid)", flush=True)

    # A point is comparable only where every repeat produced a valid run.
    complete = np.isfinite(margins).all(axis=0)
    M = margins[:, complete]
    fails = M < thr

    print()
    w = 78
    print("=" * w)
    print(" REPEATABILITY — identical points, independent evaluations")
    print("=" * w)
    print(f"   points comparable across all repeats : {int(complete.sum())}/{args.n}")

    if M.shape[1] == 0:
        print("   no point completed every repeat — nothing to compare.")
        return

    always_fail = int((fails.all(axis=0)).sum())
    always_pass = int(((~fails).all(axis=0)).sum())
    flipped = int(M.shape[1] - always_fail - always_pass)
    flip_rate = flipped / M.shape[1]

    sd = M.std(axis=0, ddof=1)
    rng = M.max(axis=0) - M.min(axis=0)
    p_per_repeat = fails.mean(axis=1)

    print(f"   always failed                        : {always_fail}")
    print(f"   always passed                        : {always_pass}")
    print(f"   FLIPPED verdict between repeats      : {flipped}  ({flip_rate:.0%})")
    print()
    print(f"   P(fail) per repeat                   : "
          f"{', '.join(f'{p:.1%}' for p in p_per_repeat)}")
    print(f"   spread of P(fail) across repeats     : "
          f"{p_per_repeat.max() - p_per_repeat.min():.1%}")
    print()
    print(f"   per-point margin std                 : "
          f"median {np.median(sd):.3f}   max {sd.max():.3f}")
    print(f"   per-point margin range               : "
          f"median {np.median(rng):.3f}   max {rng.max():.3f}")
    print()

    # The margin threshold is 0, so noise only matters relative to how close the
    # margins sit to it. A large std on a point at margin +2 is harmless.
    near = np.abs(M.mean(axis=0)) < 0.2
    print(f"   points within 0.2 of the threshold   : {int(near.sum())}"
          f"   (of these, {int((fails[:, near].std(axis=0) > 0).sum())} flipped)")
    print()

    print("   VERDICT:")
    if flip_rate < 0.05:
        print("   The simulator is repeatable enough. Seed-to-seed differences are")
        print("   sampling variation, so adding seeds buys the power you expect.")
    elif flip_rate < 0.20:
        print(f"   {flip_rate:.0%} of the points are not reproducible. Noticeable but")
        print("   workable: keep pairing arms by seed (both arms face the same noise")
        print("   distribution) and add seeds, accepting that some of the variance")
        print("   you are averaging over is the simulator, not the design.")
    else:
        print(f"   {flip_rate:.0%} of the points flip verdict. The simulator noise")
        print("   dominates. Adding seeds will NOT fix this — each seed inherits the")
        print("   same coin flips. Either repeat every point and use the mean margin,")
        print("   or move the failure threshold away from where the noise lives.")
    print("=" * w)

    if args.csv:
        d = os.path.dirname(os.path.abspath(args.csv))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write(",".join(names + [f"margin_r{r + 1}" for r in range(args.repeats)]
                              + ["std", "range", "flipped"]) + "\n")
            for i in range(args.n):
                if not complete[i]:
                    continue
                col = margins[:, i]
                flip = bool((col < thr).any() and (col >= thr).any())
                fh.write(",".join(
                    [f"{v:.4f}" for v in theta[i]]
                    + [f"{v:.4f}" for v in col]
                    + [f"{col.std(ddof=1):.4f}", f"{col.max() - col.min():.4f}",
                       str(int(flip))]) + "\n")
        print(f"[saved] {args.csv}")

    print(f"[elapsed] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
