#!/usr/bin/env python3
"""
Validation (M4) of the Cross-Entropy estimator in pipeline/rare_event.py.

As in M3, the simulator is replaced with a synthetic, known MARGIN (so the true probability
P_true is computable by brute force), and we check that the Cross-Entropy estimator:
  1. gives an unbiased estimate of P_true (correct CI coverage),
  2. reaches it with FAR FEWER runs than plain Monte Carlo at the same budget (variance
     reduction) -- the whole point of the method.

The synthetic margin is UNIMODAL (fails when mean(angles) and max_speed are both high): the class
of regions the CE with a unimodal proposal handles well. On multimodal regions CE underestimates
(documented limitation in rare_event.py).

Usage:
    python scripts/validate_rare_event.py
    python scripts/validate_rare_event.py --reps 12 --angle-thr 53 --speed-thr 28.5

No Docker required.
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scenarios.lane_keeping.config import LaneKeepingScenario
from pipeline.rare_event import estimate_failure_probability, sample_product


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="M4 validation (Cross-Entropy vs brute force).")
    ap.add_argument("--reps", type=int, default=10, help="CE repetitions (seeds) (default 10)")
    ap.add_argument("--angle-thr", type=float, default=53.0,
                    help="mean-angle threshold of the synthetic rule (default 53: P~1e-4)")
    ap.add_argument("--speed-thr", type=float, default=28.5,
                    help="max_speed threshold of the synthetic rule (default 28.5)")
    ap.add_argument("--truth-n", type=int, default=3_000_000,
                    help="ground-truth Monte Carlo samples (default 3e6)")
    # CE budget (runs are free here, so generous values).
    ap.add_argument("--spi", type=int, default=500, help="CE samples_per_iter")
    ap.add_argument("--final", type=int, default=8000, help="final-estimate samples")
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    at, st = args.angle_thr, args.speed_thr

    def margin_fn(X):
        # UNIMODAL: margin < 0  <=>  mean(angles) > at  AND  max_speed > st
        return np.maximum(at - X[:, :5].mean(axis=1), st - X[:, 6])

    sc = LaneKeepingScenario()
    b = sc.param_bounds()
    lower, upper = b["lower"], b["upper"]
    f_dists = sc.param_distributions(lower, upper)
    d = len(lower)

    line = "=" * 70
    print(line)
    print(" M4 VALIDATION — Cross-Entropy vs Monte Carlo (real distributions)")
    print(line)

    # Ground truth.
    rng = np.random.default_rng(0)
    T = sample_product(f_dists, args.truth_n, rng, d)
    p_true = float((margin_fn(T) < 0).mean())
    print(f" Synthetic rule (unimodal): mean(angles)>{at:g} AND max_speed>{st:g}")
    print(f" P_true (MC {args.truth_n:,}) = {p_true*100:.4f}%  ({p_true:.2e})")
    print("-" * 70)

    # Cross-Entropy over several seeds.
    ce_p, ce_eval, covered = [], [], 0
    for s in range(args.reps):
        res = estimate_failure_probability(
            margin_fn, f_dists, lower, upper,
            samples_per_iter=args.spi, final_samples=args.final, seed=s)
        ce_p.append(res.p_fail)
        ce_eval.append(res.n_evaluations)
        cov = res.ci[0] <= p_true <= res.ci[1]
        covered += cov
        print(f"  CE seed {s:2d}: p={res.p_fail*100:.4f}%  "
              f"CI[{res.ci[0]*100:.4f}, {res.ci[1]*100:.4f}]  "
              f"eval={res.n_evaluations}  iter={res.iterations}  cover={cov}")
    ce_p = np.array(ce_p)
    mean_eval = int(np.mean(ce_eval))
    # Empirical relative error (std of estimates / P_true) = estimator quality.
    ce_relerr = float(ce_p.std() / p_true) if p_true > 0 else float("inf")

    # Plain Monte Carlo at the SAME budget.
    B = mean_eval
    mc_fails = (margin_fn(sample_product(f_dists, B, np.random.default_rng(12345), d)) < 0)
    k_mc = int(mc_fails.sum())
    p_mc = k_mc / B
    mc_relerr = float(np.sqrt(p_mc * (1 - p_mc) / B) / p_mc) if p_mc > 0 else float("inf")

    print("-" * 70)
    print(f" Cross-Entropy : mean est = {ce_p.mean()*100:.4f}%   "
          f"CI coverage = {covered}/{args.reps}   rel.err. = {ce_relerr*100:.1f}%")
    print(f" Plain MC      : {k_mc} failures in {B} runs   est = {p_mc*100:.4f}%   "
          f"rel.err. = {mc_relerr*100:.1f}%")
    print("-" * 70)
    speedup = (mc_relerr / ce_relerr) ** 2 if ce_relerr > 0 else float("inf")
    print(f" At the same budget ({B} runs), CE has ~{mc_relerr/max(ce_relerr,1e-9):.1f}x smaller")
    print(f" relative error -> plain MC would need ~{speedup:.0f}x more runs to match it.")
    print("-" * 70)

    ok_cov = covered >= int(0.80 * args.reps)          # coverage ~>=80% (bootstrap)
    ok_bias = abs(ce_p.mean() - p_true) < 0.5 * p_true # mean bias < 50% (unimodal region)
    ok_var = ce_relerr < mc_relerr                     # CE more precise than MC
    all_ok = ok_cov and ok_bias and ok_var
    print(" RESULT:", "ALL OK" if all_ok else "WARNING - check parameters/regime")
    print(line)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
