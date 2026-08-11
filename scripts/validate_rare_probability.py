#!/usr/bin/env python3
"""
Validation (M3) of the intermediate-level failure-probability estimate.

Does NOT validate the simulator (that is a separate matter, addressed via control fidelity):
it validates the statistical machinery added to the pipeline -- distribution-aware sampling via
ppf + P(failure) estimate + Wilson interval.

Idea: replace the simulator with a synthetic, known failure rule defined on the same 9
parameters. Under the real operational distributions (LaneKeepingScenario.param_distributions)
the true probability P_true of that rule is computable with an independent high-N Monte Carlo
(ground truth). We then check that the estimator used by the pipeline (LHS + ppf, as in
pipeline.orchestrator.run) at moderate N:
  1. CONVERGES to P_true (bias ~0),
  2. has a Wilson CI that COVERS P_true about 95% of the time (with LHS the variance is <=
     binomial, so the interval is at worst conservative).

Usage:
    python scripts/validate_rare_probability.py
    python scripts/validate_rare_probability.py --ns 50,200,1000 --reps 300

No Docker required: everything runs in-process.
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Reuse the REAL pipeline code: the scenario distributions and the Wilson CI.
from scenarios.lane_keeping.config import LaneKeepingScenario
from pipeline.orchestrator import _wilson_ci


def synthetic_fail(params: np.ndarray, angle_thr: float, speed_thr: float) -> np.ndarray:
    """
    Synthetic, known failure rule on the 9 parameters (same shape as the real scenario: worse
    with sharp curves AND high speed). It only provides a ground truth: fails if
    max(5 angles) > angle_thr AND max_speed > speed_thr. Raising the thresholds makes the event
    rarer (the default targets the rare regime, P ~ a few %), the interesting case for the CI.
    """
    max_angle = params[:, :5].max(axis=1)
    max_speed = params[:, 6]
    return (max_angle > angle_thr) & (max_speed > speed_thr)


def ppf_sample(dists, unit):
    """Transform unit samples with the ppf, exactly as the orchestrator does."""
    out = np.empty_like(unit)
    for j, d in enumerate(dists):
        out[:, j] = d.ppf(unit[:, j])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 validation of the P(failure) estimate.")
    ap.add_argument("--ns", default="50,200,1000",
                    help="N values to test, comma-separated (default 50,200,1000)")
    ap.add_argument("--reps", type=int, default=300,
                    help="repetitions (seeds) to estimate bias and coverage (default 300)")
    ap.add_argument("--truth-n", type=int, default=1_000_000,
                    help="ground-truth Monte Carlo samples (default 1e6)")
    ap.add_argument("--angle-thr", type=float, default=70.0,
                    help="angle threshold of the synthetic rule (default 70: rare regime)")
    ap.add_argument("--speed-thr", type=float, default=26.0,
                    help="speed threshold of the synthetic rule (default 26: rare regime)")
    args = ap.parse_args()

    Ns = [int(x) for x in args.ns.split(",") if x.strip()]

    sc = LaneKeepingScenario()
    b = sc.param_bounds()
    lower, upper = b["lower"], b["upper"]
    dists = sc.param_distributions(lower, upper)   # the scenario's REAL distributions
    d = len(lower)

    line = "=" * 68
    print(line)
    print(" M3 VALIDATION — P(failure) estimate (scenario's real distributions)")
    print(line)

    # Ground truth: independent high-N MC from the distributions.
    rng = np.random.default_rng(0)
    truth = np.empty((args.truth_n, d))
    for j, dist in enumerate(dists):
        truth[:, j] = dist.rvs(size=args.truth_n, random_state=rng)
    fails_truth = synthetic_fail(truth, args.angle_thr, args.speed_thr)
    p_true = float(fails_truth.mean())
    se_truth = float(np.sqrt(p_true * (1 - p_true) / args.truth_n))
    print(f" Synthetic rule: max(angles)>{args.angle_thr:g} AND max_speed>{args.speed_thr:g}")
    print(f" P_true (MC {args.truth_n:,} samples) = {p_true*100:.3f}%  (+/-{se_truth*100:.3f}%)")
    print("-" * 68)
    print(f" {'N':>6} | {'mean est':>12} | {'bias':>8} | {'CI95 coverage':>14} | {'CI width':>11}")
    print("-" * 68)

    all_ok = True
    for N in Ns:
        ests = np.empty(args.reps)
        covered = 0
        widths = np.empty(args.reps)
        for r in range(args.reps):
            unit = LatinHypercube(d=d, seed=1000 + r).random(n=N)   # as in the orchestrator
            params = ppf_sample(dists, unit)
            fails = synthetic_fail(params, args.angle_thr, args.speed_thr)
            k = int(fails.sum())
            ests[r] = k / N
            lo, hi = _wilson_ci(k, N)          # the pipeline's REAL Wilson CI
            widths[r] = hi - lo
            if lo <= p_true <= hi:
                covered += 1
        mean_est = float(ests.mean())
        bias = mean_est - p_true
        coverage = covered / args.reps
        # Criteria: small bias (decreasing with N) and coverage >= ~0.90 (Wilson is exact/
        # conservative; with LHS the variance is <= binomial, so coverage is not lower).
        ok_bias = abs(bias) < max(0.02, 2 * se_truth + 1.0 / N)
        ok_cov = coverage >= 0.90
        all_ok = all_ok and ok_bias and ok_cov
        flag = "OK" if (ok_bias and ok_cov) else "!!"
        print(f" {N:>6} | {mean_est*100:>11.2f}% | {bias*100:>+7.2f}% | "
              f"{coverage*100:>13.1f}% | {widths.mean()*100:>10.2f}%  {flag}")

    print("-" * 68)
    print(" Reading: mean est ~ P_true (bias ~0, shrinks with N) and ~95% coverage =")
    print("          the estimator and CI are correct. The convergence also shows why a small")
    print("          0/20 experiment is NOT 'P=0' but only a wide upper bound: the same machine,")
    print("          with more samples, tightens the CI.")
    print(line)
    print(" RESULT:", "ALL OK" if all_ok else "WARNING - some criterion not met")
    print(line)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
