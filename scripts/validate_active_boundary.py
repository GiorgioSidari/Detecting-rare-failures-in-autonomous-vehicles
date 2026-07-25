"""
Validate the active-boundary estimator (Part C item 2).

Two checks, on the real scenario:

1) CORRECTNESS — brute-force Monte-Carlo P(failure) under the ODD (ground truth
   with a Wilson CI) vs the active-boundary P estimate. They should agree.

2) SAMPLE EFFICIENCY — how accurately P(failure) is estimated as a function of
   the simulation budget, for plain MC COUNTING vs the GP SURROGATE (fit on the
   same points, then integrate the smooth P(fail|theta) over the ODD). The
   surrogate should reach a given accuracy with fewer simulations. This part
   REUSES the brute-force simulations (no extra runs), so it is cheap.

Run (needs the opensbt-core simulator pool up, see README):
    python scripts/validate_active_boundary.py --n-bf 200
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from scipy.stats.qmc import LatinHypercube
from sklearn.exceptions import ConvergenceWarning

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import SCENARIOS
from pipeline.active_boundary import run_active_boundary, _build_gp, _p_fail

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sample_odd(scenario, n: int, seed: int, lower, upper) -> np.ndarray:
    """Draw n points from the operational distribution (ppf of LHS), as run()."""
    dists = scenario.param_distributions(lower, upper)
    u = LatinHypercube(d=len(lower), seed=seed).random(n=n)
    theta = np.empty_like(u)
    for j, dist in enumerate(dists):
        theta[:, j] = dist.ppf(u[:, j])
    return theta


def evaluate(scenario, theta):
    traj = scenario.run_simulation(theta)
    m = np.asarray(scenario.compute_qoi(traj, theta), dtype=float)
    v = getattr(scenario, "_valid_mask", None)
    if v is None or np.asarray(v).shape[0] != m.shape[0]:
        v = ~np.isnan(m)
    return m, np.asarray(v, dtype=bool)


def surrogate_p(theta_tr, y_tr, lower, upper, odd_pts, thr) -> float:
    """GP-surrogate P(failure): fit on (theta, margin), integrate P(fail|theta)
    over ODD points (model-only, no simulation)."""
    span = np.where((upper - lower) > 0, upper - lower, 1.0)
    gp = _build_gp(theta_tr.shape[1])
    gp.fit((theta_tr - lower) / span, y_tr)
    mu, sig = gp.predict((odd_pts - lower) / span, return_std=True)
    return float(np.mean(_p_fail(mu, sig, thr)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--n-bf", type=int, default=200, help="brute-force MC simulations")
    ap.add_argument("--n-seed", type=int, default=30)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--n-iter", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    b = sc.param_bounds()
    lower, upper = np.asarray(b["lower"], float), np.asarray(b["upper"], float)
    thr = float(sc.failure_threshold())

    # ── 1. Brute-force ground truth ───────────────────────────────────────────
    print(f"[brute-force] {args.n_bf} ODD simulations ...")
    theta_bf = sample_odd(sc, args.n_bf, args.seed, lower, upper)
    m_bf, v_bf = evaluate(sc, theta_bf)
    theta_bf, m_bf = theta_bf[v_bf], m_bf[v_bf]
    fail = (m_bf < thr)
    k, n = int(fail.sum()), len(fail)
    P_bf = k / n
    lo, hi = wilson(k, n)
    print(f"  P_bruteforce = {P_bf*100:.2f}%   CI95 [{lo*100:.1f}, {hi*100:.1f}]%   (valid n={n})")

    # ── 2. Active-boundary estimate ───────────────────────────────────────────
    print(f"[active] budget ~ {args.n_seed + args.batch*args.n_iter} simulations ...")
    res = run_active_boundary(sc, seed=args.seed + 1, n_seed=args.n_seed,
                              batch=args.batch, n_iter=args.n_iter, verbose=False)
    P_act = res.p_fail
    within = lo <= P_act <= hi
    print(f"  P_active     = {P_act*100:.2f}%   CI95 {tuple(round(x*100,1) for x in res.p_fail_ci)}"
          f"   ({res.n_evaluations} sims)")
    print(f"  |P_active - P_bruteforce| = {abs(P_act-P_bf)*100:.2f} pts   "
          f"-> {'WITHIN' if within else 'OUTSIDE'} brute-force CI")

    # ── 3. Sample efficiency (reuse brute-force sims) ─────────────────────────
    print("\n[efficiency] |P_hat - P_bruteforce| vs #sims  (MC counting vs GP surrogate)")
    odd_big = sample_odd(sc, 4000, args.seed + 2, lower, upper)   # integration pts (no sim)
    rng = np.random.default_rng(args.seed)
    print(f"  {'#sims':>6} | {'MC-count err':>12} | {'GP-surrogate err':>16}")
    print("  " + "-" * 42)
    for nsub in [20, 40, 80, 160]:
        if nsub > n:
            break
        mc_e, su_e = [], []
        for _ in range(8):
            idx = rng.choice(n, nsub, replace=False)
            mc_e.append(abs((m_bf[idx] < thr).mean() - P_bf))
            su_e.append(abs(surrogate_p(theta_bf[idx], m_bf[idx], lower, upper, odd_big, thr) - P_bf))
        print(f"  {nsub:>6} | {np.mean(mc_e)*100:>11.2f}% | {np.mean(su_e)*100:>15.2f}%")
    print("\n If the GP-surrogate column has a lower error at the same #sims, the")
    print(" surrogate estimates P(failure) more efficiently than plain counting.")


if __name__ == "__main__":
    main()
