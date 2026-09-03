#!/usr/bin/env python3
"""
Docker-free validation of the search machinery on a synthetic scenario.

The scenario used here has failure regions defined analytically, so the script
runs in seconds and every printed number can be checked against ground truth.

Part 1 -- design-level detection rate (no simulation)
    For failure regions defined by 1, 2 and 4 parameters, the script repeats
    `--reps` replications in which it draws a design of n points (LHS and
    random) and records whether any point lands inside the region. It prints the
    detection rate of each design for each region dimensionality.

Part 2 -- end-to-end comparison
    Runs the full `ModelComparison` harness (active boundary, cross-entropy and
    plain sampling, each crossed with the lhs and random designs) on a
    two-region synthetic scenario and prints the failure-region report together
    with the probability that a blind draw hits each region.

Usage:
    python scripts/validate_model_comparison.py                  # ~2 min
    python scripts/validate_model_comparison.py --skip-part2     # ~5 s
    python scripts/validate_model_comparison.py --reps 500 --budget 80 --seeds 0 1 2

Runtime is dominated by the Gaussian-process fits of the active-boundary arms
(~10 s per arm per seed); the synthetic simulations themselves are immediate.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic scenario with two known failure regions
# ─────────────────────────────────────────────────────────────────────────────
class TwoRegionScenario:
    """
    A 4-parameter scenario on the unit cube with two disjoint failure regions:

        A (broad, corner)  p0 > 0.80 and p1 > 0.80             volume 4.0e-2
        B (rare, slab)     |p2 - 0.42| < 0.02 and p3 > 0.90    volume 4.0e-3

    Failure iff margin < 0, with margin = min(margin_A, margin_B). Region B is
    the interesting one: it is ten times smaller and one of its two conditions
    is a thin slab in the middle of an axis — the kind of region a coarse
    unstratified design walks straight past.
    """

    name = "two_region_synthetic"
    d = 4

    def __init__(self):
        self._valid_mask = None

    def param_bounds(self):
        return {"names": ["p0", "p1", "p2", "p3"],
                "lower": np.zeros(self.d), "upper": np.ones(self.d)}

    def param_distributions(self, lower=None, upper=None):
        from scipy import stats
        lo = np.zeros(self.d) if lower is None else np.asarray(lower, float)
        hi = np.ones(self.d) if upper is None else np.asarray(upper, float)
        return [stats.uniform(loc=lo[j], scale=max(hi[j] - lo[j], 1e-9))
                for j in range(self.d)]

    def failure_threshold(self):
        return 0.0

    def run_simulation(self, params, verbose=False):
        return np.zeros((len(np.asarray(params)), 2, 4), dtype=float)

    def compute_qoi(self, trajectories, params):
        p = np.asarray(params, float)
        self._valid_mask = np.ones(len(p), dtype=bool)
        mA = np.maximum(0.80 - p[:, 0], 0.80 - p[:, 1])
        mB = np.maximum(np.abs(p[:, 2] - 0.42) - 0.02, 0.90 - p[:, 3])
        return np.minimum(mA, mB)

    # Ground truth, for checking the estimates.
    TRUE_P_FAIL = 0.04 + 0.004 - 0.04 * 0.004      # inclusion-exclusion, independent


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — how often does a design of n points hit a region?
# ─────────────────────────────────────────────────────────────────────────────
# Ordered from "one axis decides" to "four axes decide jointly": the LHS
# advantage is largest at the top and fades as more axes enter the conjunction.
REGIONS = {
    "1 axis  |p2-0.42|<0.01         ": lambda p: np.abs(p[:, 2] - 0.42) < 0.01,
    "2 axes  |p2-0.42|<0.02 & p3>0.9": lambda p: (np.abs(p[:, 2] - 0.42) < 0.02) & (p[:, 3] > 0.90),
    "2 axes  p0>0.80 & p1>0.80      ": lambda p: (p[:, 0] > 0.80) & (p[:, 1] > 0.80),
    "4 axes  all params in [0,0.35] ": lambda p: (p < 0.35).all(axis=1),
}
VOLUMES = {
    "1 axis  |p2-0.42|<0.01         ": 0.02,
    "2 axes  |p2-0.42|<0.02 & p3>0.9": 0.04 * 0.10,
    "2 axes  p0>0.80 & p1>0.80      ": 0.20 * 0.20,
    "4 axes  all params in [0,0.35] ": 0.35 ** 4,
}


def detection_rates(n: int, reps: int, d: int = 4) -> None:
    from pipeline.samplers import get_sampler

    print("=" * 78)
    print(f" PART 1 — detection rate of a design of n={n} points ({reps} replications)")
    print("=" * 78)
    print(f" {'failure region':<34}{'volume':>10}{'analytic':>10}{'LHS':>9}{'random':>9}{'ratio':>8}")

    for label, test in REGIONS.items():
        v = VOLUMES[label]
        analytic = 1.0 - (1.0 - v) ** n            # exact for i.i.d. sampling
        rates = {}
        for sname in ("lhs", "random"):
            s = get_sampler(sname)
            hits = sum(int(test(s.unit(n, d, seed=r)).any()) for r in range(reps))
            rates[sname] = hits / reps
        ratio = rates["lhs"] / rates["random"] if rates["random"] > 0 else float("inf")
        print(f" {label:<34}{v:>10.4f}{analytic:>10.2%}{rates['lhs']:>9.2%}"
              f"{rates['random']:>9.2%}{ratio:>8.2f}")

    print()
    print(" How to read this. 'analytic' is the exact i.i.d. result 1-(1-v)^n, so the")
    print(" 'random' column tracking it is a sanity check on the harness. The LHS")
    print(" advantage is a MARGINAL-coverage effect: with n points LHS fills all n")
    print(" strata of every axis, so a region a stratum or two wide on a single axis is")
    print(" hit far more reliably, while random search leaves ~37% of the strata empty")
    print(" and misses it outright. As the failure condition involves more axes JOINTLY,")
    print(" the guarantee weakens and the two designs converge — LHS controls the")
    print(" 1-D projections, not the joint occupancy of the cube.")
    print(" Practical consequence for lane keeping: stratification is worth most when")
    print(" the failure is driven by one dominant parameter (a sharp angle, a speed")
    print(" band), and worth little when it needs several parameters to conspire.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — the full comparison harness
# ─────────────────────────────────────────────────────────────────────────────
def end_to_end(budget: int, seeds: list, out: str | None) -> None:
    from pipeline.model_comparison import ModelComparison

    sc = TwoRegionScenario()
    print("=" * 78)
    print(f" PART 2 — end-to-end comparison, budget={budget}, seeds={seeds}")
    print("=" * 78)
    print(f" ground truth P(failure) under the uniform ODD: {sc.TRUE_P_FAIL:.4f}")
    print()

    # pool_size/odd_samples are model-only, but the GP credible interval is an
    # O(odd_samples^3) Cholesky: keep them small here, this is a smoke test.
    cmp_ = ModelComparison.default(sc, budget=budget, seeds=seeds,
                                   include_baseline=True, pool_size=1500,
                                   odd_samples=1500, verbose=False)
    res = cmp_.run()
    print(res.report())

    print()
    print(" accuracy against ground truth (P(fail) = "
          f"{sc.TRUE_P_FAIL:.4f}):")
    for lab in res.arm_labels:
        if lab.startswith("qoi_"):
            continue        # an optimiser's failure fraction is not a probability
        p = res.per_arm[lab]["mean_p_fail"]
        print(f"   {lab:<28}{p:>10.4f}   error {abs(p - sc.TRUE_P_FAIL):>8.4f}")
    print()
    print(" Expect the cross-entropy arms to UNDERESTIMATE here, and badly. This")
    print(" scenario has two disjoint failure regions, while the CE proposal is a")
    print(" product of independent unimodal truncated normals: once it commits to")
    print(" one mode it stops seeing the other. That limitation is documented in")
    print(" pipeline/rare_event.py and this synthetic case is a clean demonstration")
    print(" of it — not a bug in the comparison. On a single-mode ODD the CE arms")
    print(" recover the right probability (see tests/test_random_search_comparison.py).")

    if out:
        for f in res.save(out):
            print(f"[saved] {f}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Docker-free validation of the comparison.")
    ap.add_argument("--n", type=int, default=64, help="design size for part 1")
    ap.add_argument("--reps", type=int, default=400, help="replications for part 1")
    ap.add_argument("--budget", type=int, default=96, help="budget per arm for part 2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--skip-part1", action="store_true")
    ap.add_argument("--skip-part2", action="store_true")
    ap.add_argument("--out", default=None, help="path prefix for JSON/CSV output")
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if not args.skip_part1:
        detection_rates(args.n, args.reps)
    if not args.skip_part2:
        end_to_end(args.budget, args.seeds, args.out)


if __name__ == "__main__":
    main()
