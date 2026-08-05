#!/usr/bin/env python3
"""
Find the worst case: minimise the scenario's QoI safety margin.

Where run_rare_event.py answers "how OFTEN does it fail" and
run_active_boundary.py answers "WHERE is the fail/safe boundary", this script
answers "how BAD can it get" — the parameter combination that drives the
composite lane-keeping margin as low as it goes.

Two optimisers, both budgeted in simulations:
    --method bayes   GP surrogate + Expected Improvement. Most sample-efficient;
                     also reports which parameters drive the worst case (ARD).
    --method cmaes   CMA-ES. Evaluates a full generation per step, so it uses
                     the parallel simulator pool well and copes with rugged,
                     noisy landscapes.

Prerequisites: the opensbt-core simulator containers must be running.

Usage:
    python scripts/run_qoi_optimizer.py --budget 120
    python scripts/run_qoi_optimizer.py --method cmaes --budget 200 --workers 6
    python scripts/run_qoi_optimizer.py --preset realistic --max-angle 20
    python scripts/run_qoi_optimizer.py --compare --budget 120     # both, side by side

Caveat worth repeating: the minimiser is the WORST case, not the likeliest one.
A worst case sitting in a corner of the ODD with probability 1e-9 is a good
stress test and a bad risk estimate — read it next to the rare-event probability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimise the QoI margin (worst-case search).")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--method", choices=["bayes", "cmaes"], default="bayes")
    ap.add_argument("--compare", action="store_true",
                    help="run BOTH optimisers at the same budget and compare")
    ap.add_argument("--budget", type=int, default=120, help="total simulations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4,
                    help="bayes: evaluations per iteration (match your worker count)")
    ap.add_argument("--n-init", type=int, default=0,
                    help="bayes: initial design size (default max(2d, budget/4))")
    ap.add_argument("--popsize", type=int, default=0,
                    help="cmaes: generation size (default 4 + 3 ln d)")
    ap.add_argument("--sigma0", type=float, default=0.3,
                    help="cmaes: initial step, fraction of each parameter range")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel simulator containers (default: NUM_WORKERS or 4)")
    ap.add_argument("--out", default=None, help="write the result summary to this JSON file")
    ap.add_argument("--quiet", action="store_true")

    from pipeline.odd_presets import add_odd_args, resolve_bounds
    add_odd_args(ap)
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from pipeline.qoi_optimizer import BayesianQoIOptimizer, CMAESQoIOptimizer
    from scenarios import SCENARIOS

    scenario = SCENARIOS[args.scenario]
    lower, upper = resolve_bounds(scenario, args)
    verbose = not args.quiet

    print(f"[scenario] {args.scenario}")
    print(f"[budget]   {args.budget} simulations")
    if lower is not None:
        print(f"[ODD]      narrowed  lower={lower.tolist()}")
        print(f"[ODD]      narrowed  upper={upper.tolist()}")

    def make(method):
        if method == "bayes":
            return BayesianQoIOptimizer(
                scenario, budget=args.budget, n_init=args.n_init, batch=args.batch,
                param_lower=lower, param_upper=upper, verbose=verbose)
        return CMAESQoIOptimizer(
            scenario, budget=args.budget, popsize=args.popsize, sigma0=args.sigma0,
            param_lower=lower, param_upper=upper, verbose=verbose)

    methods = ["bayes", "cmaes"] if args.compare else [args.method]
    results = {}
    for m in methods:
        t0 = time.time()
        res = make(m).run(seed=args.seed)
        print()
        print(res.report())
        print(f"[elapsed] {(time.time() - t0) / 60:.1f} min")
        results[m] = res

    if args.compare and len(results) == 2:
        b, c = results["bayes"], results["cmaes"]
        best = "bayes" if b.best_margin <= c.best_margin else "cmaes"
        print()
        print("=" * 78)
        print(" WORST-CASE SEARCH — HEAD TO HEAD (same budget)")
        print("=" * 78)
        print(f"   {'optimiser':<16}{'evals':>8}{'best margin':>14}{'failures':>10}")
        for name, r in results.items():
            print(f"   {name:<16}{r.n_evaluations:>8}{r.best_margin:>14.4f}"
                  f"{int(r.labels.sum()):>10}")
        print(f"   -> deeper worst case found by: {best}")
        print("=" * 78)

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {m: r.summary | {"history": r.history} for m, r in results.items()}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
