#!/usr/bin/env python3
"""
Compare every search method at a matched budget on the Udacity backend.

Per seed the script runs the six arms

    active_boundary[lhs]   active_boundary[random]
    cross_entropy[lhs]     cross_entropy[random]
    plain_sampling[lhs]    plain_sampling[random]

through `pipeline.model_comparison.ModelComparison`, then extracts one shared
map of failure regions from the union of all the failures found and reports,
per region, the probability that a draw from the operational distribution lands
in it. `build_arms` constructs the arm list and is reused by
`scripts/run_model_comparison_md.py`.

With `--out` the campaign is saved as `<out>_raw.npz` (the raw clouds) and
`<out>.json` (metadata and per-seed summaries), which `scripts/rank_arms.py`
and `scripts/reanalyze_regions.py` read back.

Prerequisites: the opensbt-core simulator containers must be running (see the
README). Cost scales as budget x seeds x arms simulations, at roughly 10 s per
simulation.

Usage:
    python scripts/run_model_comparison.py --budget 120 --seeds 0 1 2
    python scripts/run_model_comparison.py --preset realistic --max-angle 20
    python scripts/run_model_comparison.py --arms ab_lhs ab_random --budget 80
    python scripts/run_model_comparison.py --out results/cmp_run1

`scripts/validate_model_comparison.py` runs the same machinery on a synthetic
scenario, without Docker.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

from pipeline.odd_presets import add_odd_args, resolve_bounds
from pipeline.model_comparison import ModelComparison

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

ARM_CHOICES = ["ab_lhs", "ab_random", "ce_lhs", "ce_random",
               "plain_lhs", "plain_random", "qoi_bayes", "qoi_cmaes",
               # Control arms: the active-boundary machinery with the active step
               # DISABLED. Same seed design, same candidate pool, same budget --
               # each batch is drawn from the pool at random instead of by
               # entropy. They separate two effects the main campaign confounds:
               # active_boundary searches uniformly over the box while
               # plain_sampling draws from the ODD, so part of its rare-failure
               # advantage may come from WHERE it samples rather than from what
               # it learns. See pipeline/active_boundary_random.py, acquisition=.
               "ab_lhs_randacq", "ab_random_randacq"]


def build_arms(names, scenario, budget, n_iter, ce_max_iter, lower, upper, verbose):
    """Instantiate the requested arms at a matched simulation budget."""
    from pipeline.active_boundary_random import (
        LHSActiveBoundary, RandomSearchActiveBoundary)
    from pipeline.model_comparison import PlainSamplingBaseline
    from pipeline.qoi_optimizer import BayesianQoIOptimizer, CMAESQoIOptimizer
    from pipeline.rare_event_random import LHSCrossEntropy, RandomSearchCrossEntropy

    n_seed = max(8, budget // 3)
    batch = max(1, (budget - n_seed) // max(1, n_iter))
    ab_kw = dict(n_seed=n_seed, batch=batch, n_iter=n_iter,
                 param_lower=lower, param_upper=upper, verbose=verbose)

    spi = max(8, budget // (ce_max_iter + 3))
    final = max(8, budget - spi * ce_max_iter)
    ce_kw = dict(samples_per_iter=spi, max_iter=ce_max_iter, final_samples=final,
                 lower=lower, upper=upper, verbose=verbose)

    opt_kw = dict(budget=budget, param_lower=lower, param_upper=upper, verbose=verbose)

    factory = {
        "ab_lhs":       lambda: LHSActiveBoundary(scenario, **ab_kw),
        "ab_random":    lambda: RandomSearchActiveBoundary(scenario, **ab_kw),
        # Control arms -- identical to the two above except acquisition="random".
        "ab_lhs_randacq":    lambda: LHSActiveBoundary(
            scenario, acquisition="random", **ab_kw),
        "ab_random_randacq": lambda: RandomSearchActiveBoundary(
            scenario, acquisition="random", **ab_kw),
        "ce_lhs":       lambda: LHSCrossEntropy(scenario, **ce_kw),
        "ce_random":    lambda: RandomSearchCrossEntropy(scenario, **ce_kw),
        "plain_lhs":    lambda: PlainSamplingBaseline(scenario, "lhs", n_samples=budget,
                                                      param_lower=lower, param_upper=upper),
        "plain_random": lambda: PlainSamplingBaseline(scenario, "random", n_samples=budget,
                                                      param_lower=lower, param_upper=upper),
        "qoi_bayes":    lambda: BayesianQoIOptimizer(scenario, **opt_kw),
        "qoi_cmaes":    lambda: CMAESQoIOptimizer(scenario, **opt_kw),
    }
    return [factory[n]() for n in names]


def _build_parser() -> argparse.ArgumentParser:
    """The command line of this script."""
    ap = argparse.ArgumentParser(
        description="Matched-budget comparison of the failure-search methods.")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--budget", type=int, default=120,
                    help="simulations per arm per seed (default 120)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="seeds to repeat each arm on (default: 0)")
    ap.add_argument("--arms", nargs="+", choices=ARM_CHOICES,
                    default=["ab_lhs", "ab_random", "ce_lhs", "ce_random",
                             "plain_lhs", "plain_random"],
                    help="which arms to run")
    ap.add_argument("--n-iter", type=int, default=5,
                    help="active-boundary iterations (default 5)")
    ap.add_argument("--ce-max-iter", type=int, default=5,
                    help="max cross-entropy iterations (default 5)")
    ap.add_argument("--eps", type=float, default=None,
                    help="DBSCAN radius in the normalised cube "
                         "(default: estimated from the data — leave it alone)")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="DBSCAN min_samples (default 2)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel simulator containers (default: NUM_WORKERS or 4)")
    ap.add_argument("--out", default=None,
                    help="path prefix for the JSON/CSV output (e.g. results/cmp)")
    ap.add_argument("--warmup", type=int, default=40,
                    help="throwaway simulations before the campaign starts. The "
                         "first pass over a freshly started pool is measurably "
                         "harsher than the steady state (20-38%% failures against "
                         "8-11%% later), so those runs are burned rather than "
                         "counted. 0 disables it")
    ap.add_argument("--order", choices=["shuffled", "interleaved", "sequential"],
                    default="shuffled",
                    help="execution order of the (arm, seed) runs. 'shuffled' "
                         "(default) protects the comparison from any drift over "
                         "the session; 'sequential' is arm-major and only there "
                         "to reproduce an older campaign")
    ap.add_argument("--order-seed", type=int, default=0,
                    help="seed of the execution-order shuffle")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the one-simulation-per-container check before starting")
    ap.add_argument("--quiet", action="store_true")

    add_odd_args(ap)
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from scenarios import SCENARIOS

    scenario = SCENARIOS[args.scenario]
    lower, upper = resolve_bounds(scenario, args)

    print(f"[scenario] {args.scenario}")
    print(f"[budget]   {args.budget} sims x {len(args.seeds)} seeds x "
          f"{len(args.arms)} arms = ~{args.budget * len(args.seeds) * len(args.arms)} runs")
    if lower is not None:
        print(f"[ODD]      narrowed  lower={lower.tolist()}")
        print(f"[ODD]      narrowed  upper={upper.tolist()}")

    # One real simulation per container before committing to hours of work: /health
    # is answered by a container whose Unity instance is hung, a job is not.
    if not args.no_preflight and hasattr(scenario, "probe_workers"):
        status = scenario.probe_workers(verbose=True)
        good = [u for u, m in status.items() if m == "ok"]
        if not good:
            print("[preflight] no container completes a simulation -- stopping.")
            sys.exit(1)
        if len(good) < len(status):
            print(f"[preflight] continuing with {len(good)}/{len(status)} workers.")

    # Burn the warm-up transient before any arm is measured. Two independent
    # repeatability sessions both showed the first pass over the points failing
    # far more often than every later pass (37.9% then 13.8% average; 20.7% then
    # 11.0%), settling afterwards. Whichever arm went first inherited that.
    if args.warmup > 0:
        import numpy as _np
        from pipeline.samplers import get_sampler as _gs
        _b = scenario.param_bounds()
        _lo = lower if lower is not None else _np.asarray(_b["lower"], float)
        _hi = upper if upper is not None else _np.asarray(_b["upper"], float)
        print(f"[warmup]   {args.warmup} simulazioni di riscaldamento (scartate)...")
        _t = time.time()
        try:
            if hasattr(scenario, "param_distributions"):
                _th = _gs("lhs").from_dists(args.warmup,
                                            scenario.param_distributions(_lo, _hi),
                                            seed=12345)
            else:
                _th = _gs("lhs").bounded(args.warmup, _lo, _hi, seed=12345)
            scenario.run_simulation(_th)
            print(f"[warmup]   done in {(time.time() - _t) / 60:.1f} min")
        except Exception as exc:
            print(f"[warmup]   saltato ({type(exc).__name__}: {exc})")

    arms = build_arms(args.arms, scenario, args.budget, args.n_iter,
                      args.ce_max_iter, lower, upper, not args.quiet)

    t0 = time.time()
    cmp_ = ModelComparison(scenario, arms, seeds=args.seeds, budget=args.budget,
                           param_lower=lower, param_upper=upper, eps=args.eps,
                           min_samples=args.min_samples, order=args.order,
                           order_seed=args.order_seed, verbose=not args.quiet)
    result = cmp_.run()
    dt = time.time() - t0

    print()
    print(result.report())
    print(f"\n[elapsed] {dt / 60:.1f} min")

    if args.out:
        for f in result.save(args.out):
            print(f"[saved] {f}")


if __name__ == "__main__":
    main()
