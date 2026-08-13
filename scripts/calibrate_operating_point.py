#!/usr/bin/env python3
"""
Calibrate a backend's operating point.

Searches for the `speed_scale` that brings the failure rate into the 10-20%
band under uniform sampling. A backend at 0% or at 100% carries no information:
the boundary is not learnable and the comparison between search methods becomes
empty.

Osservato a `speed_scale = 1.0`:
    Udacity  (C2, 1 worker)   5/5 falliti
    MetaDrive (C2, smoke)     2/2 falliti

Uso
---
    # MetaDrive: runs locally, no Docker
    python scripts/calibrate_operating_point.py lane_keeping_md

    # Udacity: richiede i container C2 attivi
    python scripts/calibrate_operating_point.py lane_keeping --n 40

    # different band, fewer samples for a quick check
    python scripts/calibrate_operating_point.py lane_keeping_md --n 12 --low 0.2 --high 0.4

The value found must then be set on the backend:
    MetaDrive -> LaneKeepingMetaDriveScenario(speed_scale=...)
    Udacity   -> LK_SPEED_SCALE in the C2 compose file

And it must be DECLARED in the report: it is a parameter of the experiment. Two
backends tuned to different values remain comparable on the SHAPE of the failure
set, not on absolute failure rates.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import sys

# Project root on sys.path (runnable from any directory).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.operating_point import calibrate, scenario_evaluator   # noqa: E402


def _scenario_builder(name: str):
    """
    Returns `speed_scale -> BaseScenario` for the requested backend.

    Each backend exposes the lever its own way: MetaDrive takes it in the
    constructor, Udacity reads it from the environment because the controller
    lives inside the container. This is the only point where the two differ.
    """
    if name in ("lane_keeping_md", "metadrive", "md"):
        from scenarios.lane_keeping_md import LaneKeepingMetaDriveScenario
        return lambda s: LaneKeepingMetaDriveScenario(speed_scale=s), "lane_keeping_md"

    if name in ("lane_keeping", "udacity"):
        # The controller runs inside the container: the lever comes from the
        # environment and the containers must be RESTARTED at every evaluation,
        # which this script cannot do. It fails explicitly rather than
        # calibrating something meaningless.
        raise SystemExit(
            "The Udacity backend reads speed_scale from LK_SPEED_SCALE inside\n"
            "the container, so automatic calibration would require restarting\n"
            "the containers at every evaluation.\n"
            "\n"
            "Manual procedure (few iterations, the band is wide):\n"
            "  1. cd opensbt-core\n"
            "  2. python gen_c2_compose.py 1 --speed-scale 0.6\n"
            "  3. docker compose -f docker-compose.c2.parallel.yml up --build\n"
            "  4. python scripts/run_lanekeeping.py --n 40\n"
            "  5. read P(failure); if >20% lower it, if <10% raise it\n"
            "\n"
            "Tip: start from the value found on MetaDrive, which calibrates\n"
            "automatically and is much faster.")

    raise SystemExit(f"backend sconosciuto: {name!r} "
                     f"(usa lane_keeping_md | lane_keeping)")


def _obs_lag_builder(name: str, speed_scale: float):
    """
    Returns `obs_lag -> BaseScenario`: the second lever of the operating point.

    Why it exists
    -------------
    `speed_scale` is not always enough. On a narrow ODD (gentle curves, tight
    speed band) a lateral controller reading the EXACT state does not fail even at
    `speed_scale = 1.0`: there is no admissible speed that takes it out of the
    lane, so there is no boundary to learn and the comparison between search
    methods stays empty.

    `obs_lag_tau` acts on the right mechanism instead of on speed: the
    controller steers on stale observations, which is the failure mode of the
    image-based network -- not "it drives too fast". And because a latency
    fixed in time becomes a staleness growing in metres as speed rises, the
    failures stay *scenario-dependent*, which is the condition for a search to
    have anything to find.

    NOTE: enabling it changes the system under test. A campaign with
    `obs_lag > 0` is not comparable with one on an ideal controller, and the
    value must be declared next to the results exactly like `speed_scale`.
    """
    if name not in ("lane_keeping_md", "metadrive", "md"):
        raise SystemExit(
            f"the obs_lag lever is implemented for MetaDrive only, not for {name!r}: "
            "on the other backends the driver cannot be injected from here.")
    from scenarios.lane_keeping_md import LaneKeepingMetaDriveScenario
    from scenarios.lane_keeping_md.driver import LateralFeedbackDriver

    def _build(lag: float):
        return LaneKeepingMetaDriveScenario(
            speed_scale=speed_scale,
            driver_factory=lambda: LateralFeedbackDriver(obs_lag_tau=float(lag)))

    return _build, "lane_keeping_md"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", help="lane_keeping_md | lane_keeping")
    ap.add_argument("--n", type=int, default=40, dest="n_samples",
                    help="samples per evaluation (default 40). "
                         "With 40 the uncertainty on the rate is ~+/-13%%")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed of the LHS design, FIXED across evaluations")
    ap.add_argument("--low", type=float, default=0.10, help="lower end of the band")
    ap.add_argument("--high", type=float, default=0.20, help="upper end of the band")
    ap.add_argument("--scale-min", type=float, default=None, dest="scale_min")
    ap.add_argument("--scale-max", type=float, default=None, dest="scale_max")
    ap.add_argument("--lever", choices=["speed_scale", "obs_lag"],
                    default="speed_scale",
                    help="which parameter to calibrate. 'speed_scale' raises the "
                         "speed; 'obs_lag' makes the controller's observations "
                         "stale (seconds) -- use it when speed_scale does not "
                         "reach the band even at its highest, as on narrow ODDs. "
                         "It changes the system under test: declare it.")
    ap.add_argument("--sampling", choices=["uniform", "odd"], default="uniform",
                    help="which distribution the design is drawn from. 'uniform' "
                         "(historical default) is uniform over the box; 'odd' uses "
                         "the operational marginals, i.e. what the campaign's "
                         "plain_sampling arms measure. The two rates differ by a "
                         "factor of ~5: use 'odd' when the target band is the one "
                         "of blind sampling.")
    ap.add_argument("--fixed-speed-scale", type=float, default=1.0,
                    dest="fixed_speed_scale",
                    help="with --lever obs_lag, the speed_scale held fixed (default 1.0)")
    ap.add_argument("--max-iter", type=int, default=8, dest="max_iter")
    ap.add_argument("--out", default=None,
                    help="save the result as JSON (default: results/taratura_<backend>.json)")

    # The operating point is NOT a property of the backend: it depends on the
    # ODD. Without these flags the calibration always ran on the default
    # bounds, and the value found was then reused on narrow ODDs where it means
    # something entirely different -- which is how a whole campaign can return
    # zero failures out of 8198 simulations.
    from pipeline.odd_presets import add_odd_args, resolve_bounds
    add_odd_args(ap)
    args = ap.parse_args()

    if args.lever == "obs_lag":
        build, name = _obs_lag_builder(args.backend, args.fixed_speed_scale)
        lo_default, hi_default = 0.0, 3.0
        print(f"Leva: obs_lag (secondi), speed_scale fisso a "
              f"{args.fixed_speed_scale}.\n"
              "NOTE: degrading the observation CHANGES the system under test.\n"
              "The results are not comparable with a campaign on an ideal "
              "controller.\n")
    else:
        build, name = _scenario_builder(args.backend)
        lo_default, hi_default = 0.15, 1.0
    if args.scale_min is None:
        args.scale_min = lo_default
    if args.scale_max is None:
        args.scale_max = hi_default

    # Cost warning: every evaluation is n_samples simulations.
    estimate = args.n_samples * (args.max_iter + 2)
    print(f"Up to ~{estimate} simulations in total "
          f"({args.n_samples} per evaluation, at most {args.max_iter + 2} evaluations).\n")

    from scenarios import SCENARIOS
    lower, upper = resolve_bounds(SCENARIOS[args.backend], args)
    if lower is None:
        b = SCENARIOS[args.backend].param_bounds()
        lower, upper = np.asarray(b["lower"], float), np.asarray(b["upper"], float)
    print(f"Design: {args.sampling}"
          + ("  (uniform over the box: NOT the rate the plain_sampling arms "
             "will measure)" if args.sampling == "uniform" else
             "  (operational marginals: same rate as the plain_sampling arms)"))
    print(f"ODD: lower={np.asarray(lower).tolist()}")
    print(f"     upper={np.asarray(upper).tolist()}\n")

    evaluate = scenario_evaluator(build, n_samples=args.n_samples,
                                 seed=args.seed, lower=lower, upper=upper,
                                 sampling=args.sampling)
    res = calibrate(evaluate, backend=name,
                  target_low=args.low, target_high=args.high,
                  scale_min=args.scale_min, scale_max=args.scale_max,
                  max_iter=args.max_iter, n_samples=args.n_samples,
                  seed=args.seed)
    res.odd_lower = np.asarray(lower, float).tolist()
    res.odd_upper = np.asarray(upper, float).tolist()
    res.lever = args.lever
    res.fixed_speed_scale = (float(args.fixed_speed_scale)
                             if args.lever == "obs_lag" else None)

    print()
    print(res.report())

    out = args.out or os.path.join(_PROJECT_ROOT, "results", f"taratura_{name}.json")
    res.save(out)
    print(f"\nSalvato in {out}")

    if not res.centered:
        print("\nThe band was not centred: read the note above before launching\n"
              "the campaign. Calibrating further will not help -- the problem is\n"
              "not the speed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
