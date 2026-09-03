#!/usr/bin/env python3
"""
Search-method comparison on the MetaDrive backend.

The script reuses `build_arms` from `scripts/run_model_comparison.py` and the
same `ModelComparison` harness; what differs is the backend
(`LaneKeepingMetaDriveScenario`) and the operating point.

The scenario is instantiated here rather than taken from the registry, because
the registry builds it with its default `speed_scale = 1.0`, while this campaign
runs at the calibrated value produced by
`scripts/calibrate_operating_point.py`.

The controller driving here is the state-based lateral controller of
`scenarios/common/driver.py`, not the `mixed-chauffeur.h5` network used by the
Udacity campaign.

Usage:
    python scripts/run_model_comparison_md.py --budget 120 --seeds 0 1 2
    python scripts/run_model_comparison_md.py --budget 60 --seeds 0   # quick check
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
import warnings

import numpy as np

from run_model_comparison import ARM_CHOICES, build_arms
from pipeline.odd_presets import add_odd_args, resolve_bounds
from pipeline.model_comparison import ModelComparison

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Operating point measured with scripts/calibrate_operating_point.py on 60
# samples, seed 42: 17.2% failures. Declare it in the report.
SPEED_SCALE = 0.3625


def _build_parser() -> argparse.ArgumentParser:
    """The command line of this script."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=120,
                    help="simulations per arm per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--speed-scale", type=float, default=SPEED_SCALE,
                    dest="speed_scale",
                    help=f"operating point (default {SPEED_SCALE}, calibrated)")
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--ce-max-iter", type=int, default=5)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--out", default="results/cmp_md")
    ap.add_argument("--order", choices=["shuffled", "interleaved", "sequential"],
                    default="shuffled")
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--arms", nargs="+", choices=ARM_CHOICES,
                    default=["ab_lhs", "ab_random", "ce_lhs", "ce_random",
                             "plain_lhs", "plain_random"])

    # The operating point in full, read from the calibration file. It exists
    # because passing its pieces by hand already cost a campaign: a calibration
    # done at speed_scale=1.0 applied to a campaign started with the 0.3625
    # default cuts the staleness in metres to a third, and 8198 simulations
    # produce not a single failure. obs_lag and speed_scale are ONE operating
    # point, not two flags.
    ap.add_argument("--from-calibration", default=None, dest="from_calibration",
                    help="calibration JSON: reads lever, value, speed_scale and "
                         "ODD bounds from it, and applies them all together. "
                         "The recommended way: it removes the possibility of "
                         "applying only part of it.")

    # Controller degradation: the second lever of the operating point. Needed
    # when speed_scale does not bring the failure rate into the useful band even
    # at its highest -- on a narrow ODD a lateral controller on exact state
    # does not fail at any admissible speed. Calibrate it with
    #   scripts/calibrate_operating_point.py ... --lever obs_lag
    ap.add_argument("--obs-lag", type=float, default=0.0, dest="obs_lag",
                    help="time constant (s) of the observation sluggishness. "
                         "0 = ideal controller (default)")
    ap.add_argument("--obs-latency", type=int, default=0, dest="obs_latency",
                    help="steps of observation delay (0 = off)")
    ap.add_argument("--steer-noise", type=float, default=0.0, dest="steer_noise",
                    help="deviazione standard del rumore sullo sterzo (0 = off)")

    add_odd_args(ap)
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from scenarios.lane_keeping_md import LaneKeepingMetaDriveScenario

    # A dedicated instance, NOT the registry's: the operating point is part of
    # the experiment and must be explicit in the command that launches it.
    if args.from_calibration:
        with open(args.from_calibration, encoding="utf-8") as fh:
            cal = json.load(fh)
        lever = cal.get("lever", "speed_scale")
        value = float(cal.get("lever_value", cal["speed_scale"]))
        if lever == "obs_lag":
            args.obs_lag = value
            args.speed_scale = float(cal.get("fixed_speed_scale") or 1.0)
        else:
            args.speed_scale = value
        if cal.get("odd_lower"):
            cal_lo = np.asarray(cal["odd_lower"], float)
            cal_hi = np.asarray(cal["odd_upper"], float)
        else:
            cal_lo = cal_hi = None
        print(f"[calibration] from {args.from_calibration}")
        print(f"              lever={lever}  value={value}  "
              f"speed_scale={args.speed_scale}")
        print(f"              rate measured during calibration: "
              f"{cal.get('failure_rate', float('nan')):.1%} "
              f"(design {cal.get('sampling', 'uniform')})")
        if not cal.get("centered", cal.get("centrato", True)):
            raise SystemExit(
                "this calibration did NOT centre the band: launching the campaign "
                "on it will produce an unusable regime. Recalibrate first.")
    else:
        cal_lo = cal_hi = None

    degraded = bool(args.obs_lag or args.obs_latency or args.steer_noise)
    if degraded:
        from scenarios.lane_keeping_md.driver import LateralFeedbackDriver
        scenario = LaneKeepingMetaDriveScenario(
            speed_scale=args.speed_scale,
            driver_factory=lambda: LateralFeedbackDriver(
                obs_lag_tau=args.obs_lag, obs_latency=args.obs_latency,
                steer_noise=args.steer_noise, seed=args.order_seed))
    else:
        scenario = LaneKeepingMetaDriveScenario(speed_scale=args.speed_scale)
    lower, upper = resolve_bounds(scenario, args)
    if cal_lo is not None:
        # The calibration bounds are as much part of the operating point as the
        # lever: a value tuned on one box means nothing on another.
        if lower is None:
            lower, upper = cal_lo, cal_hi
        elif not (np.allclose(lower, cal_lo) and np.allclose(upper, cal_hi)):
            raise SystemExit(
                "the ODD flags passed do not match the ODD the calibration was "
                "done on.\n"
                f"  calibration: lower={cal_lo.tolist()}\n"
                f"               upper={cal_hi.tolist()}\n"
                f"  campaign   : lower={np.asarray(lower).tolist()}\n"
                f"               upper={np.asarray(upper).tolist()}\n"
                "Drop the ODD flags (they come from the calibration) or recalibrate.")

    print(f"[scenario]    {scenario.name}  (geometry={scenario.geometry})")
    print(f"[operating]   speed_scale={args.speed_scale}  "
          f"-> ~17.2% failures at n=60 when 0.3625 on the default ODD")
    if degraded:
        print(f"[driver]      DEGRADED: obs_lag={args.obs_lag}s  "
              f"obs_latency={args.obs_latency} steps  "
              f"steer_noise={args.steer_noise}")
        print("              The system under test is NOT the same as an")
        print("              ideal-controller campaign: declare it next to the")
        print("              results, like speed_scale.")
    else:
        print("[driver]      ideal (no degradation)")
    print(f"[control_hz]  {scenario.control_hz_nominal:.1f} Hz exactly")
    print(f"[arms]        {', '.join(args.arms)}")
    print(f"[budget]      {args.budget} simulations/arm/seed, "
          f"seeds {args.seeds}")
    tot = args.budget * len(args.arms) * len(args.seeds)
    print(f"[total]       ~{tot} simulations")
    print()

    # No preflight and no warm-up: there is no container pool to warm here. On
    # Udacity they were needed because the first pass over a freshly started
    # pool is measurably harder (20-38% failures against 8-11% at steady state);
    # MetaDrive runs in-process and has no such transient.
    arms = build_arms(args.arms, scenario, args.budget, args.n_iter,
                      args.ce_max_iter, lower, upper, not args.quiet)

    t0 = time.time()
    cmp_ = ModelComparison(scenario, arms, seeds=args.seeds, budget=args.budget,
                           param_lower=lower, param_upper=upper, eps=args.eps,
                           min_samples=args.min_samples, order=args.order,
                           order_seed=args.order_seed, verbose=not args.quiet)
    result = cmp_.run()
    result.metadata = {
        "backend": "lane_keeping_md",
        "speed_scale": float(args.speed_scale),
        "obs_lag": float(args.obs_lag),
        "obs_latency": int(args.obs_latency),
        "steer_noise": float(args.steer_noise),
        "driver": "degraded" if degraded else "ideal",
        "calibration_file": args.from_calibration,
        "odd_lower": np.asarray(lower, float).tolist() if lower is not None else None,
        "odd_upper": np.asarray(upper, float).tolist() if upper is not None else None,
        "budget": int(args.budget),
        "seeds": list(args.seeds),
    }
    dt = time.time() - t0

    print()
    print(result.report())
    print(f"\n[elapsed] {dt / 60:.1f} min")

    if args.out:
        for f in result.save(args.out):
            print(f"[saved] {f}")


if __name__ == "__main__":
    main()
