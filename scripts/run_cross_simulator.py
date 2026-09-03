#!/usr/bin/env python3
"""
Cross-simulator comparison -- Udacity vs MetaDrive.

The two backends do not share a Python environment, so the script has two
subcommands and each campaign is collected separately.

1) COLLECT -- run a campaign on ONE backend and save the result
---------------------------------------------------------------
   Launched in the environment of that backend:

       # MetaDrive (venv with Python <3.12)
       python scripts/run_cross_simulator.py collect lane_keeping_md \
              --n 60 --speed-scale 0.1766

       # Udacity (containers up, NUM_WORKERS=1)
       python scripts/run_cross_simulator.py collect lane_keeping \
              --n 60 --speed-scale 0.42

   `--seed` must be the same on every backend: it selects the LHS design, and
   the comparison pairs scenario i of one backend with scenario i of the other.
   The compare step verifies that the designs match and aborts if they do not.

2) COMPARE -- read the files back and produce the comparison
------------------------------------------------------------
       python scripts/run_cross_simulator.py compare \
              results/cross_lane_keeping_md.json \
              results/cross_lane_keeping.json

   This step reads only the saved JSON files and runs no simulation.

The comparison reports:

  * the failure rate of each backend, as a check that neither campaign is
    degenerate (all or nothing);
  * the Spearman rank correlation between the two QoI vectors, i.e. whether the
    backends order the scenarios by difficulty in the same way;
  * the Jaccard overlap of the failure regions found on each side.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.cross_simulator import (          # noqa: E402
    CrossSimulatorComparison,
    BackendRun,
)


def _build_backend(name: str, speed_scale: float):
    """Instantiates the backend at the requested operating point."""
    if name in ("lane_keeping_md", "metadrive", "md"):
        from scenarios.lane_keeping_md import LaneKeepingMetaDriveScenario
        return LaneKeepingMetaDriveScenario(speed_scale=speed_scale)

    if name in ("lane_keeping", "udacity"):
        # The controller lives in the container: `speed_scale` comes from
        # LK_SPEED_SCALE, not from the constructor. Here it only serves to record
        # it in the result -- if the two values diverged, the report would
        # declare an operating point different from the actual one.
        from scenarios.lane_keeping.config import LaneKeepingScenario
        env = os.environ.get("LK_SPEED_SCALE")
        if env is not None and abs(float(env) - speed_scale) > 1e-9:
            raise SystemExit(
                f"--speed-scale={speed_scale} but the containers run with "
                f"LK_SPEED_SCALE={env}.\n"
                f"The value declared in the result must be the real one, "
                f"otherwise the report lies about the operating point.\n"
                f"Regenerate the compose file:\n"
                f"  cd opensbt-core\n"
                f"  python gen_c2_compose.py 1 --speed-scale {speed_scale}")
        return LaneKeepingScenario()

    raise SystemExit(f"unknown backend: {name!r}")


def _collect(args) -> int:
    from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

    scenario = _build_backend(args.backend, args.speed_scale)
    bounds = scenario.param_bounds()

    # Shared design: same seed = same thetas on every backend.
    sampler = LatinHypercube(d=len(bounds["lower"]), seed=args.seed)
    params = qmc_scale(sampler.random(n=args.n), bounds["lower"], bounds["upper"])

    print(f"[{scenario.name}] {args.n} samples, seed {args.seed}, "
          f"speed_scale={args.speed_scale}", flush=True)

    traj = scenario.run_simulation(params, verbose=args.verbose)
    qoi = np.asarray(scenario.compute_qoi(traj, params), dtype=float)

    res = BackendRun.from_scenario(
        scenario, theta=params, qoi=qoi, speed_scale=args.speed_scale,
        note=args.note or f"LHS design seed={args.seed}, n={args.n}")

    out = args.out or os.path.join(
        _PROJECT_ROOT, "results", f"cross_{scenario.name}.json")
    res.save(out)

    print(f"\n  valid runs   : {res.n_valid}/{args.n}")
    print(f"  failure rate : {res.failure_rate:.1%}")
    print(f"  saved to     : {out}")

    if res.failure_rate in (0.0, 1.0):
        print("\n  !  Degenerate rate: the comparison will be unable to say\n"
              "     anything from this backend. Check the calibration, and if the\n"
              "     rate is 100% check that the ODD is physically drivable:\n"
              "        python scripts/diag_odd_feasibility.py")
        return 1
    return 0


def _expand_globs(pattern: Sequence[str]) -> List[str]:
    """
    Espande i glob a mano.

    On Unix the shell does it before the script starts; **PowerShell does
    not**, it passes `results/cross_*.json` as a literal string and `open()`
    fails with an unhelpful "Invalid argument". Expanding here makes the command
    identical
    sui due sistemi.
    """
    import glob as _glob

    out: List[str] = []
    for p in pattern:
        if any(ch in p for ch in "*?["):
            trovati = sorted(_glob.glob(p))
            if not trovati:
                raise SystemExit(f"no file matches {p!r}")
            out.extend(trovati)
        else:
            if not os.path.isfile(p):
                raise SystemExit(f"file not found: {p}")
            out.append(p)

    # A glob can also pick up the previous comparison's output, which is not a
    # backend result: better to say so now than to fail during parsing.
    out = [p for p in out if not os.path.basename(p).startswith("confronto")]
    if len(out) < 2:
        raise SystemExit(
            f"at least two results are needed, found {len(out)}: {out}\n"
            f"Missing one? Each backend must be collected in its own environment:\n"
            f"  python scripts/run_cross_simulator.py collect <backend> "
            f"--n 60 --speed-scale <value>")
    return out


def _compare(args) -> int:
    cmp = CrossSimulatorComparison()
    for path in _expand_globs(args.file):
        r = BackendRun.load(path)
        cmp.add(r)
        print(f"loaded {r.backend}: {r.n_valid} valid runs, "
              f"speed_scale={r.speed_scale}, failure {r.failure_rate:.1%}")

    # The bounds come from the registry, not from the files: they must be the
    # canonical ones of the parameter space, the same for every backend.
    from scenarios.lane_keeping.config import LaneKeepingScenario
    b = LaneKeepingScenario().param_bounds()

    outcome = cmp.compare(
        lower=b["lower"], upper=b["upper"], param_names=b["names"],
        dists=(LaneKeepingScenario().param_distributions() if args.odd else None),
    )

    print()
    print(cmp.report())

    if args.regions:
        print()
        print(cmp.report_regions())

    out = args.out or os.path.join(_PROJECT_ROOT, "results", "confronto_cross.json")
    outcome.save(out)
    print(f"\nSalvato in {out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("collect", help="run a campaign on one backend and save it")
    r.add_argument("backend", help="lane_keeping_md | lane_keeping")
    r.add_argument("--n", type=int, default=60, help="samples of the shared design")
    r.add_argument("--seed", type=int, default=42,
                   help="MUST be the same on every backend")
    r.add_argument("--speed-scale", type=float, default=1.0, dest="speed_scale",
                   help="calibrated operating point (see calibrate_operating_point.py)")
    r.add_argument("--out", default=None)
    r.add_argument("--note", default=None)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=_collect)

    c = sub.add_parser("compare", help="read the files back and compare them")
    c.add_argument("file", nargs="+", help="two or more result JSON files")
    c.add_argument("--regions", action="store_true",
                   help="also print the detailed region report")
    c.add_argument("--odd", action="store_true",
                   help="weight region probabilities with the operational distribution")
    c.add_argument("--out", default=None)
    c.set_defaults(func=_compare)

    return ap


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
