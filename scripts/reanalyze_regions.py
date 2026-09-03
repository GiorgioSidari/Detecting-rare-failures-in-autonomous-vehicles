#!/usr/bin/env python3
"""
Re-run the failure-region analysis on a saved campaign, without the simulator.

The script reads the `*_raw.npz` written by `run_model_comparison.py --out` and
redoes everything downstream of the simulations: clustering of the failing
points, per-region bounds and per-axis spread, and the probability that a draw
from the operational distribution lands in each region.

Options that change the analysis: `--eps` (DBSCAN radius), `--max-span` (how
narrow a region must be on an axis before that axis counts as a constraint) and
the ODD flags, which set the distribution the region probabilities are computed
against.

Usage:
    python scripts/reanalyze_regions.py results/cmp_raw.npz
    python scripts/reanalyze_regions.py results/cmp_raw.npz --eps 0.6
    python scripts/reanalyze_regions.py results/cmp_raw.npz --max-span 0.3 \
        --out results/cmp_v2

    # campaign saved before _raw.npz existed: rebuild the failure cloud from the
    # regions CSV, whose singleton centroids are the failing points themselves
    python scripts/reanalyze_regions.py results/cmp_regions.csv --from-csv

The output starts with the per-axis spread table: an axis whose failures span
the whole ODD is not a constraint of the region, whatever the clustering
parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_npz(path: str):
    """Return ({label: (theta, margins)}, param_names) from a saved campaign."""
    z = np.load(path, allow_pickle=True)
    labels = [str(x) for x in z["labels"]]
    names = [str(x) for x in z["param_names"]] if "param_names" in z else None
    runs = {lab: (z[f"theta_{i}"], z[f"margins_{i}"]) for i, lab in enumerate(labels)}
    return runs, names


def load_regions_csv(path: str):
    """
    Rebuild the failure cloud from a ``*_regions.csv``.

    Lossy but useful for campaigns saved before the raw dump existed: every
    single-point region's centroid IS a failing point, and multi-point regions
    collapse to their centroid (so a handful of points are lost). Only failures
    are recoverable — the safe evaluations are not in that file — which is enough
    for the region analysis but not for failure rates.
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = [c for c in rows[0] if c.startswith("centroid_")]
    names = [c[len("centroid_"):] for c in cols]
    runs: dict = {}
    for r in rows:
        theta = [float(r[c]) for c in cols]
        margin = float(r["worst_margin"])
        for lab in (r.get("found_by") or "").split("|"):
            if not lab:
                continue
            runs.setdefault(lab, ([], []))
            runs[lab][0].append(theta)
            runs[lab][1].append(margin)
    return ({lab: (np.array(t, float), np.array(m, float))
             for lab, (t, m) in runs.items()}, names)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Re-analyse a saved campaign offline.")
    ap.add_argument("path", help="results/<prefix>_raw.npz (or _regions.csv with --from-csv)")
    ap.add_argument("--from-csv", action="store_true",
                    help="rebuild the failure cloud from a *_regions.csv instead")
    ap.add_argument("--scenario", default="lane_keeping",
                    help="scenario whose bounds/ODD to analyse against")
    ap.add_argument("--eps", type=float, default=None,
                    help="DBSCAN radius in the normalised cube (default: estimated)")
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--max-span", type=float, default=0.5,
                    help="an axis is reported as constraining when the group "
                         "covers less than this fraction of its range (0.5)")
    ap.add_argument("--min-points", type=int, default=4,
                    help="groups smaller than this get no reported bounds (4)")
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--weighting", choices=["odd", "uniform"], default="odd")
    ap.add_argument("--uniform-odd", action="store_true",
                    help="ignore param_distributions and treat the ODD as uniform")
    ap.add_argument("--verbose", action="store_true",
                    help="add the pairwise matrix and the non-localised groups")
    ap.add_argument("--out", default=None, help="write the report and JSON here")
    return ap


def main() -> None:
    ap = _build_parser()

    from pipeline.odd_presets import add_odd_args, resolve_bounds
    add_odd_args(ap)
    args = ap.parse_args()

    from pipeline.region_comparison import compare_failure_regions
    from scenarios import SCENARIOS

    runs, names = (load_regions_csv(args.path) if args.from_csv
                   else load_npz(args.path))
    scenario = SCENARIOS[args.scenario]
    lower, upper = resolve_bounds(scenario, args)
    if lower is None:
        b = scenario.param_bounds()
        lower, upper = np.array(b["lower"], float), np.array(b["upper"], float)
    names = names or list(scenario.param_bounds()["names"])
    dists = (None if args.uniform_odd or not hasattr(scenario, "param_distributions")
             else scenario.param_distributions(lower, upper))

    n_pts = sum(len(m) for _, m in runs.values())
    print(f"[loaded] {len(runs)} arms, {n_pts} evaluations from {args.path}")
    print(f"[ODD]    lower={np.asarray(lower).tolist()}")
    print(f"[ODD]    upper={np.asarray(upper).tolist()}")
    print()

    cmp_ = compare_failure_regions(
        runs, lower, upper, threshold=float(scenario.failure_threshold()),
        param_names=names, dists=dists, eps=args.eps,
        min_samples=args.min_samples, pad=args.pad,
        max_span=args.max_span, min_points=args.min_points,
        weighting=args.weighting,
    )
    print(cmp_.report(verbose=args.verbose))

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
            json.dump(cmp_.to_dict(), fh, indent=2, default=float)
        with open(f"{args.out}.txt", "w", encoding="utf-8") as fh:
            fh.write(cmp_.report(verbose=True))
        print(f"\n[saved] {args.out}.json\n[saved] {args.out}.txt")


if __name__ == "__main__":
    main()
