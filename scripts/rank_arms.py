#!/usr/bin/env python3
"""
Rank the arms of a saved campaign by rare-failure yield, without the simulator.

    python scripts/rank_arms.py results/cmp_md_raw.npz
    python scripts/rank_arms.py results/cmp_rare_raw.npz \
        --max-angle 8 --max-speed 9.6 --max-seg 12
    python scripts/rank_arms.py results/cmp_md_raw.npz --out results/rank_md

Rarity is defined against the operational distribution, so the ODD flags select
which failures count as rare: they must be the ones the campaign was run with.
The script prints the bounds it used at the top of the report.

Per-seed provenance
-------------------
The paired test needs to know which run produced each point. Campaigns saved
with the current code record it (`seeds_<i>` in the .npz). For older campaigns
the script reconstructs it from the sibling .json: the clouds were appended in
execution order, so ordering `per_seed` by `run_index` recovers the block order,
and the blocks are cut apart when the recorded evaluation counts add up to the
number of saved points. An arm whose counts do not add up -- which happens when
invalid runs were dropped -- is scored but excluded from the paired test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.odd_presets import add_odd_args, resolve_bounds


def load_campaign(npz_path: str):
    """Return (clouds, cloud_seeds, param_names, per_arm, notes)."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(x) for x in z["labels"]]
    names = [str(x) for x in z["param_names"]] if "param_names" in z else None
    clouds, seeds, notes = {}, {}, []
    for i, lab in enumerate(labels):
        clouds[lab] = (z[f"theta_{i}"], z[f"margins_{i}"])
        if f"seeds_{i}" in z:
            seeds[lab] = z[f"seeds_{i}"]

    json_path = npz_path.replace("_raw.npz", ".json")
    per_arm, meta = {}, {}
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        per_arm = doc.get("per_arm", {})
        meta = doc.get("metadata") or {}
        if not seeds:
            seeds, rec_notes = _reconstruct_seeds(clouds, doc.get("per_seed", {}))
            notes += rec_notes
    elif not seeds:
        notes.append(f"no {os.path.basename(json_path)} beside the .npz: "
                     "per-seed provenance could not be recovered.")
    return clouds, seeds, names, per_arm, notes, meta


def _reconstruct_seeds(clouds: dict, per_seed: dict):
    """
    Rebuild the per-point seed array from the recorded run order.

    Exact or nothing: an arm whose evaluation counts do not sum to the number of
    saved points is left out, because a proportional split would invent a
    provenance and every downstream p-value would inherit the invention.
    """
    out, notes, no_index = {}, [], []
    for label, (_, margins) in clouds.items():
        rows = per_seed.get(label) or []
        if not rows or any("run_index" not in r for r in rows):
            no_index.append(label)
            continue
        ordered = sorted(rows, key=lambda r: r["run_index"])
        counts = [int(r["n_evaluations"]) for r in ordered]
        if sum(counts) != len(margins):
            notes.append(
                f"{label}: recorded evaluations ({sum(counts)}) do not match the "
                f"{len(margins)} saved points -- invalid runs were dropped, so the "
                "blocks cannot be cut apart exactly. Scored, but excluded from the "
                "paired test.")
            continue
        sd = np.concatenate([np.full(c, int(r["seed"]), dtype=int)
                             for c, r in zip(counts, ordered)])
        out[label] = sd
    if no_index:
        notes.append(
            "this campaign predates run_index bookkeeping, so the pooled clouds "
            f"cannot be split by seed for: {', '.join(no_index)}.")
    return out, notes


def _build_parser() -> argparse.ArgumentParser:
    """The command line of this script."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="results/<prefix>_raw.npz")
    ap.add_argument("--scenario", default="lane_keeping",
                    help="scenario whose ODD marginals define rarity")
    ap.add_argument("--rarity-q", type=float, default=0.10,
                    help="a failure is rare below this ODD quantile (default 0.10)")
    ap.add_argument("--reference-n", type=int, default=200_000,
                    help="ODD draws used to calibrate the cut (costs no simulation)")
    ap.add_argument("--metric", default="rare_per_100",
                    choices=["rare_per_100", "n_rare", "failures_per_100",
                             "n_failures", "n_exclusive_regions"],
                    help="what to rank on (default: rare failures per 100 sims)")
    ap.add_argument("--regions", action="store_true",
                    help="also cluster the failures to fill the region columns "
                         "(slower; needs scikit-learn)")
    ap.add_argument("--plan", default=None,
                    help="a pre-registration file (see docs/preregistrazione_"
                         "classifica.json): runs the declared hypotheses in the "
                         "declared order, stopping at the first failure")
    ap.add_argument("--out", default=None, help="write <out>.json and <out>.txt")

    add_odd_args(ap)
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    from pipeline.arm_ranking import (fixed_sequence_report, fixed_sequence_test,
                                      load_plan, rank_arms)
    from scenarios import SCENARIOS

    clouds, cloud_seeds, names, per_arm, notes, meta = load_campaign(args.path)
    scenario = SCENARIOS[args.scenario]
    lower, upper = resolve_bounds(scenario, args)

    # The campaign's ODD, when the file records it. Passing it again by hand on
    # the command line has already cost a whole analysis: without the flags the
    # default is the wide ODD, every point of a narrow campaign falls outside
    # the support of the marginals, its log-density is -inf and EVERY failure
    # counts as "rare". The metric degenerates and the pre-registered sequence
    # reads a comparison that does not exist.
    meta_lo = meta.get("odd_lower")
    meta_hi = meta.get("odd_upper")
    if meta_lo and meta_hi:
        meta_lo = np.asarray(meta_lo, float)
        meta_hi = np.asarray(meta_hi, float)
        if lower is None:
            lower, upper = meta_lo, meta_hi
            print("[ODD]    taken from the campaign metadata "
                  "(no ODD flag passed)")
        elif not (np.allclose(lower, meta_lo) and np.allclose(upper, meta_hi)):
            raise SystemExit(
                "the ODD flags passed do not match the ODD recorded in the "
                "campaign.\n"
                f"  campaign: lower={meta_lo.tolist()}\n"
                f"            upper={meta_hi.tolist()}\n"
                f"  passed  : lower={np.asarray(lower).tolist()}\n"
                f"            upper={np.asarray(upper).tolist()}\n"
                "Drop the flags: they are read from the metadata.")
    if lower is None:
        b = scenario.param_bounds()
        lower, upper = np.asarray(b["lower"], float), np.asarray(b["upper"], float)
        if not meta:
            print("[ODD]    NO metadata in the campaign and no flag: the default "
                  "bounds are used.\n"
                  "         If the campaign ran on a narrow ODD, this analysis "
                  "is meaningless.")
    names = names or list(scenario.param_bounds()["names"])
    dists = scenario.param_distributions(lower, upper)
    thr = float(scenario.failure_threshold())

    print(f"[loaded] {len(clouds)} arms, "
          f"{sum(len(m) for _, m in clouds.values())} points from {args.path}")
    print(f"[ODD]    lower={np.asarray(lower).tolist()}")
    print(f"[ODD]    upper={np.asarray(upper).tolist()}")
    print(f"[seeds]  per-point provenance for "
          f"{len(cloud_seeds)}/{len(clouds)} arms")
    _sanity_check_bounds(clouds, lower, upper)
    print()

    regions = None
    if args.regions:
        from pipeline.region_comparison import compare_failure_regions
        regions = compare_failure_regions(clouds, lower, upper, threshold=thr,
                                          param_names=names, dists=dists)

    rk = rank_arms(clouds, lower, upper, dists, threshold=thr,
                   cloud_seeds=cloud_seeds, regions=regions, per_arm=per_arm,
                   q=args.rarity_q, reference_n=args.reference_n,
                   metric=args.metric)
    for n in notes:
        rk.notes.append(n)
    print(rk.report())

    sequence = None
    if args.plan:
        plan = load_plan(args.plan)
        if plan.get("metric") and plan["metric"] != rk.metric:
            raise SystemExit(
                f"the plan pre-registered '{plan['metric']}' as the endpoint but "
                f"this run ranked on '{rk.metric}'. Changing the endpoint after "
                "the fact is the thing the plan exists to prevent -- pass "
                f"--metric {plan['metric']}.")
        sequence = fixed_sequence_test(rk, plan)
        print()
        print(fixed_sequence_report(sequence))

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        doc = rk.to_dict()
        if sequence is not None:
            doc["preregistered_sequence"] = sequence
            doc["plan_file"] = os.path.abspath(args.plan)
        with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=float)
        with open(f"{args.out}.txt", "w", encoding="utf-8") as fh:
            fh.write(rk.report())
            if sequence is not None:
                fh.write("\n\n" + fixed_sequence_report(sequence))
        print(f"\n[saved] {args.out}.json\n[saved] {args.out}.txt")


def _sanity_check_bounds(clouds: dict, lower, upper) -> None:
    """
    Shout when the points fall outside the ODD they are being scored against.

    This is the failure mode the ODD flags exist to prevent: score a narrow-ODD
    campaign against the wide default and every failure looks ordinary, because
    the reference distribution is the wrong one.
    """
    theta = np.vstack([t for t, _ in clouds.values()])
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    out = ((theta < lo - 1e-6) | (theta > hi + 1e-6)).any(axis=1).mean()
    if out > 0.01:
        print(f"[WARNING] {out:.0%} of the evaluated points lie OUTSIDE these "
              "bounds.\n          The campaign almost certainly ran on a "
              "different ODD, and\n          rarity scored against this one is "
              "meaningless. Pass the\n          campaign's own --max-angle / "
              "--max-speed / --max-seg flags.")
        return

    # The other half of the same mistake, and the one that does not announce
    # itself: a NARROWED campaign scored against the wide default. Every point
    # is inside the bounds, so nothing looks wrong -- but the reference
    # distribution is far wider than the one the campaign sampled, and every
    # failure it found gets scored as unremarkable. Detect it by how little of
    # each axis the evaluated points actually cover.
    span = (theta.max(axis=0) - theta.min(axis=0)) / np.where(hi - lo > 0, hi - lo, 1)
    narrow = np.where(span < 0.3)[0]
    if narrow.size:
        print("[WARNING] the evaluated points cover less than 30% of the given "
              "range on\n          "
              + ", ".join(f"axis {j} ({span[j]:.0%})" for j in narrow)
              + ".\n          This campaign was almost certainly run on a "
                "NARROWED ODD. Scoring\n          it against these wider bounds "
                "makes every failure look ordinary.\n          Pass the "
                "campaign's own ODD flags.")


if __name__ == "__main__":
    main()
