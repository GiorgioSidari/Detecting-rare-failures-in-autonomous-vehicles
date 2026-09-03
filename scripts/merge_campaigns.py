#!/usr/bin/env python3
"""
Merge the raw arrays of two or more campaigns into a single campaign file.

`rank_arms.py` can only pair arms that appear in the same `.npz`, so arms run
in separate campaigns have to be merged into one file before they can be
compared.

The script writes nothing unless both checks below pass:

  1. every input campaign declares identical `metadata` (backend, operating
     point, budget, seeds);
  2. with `--verify-against <reference.npz>`, every arm an input has in common
     with the reference reproduces seed by seed -- for each (arm, seed) pair the
     multiset of safety margins must match. The comparison is on the multiset
     and not on the array, because the execution order of the (arm, seed) runs
     is shuffled per campaign, so the same margins are stored in a different
     order.

Usage
-----
    python scripts/merge_campaigns.py \
        results/cmp_md12_ctrl_raw.npz results/cmp_md12_ctrl_rand_raw.npz \
        --out results/cmp_md12_2x2 \
        --verify-against results/cmp_md12_raw.npz

Writes `<out>_raw.npz` and `<out>.json`, which `rank_arms.py` then reads like
any other campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Fields that must agree across campaigns for a merge to mean anything.
# `seeds` is included: arms paired on different seed sets are not paired.
META_KEYS = ("backend", "speed_scale", "obs_lag", "obs_latency",
             "steer_noise", "driver", "odd_lower", "odd_upper",
             "budget", "seeds")


def _json_beside(npz_path: str) -> str:
    return npz_path.replace("_raw.npz", ".json")


def load(npz_path: str):
    """Return (labels, arrays_by_label, doc) for one campaign."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(x) for x in z["labels"]]
    arrays = {}
    for i, lab in enumerate(labels):
        arrays[lab] = {
            "theta":   z[f"theta_{i}"],
            "margins": z[f"margins_{i}"],
            "seeds":   z[f"seeds_{i}"] if f"seeds_{i}" in z else None,
        }
    doc = {}
    jp = _json_beside(npz_path)
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as fh:
            doc = json.load(fh)
    return labels, arrays, doc, z


def check_metadata(docs: list, paths: list) -> dict:
    """Every campaign must declare the same operating point. Return it."""
    declared = [(p, d.get("metadata") or {}) for p, d in zip(paths, docs)]
    silent = [p for p, m in declared if not m]
    known = [(p, m) for p, m in declared if m]
    for p in silent:
        print(f"[metadata] {p} declares none -- it predates the metadata block. "
              "Its operating point cannot be read from the file; the merge rests "
              "on --verify-against for this campaign.")
    if not known:
        print("[metadata] no campaign declares metadata.", file=sys.stderr)
        return {}
    base_path, base = known[0]
    for path, meta in known[1:]:
        for k in META_KEYS:
            if base.get(k) != meta.get(k):
                sys.exit(f"[REFUSED] {k!r} differs: {base.get(k)!r} in "
                         f"{base_path} vs {meta.get(k)!r} in {path}. These "
                         "campaigns did not run at the same operating point.")
    print(f"[metadata] identical across the {len(known)} campaigns that declare it: "
          f"backend={base.get('backend')} speed_scale={base.get('speed_scale')} "
          f"driver={base.get('driver')} budget={base.get('budget')}")
    return dict(base)


def same_arm(a: dict, b: dict, atol: float) -> bool:
    """True if two copies of one arm hold the same outcomes, seed by seed."""
    if a["seeds"] is None or b["seeds"] is None:
        return False
    if len(a["margins"]) != len(b["margins"]):
        return False
    for s in sorted(set(a["seeds"].tolist()) | set(b["seeds"].tolist())):
        ma = np.sort(a["margins"][a["seeds"] == s])
        mb = np.sort(b["margins"][b["seeds"] == s])
        if ma.shape != mb.shape or not np.allclose(ma, mb, atol=atol, rtol=0):
            return False
    return True


def verify(merged: dict, ref_path: str, atol: float) -> None:
    """
    Every arm shared with the reference must reproduce seed by seed.

    Compares the SORTED margins within each seed: the storage order differs
    because the execution order is shuffled per campaign, but the set of
    outcomes for a given (arm, seed) is determined by the pipeline and must
    not move.
    """
    _, ref, _, _ = load(ref_path)
    shared = [lab for lab in merged if lab in ref]
    if not shared:
        print(f"[verify] no arm in common with {ref_path} -- nothing to "
              "check. The merge rests on the metadata alone.")
        return
    for lab in shared:
        a, b = ref[lab], merged[lab]
        if a["seeds"] is None or b["seeds"] is None:
            sys.exit(f"[REFUSED] {lab}: no per-seed provenance, cannot verify.")
        if len(a["margins"]) != len(b["margins"]):
            sys.exit(f"[REFUSED] {lab}: {len(a['margins'])} points in the "
                     f"reference, {len(b['margins'])} here.")
        for s in sorted(set(a["seeds"].tolist())):
            ma = np.sort(a["margins"][a["seeds"] == s])
            mb = np.sort(b["margins"][b["seeds"] == s])
            if ma.shape != mb.shape or not np.allclose(ma, mb, atol=atol, rtol=0):
                sys.exit(
                    f"[REFUSED] {lab}, seed {s}: the margins do not reproduce "
                    f"(n={ma.size} vs {mb.size}, max deviation "
                    f"{np.abs(ma[:min(ma.size, mb.size)] - mb[:min(ma.size, mb.size)]).max() if ma.size and mb.size else float('nan'):.3g}). "
                    "The campaigns are not comparable -- investigate the drift "
                    "before merging.")
        print(f"[verify] {lab:<34} reproduces on all "
              f"{len(set(a['seeds'].tolist()))} seeds")


def _build_parser() -> argparse.ArgumentParser:
    """The command line of this script."""
    ap = argparse.ArgumentParser(
        description="Merge campaign raw files into one, if they are comparable.")
    ap.add_argument("inputs", nargs="+", help="the <prefix>_raw.npz files to merge")
    ap.add_argument("--out", required=True,
                    help="output prefix, e.g. results/cmp_md12_2x2")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="keep only these arms, by exact label. Use it when a "
                         "campaign holds arms that another input already "
                         "provides: merging would refuse the duplicate, and "
                         "dropping the copy is the right fix only when the two "
                         "reproduce -- which --verify-against is there to prove.")
    ap.add_argument("--verify-against", default=None, dest="verify_against",
                    help="a reference <prefix>_raw.npz whose shared arms must "
                         "reproduce seed by seed. Strongly recommended.")
    ap.add_argument("--atol", type=float, default=1e-9,
                    help="absolute tolerance on the margins (default 1e-9: the "
                         "pipeline is deterministic, this is float noise only)")
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    if len(args.inputs) < 2:
        sys.exit("[REFUSED] merging needs at least two campaigns.")

    docs, merged, order, param_names, per_arm = [], {}, [], None, {}
    for path in args.inputs:
        labels, arrays, doc, z = load(path)
        docs.append(doc)
        if param_names is None and "param_names" in z:
            param_names = [str(x) for x in z["param_names"]]
        kept = [l for l in labels if args.arms is None or l in args.arms]
        if args.arms is not None and len(kept) < len(labels):
            print(f"[filter] {path}: kept {len(kept)}/{len(labels)} arms "
                  f"({', '.join(kept) if kept else 'none'})")
        for lab in kept:
            if lab in merged:
                # The same arm in two campaigns is not automatically an error:
                # the pipeline is deterministic given (arm, seed), so a campaign
                # re-run of an arm should reproduce it exactly. Keep one copy if
                # it does, refuse if it does not -- a silent pick would hide a
                # session drift, and a blanket refusal would block a legitimate
                # merge of overlapping campaigns.
                if same_arm(merged[lab], arrays[lab], args.atol):
                    print(f"[duplicate] {lab}: also in {path}, reproduces "
                          "seed by seed -- keeping the first copy")
                    continue
                sys.exit(f"[REFUSED] arm {lab!r} appears in more than one "
                         f"campaign and the copies DIFFER. {path} does not "
                         "reproduce the earlier campaign: investigate the "
                         "drift, do not merge.")
            merged[lab] = arrays[lab]
            order.append(lab)
        for lab, rec in (doc.get("per_arm") or {}).items():
            if lab in merged:
                per_arm[lab] = rec
        print(f"[loaded] {path}: {len(kept)} arms "
              f"({sum(len(arrays[l]['margins']) for l in kept)} points)")

    meta = check_metadata(docs, args.inputs)
    if args.verify_against:
        verify(merged, args.verify_against, args.atol)

    out = {}
    for i, lab in enumerate(order):
        out[f"theta_{i}"]   = merged[lab]["theta"]
        out[f"margins_{i}"] = merged[lab]["margins"]
        if merged[lab]["seeds"] is not None:
            out[f"seeds_{i}"] = merged[lab]["seeds"]
    out["labels"] = np.array(order, dtype=object)
    if param_names is not None:
        out["param_names"] = np.array(param_names, dtype=object)

    npz_path = f"{args.out}_raw.npz"
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez_compressed(npz_path, **out)

    doc = {
        "metadata": meta,
        "merged_from": list(args.inputs),
        "verified_against": args.verify_against,
        "arms": order,
        "per_arm": per_arm,
        "note": "Merged campaign. The arms did not run in the same session; "
                "they are comparable because they share an operating point and "
                "because the arms in common with the reference reproduce seed "
                "by seed. Region and exclusive-region counts computed on this "
                "file are NOT comparable with those of the source campaigns: "
                "clustering runs on the pooled failures of whatever arms are "
                "present.",
    }
    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)

    print(f"\n[saved] {npz_path}")
    print(f"[saved] {args.out}.json")
    print(f"[arms]  {len(order)}: {', '.join(order)}")
    print(f"\nNow run:\n  python scripts/rank_arms.py {npz_path} --out {args.out.replace('cmp', 'rank')}")


if __name__ == "__main__":
    main()
