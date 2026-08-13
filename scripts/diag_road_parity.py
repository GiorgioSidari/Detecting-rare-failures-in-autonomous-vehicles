#!/usr/bin/env python3
"""
Geometric parity gate between backends -- "same theta, same road?"

Perché esiste
-------------
The cross-simulator comparison rests on an assumption that is easy to take for
granted and that nobody was checking: that at the same theta the backends drive
on the SAME road. If that is not true, every comparison metric (Spearman on the
QoI, failure-region overlap) correlates different scenarios, and the result that
comes out looks publishable but means nothing.

And it really was broken. Before the fix, on the 60-point LHS design:

    road length         Udacity   202.7 m  (range 201-205)
                        MetaDrive 124.0 m  (range  51-200)
    Spearman between the lengths: -0.026
    -> confronto risultante: rho=-0.020, Fisher p=1.000

i.e. "the simulators disagree completely", which looks like an interesting
result and was instead a parameterisation bug: MetaDrive built the road with its
PGBlocks, collapsing the 5 angles into their mean.

This script is the check that would have caught all of it before burning a
campaign.

Uso
---
    python scripts/diag_road_parity.py                  # 20 thetas, every backend
    python scripts/diag_road_parity.py --n 60 --seed 42 # same design as the campaign
    python scripts/diag_road_parity.py --backend metadrive
    python scripts/diag_road_parity.py --legacy         # show the old path fails

Exit code 1 if a backend exceeds the threshold: usable in CI or before
launching a collection.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scenarios.common.road_geometry import road_polyline        # noqa: E402

# Parity threshold. 10 cm is ~1/60 of the lane width and ~1/25 of the MAX_XTE
# failure threshold (2.5 m): a divergence below that level cannot move a
# scenario's verdict. Above it, it can.
THRESHOLD_M = 0.10

# The ends must be excluded: MetaDrive extrapolates beyond the first and last
# point of the polyline, and the deviation rises there (measured: 14 cm at
# s=0.5 m against 1.9 cm inside). It is a boundary artefact, not a divergence in
# shape, and the vehicle never meaningfully drives there.
EDGE_MARGIN_M = 2.0


def point_to_polyline_distance(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Distance of every point of `P` from the polyline `C`, point-to-SEGMENT.

    Using point-to-VERTEX distance is the easy mistake: the centreline vertices
    are ~1.34 m apart, so half that spacing lands in the metric and a perfect
    alignment measures ~34 cm instead of ~0. It is a mistake that makes correct
    things look broken.
    """
    P = np.asarray(P, dtype=float)
    C = np.asarray(C, dtype=float)
    a, b = C[:-1], C[1:]
    ab = b - a
    L2 = (ab ** 2).sum(1)
    t = (((P[:, None, :] - a[None]) * ab[None]).sum(-1)
         / np.maximum(L2[None], 1e-12)).clip(0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(1)


def _length(C: np.ndarray) -> float:
    d = np.diff(np.asarray(C, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


# -- backends: each returns the centreline the simulator will actually use ----

def _reference_centerline(row: np.ndarray) -> np.ndarray:
    """Lo standard: la spline Catmull-Rom di Udacity, in coordinate locali."""
    poly = np.asarray(road_polyline(np.asarray(row, float)), dtype=float)
    return poly - poly[0]


def _centerline_metadrive(row: np.ndarray, legacy: bool = False) -> np.ndarray:
    """
    The lane MetaDrive actually builds, resampled.

    With `legacy=True` it reconstructs the PGBlock arc chain -- the old path,
    kept here so the gate can DEMONSTRATE that it rejects it rather than merely
    asserting so.
    """
    if legacy:
        from scenarios.lane_keeping_md.map_builder import build_scenario_spec

        spec = build_scenario_spec(np.asarray(row, float))
        pts, x, y, hdg = [np.array([0.0, 0.0])], 0.0, 0.0, 0.0
        for b in spec.blocks:
            n = max(2, int(b.length))
            for _ in range(n):
                ds = b.length / n
                if b.kind == "C" and abs(b.angle) > 1e-9:
                    dk = ds / b.radius * np.sign(b.angle)
                    hdg += dk
                x += ds * np.cos(hdg)
                y += ds * np.sin(hdg)
                pts.append(np.array([x, y]))
        return np.asarray(pts)

    from scenarios.lane_keeping_md.scenario_map import lane_of, make_online_env

    env = make_online_env(row, decision_repeat=5, physics_world_step_size=0.02,
                          max_steps=10)
    try:
        env.reset()
        ln = lane_of(env)
        s = np.linspace(0.0, float(ln.length), 400)
        return np.array([ln.position(float(x), 0.0) for x in s])
    finally:
        try:
            env.close()
        except Exception:
            pass


BACKEND = {
    "metadrive": _centerline_metadrive,
}


def compare_one(row: np.ndarray, fn, **kw) -> dict:
    ref = _reference_centerline(row)
    got = np.asarray(fn(row, **kw), dtype=float)

    err = point_to_polyline_distance(got, ref)
    # approximate arc position of every point, used to exclude the edges
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(got, axis=0).T))])
    inside = (s > EDGE_MARGIN_M) & (s < s[-1] - EDGE_MARGIN_M)
    if inside.sum() < 10:
        inside = np.ones_like(s, dtype=bool)

    return {
        "mean_dev": float(err[inside].mean()),
        "max": float(err[inside].max()),
        "max_bordi_inclusi": float(err.max()),
        "len_ref": _length(ref),
        "len_got": _length(got),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="scenarios to sample")
    ap.add_argument("--seed", type=int, default=42,
                    help="same seed as the campaign = same thetas")
    ap.add_argument("--backend", default=None, choices=sorted(BACKEND),
                    help="default: every importable backend")
    ap.add_argument("--legacy", action="store_true",
                    help="use the old PGBlock geometry (must FAIL the gate)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_M)
    args = ap.parse_args()

    from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

    from scenarios.lane_keeping.config import LaneKeepingScenario
    b = LaneKeepingScenario().param_bounds()
    sampler = LatinHypercube(d=len(b["lower"]), seed=args.seed)
    theta = qmc_scale(sampler.random(n=args.n), b["lower"], b["upper"])

    names = [args.backend] if args.backend else sorted(BACKEND)

    print("=" * 74)
    print(" GEOMETRIC PARITY -- does the same theta produce the same road?")
    print("=" * 74)
    print(f" reference : Udacity Catmull-Rom centreline")
    print(f" design    : LHS seed={args.seed}, n={args.n}")
    print(f" threshold : {args.threshold*100:.0f} cm  ({EDGE_MARGIN_M:.0f} m of head/tail excluded)")
    if args.legacy:
        print(" mode      : LEGACY (PGBlock) -- FAILURE expected")
    print()

    exit_code = 0
    skipped = []
    verified = []
    for name in names:
        kw = {"legacy": True} if (args.legacy and name == "metadrive") else {}
        try:
            res = [compare_one(t, BACKEND[name], **kw) for t in theta]
        except ImportError as e:
            # NOT an "OK with a warning": a gate that verified nothing cannot
            # declare that all is well. If it did, running it in the wrong
            # interpreter would be enough to get a free pass -- i.e. the gate
            # would protect only those who do not need it.
            print(f" {name:<12} NOT VERIFIED -- missing dependency ({e})")
            skipped.append((name, str(e)))
            continue
        except Exception as e:
            print(f" {name:<12} ERROR -- {type(e).__name__}: {e}")
            exit_code = 1
            continue

        verified.append(name)

        mean_dev = np.mean([r["mean_dev"] for r in res])
        worst = max(r["max"] for r in res)
        rap = np.array([r["len_got"] / max(r["len_ref"], 1e-9) for r in res])
        ok = worst <= args.threshold

        print(f" {name:<12} {'OK ' if ok else 'FALLITO'}")
        print(f"   scostamento mean_dev   {mean_dev*100:8.3f} cm")
        print(f"   scostamento highest {worst*100:8.2f} cm   (threshold {args.threshold*100:.0f})")
        print(f"   lunghezza got/ref   {rap.mean():8.3f}x   range [{rap.min():.2f}, {rap.max():.2f}]")
        if not ok:
            exit_code = 1
            print(f"   -> the roads DIVERGE. A cross-simulator comparison on these")
            print(f"      backends would correlate different scenarios: the numbers it")
            print(f"      produces (Spearman, Jaccard) are not interpretable.")
        print()

    print("=" * 74)

    if skipped:
        exit_code = 1
        skipped_names = ", ".join(n for n, _ in skipped)
        print(f" GATE INCONCLUSIVE -- not verified: {skipped_names}")
        print()
        print(" The backend is not importable from THIS interpreter, so the check")
        print(" was not run. That is not a pass.")
        print()
        print(f"   interpreter : {sys.executable}")
        print(f"   version     : {sys.version.split()[0]}")
        if any(n == "metadrive" for n, _ in skipped):
            print()
            print(" MetaDrive requires Python < 3.12 and lives in its own venv.")
            print(" On Windows:")
            print(r"   .venv310\Scripts\python.exe scripts\diag_road_parity.py --backend metadrive")
        return exit_code

    if not verified:
        print(" GATE INCONCLUSIVE -- no backend verified")
        return 1

    if exit_code == 0:
        print(f" OK -- verified: {', '.join(verified)}")
    else:
        print(" GATE FAILED -- do not collect data until it is green again")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
