"""
Plot the learned fail/safe BOUNDARY in the (curvature, speed) plane — the actual
product of Step B (Part C item 5).

Runs the active-boundary learner, then projects the fitted GP onto two axes:
  x = curvature   (all 5 road angles set equal, so the curve radius follows)
  y = max_speed
holding the other parameters at their ODD median. It draws the P(failure) heatmap
with the P=0.5 decision boundary, and overlays the evaluated scenarios coloured by
fail/safe (they concentrate along the boundary by construction).

Run (needs MetaDrive):
    python scripts/plot_boundary_md.py --speed-scale 0.3625     # operating point of cmp_md12
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scenarios import SCENARIOS
from pipeline.active_boundary import run_active_boundary, failure_probability

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lane_keeping_md")
    ap.add_argument("--speed-scale", type=float, default=0.3)
    ap.add_argument("--n-seed", type=int, default=40)
    ap.add_argument("--batch", type=int, default=14)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="boundary_md.png")
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    if hasattr(sc, "speed_scale"):
        sc.speed_scale = args.speed_scale
    print("[active] learning the boundary ...")
    res = run_active_boundary(sc, seed=args.seed, n_seed=args.n_seed,
                              batch=args.batch, n_iter=args.n_iter, verbose=True)
    gp = res.model
    b = sc.param_bounds()
    lower, upper = np.asarray(b["lower"], float), np.asarray(b["upper"], float)
    span = np.where((upper - lower) > 0, upper - lower, 1.0)
    thr = float(sc.failure_threshold())

    # Other parameters held at their ODD median.
    dists = sc.param_distributions(lower, upper)
    med = np.array([float(d.median()) for d in dists])

    ax_angle = np.linspace(lower[0], upper[0], 70)     # curvature axis (deg)
    ax_speed = np.linspace(lower[6], upper[6], 70)     # max_speed axis (m/s)
    AA, SS = np.meshgrid(ax_angle, ax_speed)
    grid = np.tile(med, (AA.size, 1))
    grid[:, 0:5] = AA.ravel()[:, None]                 # all 5 angles = curvature
    grid[:, 6] = SS.ravel()                            # max_speed
    grid[:, 5] = np.minimum(grid[:, 5], grid[:, 6])    # keep min_speed <= max_speed

    mu, sig = gp.predict((grid - lower) / span, return_std=True)
    P = failure_probability(mu, sig, thr).reshape(AA.shape)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cf = ax.contourf(AA, SS, P, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r", alpha=0.9)
    ax.contour(AA, SS, P, levels=[0.5], colors="black", linewidths=2.2)
    # Evaluated scenarios projected onto the plane, coloured by outcome.
    th, lab = res.theta_evaluated, res.labels
    ax.scatter(th[:, :5].mean(axis=1), th[:, 6], c=lab, cmap="bwr",
               edgecolor="k", linewidth=0.4, s=26, alpha=0.7, label="evaluated")
    cb = fig.colorbar(cf, ax=ax); cb.set_label("P(failure)")
    ax.set_xlabel("curvature  (mean angle, °)  →  sharper curves")
    ax.set_ylabel("max_speed (m/s)")
    drv = "pure-pursuit"
    ax.set_title(f"Learned fail/safe boundary — {drv}, speed_scale={args.speed_scale}\n"
                 f"P(failure) ODD ≈ {res.p_fail*100:.0f}%  ·  {res.n_evaluations} simulations  ·  "
                 "black line = P=0.5")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[done] saved: {args.out}")
    top = ", ".join(f"{n} {v*100:.0f}%" for n, v in res.boundary_summary["top_params"][:3])
    print(f"       dominant parameters: {top}")


if __name__ == "__main__":
    main()
