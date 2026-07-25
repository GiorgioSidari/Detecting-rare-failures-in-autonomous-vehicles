"""
Train the behavioral-cloning (BC) state controller for MetaDrive lane-keeping
(Part C item 1 / variant A2).

The teacher is the robust PurePursuitDriver. We collect its (state -> steering)
decisions ONLY on an EASY training distribution (gentle curves, low speed), then
fit a small MLP. Because the MLP is trained in-distribution and extrapolates
poorly, when later used to drive the FULL ODD it fails on the hard scenarios
(sharp curves / high speed) — a genuine generalization failure, not injected.

Run (needs MetaDrive):
    python scripts/train_bc_md.py --n-scenarios 60 --epochs 60
Then use the learned driver:
    python scripts/run_active_boundary.py --learned-model scenarios/lane_keeping_md/models/bc_state_mlp.keras
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario
from scenarios.lane_keeping_md.driver import PurePursuitDriver
from scenarios.lane_keeping_md.bc_controller import (
    build_bc_mlp, save_model, state_to_features,
)

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "scenarios", "lane_keeping_md", "models", "bc_state_mlp.joblib")


def _easy_training_params(n: int, angle_max: float, speed_max: float, seed: int) -> np.ndarray:
    """LHS over a RESTRICTED (easy) ODD: gentle angles, low speeds. This is the
    training support; the model will fail OUTSIDE it."""
    # [a1..a5, min_speed, max_speed, seg_length, map_size]
    lower = np.array([0, 0, 0, 0, 0, 5.0, 8.0, 15.0, 200.0])
    upper = np.array([angle_max]*5 + [max(6.0, speed_max - 3.0), speed_max, 30.0, 300.0])
    u = LatinHypercube(d=9, seed=seed).random(n=n)
    return scale(u, lower, upper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=60, help="teacher rollouts to collect")
    ap.add_argument("--angle-max", type=float, default=25.0, help="max training curve angle (deg)")
    ap.add_argument("--speed-max", type=float, default=12.0, help="max training speed (m/s)")
    ap.add_argument("--epochs", type=int, default=300, help="max_iter for the sklearn MLP")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-speed-scale", type=float, default=0.3,
                    help="drive the teacher slowly so it stays near centre = GOOD demos")
    ap.add_argument("--lat-max", type=float, default=1.0,
                    help="keep only in-distribution states |lateral_error| < lat_max (m)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    # Teacher scenario: robust pure-pursuit driven SLOWLY on the easy ODD, so it
    # stays near the lane centre and produces clean, in-distribution demos.
    sc = LaneKeepingMetaDriveScenario(
        max_steps=300, n_jobs=1, speed_scale=args.teacher_speed_scale,
        driver_factory=lambda: PurePursuitDriver(),
    )
    params = _easy_training_params(args.n_scenarios, args.angle_max, args.speed_max, args.seed)

    print(f"[collect] {args.n_scenarios} teacher rollouts (speed_scale="
          f"{args.teacher_speed_scale}, angle<= {args.angle_max}) ...")
    sc._record = []
    sc.run_simulation(params, verbose=True)
    record = sc._record
    print(f"[collect] {len(record)} raw (state, steering) samples")

    # Restrict to the in-distribution support: near-centre states only. The model
    # never learns to recover from large errors, so it FAILS out-of-distribution
    # (sharp curves / high speed) — a genuine generalization failure.
    kept = [(s, st) for s, st in record if abs(s.get("lateral_error", 0.0)) < args.lat_max]
    print(f"[filter] {len(kept)}/{len(record)} samples kept (|lateral| < {args.lat_max} m)")

    X = np.stack([state_to_features(s) for s, _ in kept])
    y = np.array([steer for _, steer in kept], dtype=np.float64)

    model = build_bc_mlp(hidden=(32, 32), max_iter=args.epochs)
    print(f"[train] sklearn MLP on {len(X)} samples (max_iter={args.epochs}) ...")
    model.fit(X, y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model(model, args.out)
    mse = float(np.mean((model.predict(X) - y) ** 2))
    print(f"[done] train MSE={mse:.5f}  ->  saved to {args.out}")
    print(" Now use the learned driver over the full ODD:")
    print(f"   python scripts/run_active_boundary.py --learned-model {args.out}")


if __name__ == "__main__":
    main()
