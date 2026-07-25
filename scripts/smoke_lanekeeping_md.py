"""
Smoke test M0 for the MetaDrive lane-keeping backend.

Runs a couple of REAL headless MetaDrive episodes through
LaneKeepingMetaDriveScenario and prints diagnostics, so we can validate the
MetaDrive glue (_build_md_config / _extract_state / _simulate_one) against the
installed MetaDrive version. It deliberately catches and prints any exception
with a full traceback, so if something is off you can paste the output.

Prereqs (run once):
    pip install metadrive-simulator
    python -m metadrive.pull_asset        # downloads the assets

Run:
    python scripts/smoke_lanekeeping_md.py
"""
import sys
import traceback

import numpy as np


def main() -> int:
    try:
        import metadrive
        print(f"metadrive version: {getattr(metadrive, '__version__', '?')}")
    except Exception:
        print("MetaDrive is not importable. Install with:\n"
              "  pip install metadrive-simulator\n"
              "  python -m metadrive.pull_asset")
        traceback.print_exc()
        return 2

    from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario

    sc = LaneKeepingMetaDriveScenario(
        decision_repeat=5,          # 5 * 0.02s -> 10 Hz control rate (exact)
        physics_world_step_size=0.02,
        max_steps=200,
        n_jobs=1,                   # sequential: clearer errors for the smoke test
    )
    print(f"control rate (nominal): {sc.control_hz_nominal:.1f} Hz")

    # Two scenarios: a gentle one and a sharp/fast one.
    params = np.array([
        [10, 0,  10, 0,  10, 6.0, 12.0, 25.0, 250.0],   # gentle curves, moderate speed
        [70, 40, 60, 30, 80, 8.0, 24.0, 25.0, 250.0],   # sharp curves, higher speed
    ], dtype=float)

    try:
        traj = sc.run_simulation(params, verbose=True)
    except Exception:
        print("\n--- ERROR during run_simulation (MetaDrive glue) ---")
        traceback.print_exc()
        return 1

    print(f"\ntrajectories shape : {traj.shape}   (expected (2, T, 4))")
    print(f"run lengths        : {sc._run_lengths}")
    print(f"control_hz         : {np.round(sc._control_hz, 2)}")
    print(f"meters_per_step    : {np.round(sc._meters_per_step, 3)}")

    qoi = sc.compute_qoi(traj, params)
    print(f"QoI (safety margin): {np.round(qoi, 3)}   (>0 safe, <0 failure, NaN invalid)")
    print(f"valid_mask         : {sc._valid_mask}")
    print(f"n_invalid          : {sc._n_invalid}  (degenerate={sc._n_degenerate})")

    # First few steps of scenario 0 to eyeball [x, y, xte, steering].
    L0 = min(5, sc._run_lengths[0])
    print("\nfirst steps of scenario 0  [x, y, xte, steering]:")
    for t in range(L0):
        x, y, xte, steer = traj[0, t]
        print(f"  t={t:3d}  x={x:8.2f}  y={y:8.2f}  xte={xte:+.3f}  steer={steer:+.3f}")

    print("\nSMOKE OK: the MetaDrive glue runs and produces trajectories/QoI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
