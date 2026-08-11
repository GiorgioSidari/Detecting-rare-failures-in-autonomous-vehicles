from __future__ import annotations

"""
Data collection for the emergency_braking CARLA CNN (Fase 2 del piano).

Runs N episodes with the analytic "expert" controller (see
opensbt-core/Simulator/emergency_braking/carla_episode.py::expert_controller)
and records every (camera frame, braking_force) pair to build the
behavioral-cloning dataset. Mirrors dataset_utils.save_archive() in
opensbt-core/Simulator/lanekeeping, generalized via simulators/common
(this is NOT a video file — it's a sequence of numpy frames, see
simulators/common/camera_dataset.py).

Usage:
    python -m scenarios.emergency_braking.collect_carla_dataset --n 50
    python -m scenarios.emergency_braking.collect_carla_dataset --n 200 --host localhost --port 2000
"""

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OPENSBT_SIM = os.path.join(_REPO_ROOT, "opensbt-core", "Simulator")
for _p in (_REPO_ROOT, _OPENSBT_SIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emergency_braking.carla_episode import EmergencyBrakingCarlaEpisode, expert_controller  # noqa: E402
from simulators.emergency_braking import EmergencyBrakingSimulator  # noqa: E402
from simulators.common.camera_dataset import save_archive  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_carla")
ARCHIVE_PATH = os.path.join(DATASET_DIR, "emergency_braking_carla.npz")
N_DEFAULT = 50
SEED = 42


def collect(
    n: int = N_DEFAULT,
    seed: int = SEED,
    host: str = "localhost",
    port: int = 2000,
    archive_path: str = ARCHIVE_PATH,
) -> None:
    """
    Sample n parameter combinations (same LHS bounds as the physics simulator),
    run one CARLA episode each with the expert controller, and save every
    (frame, braking_force) pair collected along the way.
    """
    from scipy.stats.qmc import LatinHypercube, scale

    bounds = EmergencyBrakingSimulator().ParamBounds()
    sampler = LatinHypercube(d=4, seed=seed)
    unit = sampler.random(n=n)
    params = scale(unit, bounds["lower"], bounds["upper"])

    episode = EmergencyBrakingCarlaEpisode(host=host, port=port)
    episode.connect()

    observations: list[np.ndarray] = []
    actions: list[float] = []
    n_collisions = 0

    try:
        for i, row in enumerate(params):
            frame_buf: list[np.ndarray] = []
            label_buf: list[float] = []

            def controller(frame, ego_speed, t, lead_braking, _fb=frame_buf, _lb=label_buf):
                label = expert_controller(frame, ego_speed, t, lead_braking)
                if frame is not None:
                    arr = np.frombuffer(frame.raw_data, dtype=np.uint8)
                    arr = arr.reshape(frame.height, frame.width, 4)[:, :, :3]  # BGRA -> BGR
                    _fb.append(arr.copy())
                    _lb.append(label)
                return label

            result = episode.run_episode(row, controller, record_frames=True)
            observations.extend(frame_buf)
            actions.extend(label_buf)
            n_collisions += int(result["collided"])
            print(
                f"  [{i + 1:3d}/{n}] steps={result['iterations']:3d}  "
                f"collided={result['collided']}  frames={len(frame_buf)}",
                flush=True,
            )
    finally:
        episode.close()

    X = np.asarray(observations, dtype=np.uint8)
    y = np.asarray(actions, dtype=np.float32).reshape(-1, 1)

    print(f"\nDataset: {X.shape[0]} coppie frame-label  (frame shape: {X.shape[1:]})")
    print(f"Episodi con collisione: {n_collisions}/{n}")
    print(f"Frazione frenata: {y.mean():.1%}")

    save_archive(list(X), list(y), archive_path)
    print(f"Salvato: {archive_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raccoglie il dataset frame-camera per emergency_braking via CARLA."
    )
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="numero di episodi (default 50)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out", default=ARCHIVE_PATH, help="percorso di output .npz")
    args = parser.parse_args()
    collect(n=args.n, seed=args.seed, host=args.host, port=args.port, archive_path=args.out)
