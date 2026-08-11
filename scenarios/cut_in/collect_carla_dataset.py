from __future__ import annotations

"""
Data collection for the cut_in CARLA CNN (Fase 3 del piano), mirrors
scenarios/emergency_braking/collect_carla_dataset.py — see that file's
docstring for the general rationale (not a video file, frame sequences).

Usage:
    python -m scenarios.cut_in.collect_carla_dataset --n 50
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

from cut_in.carla_episode import CutInCarlaEpisode, expert_controller  # noqa: E402
from scenarios.cut_in.simulator import CutInSimulator  # noqa: E402
from simulators.common.camera_dataset import save_archive  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_carla")
ARCHIVE_PATH = os.path.join(DATASET_DIR, "cut_in_carla.npz")
N_DEFAULT = 50
SEED = 42


def collect(
    n: int = N_DEFAULT,
    seed: int = SEED,
    host: str = "localhost",
    port: int = 2000,
    archive_path: str = ARCHIVE_PATH,
) -> None:
    from scipy.stats.qmc import LatinHypercube, scale

    bounds = CutInSimulator().ParamBounds()
    sampler = LatinHypercube(d=4, seed=seed)
    unit = sampler.random(n=n)
    params = scale(unit, bounds["lower"], bounds["upper"])

    episode = CutInCarlaEpisode(host=host, port=port)
    episode.connect()

    observations: list[np.ndarray] = []
    actions: list[float] = []
    n_collisions = 0

    n_skipped = 0
    try:
        for i, row in enumerate(params):
            frame_buf: list[np.ndarray] = []
            label_buf: list[float] = []

            def controller(frame, ego_speed, t, cutter_in_lane, _fb=frame_buf, _lb=label_buf):
                label = expert_controller(frame, ego_speed, t, cutter_in_lane)
                if frame is not None:
                    arr = np.frombuffer(frame.raw_data, dtype=np.uint8)
                    arr = arr.reshape(frame.height, frame.width, 4)[:, :, :3]  # BGRA -> BGR
                    _fb.append(arr.copy())
                    _lb.append(label)
                return label

            # The cutter's scripted spawn point (ego position + lateral_gap offset)
            # occasionally lands on map geometry (barrier, sign) for some sampled
            # lateral_gap values -> CARLA raises RuntimeError. Skip that one episode
            # rather than losing the whole collection run.
            try:
                result = episode.run_episode(row, controller, record_frames=True)
            except RuntimeError as e:
                n_skipped += 1
                print(f"  [{i + 1:3d}/{n}] SALTATO ({e})", flush=True)
                continue

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

    if n_skipped:
        print(f"\nEpisodi saltati per fallimento spawn: {n_skipped}/{n}")

    X = np.asarray(observations, dtype=np.uint8)
    y = np.asarray(actions, dtype=np.float32).reshape(-1, 1)

    print(f"\nDataset: {X.shape[0]} coppie frame-label  (frame shape: {X.shape[1:]})")
    print(f"Episodi con collisione: {n_collisions}/{n}")
    print(f"Frazione frenata: {y.mean():.1%}")

    save_archive(list(X), list(y), archive_path)
    print(f"Salvato: {archive_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raccoglie il dataset frame-camera per cut_in via CARLA.")
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out", default=ARCHIVE_PATH)
    args = parser.parse_args()
    collect(n=args.n, seed=args.seed, host=args.host, port=args.port, archive_path=args.out)
