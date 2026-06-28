"""
Lane-Keeping scenario configuration.

Uses the existing Udacity/Unity simulator (opensbt-core) via Docker.
The DNN autopilot (SupervisedAgent / AutopilotModel) drives the vehicle;
we test how robust it is across a wider parameter space than the original
5-angle configuration.

Extended parameter space (vs original project)
----------------------------------------------
Original : 5 road angles in [0°, 85°]
Extended : 5 road angles + min_speed + max_speed + segment_length

QoI      : negative mean XTE (Cross-Track Error) over the run
           → positive = stayed near centre, negative = deviated

Failure  : max XTE exceeded MAX_XTE threshold (car left the road)

NOTE: This scenario requires Docker (opensbt-core) to be running.
      It is much slower than the mathematical simulators (~30s per run).

TODO (Blocco 3): implement run_simulation() as async call to SimulatorServer.
"""

import numpy as np
from scenarios.base_scenario import BaseScenario

MAX_XTE     = 3.0   # metres — matches opensbt-core/Simulator/lanekeeping/config.py
STEER_NORM  = 0.3   # reference steering std (calibrate after first runs)
EARLY_FRAC  = 0.7   # XTE fraction that triggers early-exit penalty


class LaneKeepingScenario(BaseScenario):
    name = "lane_keeping"
    description = (
        "The Udacity DNN autopilot drives on a procedurally generated road. "
        "Tests whether the neural network can stay within lane boundaries "
        "across varying road geometries and speed settings."
    )

    def param_bounds(self) -> dict:
        return {
            "names": [
                "angle_1 (°)", "angle_2 (°)", "angle_3 (°)",
                "angle_4 (°)", "angle_5 (°)",
                "min_speed (m/s)", "max_speed (m/s)",
                "segment_length (m)",
            ],
            "lower": np.array([0,   0,   0,   0,   0,   10.0, 10.0, 10.0]),
            "upper": np.array([85,  85,  85,  85,  85,  15.0, 30.0, 40.0]),
        }

    def run_simulation(self, params: np.ndarray) -> np.ndarray:
        """
        Calls SimulatorServer POST /simulate for each param row.

        Returns trajectories (N, T, 4): [x, z, xte, steering] per timestep.
        position comes from udacity_sim.py as (pos_x, pos_z, pos_y) — a list,
        not a dict. Index 0=x, 1=z (height), 2=xte, 3=steering.
        T may vary per run — padded to the longest run.

        TODO (Step A): implement HTTP calls to opensbt-core SimulatorServer.
        """
        raise NotImplementedError(
            "LaneKeepingScenario requires Docker (opensbt-core). "
            "Make sure `docker compose up --build` is running in opensbt-core/."
        )

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Composite safety score combining three metrics:
          M1 (0.6): XTE margin — primary failure signal
          M2 (0.2): steering oscillation penalty — instability proxy
          M3 (0.2): early XTE violation penalty — trajectory-to-failure signal

        M2 and M3 require the steering channel (index 3), added in Step A.
        Falls back to M1 only if trajectories have fewer than 4 channels.
        """
        xte = trajectories[:, :, 2]   # (N, T)
        T = xte.shape[1]

        # M1: XTE margin
        max_xte = np.abs(xte).max(axis=1)
        m1 = MAX_XTE - max_xte   # (N,)

        if trajectories.shape[2] < 4:
            return m1

        # M2: steering oscillation penalty — high std = unstable driving
        steering = trajectories[:, :, 3]   # (N, T)
        steer_std = steering.std(axis=1)
        m2 = -np.clip(steer_std / STEER_NORM, 0.0, 1.0)   # (N,) in [-1, 0]

        # M3: early XTE violation penalty
        over_threshold = np.abs(xte) > EARLY_FRAC * MAX_XTE   # (N, T) bool
        first_violation = np.where(
            over_threshold.any(axis=1),
            over_threshold.argmax(axis=1).astype(float),
            float(T),
        )
        m3 = -(T - first_violation) / T   # (N,) in [-1, 0]

        return 0.6 * m1 + 0.2 * m2 + 0.2 * m3

    def failure_threshold(self) -> float:
        return 0.0
