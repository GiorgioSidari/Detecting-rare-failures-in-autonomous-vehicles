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

MAX_XTE = 2.5   # metres — from opensbt-core config


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
            "lower": np.array([0,   0,   0,   0,   0,   5.0,  10.0, 10.0]),
            "upper": np.array([85,  85,  85,  85,  85,  15.0, 30.0, 40.0]),
        }

    def run_simulation(self, params: np.ndarray) -> np.ndarray:
        """
        Calls SimulatorServer POST /simulate for each param row.

        Returns trajectories (N, T, 3): [x, y, xte] per timestep.
        T may vary per run — will be padded to the longest run.

        TODO (Blocco 3): implement HTTP calls to opensbt-core API.
        """
        raise NotImplementedError(
            "LaneKeepingScenario requires Docker (opensbt-core). "
            "Make sure `docker compose up --build` is running in opensbt-core/."
        )

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Safety margin = MAX_XTE - max(|xte|) over the run.
        Positive → stayed in lane, Negative → left the road.
        """
        xte = trajectories[:, :, 2]          # (N, T)
        max_xte_per_run = np.abs(xte).max(axis=1)   # (N,)
        return MAX_XTE - max_xte_per_run

    def failure_threshold(self) -> float:
        return 0.0
