"""
Cut-In scenario configuration.
"""

from __future__ import annotations

import inspect
import numpy as np
from scenarios.base_scenario import BaseScenario
from scenarios.cut_in.simulator import CutInSimulator
from scenarios.cut_in.qoi import compute_min_gap


class CutInScenario(BaseScenario):
    name = "cut_in"
    description = (
        "A vehicle cuts in front of the ego AV from an adjacent lane. "
        "Tests whether the ego can brake and steer to avoid a collision "
        "under varying speed, lateral gap and reaction-delay conditions."
    )

    def __init__(self, use_nn: bool | str = False):
        """
        use_nn : False   -> ideal physics (CutInSimulator)
                 "carla" -> video-CNN driving from camera frames inside CARLA
                            (Fase 3 del piano; requires CARLA + SimulatorServer
                            running, see opensbt-core/Simulator/cut_in/, and a
                            model trained with train_cnn.py)
        """
        self.use_nn = use_nn
        if use_nn == "carla":
            from scenarios.cut_in.nn_simulator_carla import CutInCarlaNNSimulator
            self._sim = CutInCarlaNNSimulator()
        else:
            self._sim = CutInSimulator()

    def param_bounds(self) -> dict:
        bounds = self._sim.ParamBounds()
        return {
            "names": [
                "ego_speed (m/s)",
                "cutter_speed (m/s)",
                "lateral_gap (m)",
                "reaction_delay (s)",
            ],
            "lower": bounds["lower"],
            "upper": bounds["upper"],
        }

    def run_simulation(self, params: np.ndarray, verbose: bool = False) -> np.ndarray:
        # Only the CARLA simulator's run() accepts verbose (per-job progress);
        # physics runs silently regardless.
        if "verbose" in inspect.signature(self._sim.run).parameters:
            return self._sim.run(params, verbose=verbose)
        return self._sim.run(params)

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        margins = compute_min_gap(trajectories)

        # CARLA path only — see scenarios/emergency_braking/config.py's compute_qoi
        # for why a failed job becomes an invalid (NaN) sample here.
        invalid = getattr(self._sim, "_invalid_mask", None)
        if invalid is not None:
            invalid = np.asarray(invalid, dtype=bool)
            N = len(margins)
            if len(invalid) != N:
                invalid = invalid[-N:]
            margins = np.where(invalid, np.nan, margins)

        return margins

    def failure_threshold(self) -> float:
        return 0.0
