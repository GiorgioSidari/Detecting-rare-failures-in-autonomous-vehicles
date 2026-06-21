"""
Cut-In scenario configuration.

TODO (Blocco 2): switch use_nn=True after training.
"""

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

    def __init__(self, use_nn: bool = False):
        # TODO (Blocco 2c): when use_nn=True, use CutInNNSimulator
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

    def run_simulation(self, params: np.ndarray) -> np.ndarray:
        return self._sim.run(params)

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        return compute_min_gap(trajectories)

    def failure_threshold(self) -> float:
        return 0.0
