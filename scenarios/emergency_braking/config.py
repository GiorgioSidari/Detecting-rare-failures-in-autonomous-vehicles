"""
Emergency Braking scenario configuration.

Wires together the simulator (physics or NN), the QoI and the metadata
that the frontend needs to render the parameter form.

BLOCCO 1 — implementation order:
  1. Use EmergencyBrakingSimulator (physics, already exists) → validate pipeline
  2. Train MLP controller (train.py) → switch to NNSimulator
"""

import os
import numpy as np
from scenarios.base_scenario import BaseScenario
from simulators.emergency_braking import EmergencyBrakingSimulator
from evaluation.qoi import compute_safety_margin, failure_indicator

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "emergency_braking_mlp.keras"
)


class EmergencyBrakingScenario(BaseScenario):
    name = "emergency_braking"
    description = (
        "A vehicle detects an obstacle ahead and performs an emergency stop. "
        "Tests whether the braking system can bring the vehicle to a halt "
        "before impact under varying speed, friction and sensor-delay conditions."
    )

    def __init__(self, use_nn: bool = False, model_path: str = _DEFAULT_MODEL_PATH):
        """
        Parameters
        ----------
        use_nn     : if True, use the trained MLP controller instead of ideal physics.
                     Requires `python -m scenarios.emergency_braking.train` to have run first.
        model_path : path to the saved Keras model (only used when use_nn=True).
        """
        self.use_nn = use_nn
        if use_nn:
            from scenarios.emergency_braking.nn_simulator import EmergencyBrakingNNSimulator
            self._sim = EmergencyBrakingNNSimulator(model_path=model_path)
        else:
            self._sim = EmergencyBrakingSimulator()

    def param_bounds(self) -> dict:
        bounds = self._sim.ParamBounds()
        return {
            "names": [
                "initial_speed (m/s)",
                "friction_coefficient",
                "detection_distance (m)",
                "nominal_delay (s)",
            ],
            "lower": bounds["lower"],
            "upper": bounds["upper"],
        }

    def run_simulation(self, params: np.ndarray) -> np.ndarray:
        return self._sim.run(params)

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        detection_distances = params[:, 2]
        return compute_safety_margin(trajectories, detection_distance=detection_distances)

    def failure_threshold(self) -> float:
        return 0.0
