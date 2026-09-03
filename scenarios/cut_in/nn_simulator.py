"""
Cut-In simulator driven by the trained LSTM controller.

TODO (Blocco 2b): implement after EvasiveLSTM is trained.
"""

import numpy as np
from simulators.base_simulator import BaseSimulator
from scenarios.cut_in.nn_controller import EvasiveLSTM


class CutInNNSimulator(BaseSimulator):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.controller = EvasiveLSTM(model_path=model_path)

    def run(self, controllable_parameters: np.ndarray) -> np.ndarray:
        """
        Same signature as CutInSimulator.run():
            params (N, 4) → trajectories (N, T, 2)

        At each timestep, feeds the last SEQUENCE_LEN states to the LSTM
        and uses its output as the ego's action.

        TODO (Blocco 2b).
        """
        raise NotImplementedError

    def ParamBounds(self) -> dict:
        return {
            "lower": np.array([10.0, 10.0, 1.0, 0.05]),
            "upper": np.array([40.0, 40.0, 4.0, 0.80]),
        }
