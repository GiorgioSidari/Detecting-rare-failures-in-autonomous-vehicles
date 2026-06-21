"""
LSTM controller for the Cut-In scenario.

Uses a recurrent architecture because the correct action depends on how
the cutter's trajectory has evolved over the last k timesteps, not just
the current snapshot.

Architecture
------------
Input  : sequence of k states, each = [ego_vel, long_gap, lat_gap]  — shape (k, 3)
LSTM   : 64 units
Output : [braking_force, steering_correction]  — shape (2,)
         braking_force      in [0, 1]
         steering_correction in [-1, 1]  (positive = steer away from cutter)

TODO (Blocco 2b): implement build(), predict(), load(), save().
"""

import numpy as np

SEQUENCE_LEN = 20   # number of past timesteps fed to the LSTM


class EvasiveLSTM:
    """Lightweight LSTM wrapper — framework-agnostic interface."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path is not None:
            self.load(model_path)

    def predict(self, state_sequence: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        state_sequence : (N, k, 3)

        Returns
        -------
        actions : (N, 2)  — [braking_force, steering_correction]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        raise NotImplementedError

    def build(self) -> None:
        """
        Build the Keras LSTM model.
        Input  : (SEQUENCE_LEN, 3)
        LSTM   : 64 units
        Dense  : 32, relu
        Output : Dense(2, tanh) then clipped to correct ranges
        """
        raise NotImplementedError

    def load(self, model_path: str) -> None:
        raise NotImplementedError

    def save(self, model_path: str) -> None:
        raise NotImplementedError
