from abc import ABC, abstractmethod
import numpy as np

class BaseSimulator(ABC):
    def __init__(self, dt: float = 0.01, t_max: float = 10.0):
        """
        Simulation config shared by all scenarios.
        Scenario parameters (speed, friction, etc.) vary per sample and are
        passed as a batch array to run(), not stored on the instance.
        """
        self.dt = dt
        self.t_max = t_max
        self.T = int(t_max / dt)

    @abstractmethod
    def run(self, controllable_parameters: np.ndarray) -> np.ndarray:
        """
        Generates trajectories based on input parameters.
        controllable_parameters: (num_scenarios, 4) -> trajectories: (num_scenarios, num_timesteps, 2)
        """
        pass

    @abstractmethod
    def ParamBounds(self) -> dict:
        """Returns dict with 'lower' and 'upper' arrays of shape (4,)"""
        pass
