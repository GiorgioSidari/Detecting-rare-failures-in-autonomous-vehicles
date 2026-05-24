from abc import ABC, abstractmethod
import numpy as np

class BaseSimulator(ABC):
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
