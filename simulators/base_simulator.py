from abc import ABC, abstractmethod
import numpy as np

class BaseSimulator(ABC):
    def __init__(self, initial_speed: float = 30.0, friction_coefficient: float = 0.8,
                 detection_distance: float = 50.0, nominal_delay: float = 0.1):
        """
        Initializes the base parameters of the AV System common to all scenarios.
        These parameters will be inherited by all child simulators.
        """
        self.initial_speed = initial_speed
        self.friction_coefficient = friction_coefficient
        self.detection_distance = detection_distance
        self.nominal_delay = nominal_delay

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
