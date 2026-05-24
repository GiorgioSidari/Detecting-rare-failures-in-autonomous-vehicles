from scipy.stats import truncnorm
import numpy as np
from simulators.base_simulator import BaseSimulator

# --------------------SCENARIO PARAMETERS-----------------

# ---------------BRAKING EFFICIENCY---------------- 
# How close the braking system gets to its theoretical maximum deceleration.
# Braking efficiency: TruncatedNormal(mean=0.97, std=0.02, low=0.85, high=1.0)
BRAKING_EFFICIENCY_MEAN  = 0.97
BRAKING_EFFICIENCY_STD   = 0.02
BRAKING_EFFICIENCY_LOW   = 0.85
BRAKING_EFFICIENCY_HIGH  = 1.0

# ----------------RAW DELAY NOISE----------------
# The time between the detection of the information and the actual responde made by the AV system
DELAY_NOISE_LOG_SIGMA    = 0.1   # log-normal shape parameter
HARDWARE_MIN_DELAY       = 0.05  # minimum physically possible delay (seconds)

# -------------------- MODELING OUR AV SYSTEM -----------------

class EmergencyBrakingSimulator(BaseSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, controllable_parameters: np.ndarray) -> np.ndarray:
        """
        Simulate the emergency braking scenario.
        controllable_parameters: (num_scenarios, 4) -> trajectories: (num_scenarios, num_timesteps, 2)
        """
        # TODO: Implement the physics for emergency braking
        pass

    def ParamBounds(self) -> dict:
        # TODO: Define the actual lower and upper bounds for the 4 controllable parameters
        return {"lower": np.zeros(4), "upper": np.ones(4)}
