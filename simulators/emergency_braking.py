# Noise model — AV-specific distributions
# Braking efficiency: TruncatedNormal(mean=0.97, std=0.02, low=0.85, high=1.0)
#   Justified by: ISO 26262 brake-by-wire tolerance bands
# Delay noise: LogNormal(mu=0, sigma=0.1), zero-mean shifted
#   Justified by: Pendleton et al. (2017), Betz et al. (2022)

from scipy.stats import truncnorm
import numpy as np
from simulators.base_simulator import BaseSimulator

# --------------------SCENARIO PARAMETERS-----------------

# Braking efficiency: how close the system gets to its theoretical max deceleration
BRAKING_EFFICIENCY_MEAN = 0.97
BRAKING_EFFICIENCY_STD  = 0.02
BRAKING_EFFICIENCY_LOW  = 0.85
BRAKING_EFFICIENCY_HIGH = 1.0

# Delay noise: reaction-time jitter from sensor/actuator stack
DELAY_NOISE_LOG_SIGMA = 0.1    # log-normal shape parameter
HARDWARE_MIN_DELAY    = 0.05   # minimum physically possible delay (seconds)

# Precomputed truncnorm clip bounds (standard-normal units)
_BRAKING_A = (BRAKING_EFFICIENCY_LOW  - BRAKING_EFFICIENCY_MEAN) / BRAKING_EFFICIENCY_STD
_BRAKING_B = (BRAKING_EFFICIENCY_HIGH - BRAKING_EFFICIENCY_MEAN) / BRAKING_EFFICIENCY_STD
# Mean of LogNormal(0, sigma) = exp(sigma^2 / 2), used to zero-center delay noise
_LOGNORMAL_MEAN = np.exp(DELAY_NOISE_LOG_SIGMA ** 2 / 2)

class EmergencyBrakingSimulator(BaseSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, controllable_parameters: np.ndarray) -> np.ndarray:
        """
        Simulate N emergency braking scenarios in batch.

        controllable_parameters: (N, 4)
            columns: [initial_speed (m/s), friction_coefficient (-),
                      detection_distance (m), nominal_delay (s)]

        Returns trajectories: (N, T, 2)
            last dim: [position (m), velocity (m/s)]
        """
        N = controllable_parameters.shape[0]
        initial_speeds      = controllable_parameters[:, 0]
        friction_coeffs     = controllable_parameters[:, 1]
        detection_distances = controllable_parameters[:, 2]
        nominal_delays      = controllable_parameters[:, 3]

        # --- Sample stochastic noise for all N trajectories ---
        braking_efficiency = truncnorm.rvs(
            _BRAKING_A, _BRAKING_B,
            loc=BRAKING_EFFICIENCY_MEAN,
            scale=BRAKING_EFFICIENCY_STD,
            size=N,
        )
        raw_delay_noise = np.random.lognormal(mean=0.0, sigma=DELAY_NOISE_LOG_SIGMA, size=N)
        delay_noise = raw_delay_noise - _LOGNORMAL_MEAN  # zero-center

        actual_delays = np.maximum(HARDWARE_MIN_DELAY, nominal_delays + delay_noise)
        actual_decels = friction_coeffs * 9.81 * braking_efficiency

        # --- Time integration (vectorised across N) ---
        trajectories = np.zeros((N, self.T, 2))
        positions    = np.zeros(N)
        velocities   = initial_speeds.copy()
        active       = np.ones(N, dtype=bool)  # trajectories still in motion

        for t_idx in range(self.T):
            t = t_idx * self.dt

            # Record current state before updating
            trajectories[:, t_idx, 0] = positions
            trajectories[:, t_idx, 1] = velocities

            # Delay phase — constant speed, no braking yet
            delay_mask = active & (t < actual_delays)
            positions[delay_mask] += velocities[delay_mask] * self.dt

            # Braking phase — decelerate (update velocity first, then position per spec)
            braking_mask = active & (t >= actual_delays)
            new_v = np.maximum(0.0, velocities[braking_mask] - actual_decels[braking_mask] * self.dt)
            positions[braking_mask] += new_v * self.dt
            velocities[braking_mask] = new_v

            # Deactivate trajectories that have crashed or fully stopped
            active[positions >= detection_distances] = False
            active[velocities == 0.0] = False

            if not np.any(active):
                # Pad all remaining timesteps with the frozen final states
                for t_rem in range(t_idx + 1, self.T):
                    trajectories[:, t_rem, 0] = positions
                    trajectories[:, t_rem, 1] = velocities
                break

        return trajectories

    def ParamBounds(self) -> dict:
        """
        Physical parameter bounds for the 4 controllable inputs.
        Order: [initial_speed, friction_coefficient, detection_distance, nominal_delay]
        """
        return {
            "lower": np.array([5.0,  0.3,  10.0, 0.05]),  # [m/s, -, m, s]
            "upper": np.array([50.0, 1.0, 100.0, 0.50]),
        }
