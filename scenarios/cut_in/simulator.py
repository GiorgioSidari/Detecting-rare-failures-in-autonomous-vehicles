"""
Cut-In scenario — physics simulator (optimal controller).

Two vehicles moving on a straight road:
  - ego   : the AV under test, travelling at constant speed until the cutter
             enters its lane, then braking at maximum deceleration.
  - cutter: a vehicle that changes lane in front of the ego.

State recorded per timestep (D=2):
    [longitudinal_gap (m), lateral_gap (m)]
    between the front bumper of the ego and the rear bumper of the cutter.

Failure: longitudinal_gap < IMPACT_THRESHOLD while lateral_gap < LANE_WIDTH/2.

Simplifying assumptions (fixed, not sampled — the 4 controllable parameters
are ego_speed, cutter_speed, lateral_gap, reaction_delay):
  - The cutter starts INITIAL_LONG_GAP ahead of the ego (bumper-to-bumper) in
    the adjacent lane, already travelling — a cut-in is only physically
    possible once the cutter is slightly ahead.
  - The lateral maneuver follows a smooth sigmoid from lateral_gap (at t=0)
    to 0, centred at CUTIN_T0 with steepness CUTIN_STEEPNESS.
  - The cutter's speed is constant (it doesn't brake); only the ego reacts.
  - The ego brakes at a fixed MAX_DECEL, reaction_delay seconds after the
    cutter first crosses into the ego's lane (lateral_gap < LANE_WIDTH/2).
"""

import numpy as np
from simulators.base_simulator import BaseSimulator

IMPACT_THRESHOLD = 1.5   # metres — minimum safe bumper-to-bumper distance
LANE_WIDTH       = 3.5   # metres
G                = 9.81

INITIAL_LONG_GAP  = 15.0   # metres — ego-front-to-cutter-rear gap at t=0
CUTIN_T0          = 1.5    # seconds — time at which the lateral maneuver is centred
CUTIN_STEEPNESS   = 2.5    # 1/s — steepness of the lateral sigmoid
MAX_DECEL         = 0.8 * G  # m/s^2 — ego's max braking deceleration once triggered


class CutInSimulator(BaseSimulator):
    """
    Optimal-controller cut-in simulator.

    Parameters (d=4)
    ----------------
    ego_speed        (m/s)  : initial speed of the ego vehicle
    cutter_speed     (m/s)  : speed of the cutting vehicle
    lateral_gap      (m)    : lateral distance between vehicles at cut-in start
    reaction_delay   (s)    : ego's braking reaction delay after cutter enters lane
    """

    def run(self, controllable_parameters: np.ndarray) -> np.ndarray:
        """
        params (N, 4) → trajectories (N, T, 2)

        For each timestep:
          1. Move cutter laterally along a sigmoid trajectory until it is
             centred in the ego lane (lateral_gap → 0).
          2. Ego travels at ego_speed until cutter enters lane
             (lateral_gap < LANE_WIDTH/2), then brakes after reaction_delay.
          3. Record [longitudinal_gap, lateral_gap] at each step.
        """
        N = controllable_parameters.shape[0]
        ego_speeds      = controllable_parameters[:, 0]
        cutter_speeds   = controllable_parameters[:, 1]
        lateral_gaps0   = controllable_parameters[:, 2]
        reaction_delays = controllable_parameters[:, 3]

        trajectories = np.zeros((N, self.T, 2))
        long_gap = np.full(N, INITIAL_LONG_GAP)
        ego_vel  = ego_speeds.copy()

        triggered    = np.zeros(N, dtype=bool)   # cutter has entered the ego's lane
        trigger_time = np.full(N, np.inf)
        braking      = np.zeros(N, dtype=bool)   # ego has started braking

        for t_idx in range(self.T):
            t = t_idx * self.dt

            # 1. Lateral maneuver: decreasing sigmoid from lateral_gap0 to 0.
            lateral_gap = lateral_gaps0 / (1.0 + np.exp(CUTIN_STEEPNESS * (t - CUTIN_T0)))

            # First time crossing into the ego's lane.
            newly_triggered = (~triggered) & (lateral_gap < LANE_WIDTH / 2.0)
            trigger_time = np.where(newly_triggered, t, trigger_time)
            triggered = triggered | newly_triggered

            # Ego starts braking reaction_delay seconds after being triggered.
            braking = braking | (triggered & (t >= trigger_time + reaction_delays))

            # Record current state.
            trajectories[:, t_idx, 0] = long_gap
            trajectories[:, t_idx, 1] = lateral_gap

            # 2. Update ego speed (brake if triggered) and longitudinal gap.
            ego_vel = np.where(braking, np.maximum(0.0, ego_vel - MAX_DECEL * self.dt), ego_vel)
            long_gap = long_gap + (cutter_speeds - ego_vel) * self.dt

        return trajectories

    def ParamBounds(self) -> dict:
        return {
            "lower": np.array([10.0, 10.0,  1.0, 0.05]),   # [m/s, m/s, m, s]
            "upper": np.array([40.0, 40.0,  4.0, 0.80]),
        }
