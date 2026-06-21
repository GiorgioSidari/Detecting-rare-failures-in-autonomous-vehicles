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

TODO (Blocco 2a): implement run() physics.
"""

import numpy as np
from simulators.base_simulator import BaseSimulator

IMPACT_THRESHOLD = 1.5   # metres — minimum safe bumper-to-bumper distance
LANE_WIDTH       = 3.5   # metres
G                = 9.81


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

        TODO (Blocco 2a):
          For each timestep:
            1. Move cutter laterally along a sigmoid trajectory until it is
               centred in the ego lane (lateral_gap → 0).
            2. Ego travels at ego_speed until cutter enters lane
               (lateral_gap < LANE_WIDTH/2), then brakes after reaction_delay.
            3. Record [longitudinal_gap, lateral_gap] at each step.
        """
        raise NotImplementedError("CutInSimulator.run() not yet implemented.")

    def ParamBounds(self) -> dict:
        return {
            "lower": np.array([10.0, 10.0,  1.0, 0.05]),   # [m/s, m/s, m, s]
            "upper": np.array([40.0, 40.0,  4.0, 0.80]),
        }
