"""
Components shared across the simulation backends.

Holds the parts that **must** be identical for the cross-simulator comparison to
mean anything:

  * `driver`        -- the state-based controller of the C2 arm
  * `road_frame`    -- lateral and heading error with respect to the centreline
  * `road_geometry` -- the single road generator
  * `episode_budget`-- how long an episode may last, derived from the scenario

They are all pure (numpy only) and import no simulator, so they can be tested
without Docker, without a GPU and without MetaDrive installed.
"""

from scenarios.common.driver import Driver, LateralFeedbackDriver, target_speed
from scenarios.common.road_frame import RoadFrame, road_frame, yaw_from_positions

__all__ = [
    "Driver",
    "LateralFeedbackDriver",
    "target_speed",
    "RoadFrame",
    "road_frame",
    "yaw_from_positions",
]
