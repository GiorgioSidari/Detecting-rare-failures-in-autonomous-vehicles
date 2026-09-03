"""
State-based agent for Udacity: drives from telemetry and road geometry rather
than from camera frames.

`predict(obs, state)` ignores the image and computes the two control errors from
the vehicle position in `state` and the run's centreline:

  * the lateral and heading errors come from `shared.road_frame`, which projects
    the position onto the centreline polyline -- the same projection the
    MetaDrive backend uses;
  * the heading itself is estimated from consecutive positions
    (`yaw_from_positions`), because the Udacity telemetry exposes no vehicle
    yaw. It is therefore the heading of the trajectory, which equals the vehicle
    attitude only in the absence of slip. On the first step, with no previous
    position available, the road tangent is used as the initial estimate;
  * the errors are passed to `shared.driver.LateralFeedbackDriver`, which
    returns `(steering, throttle)`.

The agent exposes the same `predict(obs, state)` interface as `SupervisedAgent`,
so `UdacitySimulatorC2` can use either.

`_validate_against_unity` compares the lateral error obtained from the
projection with the `cte` reported by the Unity telemetry and stores the
difference in `last_cte_mismatch`; `_resolve_dt` returns the time step, measured
from the telemetry timestamps when available and falling back to the driver's
`reference_dt` otherwise.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from ..shared.driver import LateralFeedbackDriver, target_speed
from ..shared.road_frame import road_frame, yaw_from_positions
from .config import STEERING_SIGN
from ..lanekeeping.global_log import GlobalLog
from ..lanekeeping.self_driving.agent import Agent


class StateBasedAgent(Agent):

    def __init__(self, env_name: str, min_speed: float = 5.0,
                 max_speed: float = 15.0, speed_scale: float = 1.0,
                 driver: Optional[LateralFeedbackDriver] = None,
                 steering_sign: float = STEERING_SIGN):
        """
        Adapts the shared controller to the Udacity telemetry.

        It exposes the same interface as `SupervisedAgent` -- `predict(obs, state)`
        returning `[[steering, throttle]]` -- so the simulation loop changes as
        little as possible.

        Parameters
        ----------
        speed_scale : operating-point calibration lever: lowered until the backend
                      sits at 10-20% failures under uniform sampling.
        driver      : injectable controller. By default the shared lateral one; it
                      can be a degraded variant of it (obs_latency, obs_lag,
                      steer_noise) to make failure scenario-dependent.
        """
        super().__init__(env_name=env_name)
        self.logger = GlobalLog("state_based_agent")

        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.speed_scale = float(speed_scale)
        # -1.0 when Unity's convention is the opposite of ours. See the note on
        # STEERING_SIGN in c2/config.py.
        self.steering_sign = float(steering_sign)
        self.driver = driver or LateralFeedbackDriver()

        self._road_xy: Optional[np.ndarray] = None
        self._prev_pos: Optional[Tuple[float, float]] = None
        self._last_yaw: float = 0.0
        #: Timestamp of the last call, to measure `dt`.
        #:
        #: On Udacity the control rate is NOT configurable: it depends on load
        #: and on the number of workers (measured: 20.8 Hz with 1 container,
        #: 8.5 Hz with 4). The shared controller expresses the steering limit in
        #: units per second and needs the real `dt` to convert it: without that,
        #: its responsiveness would change with how the campaign is launched.
        self._last_t: Optional[float] = None
        #: True once the vehicle has driven past the end of the road. From then
        #: on the lateral error is not a lateral error any more (see
        #: `RoadFrame.beyond_end`) and the run must be ended, not corrected. The
        #: roads are ~100 m long: at 15 m/s they are over in under 7 s, against
        #: the 30 s of maxTime, so it happens nearly always.
        self.reached_end: bool = False
        #: Sign agreement between our lateral error and Unity's `cte`. It tells
        #: apart two faults that produce the SAME symptom (the car always leaves
        #: the road):
        #:   - signs agree + small mismatch -> the geometry is right, it is the
        #:     STEERING that is inverted: act on LK_STEERING_SIGN;
        #:   - signs disagree, or large mismatch -> the GEOMETRY is wrong
        #:     (position axes, or a different road from the one Unity sees), and
        #:     flipping the steering would fix nothing.
        self.n_sign_agree: int = 0
        self.n_sign_disagree: int = 0
        #: Largest gap observed between our lateral error and Unity's `cte` in
        #: the current run. A validation measurement, not a control one: if it
        #: grows, our idea of where the road is has come apart from the
        #: simulator's.
        self.last_cte_mismatch: float = 0.0

    # -- lifecycle -----------------------------------------------------------

    def setSpeedLimits(self, minSpeed: float, maxSpeed: float) -> None:
        """Same signature as SupervisedAgent, so the loop cannot tell them apart."""
        self.min_speed = float(minSpeed)
        self.max_speed = float(maxSpeed)

    def set_road(self, road_xy: np.ndarray) -> None:
        """
        Sets the run's centreline and clears the state.

        Must be called at every reset: the road changes with every sample, and
        the driver must not inherit the previous run's steering.
        """
        xy = np.asarray(road_xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError(f"expected a centreline (M>=2, 2), got {xy.shape}")
        self._road_xy = xy
        self._prev_pos = None
        self._last_yaw = 0.0
        self.last_cte_mismatch = 0.0
        self.n_sign_agree = 0
        self.n_sign_disagree = 0
        self.reached_end = False
        self._last_t = None
        self.driver.reset()

    # -- control ---------------------------------------------------------------

    def _validate_against_unity(self, cte: float, lateral_error: float) -> None:
        """
        Compare our lateral error with the simulator's, without using it.

        Purely diagnostic: it accumulates the largest mismatch and how often the
        two agree in sign, which is what tells an inverted steering convention
        apart from a wrong geometry. The control law never reads `cte`.
        """
        if cte is None:
            return
        cte = float(cte)
        self.last_cte_mismatch = max(
            self.last_cte_mismatch, abs(abs(cte) - abs(lateral_error)))
        if abs(cte) > 0.05 and abs(lateral_error) > 0.05:
            if cte * lateral_error > 0:
                self.n_sign_agree += 1
            else:
                self.n_sign_disagree += 1

    def _resolve_dt(self, supplied) -> float:
        """
        Seconds since the previous decision, from the caller or the clock.

        A supplied `dt` is preferred: the loop knows how much time really passed
        between two commands applied to the vehicle. Measuring it here works in
        production, where the loop is paced by Unity, but NOT on a test bench
        where calls follow each other in microseconds -- there the steering rate
        limiter would choke and the car could not correct. The clamp protects
        against abnormal pauses (GC, CPU contention between containers).
        """
        if supplied is None:
            now = time.perf_counter()
            supplied = ((now - self._last_t) if self._last_t is not None
                        else self.driver.reference_dt)
            self._last_t = now
        return min(max(float(supplied), 1e-3), 1.0)

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        """
        `obs` is ignored: this arm does not use perception. It stays in the
        signature to remain interchangeable with `SupervisedAgent`.

        `state` must contain `pos` -- the position from the telemetry -- besides
        `speed` (km/h, as Unity exposes it). `cte`, when present, is used only
        for validation.
        """
        if self._road_xy is None:
            raise RuntimeError(
                "set_road() was not called: the agent does not know where the road is")

        pos = state.get("pos")
        if pos is None:
            raise RuntimeError("state['pos'] missing: the vehicle position is required")
        # Unity returns (x, z, y) with the axes swapped (see the comment in
        # udacity_sim.observe): the road plane is (pos[0], pos[1]).
        px, py = float(pos[0]), float(pos[1])

        # Heading from the trajectory. At the first step, or when stopped, it
        # falls back to the road tangent: the car is spawned aligned with the
        # centreline.
        if self._prev_pos is None:
            yaw = road_frame(self._road_xy, px, py, 0.0).tangent_yaw
        else:
            yaw = yaw_from_positions(self._prev_pos, (px, py), fallback=self._last_yaw)
        self._last_yaw = yaw
        self._prev_pos = (px, py)

        frame = road_frame(self._road_xy, px, py, yaw)

        # End of road reached: from here the lateral error is no longer one. The
        # loop is signalled and must end the run; meanwhile the agent holds
        # straight instead of chasing a target that does not exist.
        if frame.beyond_end:
            self.reached_end = True
            return np.asarray([[0.0, 0.0]], dtype=np.float32)

        self._validate_against_unity(state.get("cte"), frame.lateral_error)

        dt = self._resolve_dt(state.get("dt"))

        speed_kmh = float(state.get("speed", 0.0) or 0.0)
        steering, throttle = self.driver.act({
            "lateral_error": frame.lateral_error,
            "heading_error": frame.heading_error,
            "speed": speed_kmh / 3.6,          # Unity telemetry is in km/h
            "target_speed": target_speed(self.min_speed, self.max_speed,
                                         self.speed_scale),
            "dt": dt,
        })

        # Unity has no separate braking channel: a negative throttle would be
        # read as reverse.
        return np.asarray([[self.steering_sign * steering, max(0.0, throttle)]],
                          dtype=np.float32)
