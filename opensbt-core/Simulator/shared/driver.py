"""
Lateral controller shared by every backend.

One file, imported by both Udacity and MetaDrive, so the two run the same
control law.

State schema (SI units, produced by each backend through `road_frame`):

    lateral_error : float -- signed metres from the centreline (+ = left)
    heading_error : float -- signed radians between heading and tangent
    speed         : float -- current speed (m/s)
    target_speed  : float -- desired cruising speed (m/s)
    dt            : float -- seconds elapsed since the previous step

Action: `(steering, throttle)`, both normalised to [-1, 1].

The control law is proportional feedback on the two errors:

    steering = -k_lat * lateral_error - k_head * heading_error
    throttle =  k_throttle * (target_speed - speed)

`dt` is used by everything that integrates over time. The steering rate limit
is expressed in units per SECOND and converted to a per-step delta with `dt`,
and `obs_lag` filters the observations with a time constant in seconds
(`alpha = exp(-dt/tau)`). A backend that does not supply `dt` falls back on
`reference_dt`.

Three optional degradations are applied before the control law when enabled:
`obs_latency` (integer steps of delay on the observed errors), `obs_lag_tau`
(exponential smoothing of the same errors, time constant in seconds), and
`steer_noise` (Gaussian noise added to the steering command). All three are off
by default.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class Driver:
    """Interface: from a state to a normalised action."""

    #: True when the driver needs the camera frame. Lets the backends turn
    #: rendering off when it is not needed -- on Unity it changes the control
    #: rate.
    needs_camera: bool = False

    def reset(self) -> None:
        """Called at the start of every run."""

    def act(self, state: dict) -> Tuple[float, float]:
        raise NotImplementedError


@dataclass
class LateralFeedbackDriver(Driver):
    """
    Proportional feedback on lateral and heading error, plus a speed regulator.

    The steering command before clipping and rate limiting is

        raw = -k_lat * lateral_error - k_head * heading_error

    so a positive `lateral_error` (vehicle to the left of the centreline)
    produces negative steering, i.e. to the right. The throttle command is
    `k_throttle * (target_speed - speed)`. Both are clipped to [-1, 1].

    The class is a linear feedback law: it holds no lookahead point, no
    wheelbase and no path geometry. The ratio `k_head / k_lat` (0.8 / 0.35, so
    about 2.3 m) is the only length scale in it.

    Fields
    ------
    `k_lat`, `k_head`, `k_throttle`
        Gains of the two steering terms and of the throttle term.
    `max_steer_rate`
        Highest |steering delta| per SECOND; the per-step bound is
        `max_steer_rate * dt`. 0 disables the limit.
    `obs_latency`
        Number of steps of delay: `act` uses the state stored `obs_latency`
        steps earlier, kept in a `deque` of length `obs_latency + 1`.
    `obs_lag_tau`
        Time constant in seconds of an EMA applied to the observed lateral and
        heading errors, with `alpha = exp(-dt / obs_lag_tau)`. 0 disables it.
    `steer_noise`
        Standard deviation of the Gaussian noise added to `raw`. 0 disables it.
    `seed`
        Seed of the internal RNG used by `steer_noise`; re-applied on `reset`.
    `reference_dt`
        `dt` used when the state dict carries none, or carries a non-positive
        one.

    State kept between steps: the observation buffer, the EMA state, the
    previous steering command (for the rate limit) and the RNG. `reset` clears
    all four.
    """

    needs_camera: bool = False

    k_lat: float = 0.35          # steering per metre of lateral error
    k_head: float = 0.8          # steering per radian of heading error
    k_throttle: float = 0.3      # throttle per (m/s) of speed error

    #: Highest |steering delta| per SECOND (0 = off). Converted to a per-step
    #: bound as `max_steer_rate * dt`, so the limit is the same at any rate.
    max_steer_rate: float = 4.0

    obs_latency: int = 0
    #: Time constant (seconds) of the EMA applied to the observed errors:
    #: `alpha = exp(-dt / obs_lag_tau)`. 0 = off.
    obs_lag_tau: float = 0.0
    steer_noise: float = 0.0
    seed: Optional[int] = None

    #: `dt` used when the state dict supplies none, or supplies a non-positive
    #: one. Backends normally pass their own `dt` in the state.
    reference_dt: float = 0.05

    _prev_steer: float = field(default=0.0, repr=False)
    _buf: deque = field(default=None, repr=False)
    _rng: object = field(default=None, repr=False)
    _ema: object = field(default=None, repr=False)

    def __post_init__(self):
        self._buf = deque(maxlen=max(1, int(self.obs_latency) + 1))
        self._rng = np.random.default_rng(self.seed)
        self._ema = None

    def reset(self) -> None:
        self._prev_steer = 0.0
        self._buf = deque(maxlen=max(1, int(self.obs_latency) + 1))
        self._ema = None
        # Re-seeding the RNG restarts the same noise sequence for a given seed.
        self._rng = np.random.default_rng(self.seed)

    def act(self, state: dict) -> Tuple[float, float]:
        # Observation latency: acts on the oldest state in the buffer.
        self._buf.append(state)
        used = self._buf[0]

        lateral = float(used.get("lateral_error", 0.0))
        heading = float(used.get("heading_error", 0.0))

        # Time step supplied by the backend; used by the EMA and by the
        # steering rate limit, the two terms that integrate over time.
        dt = float(state.get("dt", self.reference_dt))
        if dt <= 0.0:
            dt = self.reference_dt

        if self.obs_lag_tau and self.obs_lag_tau > 0.0:
            # EMA with a fixed time constant: alpha = exp(-dt / tau).
            a = math.exp(-dt / float(self.obs_lag_tau))
            if self._ema is None:
                self._ema = [lateral, heading]
            else:
                self._ema[0] = a * self._ema[0] + (1.0 - a) * lateral
                self._ema[1] = a * self._ema[1] + (1.0 - a) * heading
            lateral, heading = self._ema[0], self._ema[1]

        # Speed and target speed are read from the current state, not from the
        # delayed one: they do not go through the latency buffer.
        speed = float(state.get("speed", 0.0))
        target = float(state.get("target_speed", 0.0))

        raw = -(self.k_lat * lateral) - (self.k_head * heading)
        if self.steer_noise and self.steer_noise > 0.0:
            raw += float(self._rng.normal(0.0, self.steer_noise))
        steer = _clip(raw, -1.0, 1.0)

        if self.max_steer_rate and self.max_steer_rate > 0.0:
            max_delta = self.max_steer_rate * dt      # units/s -> units/step
            delta = _clip(steer - self._prev_steer, -max_delta, max_delta)
            steer = self._prev_steer + delta
        self._prev_steer = steer

        throttle = _clip(self.k_throttle * (target - speed), -1.0, 1.0)
        return float(steer), float(throttle)


def target_speed(min_speed: float, max_speed: float, speed_scale: float = 1.0) -> float:
    """
    Cruising speed from the scenario's speed band.

    Returns `0.5 * (min_speed + max_speed) * speed_scale`, i.e. the midpoint of
    the band scaled by `speed_scale`.
    """
    return 0.5 * (float(min_speed) + float(max_speed)) * float(speed_scale)
