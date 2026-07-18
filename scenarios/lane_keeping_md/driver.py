"""
Lane-keeping drivers for the MetaDrive backend.

The driver is a PLUGGABLE object (dependency-injection): the scenario extracts a
small, backend-neutral state dict from the MetaDrive env and hands it to the
driver, which returns a normalised [steering, throttle] action. Keeping the
control law free of any MetaDrive import makes it fully unit-testable.

State dict (all SI, produced by the scenario from the ego vehicle):
    lateral_error : float  — signed metres from lane centre (+ = left of centre)
    heading_error : float  — signed radians between heading and lane tangent
    speed         : float  — current speed (m/s)
    target_speed  : float  — desired cruising speed (m/s)

Action: (steering, throttle) each in [-1, 1].

A1 (state/feature-based) uses PurePursuitDriver below; no rendering is required,
so this backend runs headless at thousands of steps/s. A camera-vision driver
(A2) can implement the same .act() interface later.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


class Driver:
    """Interface: map a state dict to a normalised (steering, throttle) action."""

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def act(self, state: dict) -> tuple[float, float]:
        raise NotImplementedError


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class PurePursuitDriver(Driver):
    """
    Lateral+heading feedback controller with a speed regulator, plus OPTIONAL,
    physically-motivated degradations that reproduce the real DNN failure mode.

    Steering law (normalised, sign convention: positive steering turns left):
        raw = -k_lat * lateral_error - k_head * heading_error
    A car left of centre (lateral_error > 0) must steer right (negative). A
    `max_rate` limiter caps the change per step to damp the oscillation/
    saturation failure mode (mirrors the Unity LK_STEER_MAX_RATE); set to 0 off.
    Throttle: proportional regulator on (target_speed - speed).

    Degradations (default OFF -> robust controller):
      obs_latency : act on the state from N control steps ago. This reproduces
                    the README's failure mechanism ("the DNN steers on stale,
                    distant observations"): a FIXED step latency becomes a larger
                    SPATIAL staleness as meters-per-step grows (low control rate
                    / high speed), so failures emerge along the envelope.
      steer_noise : gaussian noise (std, in normalised steering units) added to
                    the steering command — a stochastic band around the boundary.
    """
    k_lat: float = 0.35          # steering per metre of lateral error
    k_head: float = 0.8          # steering per radian of heading error
    k_throttle: float = 0.3      # throttle per (m/s) of speed error
    max_rate: float = 0.20       # max |delta steering| per step (0 = off)
    obs_latency: int = 0         # act on the state from N steps ago (0 = off)
    obs_lag: float = 0.0         # CONTINUOUS observation sluggishness (EMA) in [0,1) (0 = off)
    steer_noise: float = 0.0     # std of gaussian steering noise (0 = off)
    seed: int | None = None      # RNG seed for reproducible noise
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

    def act(self, state: dict) -> tuple[float, float]:
        # Observation latency: act on the oldest state within the buffer window.
        self._buf.append(state)
        used = self._buf[0]

        lateral = float(used.get("lateral_error", 0.0))
        heading = float(used.get("heading_error", 0.0))

        # Continuous observation sluggishness: an EMA of the perceived errors.
        # obs_lag -> 0 is instant/robust; higher obs_lag lags the response, so a
        # weak controller fails on the HARD scenarios (sharp curves / high speed)
        # while surviving the easy ones -> a scenario-dependent failure boundary.
        if self.obs_lag and self.obs_lag > 0.0:
            a = float(self.obs_lag)
            if self._ema is None:
                self._ema = [lateral, heading]
            else:
                self._ema[0] = a * self._ema[0] + (1.0 - a) * lateral
                self._ema[1] = a * self._ema[1] + (1.0 - a) * heading
            lateral, heading = self._ema[0], self._ema[1]

        # Throttle regulates on the CURRENT speed (actuation, not perception).
        speed   = float(state.get("speed", 0.0))
        target  = float(state.get("target_speed", 0.0))

        raw = -(self.k_lat * lateral) - (self.k_head * heading)
        if self.steer_noise and self.steer_noise > 0.0:
            raw += float(self._rng.normal(0.0, self.steer_noise))
        steer = _clip(raw, -1.0, 1.0)

        if self.max_rate and self.max_rate > 0.0:
            delta = _clip(steer - self._prev_steer, -self.max_rate, self.max_rate)
            steer = self._prev_steer + delta
        self._prev_steer = steer

        throttle = _clip(self.k_throttle * (target - speed), -1.0, 1.0)
        return steer, throttle
