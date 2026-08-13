"""
Controller shared across the backends -- the C2 arm of the comparison protocol.

One file, used by Udacity and MetaDrive. This is not reuse for convenience: it
is the condition for C2 to mean anything. If each backend had its own copy of
the controller, the comparison would also measure the difference between the
copies, and nobody would notice.

State schema (SI units, produced by each backend through `road_frame`):

    lateral_error : float -- signed metres from the centreline (+ = left)
    heading_error : float -- signed radians between heading and tangent
    speed         : float -- current speed (m/s)
    target_speed  : float -- desired cruising speed (m/s)
    dt            : float -- seconds elapsed since the previous step

Action: `(steering, throttle)`, both normalised.

Why `dt` is mandatory
---------------------
The steering rate limiter used to be expressed **per step**. At a different
control rate, the same value gives a different steering authority:

    0.20/step @ 20.8 Hz  ->  4.2 units/s     (Udacity, 1 worker)
    0.20/step @  8.5 Hz  ->  1.7 units/s     (Udacity, 4 workers)
    0.20/step @ 10.0 Hz  ->  2.0 units/s     (MetaDrive)

That is: three controllers with different responsiveness, all claiming to be the
same one. On Udacity the rate even depends on the number of workers, so the
"controller" changed with how the campaign was launched -- and the difference
would have been charged to the simulator.

The limit is now in **units per second** and is converted with the step's `dt`.
The same holds for `obs_lag`, whose time constant was also per step. The backend
must supply `dt`: exact where the rate is configurable (MetaDrive), measured
where it varies (Udacity).

The parameters and the control law are taken from the MetaDrive branch's
`scenarios/lane_keeping_md/driver.py`, which is the implementation already in
use: adopting it rather than rewriting it avoids introducing a difference in
exactly the component that must stay constant.

No simulator imports. Only numpy.
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
    Proportional feedback on lateral and heading error, with a speed regulator.

        raw = -k_lat * lateral_error - k_head * heading_error

    Signs: `lateral_error > 0` (vehicle to the left) must produce negative
    steering, i.e. to the right.

    What this is, and what it is not
    --------------------------------
    This is NOT geometric pure pursuit. Pure pursuit picks a lookahead point on
    the path at distance `l_d` and steers along the arc that reaches it:

        delta = arctan(2 * L * sin(alpha) / l_d)

    There is no lookahead point here, no wheelbase and no arc -- only a linear
    combination of the two errors. The class was called `PurePursuitDriver` for
    a while, which claimed more than it did.

    The two are related, and the relation is worth knowing because it is the
    honest way to describe this controller. Linearising pure pursuit for small
    angles gives

        delta ~= -(2L / l_d^2) * lateral_error - (2L / l_d) * heading_error

    which is exactly the form above. The ratio of the gains therefore carries an
    **effective lookahead** of `k_head / k_lat` = 0.8 / 0.35 ~= 2.3 m. The
    absolute gains, however, are tuned rather than derived from the vehicle
    geometry, so the correspondence is one of form, not of derivation.

    None of this matters for the experiment's validity: what the C2 arm needs is
    not a good controller but *the same* controller on both simulators, which is
    what makes everything else attributable to the backend.

    Degradations, all OFF by default
    --------------------------------
    They exist to make the controller *fallible in a scenario-dependent way*. On
    exact state this law never errs, and an arm that never fails has no boundary
    to learn: the comparison between search methods becomes empty. With the
    degradations active, failure emerges on tight curves at high speed, which is
    the behaviour we want to study.

      `obs_latency` : acts on the state from N steps ago. Reproduces the failure
                      mechanism of interest -- the controller steers on stale
                      observations -- and since a latency fixed in STEPS becomes
                      a staleness growing in METRES as `meters_per_step` grows,
                      the failures line up along the envelope.
      `obs_lag`     : the continuous version of the same thing (EMA on the
                      perceived errors), useful to sweep difficulty smoothly
                      instead of in integer jumps.
      `steer_noise` : Gaussian noise on the steering, for a stochastic band
                      around the boundary instead of a sharp threshold.
    """

    needs_camera: bool = False

    k_lat: float = 0.35          # steering per metre of lateral error
    k_head: float = 0.8          # steering per radian of heading error
    k_throttle: float = 0.3      # throttle per (m/s) of speed error

    #: |steering delta| highest per SECOND (0 = off). The default 4.0 reproduces
    #: the historical 0.20/step at the 20 Hz reference rate, but the steering
    #: authority is now the same on every backend whatever its rate.
    max_steer_rate: float = 4.0

    obs_latency: int = 0
    #: Time constant (seconds) of the observation sluggishness. It used to be a
    #: per-step EMA coefficient, and so also rate-dependent. 0 = off.
    obs_lag_tau: float = 0.0
    steer_noise: float = 0.0
    seed: Optional[int] = None

    #: `dt` used when the state does not supply one. Not an innocuous default:
    #: it is the rate the controller was historically tuned at. The backend MUST
    #: pass `dt`; this value only exists so that ad-hoc calls do not break.
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
        # The RNG is re-seeded: two runs with the same seed must give the same
        # noise sequence, otherwise the campaign is not reproducible.
        self._rng = np.random.default_rng(self.seed)

    def act(self, state: dict) -> Tuple[float, float]:
        # Observation latency: acts on the oldest state in the buffer.
        self._buf.append(state)
        used = self._buf[0]

        lateral = float(used.get("lateral_error", 0.0))
        heading = float(used.get("heading_error", 0.0))

        # Time step: exact on backends with a configurable rate, measured on
        # Udacity where it depends on load. Everything that integrates over time
        # goes through here, so the controller is the same at different rates.
        dt = float(state.get("dt", self.reference_dt))
        if dt <= 0.0:
            dt = self.reference_dt

        if self.obs_lag_tau and self.obs_lag_tau > 0.0:
            # Fixed-time-constant EMA: alpha = exp(-dt/tau) instead of a
            # per-step coefficient, which would respond differently at
            # different rates.
            a = math.exp(-dt / float(self.obs_lag_tau))
            if self._ema is None:
                self._ema = [lateral, heading]
            else:
                self._ema[0] = a * self._ema[0] + (1.0 - a) * lateral
                self._ema[1] = a * self._ema[1] + (1.0 - a) * heading
            lateral, heading = self._ema[0], self._ema[1]

        # The throttle regulates on the CURRENT speed: it is actuation, not
        # perception, so it does not go through the latency buffer.
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

    `speed_scale` is the calibration lever of the operating point: it is lowered
    until the backend sits at 10-20% failures under uniform sampling. A backend
    at 0% or at 100% carries no information, and the boundary is not learnable.
    """
    return 0.5 * (float(min_speed) + float(max_speed)) * float(speed_scale)
