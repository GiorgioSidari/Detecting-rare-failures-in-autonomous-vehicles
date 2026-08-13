"""
How long an episode may last: a per-scenario budget, not a global constant.

The problem
-----------
With a fixed `max_steps` the horizon is a constant in STEPS, but what matters
for lane keeping is how much ROAD is covered -- and that depends on the
scenario's speed and on the backend's control rate. Measured on the 60-point LHS
design:

    backend      scale   v (m/s)  horizon   covered   coverage
    MetaDrive    0.3625    5.40      30 s      162 m      ~80%
    Udacity      0.4200    6.25      30 s      187 m      ~92%

and those are optimistic estimates: they assume the target speed all the way.
Measuring directly at scale=0.15 gives 26% where the arithmetic predicted 33,
because the car accelerates gradually and slows down in curves.

Three consequences:

  * the two backends drive DIFFERENT PORTIONS of the same road, so the
    comparison stays confounded even once the geometry is identical;
  * `angle_5` is almost never reached, and `angle_4` only partly -- which alone
    explains why the diagnostics attributed to it 33-43% of the influence of the
    strongest angle: it is not less important, it is less driven;
  * the failure rate is measured on a truncated road, so it underestimates the
    rate on the full track.

The rule
--------
The budget is derived from the scenario: time to cover the road at the target
speed, multiplied by a margin, and capped from above.

The margin is needed because the target speed is a target, not a fact: between
the initial acceleration and slowing in curves the car covers about 79% of the
nominal distance (26% measured against 33% predicted). That calls for
1/0.79 ~ 1.27; we use 1.5 so as not to truncate precisely the curviest
scenarios, which are the interesting ones.

The cap keeps a very slow scenario from generating endless episodes: at that
point the run should be declared uninformative, not made to last forever.

The budget does NOT replace termination on road completion: it is the safety net
for the case where the car does not reach the end. The normal exit must be "I
finished the road", otherwise the horizon goes back to being a hidden
variable.
"""
from __future__ import annotations

import math

# Margin on the nominal time (see above: measured ~1.27, rounded up).
MARGIN = 1.5

# Bounds in SIMULATED seconds. The lower one keeps a very short scenario from
# ending before the controller has settled; the upper one keeps the cost of the
# campaign under control.
MIN_SECONDS = 10.0
MAX_SECONDS = 120.0


def budget_seconds(length_m: float, target_speed_ms: float, *,
                   margin: float = MARGIN,
                   lowest: float = MIN_SECONDS,
                   highest: float = MAX_SECONDS) -> float:
    """
    Simulated seconds granted to a scenario of `length_m` at `target_speed`.

    The value is in seconds and not in steps on purpose: it is the only unit
    comparable across backends running at different control rates (Udacity
    ~19.5 Hz, MetaDrive exactly 10 Hz). Converting to steps is the backend's
    job, since it knows its own rate.
    """
    L = float(length_m)
    v = float(target_speed_ms)
    if not (L > 0.0) or not (v > 0.0):
        raise ValueError(f"length={L} and speed={v} must both be positive")
    return float(min(max(L / v * float(margin), float(lowest)), float(highest)))


def budget_steps(length_m: float, target_speed_ms: float, control_hz: float,
                 **kw) -> int:
    """
    The same budget expressed in the steps of a backend running at `control_hz`.

    Rounded up: truncating would mean stopping just short of the end of the road
    in exactly the worst scenarios.
    """
    hz = float(control_hz)
    if not (hz > 0.0):
        raise ValueError(f"control_hz={hz} must be positive")
    return int(math.ceil(budget_seconds(length_m, target_speed_ms, **kw) * hz))


def polyline_length(xy) -> float:
    """Arc length of a polyline `(M, 2)`."""
    import numpy as np

    p = np.asarray(xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError(f"expected a polyline (M>=2, 2), got {p.shape}")
    d = np.diff(p, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())
