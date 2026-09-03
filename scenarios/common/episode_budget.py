"""
Episode horizon derived from the scenario.

:func:`budget_seconds` returns

    clamp( length / target_speed * MARGIN, MIN_SECONDS, MAX_SECONDS )

with `MARGIN = 1.5`, `MIN_SECONDS = 10` and `MAX_SECONDS = 120`, i.e. the time
needed to cover `length` at `target_speed`, scaled by the margin and clamped.
The margin covers the difference between the target speed and the speed
actually held, which on the measured designs is about 79% of nominal.

The budget is expressed in simulated seconds, which is the unit shared by
backends running at different control rates. :func:`budget_steps` converts it to
a number of steps for a given `control_hz`, rounding up.

The budget is the cap for a run that never reaches the end of the road; a run
that completes the road terminates on that condition first.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # numpy is imported lazily below, so the
    import numpy as np                 # module stays importable without it

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


def polyline_length(xy: np.ndarray) -> float:
    """Arc length of a polyline `(M, 2)`."""
    import numpy as np

    p = np.asarray(xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError(f"expected a polyline (M>=2, 2), got {p.shape}")
    d = np.diff(p, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())
