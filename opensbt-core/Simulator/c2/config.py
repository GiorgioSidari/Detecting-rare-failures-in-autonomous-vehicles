"""
C2 arm configuration -- its own variables, not added to `lanekeeping/config.py`.

Every constant C2 shares with the Udacity pipeline is **imported** from there,
never copied: if `lanekeeping/config.py` changes, C2 follows. Only the variables
C2 adds live here.
"""
from __future__ import annotations

import os

# Constants shared with the Udacity pipeline: imported, never duplicated.
from ..lanekeeping.config import (  # noqa: F401
    CAP_XTE,
    MAX_XTE,
    UDACITY_SIM_NAME,
)

# ─────────────────────────────────────────────────────────────────────────────
# C2-specific variables
# ─────────────────────────────────────────────────────────────────────────────

#: Operating-point calibration: scales the target speed until the backend sits
#: at 10-20% failures under uniform sampling. A backend at 0% or at 100% carries
#: no information and the boundary is not learnable. The chosen value must be
#: DECLARED in the report: it is a parameter of the experiment.
SPEED_SCALE = float(os.environ.get("LK_SPEED_SCALE", "1.0"))

#: Steering sign on Unity.
#:
#: Unity's convention is undocumented and was not verifiable offline. Symptom of
#: an inverted sign: the car **goes along with the drift** instead of correcting
#: it, and every run diverges monotonically to |XTE| = maxXTE within 1-2 seconds
#: whatever the road.
#:
#: Measured in practice: with +1 the pipeline reported "steers the wrong way";
#: with -1 the failure mode became "unstable oscillation", i.e. a closed loop
#: with the right sign but under-damped. So -1 is the correct value for Unity.
STEERING_SIGN = float(os.environ.get("LK_STEERING_SIGN", "-1.0"))

#: Default speed band, overwritten at every run by `setSpeedLimits`.
MIN_SPEED = float(os.environ.get("LK_MIN_SPEED", "5.0"))
MAX_SPEED = float(os.environ.get("LK_MAX_SPEED", "15.0"))

#: Controller degradations (default OFF = robust controller). They make failure
#: scenario-dependent: on exact state this control law never errs, and an arm
#: that never fails has no boundary to learn. See `scenarios/common/driver.py`.
OBS_LATENCY = int(os.environ.get("LK_OBS_LATENCY", "0"))
OBS_LAG_TAU = float(os.environ.get("LK_OBS_LAG_TAU", "0.0"))
STEER_NOISE = float(os.environ.get("LK_STEER_NOISE", "0.0"))
SEED_ENV = os.environ.get("LK_DRIVER_SEED")
DRIVER_SEED = int(SEED_ENV) if SEED_ENV not in (None, "") else None
