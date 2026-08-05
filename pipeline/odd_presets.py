"""
ODD (Operational Design Domain) presets and narrowing helpers.

Every runner script needs the same thing: take the scenario's default parameter
bounds and optionally shrink them toward the regime where failures are RARE
(gentle curves, moderate speeds), because that is the regime where the whole
question — "does stratified sampling find the rare failure region?" — is
actually interesting. On the wide default ODD failures are common and every
method finds them.

The narrowing rules mirror the ones already used by ``scripts/run_active_boundary.py``
and ``scripts/run_rare_event.py``; they live here so the new scripts do not
duplicate them a third time.

Lane-keeping parameter layout (9 dimensions):
    0-4 : road angles 1..5 (deg)
    5   : min_speed (m/s)
    6   : max_speed (m/s)
    7   : segment_length (m)
    8   : map_size (m)
"""
from __future__ import annotations

import numpy as np

# "realistic" preset, identical to scripts/run_rare_event.py.
REALISTIC_LOWER = [0, 0, 0, 0, 0, 5.0, 9.0, 20.0, 200.0]
REALISTIC_UPPER = [45, 45, 45, 45, 45, 8.0, 14.0, 40.0, 350.0]

# Column indices of the lane-keeping parameter vector.
IDX_ANGLES = range(5)
IDX_MIN_SPEED = 5
IDX_MAX_SPEED = 6
IDX_SEGMENT = 7


def narrow_bounds(lower, upper, *, max_angle=None, max_speed=None,
                  min_speed=None, max_seg=None, min_seg=None):
    """
    Shrink the ODD toward the rare-failure regime.

    Returns ``(lower, upper, narrowed)``; ``narrowed`` is False when no option
    was given, so callers can pass None downstream and keep scenario defaults.

    Each cap also lowers the matching lower bound where needed: capping
    ``max_speed`` at 8 m/s while the default band is [10, 30] would otherwise
    invert the interval.
    """
    lower = np.asarray(lower, float).copy()
    upper = np.asarray(upper, float).copy()
    narrowed = False

    if max_angle is not None:
        for k in IDX_ANGLES:
            upper[k] = max_angle
            lower[k] = min(lower[k], upper[k])
        narrowed = True

    if max_speed is not None:
        X = float(max_speed)
        upper[IDX_MAX_SPEED] = X
        lower[IDX_MAX_SPEED] = max(1.0, min(lower[IDX_MAX_SPEED], X - 1.0))
        upper[IDX_MIN_SPEED] = min(upper[IDX_MIN_SPEED], X)
        lower[IDX_MIN_SPEED] = max(1.0, min(lower[IDX_MIN_SPEED],
                                            upper[IDX_MIN_SPEED] - 1.0))
        narrowed = True

    if min_speed is not None:
        Y = float(min_speed)
        upper[IDX_MIN_SPEED] = Y
        lower[IDX_MIN_SPEED] = max(0.5, min(lower[IDX_MIN_SPEED], Y - 1.0))
        narrowed = True

    if max_seg is not None:
        upper[IDX_SEGMENT] = float(max_seg)
        lower[IDX_SEGMENT] = min(lower[IDX_SEGMENT], upper[IDX_SEGMENT] - 1.0)
        narrowed = True

    if min_seg is not None:
        lower[IDX_SEGMENT] = float(min_seg)
        upper[IDX_SEGMENT] = max(upper[IDX_SEGMENT], lower[IDX_SEGMENT] + 1.0)
        narrowed = True

    return lower, upper, narrowed


def add_odd_args(ap) -> None:
    """Attach the standard ODD flags to an argparse parser."""
    ap.add_argument("--preset", choices=["full", "realistic"], default="full",
                    help="'full' = scenario defaults (wide); "
                         "'realistic' = narrow operational band")
    ap.add_argument("--max-angle", type=float, default=None,
                    help="cap all 5 road angles (deg) — gentler curves")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="cap max_speed (m/s)")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="cap min_speed (m/s)")
    ap.add_argument("--max-seg", type=float, default=None,
                    help="cap segment_length (m)")
    ap.add_argument("--min-seg", type=float, default=None,
                    help="raise the segment_length lower bound (m)")


def resolve_bounds(scenario, args):
    """
    Turn parsed ODD flags into ``(param_lower, param_upper)``.

    Returns ``(None, None)`` when the scenario defaults should be used, which is
    what the pipeline classes expect for "no override".
    """
    b = scenario.param_bounds()
    if args.preset == "realistic":
        lower = np.array(REALISTIC_LOWER, float)
        upper = np.array(REALISTIC_UPPER, float)
        narrowed = True
    else:
        lower = np.array(b["lower"], float)
        upper = np.array(b["upper"], float)
        narrowed = False

    lower, upper, extra = narrow_bounds(
        lower, upper, max_angle=args.max_angle, max_speed=args.max_speed,
        min_speed=args.min_speed, max_seg=args.max_seg, min_seg=args.min_seg)
    narrowed = narrowed or extra
    return (lower, upper) if narrowed else (None, None)
