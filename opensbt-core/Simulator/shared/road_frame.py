"""
Lateral and heading error from the centreline -- one computation for every backend.

Why this exists
---------------
The C2 arm of the comparison protocol is "the same controller on different
simulators". But a controller is only identical if it receives the same *input*,
and the input of a state-based lane keeper is the pair (lateral error, heading
error). If each backend computes it its own way, C2 no longer isolates the
effect of the simulator: it also folds in the difference between competing
definitions of "how far off the road I am".

Left to themselves, the backends would compute it in different ways:

  * **Udacity** exposes `cte` in the telemetry (computed by Unity), but does
    **not expose the lane tangent**, so the heading error is simply missing;
  * **MetaDrive** has `vehicle.lane` with its own projection API.

This module replaces both: it projects the vehicle position onto the road
polyline -- which we **know**, because we generate it ourselves with
`road_geometry.road_polyline` from the same 9 parameters -- and derives both
errors from a single definition.

Side benefit: on Udacity the telemetry `cte` becomes an **independent
reference**. If our lateral error matches what Unity reports, the whole
geometric chain (road generation -> polyline -> projection) is validated against
the simulator. See
`tests/test_road_frame.py::test_agrees_with_the_udacity_cte`.

Convenzioni
-----------
    lateral_error > 0  ->  vehicle LEFT of the centreline
    heading_error > 0  ->  vehicle rotated counter-clockwise from the tangent

Consistent with `lane_keeping_md/driver.py`: a positive lateral error must
produce negative steering (to the right).

No dependency beyond numpy. No simulator imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RoadFrame:
    """Vehicle pose relative to the centreline, at the projected point."""

    lateral_error: float     # m, firmato (+ = sinistra)
    heading_error: float     # rad, firmato, in [-pi, pi]
    s: float                 # arc length of the projected point (m)
    tangent_yaw: float       # rad, road direction at the projected point
    segment_index: int       # index of the segment the projection falls on

    #: True when the projection is clamped to the final vertex of the polyline,
    #: i.e. the vehicle has **driven past the end of the road**.
    #:
    #: In that case `lateral_error` is NOT a lateral error any more: the
    #: projection stays pinned to the last point and the "distance from the
    #: centreline" becomes the distance from the end of the road, which grows
    #: without bound as the vehicle drives away. A controller that keeps feeding
    #: back on that number is steering on meaningless information.
    #:
    #: Callers must **end the run** when this becomes True, not correct the
    #: trajectory. The generated roads are ~4 x seg_length long (100 m with the
    #: defaults) and at 15 m/s they are exhausted in under 7 seconds, well
    #: before the 30 s of `maxTime`: this is not a rare edge case.
    beyond_end: bool = False

    #: Symmetric: projection clamped to the initial vertex (vehicle behind the
    #: start of the road). Normal at the first step, when the spawn point sits
    #: just before the first vertex.
    before_start: bool = False


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def project_to_polyline(xy: np.ndarray, px: float, py: float) -> Tuple[int, float, float]:
    """
    Projects the point `(px, py)` onto the polyline `xy` of shape `(M, 2)`.

    Returns
    -------
    (segment_index, t, distance)
        `t` in [0, 1] is the position along the segment, `distance` is the
        Euclidean (unsigned) one.

    The projection is onto the nearest SEGMENT, not the nearest vertex: with
    different samplings the nearest vertex jumps around, the segment does not.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError(f"attesa polilinea (M>=2, 2), ricevuto {xy.shape}")

    a = xy[:-1]
    b = xy[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 1e-18, denom, 1e-18)

    ap = np.array([px, py], dtype=float) - a
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.hypot(proj[:, 0] - px, proj[:, 1] - py)

    i = int(np.argmin(d))
    return i, float(t[i]), float(d[i])


def cumulative_s(xy: np.ndarray) -> np.ndarray:
    """Arc length of every vertex of the polyline."""
    xy = np.asarray(xy, dtype=float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def road_frame(xy: np.ndarray, px: float, py: float, yaw_rad: float) -> RoadFrame:
    """
    Pose of the vehicle `(px, py, yaw_rad)` relative to the centreline `xy`.

    The sign of the lateral error comes from the cross product of the tangent
    and the centreline-to-vehicle vector, so it is independent of the direction
    of travel and of the quadrant.
    """
    i, t, dist = project_to_polyline(xy, px, py)

    xy = np.asarray(xy, dtype=float)
    ax, ay = xy[i]
    bx, by = xy[i + 1]
    tx, ty = bx - ax, by - ay
    norm = math.hypot(tx, ty)
    if norm < 1e-12:
        raise ValueError(f"segmento degenere all'indice {i}")
    tx, ty = tx / norm, ty / norm

    projx, projy = ax + t * (bx - ax), ay + t * (by - ay)
    # Prodotto vettoriale 2D tangente × (veicolo - proiezione): > 0 = sinistra.
    cross = tx * (py - projy) - ty * (px - projx)
    lateral = math.copysign(dist, cross) if dist > 0 else 0.0

    tangent_yaw = math.atan2(ty, tx)
    s = float(cumulative_s(xy)[i] + t * norm)

    # Projection clamped to an endpoint: the vehicle is outside the extent of
    # the road, not merely off the centreline. See the note on `beyond_end`.
    n_seg = len(xy) - 1
    return RoadFrame(
        lateral_error=lateral,
        heading_error=_wrap_pi(yaw_rad - tangent_yaw),
        s=s,
        tangent_yaw=tangent_yaw,
        segment_index=i,
        beyond_end=(i == n_seg - 1 and t >= 1.0 - 1e-9),
        before_start=(i == 0 and t <= 1e-9),
    )


def yaw_from_positions(prev_xy: Tuple[float, float],
                       curr_xy: Tuple[float, float],
                       fallback: float = 0.0) -> float:
    """
    Heading estimated from two consecutive positions.

    Needed on Udacity, whose telemetry exposes the position but **not the
    attitude**. Same idea as `agent/agent_utils.calc_yaw_ego`, but in radians
    and with explicit handling of the "vehicle stopped" case: at zero speed the
    displacement is noise and the estimated heading would be random, so the last
    valid value is kept (`fallback`).

    A limit worth declaring: this is the heading of the TRAJECTORY, not of the
    vehicle's attitude. They coincide only without slip. At the speeds of this
    scenario the difference is small, but it is a difference between the C2 arm
    on Udacity and the one on MetaDrive, where the attitude is directly
    available.
    """
    dx = curr_xy[0] - prev_xy[0]
    dy = curr_xy[1] - prev_xy[1]
    if math.hypot(dx, dy) < 1e-6:
        return fallback
    return math.atan2(dy, dx)
