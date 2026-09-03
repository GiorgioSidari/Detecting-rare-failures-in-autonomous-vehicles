"""
Lateral and heading error from the centreline, computed the same way everywhere.

The vehicle position is projected onto the road polyline -- the same polyline
`road_geometry.road_polyline` generates from the 9 scenario parameters -- and
both errors are derived from that single projection, so every backend feeds the
controller the same quantities.

The alternative sources differ between backends and are not used here: Udacity's
telemetry exposes `cte` but no lane tangent, so the heading error is missing,
and MetaDrive's `vehicle.lane` has its own projection API with the opposite sign
convention. On Udacity the telemetry `cte` is instead compared against this
computation as an independent check (see
`tests/test_road_frame.py::test_agrees_with_the_udacity_cte`).

Conventions
-----------
    lateral_error > 0  ->  vehicle LEFT of the centreline
    heading_error > 0  ->  vehicle rotated counter-clockwise from the tangent

`RoadFrame` also carries `beyond_end` and `before_start`, true when the
projection clamps to the last or first vertex, i.e. when the vehicle is outside
the extent of the road and the lateral error is no longer a lateral error.
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

    The projection is onto the nearest SEGMENT, not the nearest vertex, so the
    result does not depend on how densely the polyline is sampled.
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
        raise ValueError(f"degenerate segment at index {i}")
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

    Returns the angle of the displacement vector, in radians. When the
    displacement is shorter than the minimum distance (the vehicle is stopped or
    nearly so) the displacement direction is noise, and `fallback` is returned
    instead.

    The value is the heading of the trajectory, which equals the vehicle's
    attitude only in the absence of slip. It is used on backends whose telemetry
    exposes the position but not the attitude.
    """
    dx = curr_xy[0] - prev_xy[0]
    dy = curr_xy[1] - prev_xy[1]
    if math.hypot(dx, dy) < 1e-6:
        return fallback
    return math.atan2(dy, dx)
