"""
Shared road geometry: a copy of the Udacity road-generation chain.

The original lives in `opensbt-core/Simulator/lanekeeping/`, whose import chain
pulls in tensorflow, matplotlib, descartes and gym. This module reimplements the
same mathematics with numpy as its only dependency, and
`tests/test_road_geometry_parity.py` checks that it produces output identical to
the original.

`opensbt-core/Simulator/shared/road_geometry.py` is a byte-identical vendored
copy of this file, needed because the containers' build context cannot reach
outside `opensbt-core/`. `tests/test_shared_vendoring.py` compares the hashes:
editing one file requires re-copying it to the other.

References to the original:
  - `lanekeeping/road_generator/custom_road_generator.py`
      CustomRoadGenerator.generate_control_nodes / _get_initial_control_node /
      _get_next_node / _get_next_xy
  - `lanekeeping/self_driving/catmull_rom.py`
      catmull_rom_spline / catmull_rom_chain / catmull_rom
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

# Mirror of lanekeeping/config.py -- kept in step by the parity test.
ROAD_WIDTH = 8.0        # total carriageway width (m)
SEG_LENGTH = 25         # default segment length (m)
NUM_SPLINE_NODES = 20   # spline points PER SEGMENT (not in total)

Node = Tuple[float, float, float, float]   # (x, y, z, width)


# ─────────────────────────────────────────────────────────────────────────────
# Control nodes
# ─────────────────────────────────────────────────────────────────────────────

def _next_xy(x0: float, y0: float, angle_deg: float, seg_length: float) -> Tuple[float, float]:
    """Copy of `CustomRoadGenerator._get_next_xy`."""
    a = math.radians(angle_deg)
    return x0 + seg_length * math.cos(a), y0 + seg_length * math.sin(a)


def generate_control_nodes(
    angles: Sequence[float],
    seg_length: float = SEG_LENGTH,
    initial_node: Node = (0.0, 0.0, 0.0, ROAD_WIDTH),
) -> List[Node]:
    """
    Copy of `CustomRoadGenerator.generate_control_nodes`. Angles are CUMULATIVE:
    changing angle_1 rotates the whole road downstream. 5 angles give 7 nodes
    (ghost + initial + 5), and since a Catmull-Rom tract needs 4 points the road
    ends up with 4 drivable segments, angle_5 acting only as the final tangent.
    """
    x0, y0, z0, width = initial_node
    gx, gy = _next_xy(x0, y0, 0.0, seg_length)
    nodes: List[Node] = [(gx, gy, z0, width), (x0, y0, z0, width)]

    cumulative_angle = 0.0
    for a in angles:
        cumulative_angle += a
        px, py, pz, pw = nodes[-1]
        nx, ny = _next_xy(px, py, cumulative_angle, seg_length)
        nodes.append((nx, ny, pz, pw))
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# Spline Catmull-Rom (parametrizzazione centripeta, alpha = 0.5)
# ─────────────────────────────────────────────────────────────────────────────

def catmull_rom_spline(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray,
                       num_points: int = NUM_SPLINE_NODES) -> np.ndarray:
    """
    Copy of `catmull_rom.catmull_rom_spline`: the p1->p2 arc only, so a tract
    takes 4 points -- p0 and p3 set the tangents and are not traversed.
    Centripetal parametrisation (alpha = 0.5) keeps the curve free of the cusps
    and loops that alpha = 0 produces on turns as sharp as 85 deg.
    """
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    alpha = 0.5

    def tj(ti, p_i, p_j):
        xi, yi = p_i
        xj, yj = p_j
        return (((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5) ** alpha + ti

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    t = np.linspace(t1, t2, num_points).reshape(-1, 1)

    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3

    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def catmull_rom_chain(points: Sequence[tuple],
                      num_spline_points: int = NUM_SPLINE_NODES) -> List[np.ndarray]:
    """
    Copy of `catmull_rom.catmull_rom_chain`. The node shared by two consecutive
    tracts is emitted once.
    """
    cr: List[np.ndarray] = []
    for j in range(len(points) - 3):
        c = catmull_rom_spline(points[j], points[j + 1], points[j + 2],
                               points[j + 3], num_spline_points)
        if j > 0:
            c = np.delete(c, [0], axis=0)
        cr.extend(c)
    return cr


def catmull_rom(points: Sequence[Node],
                num_spline_points: int = NUM_SPLINE_NODES) -> List[Node]:
    """Copy of `catmull_rom.catmull_rom`: keeps z and width of the first node."""
    if len(points) < 4:
        raise ValueError("points should have at least 4 points")
    if not all(p[3] == points[0][3] for p in points):
        raise AssertionError("every node must have the same width")
    xy = catmull_rom_chain([(p[0], p[1]) for p in points], num_spline_points)
    z0, width = points[0][2], points[0][3]
    return [(float(p[0]), float(p[1]), z0, width) for p in xy]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience API
# ─────────────────────────────────────────────────────────────────────────────

def road_polyline(
    angles: Sequence[float],
    seg_length: float = SEG_LENGTH,
    num_spline_nodes: int = NUM_SPLINE_NODES,
    initial_node: Node = (0.0, 0.0, 0.0, ROAD_WIDTH),
) -> np.ndarray:
    """
    Drivable centreline for `angles`, as (M, 2): control nodes -> Catmull-Rom.
    The single entry point every backend builds its road from, which is what
    makes "same theta, same road" true.
    """
    nodes = generate_control_nodes(angles, seg_length, initial_node)
    pts = catmull_rom(nodes, num_spline_nodes)
    return np.asarray([(p[0], p[1]) for p in pts], dtype=float)


def polyline_length(xy: np.ndarray) -> float:
    """Arc length of a polyline (M, 2)."""
    d = np.diff(np.asarray(xy, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def bounding_box(xy: np.ndarray) -> Tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) of the polyline."""
    xy = np.asarray(xy, dtype=float)
    return (float(xy[:, 0].min()), float(xy[:, 1].min()),
            float(xy[:, 0].max()), float(xy[:, 1].max()))
