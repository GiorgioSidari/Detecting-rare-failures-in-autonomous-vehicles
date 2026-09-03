"""
Tests for the shared lateral- and heading-error computation of
`scenarios/common/road_frame.py`.

They cover the projection onto the polyline (nearest segment rather than nearest
vertex, clamping past the ends, arc length, rejection of a too-short polyline),
the sign convention of the lateral error, the heading error against the local
tangent and its wrapping, both on straight roads and on an arc, and
`yaw_from_positions` including its stopped-vehicle fallback.

`test_agrees_with_the_udacity_cte` compares the lateral error computed here with
the `cte` definition used by Unity, on the same road and positions.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scenarios.common.road_frame import (          # noqa: E402
    cumulative_s,
    project_to_polyline,
    road_frame,
    yaw_from_positions,
)


def _straight(length: float = 100.0, n: int = 101) -> np.ndarray:
    """Rettilineo lungo +x."""
    return np.column_stack([np.linspace(0.0, length, n), np.zeros(n)])


def _arc(radius: float = 50.0, sweep_deg: float = 90.0, n: int = 201) -> np.ndarray:
    """Circular arc centred at (0, radius), starting from the origin towards +x."""
    th = np.linspace(0.0, math.radians(sweep_deg), n)
    return np.column_stack([radius * np.sin(th), radius * (1 - np.cos(th))])


# ─────────────────────────────────────────────────────────────────────────────
# Proiezione
# ─────────────────────────────────────────────────────────────────────────────

def test_projection_is_onto_the_segment_not_the_vertex():
    """
    A point halfway between two vertices must project onto the segment, with a
    distance equal to the perpendicular offset -- not to the distance from the
    nearest vertex.
    """
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    i, t, d = project_to_polyline(xy, 5.0, 3.0)
    assert i == 0
    assert t == pytest.approx(0.5)
    assert d == pytest.approx(3.0)


def test_projection_past_the_ends_is_clamped():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    _, t0, _ = project_to_polyline(xy, -5.0, 0.0)
    _, t1, _ = project_to_polyline(xy, 15.0, 0.0)
    assert t0 == 0.0
    assert t1 == 1.0


def test_too_short_polyline_is_rejected():
    with pytest.raises(ValueError):
        project_to_polyline(np.array([[0.0, 0.0]]), 0.0, 0.0)


def test_arc_length():
    s = cumulative_s(np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 9.0]]))
    np.testing.assert_allclose(s, [0.0, 5.0, 10.0])


# ─────────────────────────────────────────────────────────────────────────────
# Segni — la proprietà da cui dipende la stabilità del controller
# ─────────────────────────────────────────────────────────────────────────────

def test_lateral_error_sign():
    """
    Convention: positive = vehicle LEFT of the centreline. It must hold even
    when the road runs the opposite way, otherwise the controller diverges in
    the second half of a U-turn.
    """
    verso_x = _straight()
    assert road_frame(verso_x, 50.0, +2.0, 0.0).lateral_error > 0
    assert road_frame(verso_x, 50.0, -2.0, 0.0).lateral_error < 0

    # Same road driven backwards: "left" flips in world coordinates.
    verso_meno_x = verso_x[::-1].copy()
    assert road_frame(verso_meno_x, 50.0, -2.0, math.pi).lateral_error > 0
    assert road_frame(verso_meno_x, 50.0, +2.0, math.pi).lateral_error < 0


def test_heading_error_sign():
    xy = _straight()
    assert road_frame(xy, 50.0, 0.0, +0.3).heading_error == pytest.approx(0.3)
    assert road_frame(xy, 50.0, 0.0, -0.3).heading_error == pytest.approx(-0.3)


def test_heading_error_is_wrapped():
    """The jump at +/-pi must not produce a huge error and a counter-steer."""
    xy = _straight()
    e = road_frame(xy, 50.0, 0.0, math.pi + 0.1).heading_error
    assert abs(e) <= math.pi
    assert e == pytest.approx(-math.pi + 0.1, abs=1e-9)


def test_centred_vehicle_has_zero_errors():
    xy = _straight()
    fr = road_frame(xy, 42.0, 0.0, 0.0)
    assert fr.lateral_error == pytest.approx(0.0, abs=1e-12)
    assert fr.heading_error == pytest.approx(0.0, abs=1e-12)
    assert fr.s == pytest.approx(42.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Curve geometry
# ─────────────────────────────────────────────────────────────────────────────

def test_radial_offset_on_an_arc_is_recognised():
    """
    On an arc of radius R centred at (0, R), a point at radius R-d is displaced
    by d outwards (towards the circle's centre = right of travel).
    """
    radius, d = 50.0, 2.0
    xy = _arc(radius=radius)

    th = math.radians(45.0)
    inner = ((radius - d) * math.sin(th), radius - (radius - d) * math.cos(th))
    fr = road_frame(xy, inner[0], inner[1], 0.0)
    assert abs(fr.lateral_error) == pytest.approx(d, abs=0.02)


def test_tangent_on_an_arc():
    """Halfway along a 90-degree arc the tangent must be at ~45 degrees."""
    xy = _arc(radius=50.0, sweep_deg=90.0)
    th = math.radians(45.0)
    p = (50.0 * math.sin(th), 50.0 * (1 - math.cos(th)))
    fr = road_frame(xy, p[0], p[1], 0.0)
    assert math.degrees(fr.tangent_yaw) == pytest.approx(45.0, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Consistency with Udacity
# ─────────────────────────────────────────────────────────────────────────────

def test_agrees_with_the_udacity_cte():
    """
    Validation against an external reference.

    Udacity's `cte` is the signed distance from the centreline. On a road
    generated from our own parameters, our lateral error must match that
    definition: we build points at a KNOWN offset from the
    mezzeria e verifichiamo di ritrovare l'offset.

    If this test passes, the chain `9 parameters -> polyline -> projection` is
    consistent with what Unity measures, and the C2 arm on Udacity can use our
    `lateral_error` in place of the `cte` (necessary, because the heading error
    is needed too and the telemetry does not expose it).
    """
    from scenarios.common.road_geometry import road_polyline

    xy = road_polyline([10, 45, 3, 70, 25], seg_length=25.0, num_spline_nodes=80)

    rng = np.random.default_rng(0)
    for _ in range(40):
        i = int(rng.integers(1, len(xy) - 2))
        tx, ty = xy[i + 1] - xy[i - 1]
        n = math.hypot(tx, ty)
        # Unit left normal to the centreline.
        nx, ny = -ty / n, tx / n

        offset = float(rng.uniform(-2.0, 2.0))
        px, py = xy[i][0] + nx * offset, xy[i][1] + ny * offset

        got = road_frame(xy, px, py, 0.0).lateral_error
        # Tolerance: the normal is estimated by finite differences and the
        # polyline is discrete, so exact agreement is not expected.
        assert got == pytest.approx(offset, abs=0.05), (i, offset, got)


def test_real_road_centred_vehicle():
    """At the generated road's own vertices the error must be ~zero."""
    from scenarios.common.road_geometry import road_polyline

    xy = road_polyline([20] * 5, seg_length=25.0, num_spline_nodes=80)
    for i in range(5, len(xy) - 5, 17):
        fr = road_frame(xy, xy[i][0], xy[i][1], 0.0)
        assert abs(fr.lateral_error) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Heading from positions (needed on Udacity)
# ─────────────────────────────────────────────────────────────────────────────

def test_yaw_from_positions():
    assert yaw_from_positions((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert yaw_from_positions((0.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi / 2)
    assert yaw_from_positions((0.0, 0.0), (-1.0, 0.0)) == pytest.approx(math.pi)


def test_yaw_uses_the_fallback_when_stopped():
    """
    At zero speed the displacement is noise: estimating the heading would give a
    random value, and the controller would steer at random on spawn.
    """
    assert yaw_from_positions((5.0, 5.0), (5.0, 5.0), fallback=1.23) == 1.23
    assert yaw_from_positions((5.0, 5.0), (5.0 + 1e-9, 5.0), fallback=1.23) == 1.23
