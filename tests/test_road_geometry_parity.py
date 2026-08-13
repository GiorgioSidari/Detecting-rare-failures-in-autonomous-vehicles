"""
Parity between the copy in `scenarios/common/road_geometry.py` and the real
Udacity generator.

`road_geometry` is a copy of the mathematics that lives in
`opensbt-core/Simulator/lanekeeping/`, because that package cannot be imported
without dragging in tensorflow, gym, matplotlib and UdacitySimulatorIO (see the
module docstring).

A copy without a parity test is debt that surfaces late and in the wrong place:
if the two generators diverge, the backends drive DIFFERENT roads and every
comparison between them loses its meaning -- with nothing to flag the problem.
This test is the guardrail.

If it fails: the Udacity generator changed. Carry the change over into
`scenarios/common/road_geometry.py`, do NOT loosen the tolerances.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIM_ROOT = os.path.join(_REPO_ROOT, "opensbt-core")
# `lanekeeping` is not inside a package: it must be imported with Simulator/ on sys.path.
_LK_ROOT = os.path.join(_REPO_ROOT, "opensbt-core", "Simulator")
for _p in (_SIM_ROOT, _LK_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenarios.common import road_geometry as rg   # noqa: E402


def _bootstrap_lanekeeping() -> bool:
    """
    Makes the Udacity generator importable outside the container, skipping the
    `__init__.py` files with side effects (UdacitySimulatorIO, tensorflow) and
    stubbing the dependencies that only serve plotting.

    Returns False if the import remains impossible: in that case the test is
    skipped rather than failed, because the absence of the Udacity source is not
    a defect of our code.
    """
    lk = os.path.join(_LK_ROOT, "lanekeeping")
    if not os.path.isdir(lk):
        return False

    for name, path in (
        ("lanekeeping", lk),
        ("lanekeeping.self_driving", os.path.join(lk, "self_driving")),
        ("lanekeeping.self_driving.utils", os.path.join(lk, "self_driving", "utils")),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod

    viz = "lanekeeping.self_driving.utils.visualization"
    if viz not in sys.modules:
        m = types.ModuleType(viz)
        m.RoadTestVisualizer = object
        sys.modules[viz] = m

    if "gym" not in sys.modules:
        g = types.ModuleType("gym")
        g.Env = object
        sys.modules["gym"] = g

    try:
        import lanekeeping.road_generator.custom_road_generator  # noqa: F401
        import lanekeeping.self_driving.catmull_rom              # noqa: F401
    except Exception:
        return False
    return True


_HAVE_UDACITY = _bootstrap_lanekeeping()
requires_udacity = pytest.mark.skipif(
    not _HAVE_UDACITY,
    reason="Udacity source not importable in this environment",
)

# Configurations covering: straight, gentle curve, sharp curve, mixed,
# asymmetric. The straight one is included because it is the case where a sign
# or initial-heading error is easiest to see.
ANGLE_CASES = [
    [0, 0, 0, 0, 0],
    [20, 20, 20, 20, 20],
    [85, 85, 85, 85, 85],
    [0, 85, 0, 85, 0],
    [10, 45, 3, 70, 25],
]
SEG_LENGTHS = [10.0, 25.0, 40.0]


@requires_udacity
@pytest.mark.parametrize("angles", ANGLE_CASES)
@pytest.mark.parametrize("seg_length", SEG_LENGTHS)
def test_control_nodes_are_identical(angles, seg_length):
    """The control nodes must agree to machine precision."""
    from lanekeeping.road_generator.custom_road_generator import CustomRoadGenerator

    gen = CustomRoadGenerator(map_size=250, num_control_nodes=len(angles),
                              seg_length=int(seg_length))
    expected = gen.generate_control_nodes(
        starting_pos=(0.0, 0.0, 0.0, rg.ROAD_WIDTH),
        angles=[int(a) for a in angles],
        seg_lengths=[int(seg_length)] * len(angles),
    )
    got = rg.generate_control_nodes(
        [int(a) for a in angles], seg_length=int(seg_length),
        initial_node=(0.0, 0.0, 0.0, rg.ROAD_WIDTH),
    )

    assert len(got) == len(expected) == len(angles) + 2
    np.testing.assert_allclose(np.asarray(got, dtype=float),
                               np.asarray(expected, dtype=float),
                               rtol=0, atol=1e-12)


@requires_udacity
@pytest.mark.parametrize("angles", ANGLE_CASES)
@pytest.mark.parametrize("seg_length", SEG_LENGTHS)
def test_spline_is_identical(angles, seg_length):
    """The full Catmull-Rom spline must agree point by point."""
    from lanekeeping.road_generator.custom_road_generator import CustomRoadGenerator
    from lanekeeping.self_driving.catmull_rom import catmull_rom

    gen = CustomRoadGenerator(map_size=250, num_control_nodes=len(angles),
                              seg_length=int(seg_length))
    nodes = gen.generate_control_nodes(
        starting_pos=(0.0, 0.0, 0.0, rg.ROAD_WIDTH),
        angles=[int(a) for a in angles],
        seg_lengths=[int(seg_length)] * len(angles),
    )
    expected = np.asarray([(p[0], p[1]) for p in catmull_rom(nodes, 20)], dtype=float)
    got = rg.road_polyline([int(a) for a in angles], seg_length=int(seg_length),
                           num_spline_nodes=20)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


@requires_udacity
def test_constants_are_aligned():
    """ROAD_WIDTH and SEG_LENGTH must mirror lanekeeping/config.py."""
    import lanekeeping.config as lk_config

    assert rg.ROAD_WIDTH == lk_config.ROAD_WIDTH
    assert rg.SEG_LENGTH == lk_config.SEG_LENGTH


def test_node_and_segment_counting():
    """
    Documents the verified counting: 5 angles -> 7 nodes -> 4 spline segments.
    Does not require the Udacity source.
    """
    angles = [20] * 5
    nodes = rg.generate_control_nodes(angles, seg_length=25)
    assert len(nodes) == 7

    xy = rg.road_polyline(angles, seg_length=25, num_spline_nodes=20)
    # 4 segmenti x 20 punti, meno 3 giunzioni deduplicate
    assert len(xy) == 4 * 20 - 3

    # The drivable road covers ~4 segments, not 5.
    length = rg.polyline_length(xy)
    assert 3.5 * 25 < length < 4.5 * 25, length


def test_a_straight_road_is_really_straight():
    """With all angles zero the spline must lie on the x axis."""
    xy = rg.road_polyline([0, 0, 0, 0, 0], seg_length=25)
    assert np.allclose(xy[:, 1], 0.0, atol=1e-9)
    assert np.all(np.diff(xy[:, 0]) > 0)      # monotona in avanti
