"""
Tests for the per-scenario episode horizon of
`scenarios/common/episode_budget.py`.

They cover:

  * `budget_seconds` -- the formula
    `clamp(length / target_speed * MARGIN, MIN_SECONDS, MAX_SECONDS)`, including
    both clamps and the monotonic behaviour in length and target speed;
  * `budget_steps` -- the conversion to steps for a given `control_hz`, and its
    rounding up;
  * that two backends running at different control rates get the same horizon in
    simulated seconds, and therefore step counts in the ratio of their rates.
"""
from __future__ import annotations

import math

import pytest

from scenarios.common.episode_budget import (                       # noqa: E402
    MARGIN, MAX_SECONDS, MIN_SECONDS,
    budget_steps, budget_seconds, polyline_length,
)


def test_budget_scales_with_road_length():
    """Twice the road, twice the time: that is the whole premise."""
    a = budget_seconds(100.0, 5.0)
    b = budget_seconds(200.0, 5.0)
    assert b == pytest.approx(2 * a)


def test_budget_scales_inversely_with_speed():
    assert budget_seconds(200.0, 10.0) == pytest.approx(budget_seconds(200.0, 5.0) / 2)


def test_margin_covers_the_measured_loss():
    """
    The margin is not arbitrary: at target speed the car covers ~79% of the
    nominal distance (26% measured against 33% predicted at scale=0.15), because
    it accelerates gradually and slows in curves. At least 1/0.79 ~ 1.27 is
    needed.
    """
    assert MARGIN >= 1.27, "margin below the measured loss: roads would be truncated"
    t = budget_seconds(200.0, 5.0)
    assert t == pytest.approx(200.0 / 5.0 * MARGIN)


def test_bounds_are_respected():
    assert budget_seconds(1.0, 100.0) == pytest.approx(MIN_SECONDS)     # scenario cortissimo
    assert budget_seconds(100000.0, 0.1) == pytest.approx(MAX_SECONDS)  # scenario lentissimo


@pytest.mark.parametrize("L,v", [(0.0, 5.0), (-1.0, 5.0), (100.0, 0.0), (100.0, -2.0)])
def test_degenerate_inputs_are_rejected(L, v):
    """An error is better than an infinite or zero horizon."""
    with pytest.raises(ValueError):
        budget_seconds(L, v)


def test_steps_are_rounded_up():
    """
    Truncating would mean stopping just short of the end of the road in exactly
    the worst scenarios, which are the interesting ones.
    """
    n = budget_steps(203.0, 5.4, 10.0)
    assert n == math.ceil(budget_seconds(203.0, 5.4) * 10.0)
    assert n >= budget_seconds(203.0, 5.4) * 10.0


def test_same_budget_in_seconds_at_different_rates():
    """
    Two backends with different control rates must receive the SAME horizon in
    seconds, and therefore a different number of steps. This is the property
    that makes Udacity (~19.5 Hz) and MetaDrive (exactly 10 Hz) comparable.
    """
    L, v = 203.0, 5.4
    n_md = budget_steps(L, v, 10.0)
    n_ud = budget_steps(L, v, 19.5)
    assert n_ud > n_md
    assert n_md / 10.0 == pytest.approx(n_ud / 19.5, rel=0.01)


def test_budget_is_enough_to_cover_the_road():
    """
    Direct counter-check on the original defect: the budget must be enough to
    drive the whole track at the target speed.
    """
    for L in (150.0, 203.0, 260.0):
        for v in (3.0, 5.4, 8.0):
            steps = budget_steps(L, v, 10.0)
            path = v * (steps / 10.0)
            assert path >= L, f"budget too small: {path:.0f}m out of {L}m"


def test_the_old_300_steps_were_not_enough():
    """
    The number that caused the problem, kept here as a reminder: 300 steps at
    10 Hz on a 203 m road at 5.4 m/s cover ~162 m, i.e. 80%.
    """
    assert 5.4 * (300 / 10.0) < 203.0
    assert budget_steps(203.0, 5.4, 10.0) > 300


def test_polyline_length():
    assert polyline_length([[0, 0], [3, 4]]) == pytest.approx(5.0)
    assert polyline_length([[0, 0], [3, 4], [3, 4 + 12]]) == pytest.approx(17.0)
    with pytest.raises(ValueError):
        polyline_length([[0, 0]])


def test_real_budget_on_the_design():
    """On the real design the budget always exceeds the 300 steps used before."""
    from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

    from scenarios.common.road_geometry import road_polyline
    from scenarios.lane_keeping.config import LaneKeepingScenario
    from scenarios.lane_keeping_md.map_builder import build_scenario_spec, target_speed

    b = LaneKeepingScenario().param_bounds()
    th = qmc_scale(LatinHypercube(d=len(b["lower"]), seed=42).random(n=20),
                   b["lower"], b["upper"])
    for row in th:
        L = polyline_length(road_polyline(row))
        v = target_speed(build_scenario_spec(row)) * 0.3625
        assert budget_steps(L, v, 10.0) > 300


def test_metadrive_scenario_uses_the_budget():
    """
    The scenario passes the same budget to the step loop and to MetaDrive's
    `horizon`, so the two truncation limits agree.
    """
    pytest.importorskip("metadrive")
    from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario as S

    sc = S(speed_scale=0.3625, max_steps=300)
    assert sc._budget_steps == 300, "before a run it holds the default"

    seen = {}

    def fake_make_online_env(row, *, decision_repeat, physics_world_step_size, max_steps, seed=0):
        seen["max_steps"] = max_steps
        raise RuntimeError("stop")            # knowing what it was passed is enough

    # config.py binds the name at import time (`from ... import make_online_env`),
    # so the patch has to replace it in config, not in scenario_map.
    import scenarios.lane_keeping_md.config as cfg
    real_make_online_env = cfg.make_online_env
    cfg.make_online_env = fake_make_online_env
    try:
        from scenarios.lane_keeping_md.map_builder import build_scenario_spec
        from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

        from scenarios.lane_keeping.config import LaneKeepingScenario
        b = LaneKeepingScenario().param_bounds()
        row = qmc_scale(LatinHypercube(d=len(b["lower"]), seed=42).random(n=1),
                        b["lower"], b["upper"])[0]
        with pytest.raises(RuntimeError):
            sc._make_env(build_scenario_spec(row), 0, row=row)
    finally:
        cfg.make_online_env = real_make_online_env

    assert seen["max_steps"] > 300
    assert seen["max_steps"] == sc._budget_steps, (
        "MetaDrive's horizon and the loop limit must agree")


def test_pgblock_mode_leaves_the_budget_alone():
    """The legacy path stays reproducible: same `max_steps` as before."""
    from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario as S

    sc = S(geometry="pgblock", max_steps=300)
    assert sc._budget_steps == 300
