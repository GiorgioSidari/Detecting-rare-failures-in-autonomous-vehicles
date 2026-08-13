"""
Tests for the shared controller (the C2 arm).

Two families of checks, with different purposes:

  * **closed-loop stability** -- the controller, coupled to a toy kinematic
    model, must bring the vehicle back into the lane. If it does not do so here,
    a failure on a real simulator would not tell "the simulator is hard" from
    "the controller is broken";
  * **the degradations do what they claim** -- they exist to make failure
    scenario-dependent. If `obs_latency` did not really degrade anything, the C2
    arm would never fail and there would be no boundary to learn: the comparison
    between search methods would be empty.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scenarios.common.driver import LateralFeedbackDriver, target_speed   # noqa: E402
from scenarios.common.road_frame import road_frame                    # noqa: E402


def _straight(length: float = 400.0, n: int = 401) -> np.ndarray:
    return np.column_stack([np.linspace(0.0, length, n), np.zeros(n)])


def _simulate(driver, xy, *, x0=0.0, y0=0.0, yaw0=0.0, v=10.0,
              dt=0.05, steps=400, wheelbase=2.5, max_steer_rad=0.5):
    """
    Kinematic bicycle, the least needed to close the loop.

    This is not a simulator: it only checks that the control law has the right
    sign and converges. It returns the trajectory of lateral errors.
    """
    x, y, yaw = x0, y0, yaw0
    errs = []
    driver.reset()
    for _ in range(steps):
        fr = road_frame(xy, x, y, yaw)
        errs.append(fr.lateral_error)
        steer, throttle = driver.act({
            "lateral_error": fr.lateral_error,
            "heading_error": fr.heading_error,
            "speed": v,
            "target_speed": v,
            "dt": dt,
        })
        delta = steer * max_steer_rad          # sterzo normalizzato -> radianti
        yaw += (v / wheelbase) * math.tan(delta) * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    return np.asarray(errs)


# ─────────────────────────────────────────────────────────────────────────────
# Closed loop
# ─────────────────────────────────────────────────────────────────────────────

def test_recentres_from_a_lateral_offset():
    """Started off to one side, the controller must come back."""
    errs = _simulate(LateralFeedbackDriver(), _straight(), y0=1.5)
    assert abs(errs[-1]) < 0.1, f"final error {errs[-1]:.3f} m"
    assert abs(errs[-1]) < abs(errs[0])


def test_recentres_from_both_sides():
    """A sign error would show up as divergence on one side only."""
    for y0 in (+1.5, -1.5):
        errs = _simulate(LateralFeedbackDriver(), _straight(), y0=y0)
        assert abs(errs[-1]) < 0.1, f"y0={y0}: final error {errs[-1]:.3f} m"


def test_recentres_from_a_heading_error():
    errs = _simulate(LateralFeedbackDriver(), _straight(), yaw0=0.3)
    assert abs(errs[-1]) < 0.15


def test_stays_in_lane_when_already_centred():
    errs = _simulate(LateralFeedbackDriver(), _straight())
    assert np.max(np.abs(errs)) < 1e-6


def test_does_not_oscillate():
    """
    The error must decrease essentially monotonically. A controller that
    oscillates around the centreline would produce failures that look like
    scenario difficulty but are controller instability.
    """
    errs = np.abs(_simulate(LateralFeedbackDriver(), _straight(), y0=1.5))
    seconda_meta = errs[len(errs) // 2:]
    assert np.max(seconda_meta) < 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Degradations
# ─────────────────────────────────────────────────────────────────────────────

def test_obs_latency_degrades_control():
    base = np.max(np.abs(_simulate(LateralFeedbackDriver(), _straight(), y0=1.5)))
    slow = np.max(np.abs(_simulate(
        LateralFeedbackDriver(obs_latency=8), _straight(), y0=1.5)))
    assert slow > base


def test_obs_latency_bites_harder_at_high_speed():
    """
    This is the envelope mechanism: a latency fixed in STEPS becomes a staleness
    growing in METRES as speed rises. If that property did not hold, the speed
    axis would not produce a boundary.
    """
    def peak(v):
        return np.max(np.abs(_simulate(
            LateralFeedbackDriver(obs_latency=6), _straight(), y0=1.0, v=v)))

    assert peak(20.0) > peak(6.0)


def test_obs_lag_degrades_control():
    """`obs_lag_tau` is a TIME constant (seconds), not a per-step coefficient:
    that way the sluggishness is the same at different rates."""
    base = np.max(np.abs(_simulate(LateralFeedbackDriver(), _straight(), y0=1.5)))
    slow = np.max(np.abs(_simulate(
        LateralFeedbackDriver(obs_lag_tau=0.5), _straight(), y0=1.5)))
    assert slow > base


def test_noise_is_reproducible_with_a_seed():
    a = _simulate(LateralFeedbackDriver(steer_noise=0.05, seed=7), _straight(), y0=1.0)
    b = _simulate(LateralFeedbackDriver(steer_noise=0.05, seed=7), _straight(), y0=1.0)
    np.testing.assert_allclose(a, b)


def test_reset_reseeds_the_noise():
    """
    Without re-seeding, the second run of a campaign would see different noise
    from the first at the same seed, and the results would not be reproducible.
    """
    d = LateralFeedbackDriver(steer_noise=0.05, seed=3)
    first = _simulate(d, _straight(), y0=1.0)
    second = _simulate(d, _straight(), y0=1.0)     # _simulate calls reset()
    np.testing.assert_allclose(first, second)


def test_defaults_are_robust():
    """Every degradation must be off by default."""
    d = LateralFeedbackDriver()
    assert d.obs_latency == 0
    assert d.obs_lag_tau == 0.0
    assert d.steer_noise == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Constraints on the action
# ─────────────────────────────────────────────────────────────────────────────

def test_rate_limiter_is_in_units_per_second():
    """
    The limit is in units/s and is converted with the step's `dt`. This is the
    property that makes C2 the SAME controller on backends with different rates:
    it used to be per step, so at 8.5 Hz it had 1.7 u/s and at 20.8 Hz it had
    4.2.
    """
    d = LateralFeedbackDriver(max_steer_rate=2.0)
    for dt in (0.05, 0.1176):
        d.reset()
        s, _ = d.act({"lateral_error": 100.0, "heading_error": 0.0,
                      "speed": 0.0, "target_speed": 10.0, "dt": dt})
        assert abs(s) == pytest.approx(2.0 * dt, rel=1e-9)
        # identical authority per SECOND, which is the point
        assert abs(s) / dt == pytest.approx(2.0, rel=1e-9)


def test_authority_is_identical_at_the_real_backend_rates():
    """
    The three rates measured in practice must give the same steering authority.
    If this test fails, C2 is no longer "the same controller".
    """
    authority = []
    for hz in (8.49, 10.0, 20.78):        # Udacity 4 workers, MetaDrive, Udacity 1 worker
        d = LateralFeedbackDriver()
        d.reset()
        dt = 1.0 / hz
        s, _ = d.act({"lateral_error": 100.0, "heading_error": 0.0,
                      "speed": 0.0, "target_speed": 10.0, "dt": dt})
        authority.append(abs(s) / dt)
    assert max(authority) - min(authority) < 1e-9, authority


def test_actions_stay_in_range():
    rng = np.random.default_rng(1)
    d = LateralFeedbackDriver()
    d.reset()
    for _ in range(500):
        s, t = d.act({
            "lateral_error": float(rng.uniform(-50, 50)),
            "heading_error": float(rng.uniform(-math.pi, math.pi)),
            "speed": float(rng.uniform(0, 60)),
            "target_speed": float(rng.uniform(0, 30)),
        })
        assert -1.0 <= s <= 1.0
        assert -1.0 <= t <= 1.0


def test_throttle_accelerates_and_slows():
    d = LateralFeedbackDriver()
    d.reset()
    _, acc = d.act({"lateral_error": 0.0, "heading_error": 0.0,
                    "speed": 0.0, "target_speed": 15.0})
    _, dec = d.act({"lateral_error": 0.0, "heading_error": 0.0,
                    "speed": 30.0, "target_speed": 15.0})
    assert acc > 0
    assert dec < 0


# ─────────────────────────────────────────────────────────────────────────────
# Target speed
# ─────────────────────────────────────────────────────────────────────────────

def test_target_speed_is_the_band_midpoint():
    assert target_speed(5.0, 15.0) == pytest.approx(10.0)


def test_speed_scale_is_monotone():
    """The calibration bisection requires monotonicity, or it does not converge."""
    values = [target_speed(5.0, 15.0, s) for s in (0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)
    assert values[0] < values[-1]
