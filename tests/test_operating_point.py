"""
Tests for the operating-point calibration.

Runs on SYNTHETIC evaluation functions of known truth, without simulators: what
is tested is the algorithm, not the backends. That is the point -- a bisection
that is wrong would leave a whole campaign badly calibrated, and without these
tests the error would only surface as absurd results downstream.
"""
from __future__ import annotations


import pytest

from pipeline.operating_point import Evaluation, calibrate   # noqa: E402


def _fake_backend(curve, n_valid=40, n_total=40):
    def _valuta(scale: float) -> Evaluation:
        """
        Valutatore sintetico: `curve(speed_scale) -> failure_rate`.

        Must be non-decreasing in speed_scale -- faster, more failures -- because
        that is the assumption the bisection rests on.
        """
        r = float(curve(scale))
        return Evaluation(speed_scale=scale, failure_rate=r,
                           n_valid=n_valid, n_total=n_total,
                           median_margin=0.5 - r)
    return _valuta


# ─────────────────────────────────────────────────────────────────────────────
# Convergenza
# ─────────────────────────────────────────────────────────────────────────────

def test_centres_the_band_on_a_linear_curve():
    res = calibrate(_fake_backend(lambda s: s), backend="fake", verbose=False)
    assert res.centered
    assert 0.10 <= res.failure_rate <= 0.20
    assert 0.10 <= res.speed_scale <= 0.21


def test_centres_the_band_on_a_steep_curve():
    """A failure rate that shoots up: the bisection must still find the band."""
    res = calibrate(_fake_backend(lambda s: min(1.0, s ** 3 * 4)),
                  backend="fake", verbose=False)
    assert res.centered
    assert 0.10 <= res.failure_rate <= 0.20


def test_uses_few_steps():
    """
    Every evaluation costs N simulations: on Udacity that means minutes. The
    bisection must converge in a few iterations, not explore.
    """
    res = calibrate(_fake_backend(lambda s: s), backend="fake",
                  max_iter=8, verbose=False)
    assert len(res.history) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# Cases that are NOT errors, but must be reported
# ─────────────────────────────────────────────────────────────────────────────

def test_backend_that_always_fails():
    """
    If it fails even at walking pace, the problem is the controller, not the
    speed. Calibrating anyway would hide the defect: it has to be said.
    """
    res = calibrate(_fake_backend(lambda s: 0.95), backend="rotto", verbose=False)
    assert not res.centered
    assert res.speed_scale == pytest.approx(0.15)     # the slowest endpoint
    assert "does not hold the lane" in res.note


def test_backend_that_never_fails():
    """If it never fails, the space is too easy: nothing to search for."""
    res = calibrate(_fake_backend(lambda s: 0.0), backend="easy", verbose=False)
    assert not res.centered
    assert res.speed_scale == pytest.approx(1.0)      # the fastest endpoint
    assert "too easy" in res.note


def test_returns_the_closest_value_when_the_band_is_unreachable():
    """
    With a step-shaped curve the band can be unreachable. Better the closest
    value, declared as such, than a hard failure.
    """
    res = calibrate(_fake_backend(lambda s: 0.02 if s < 0.5 else 0.60),
                  backend="step", max_iter=6, verbose=False)
    assert not res.centered
    assert res.failure_rate in (0.02, 0.60)
    assert "Bisection exhausted" in res.note


def test_band_already_centred_at_an_endpoint():
    """If an endpoint is already in band, no extra simulation is wasted."""
    res = calibrate(_fake_backend(lambda s: 0.15), backend="lucky",
                  verbose=False)
    assert res.centered
    assert len(res.history) == 1          # the first evaluation is enough


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility and report
# ─────────────────────────────────────────────────────────────────────────────

def test_is_deterministic():
    a = calibrate(_fake_backend(lambda s: s ** 2), backend="x", verbose=False)
    b = calibrate(_fake_backend(lambda s: s ** 2), backend="x", verbose=False)
    assert a.speed_scale == b.speed_scale
    assert [v.speed_scale for v in a.history] == [v.speed_scale for v in b.history]


def test_report_contains_the_chosen_value():
    """
    The calibrated value is a parameter of the experiment and must be
    declared: the report has to make it impossible to miss.
    """
    res = calibrate(_fake_backend(lambda s: s), backend="lane_keeping_md",
                  verbose=False)
    text = res.report()
    assert "lane_keeping_md" in text
    assert f"{res.speed_scale:.4f}" in text
    assert "OPERATING-POINT CALIBRATION" in text


def test_json_round_trip(tmp_path):
    res = calibrate(_fake_backend(lambda s: s), backend="x", verbose=False)
    p = tmp_path / "sub" / "calibration.json"
    res.save(str(p))

    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["speed_scale"] == res.speed_scale
    assert len(d["history"]) == len(res.history)


def test_band_is_configurable():
    res = calibrate(_fake_backend(lambda s: s), backend="x",
                  target_low=0.30, target_high=0.40, verbose=False)
    assert res.centered
    assert 0.30 <= res.failure_rate <= 0.40
