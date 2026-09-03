"""
Tests for the cross-simulator comparison of `pipeline.cross_simulator`.

They run on synthetic campaigns with known ground truth, so what is checked is
the comparison logic and not the backends: the Spearman correlation on a pair of
QoI vectors, the failure-region overlap, and the reported failure rates.

A second group covers the cases in which the comparison must refuse to produce a
number rather than return one: designs that are not shared between the two
campaigns, a degenerate backend (no failures or all failures), and campaigns
collected at different operating points.
"""
from __future__ import annotations


import numpy as np
import pytest

from pipeline.cross_simulator import (          # noqa: E402
    CrossSimulatorComparison,
    BackendRun,
)

D = 9
LOWER = np.array([0, 0, 0, 0, 0, 5.0, 10.0, 10.0, 150.0])
UPPER = np.array([85, 85, 85, 85, 85, 15.0, 30.0, 40.0, 350.0])


def _design(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return LOWER + rng.random((n, D)) * (UPPER - LOWER)


def _qoi_from_difficulty(theta, *, noise=0.0, offset=0.0, seed=0):
    """
    Synthetic QoI: the curvier the road, the lower the margin. Used to build
    backends that agree (same difficulty) or disagree.
    """
    rng = np.random.default_rng(seed)
    difficolta = theta[:, :5].mean(axis=1) / 85.0
    return 1.0 - 2.0 * difficolta + offset + rng.normal(0, noise, len(theta))


# ─────────────────────────────────────────────────────────────────────────────
# BackendRun
# ─────────────────────────────────────────────────────────────────────────────

def test_failure_rate_and_valid_count():
    th = _design(10)
    qoi = np.array([1, 1, -1, -1, -1, np.nan, 1, 1, 1, 1], dtype=float)
    r = BackendRun(backend="x", theta=th, qoi=qoi)
    assert r.n_valid == 9
    assert r.failure_rate == pytest.approx(3 / 9)


def test_failure_rate_is_nan_when_no_run_is_valid():
    """NaN, not 0: a zero would read as 'never fails'."""
    th = _design(4)
    r = BackendRun(backend="x", theta=th, qoi=np.full(4, np.nan))
    assert r.n_valid == 0
    assert np.isnan(r.failure_rate)


def test_inconsistent_shapes_are_rejected():
    with pytest.raises(ValueError):
        BackendRun(backend="x", theta=_design(5), qoi=np.zeros(3))
    with pytest.raises(ValueError):
        BackendRun(backend="x", theta=np.zeros(5), qoi=np.zeros(5))


def test_file_roundtrip(tmp_path):
    """
    Saving to file is not an extra: the backends live in different Python
    environments and the comparison necessarily happens on files.
    """
    th = _design(20)
    qoi = _qoi_from_difficulty(th)
    qoi[3] = np.nan
    r = BackendRun(backend="md", theta=th, qoi=qoi, speed_scale=0.18,
                         control_hz=np.full(20, 10.0), note="prova")

    p = tmp_path / "sub" / "md.json"
    r.save(str(p))
    letto = BackendRun.load(str(p))

    assert letto.backend == "md"
    assert letto.speed_scale == pytest.approx(0.18)
    assert letto.note == "prova"
    np.testing.assert_allclose(letto.theta, r.theta)
    np.testing.assert_array_equal(np.isnan(letto.qoi), np.isnan(r.qoi))
    np.testing.assert_allclose(letto.qoi[~np.isnan(letto.qoi)],
                               r.qoi[~np.isnan(r.qoi)])
    np.testing.assert_array_equal(letto.valid, r.valid)


# ─────────────────────────────────────────────────────────────────────────────
# Spearman
# ─────────────────────────────────────────────────────────────────────────────

def test_spearman_is_high_when_backends_agree():
    th = _design(80)
    a = BackendRun(backend="A", theta=th, qoi=_qoi_from_difficulty(th, noise=0.02, seed=1))
    b = BackendRun(backend="B", theta=th, qoi=_qoi_from_difficulty(th, noise=0.02, seed=2))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert e.spearman["A ~ B"] > 0.9
    assert e.shared_design["A ~ B"]


def test_spearman_is_low_when_they_order_differently():
    th = _design(80)
    rng = np.random.default_rng(5)
    a = BackendRun(backend="A", theta=th, qoi=_qoi_from_difficulty(th, seed=1))
    b = BackendRun(backend="B", theta=th, qoi=rng.normal(size=len(th)))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert abs(e.spearman["A ~ B"]) < 0.4


def test_spearman_is_invariant_to_a_difficulty_offset():
    """
    This is the property that makes Spearman the load-bearing metric: two
    backends tuned to different operating points have different failure rates
    but, if they agree on which scenarios are hard, rho stays high.
    """
    th = _design(80)
    a = BackendRun(backend="A", theta=th,
                         qoi=_qoi_from_difficulty(th, noise=0.01, seed=1))
    b = BackendRun(backend="B", theta=th,
                         qoi=_qoi_from_difficulty(th, noise=0.01, offset=-0.8, seed=1))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert e.spearman["A ~ B"] > 0.95
    assert e.failure_rates["B"] > e.failure_rates["A"]   # different rates, agreement intact


def test_spearman_is_refused_when_designs_differ():
    """
    With different designs, scenario i of A would be correlated with scenario i
    of B, which are DIFFERENT scenarios. The number would look like a
    correlation but is not one: NaN is better.
    """
    a = BackendRun(backend="A", theta=_design(60, seed=0),
                         qoi=_qoi_from_difficulty(_design(60, seed=0)))
    b = BackendRun(backend="B", theta=_design(60, seed=99),
                         qoi=_qoi_from_difficulty(_design(60, seed=99)))

    c = CrossSimulatorComparison().add(a).add(b)
    e = c.compare(LOWER, UPPER)

    assert not e.shared_design["A ~ B"]
    assert np.isnan(e.spearman["A ~ B"])
    assert any("DIFFERENT parameter designs" in x for x in e.caveats)


def test_shared_design_tolerates_rounding():
    """
    The backends round angles to integers before sending them: half a degree of
    difference is expected and must not invalidate the comparison.
    """
    th = _design(40)
    th_rounded = th.copy()
    th_rounded[:, :5] = np.round(th_rounded[:, :5])

    a = BackendRun(backend="A", theta=th, qoi=_qoi_from_difficulty(th))
    b = BackendRun(backend="B", theta=th_rounded,
                         qoi=_qoi_from_difficulty(th_rounded))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert e.shared_design["A ~ B"]
    assert not np.isnan(e.spearman["A ~ B"])


# ─────────────────────────────────────────────────────────────────────────────
# Caveats -- the cases where the comparison must warn
# ─────────────────────────────────────────────────────────────────────────────

def test_warns_on_a_backend_failing_always():
    th = _design(40)
    a = BackendRun(backend="A", theta=th, qoi=np.full(40, -1.0))
    b = BackendRun(backend="B", theta=th, qoi=_qoi_from_difficulty(th))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert any("100% failures" in x for x in e.caveats)
    assert any("diag_odd_feasibility" in x for x in e.caveats)


def test_warns_on_a_backend_never_failing():
    th = _design(40)
    a = BackendRun(backend="A", theta=th, qoi=np.full(40, 1.0))
    b = BackendRun(backend="B", theta=th, qoi=_qoi_from_difficulty(th))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert any("0% failures" in x for x in e.caveats)


def test_warns_on_different_speed_scale():
    """Failure rates are not comparable when the operating points differ."""
    th = _design(40)
    a = BackendRun(backend="A", theta=th, qoi=_qoi_from_difficulty(th, seed=1),
                         speed_scale=1.0)
    b = BackendRun(backend="B", theta=th, qoi=_qoi_from_difficulty(th, seed=2),
                         speed_scale=0.18)

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert any("NOT comparable" in x for x in e.caveats)


def test_warns_on_very_different_control_rates():
    th = _design(40)
    a = BackendRun(backend="A", theta=th, qoi=_qoi_from_difficulty(th, seed=1),
                         control_hz=np.full(40, 20.8))
    b = BackendRun(backend="B", theta=th, qoi=_qoi_from_difficulty(th, seed=2),
                         control_hz=np.full(40, 8.5))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert any("Very different control rates" in x for x in e.caveats)


def test_no_failures_means_no_regions():
    th = _design(30)
    a = BackendRun(backend="A", theta=th, qoi=np.full(30, 1.0))
    b = BackendRun(backend="B", theta=th, qoi=np.full(30, 1.0))

    e = CrossSimulatorComparison().add(a).add(b).compare(LOWER, UPPER)
    assert e.n_regions == 0
    assert any("No backend produced any failure" in x for x in e.caveats)


# ─────────────────────────────────────────────────────────────────────────────
# Contratto
# ─────────────────────────────────────────────────────────────────────────────

def test_at_least_two_backends_are_required():
    th = _design(10)
    c = CrossSimulatorComparison().add(
        BackendRun(backend="A", theta=th, qoi=_qoi_from_difficulty(th)))
    with pytest.raises(ValueError, match="at least two"):
        c.compare(LOWER, UPPER)


def test_duplicate_labels_are_rejected():
    """With two 'A' labels the pairwise comparison would become ambiguous."""
    th = _design(10)
    c = CrossSimulatorComparison().add(
        BackendRun(backend="A", theta=th, qoi=_qoi_from_difficulty(th)))
    with pytest.raises(ValueError, match="already present"):
        c.add(BackendRun(backend="A", theta=th,
                                    qoi=_qoi_from_difficulty(th)))


def test_report_before_compare_raises():
    assert pytest.raises(RuntimeError, CrossSimulatorComparison().report)


def test_report_contains_the_essentials():
    th = _design(60)
    a = BackendRun(backend="lane_keeping", theta=th, speed_scale=0.42,
                         qoi=_qoi_from_difficulty(th, noise=0.05, seed=1),
                         control_hz=np.full(60, 20.8))
    b = BackendRun(backend="lane_keeping_md", theta=th, speed_scale=0.18,
                         qoi=_qoi_from_difficulty(th, noise=0.05, seed=2),
                         control_hz=np.full(60, 10.0))

    c = CrossSimulatorComparison().add(a).add(b)
    c.compare(LOWER, UPPER)
    text = c.report()

    assert "lane_keeping_md" in text
    assert "0.4200" in text and "0.1800" in text     # the declared speed_scale
    assert "NOT comparable" in text                  # the caveat on failure rates
    assert "Spearman" in text
    assert "FAILURE REGIONS" in text


def test_outcome_is_serialisable(tmp_path):
    th = _design(40)
    c = (CrossSimulatorComparison()
         .add(BackendRun(backend="A", theta=th,
                                    qoi=_qoi_from_difficulty(th, noise=0.05, seed=1)))
         .add(BackendRun(backend="B", theta=th,
                                    qoi=_qoi_from_difficulty(th, noise=0.05, seed=2))))
    e = c.compare(LOWER, UPPER)
    p = tmp_path / "comparison.json"
    e.save(str(p))

    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["backend"] == ["A", "B"]
    assert "spearman" in d and "failure_rates" in d


def test_three_backends():
    """The comparison is pairwise: three backends need three pairs."""
    th = _design(50)
    c = CrossSimulatorComparison()
    for i, name in enumerate(("udacity", "metadrive", "carla")):
        c.add(BackendRun(backend=name, theta=th,
                                    qoi=_qoi_from_difficulty(th, noise=0.05, seed=i)))
    e = c.compare(LOWER, UPPER)
    assert len(e.spearman) == 3
    assert "udacity ~ carla" in e.spearman
