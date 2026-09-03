"""
Tests for the active-learning boundary learner (Step B).

Uses a MOCK scenario with a KNOWN analytic boundary (fail iff p0 + p1 > c), so we
can check that the learner (a) recovers P(failure) close to the ground truth and
(b) identifies the two relevant parameters via ARD importance — all WITHOUT
any simulator or the scenario registry (the mock is a plain duck-typed object).

Run:  pytest tests/test_active_boundary.py -q
"""
import numpy as np

from pipeline.active_boundary import (
    run_active_boundary, failure_probability, greedy_diverse, rbf_lengthscales,
)


class MockScenario:
    """Fail iff p0 + p1 > c. Extra dims are irrelevant. Uniform ODD => P(fail)=0.5
    for c=1 over the unit cube."""

    def __init__(self, d: int = 3, c: float = 1.0):
        self.d = d
        self.c = c
        self._valid_mask = None

    def param_bounds(self):
        return {"names": [f"p{j}" for j in range(self.d)],
                "lower": np.zeros(self.d), "upper": np.ones(self.d)}

    def param_distributions(self, lower=None, upper=None):
        from scipy import stats
        lo = np.zeros(self.d) if lower is None else np.asarray(lower, float)
        hi = np.ones(self.d) if upper is None else np.asarray(upper, float)
        return [stats.uniform(loc=lo[j], scale=max(hi[j] - lo[j], 1e-9))
                for j in range(self.d)]

    def failure_threshold(self):
        return 0.0

    def run_simulation(self, theta, verbose=False):
        theta = np.asarray(theta, float)
        return np.zeros((len(theta), 2, 4), dtype=float)  # dummy trajectories

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        margin = self.c - (theta[:, 0] + theta[:, 1])     # <0 => failure
        self._valid_mask = np.ones(len(theta), dtype=bool)
        return margin


# ── helper unit tests ─────────────────────────────────────────────────────────

def test_p_fail_monotonic():
    # More negative predicted margin -> higher P(fail).
    p_low = failure_probability(np.array([1.0]), np.array([0.5]), thr=0.0)[0]
    p_high = failure_probability(np.array([-1.0]), np.array([0.5]), thr=0.0)[0]
    assert p_high > 0.5 > p_low


def test_greedy_diverse_spreads():
    X = np.array([[0, 0], [0.01, 0], [1, 1]], float)
    scores = np.array([1.0, 0.9, 0.8])
    pick = greedy_diverse(scores, X, k=2, min_dist=0.5)
    assert 0 in pick and 2 in pick     # skips the near-duplicate of point 0


def test_rbf_lengthscales_extraction():
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    k = ConstantKernel(1.0) * RBF(length_scale=[1.0, 2.0, 3.0]) + WhiteKernel(0.1)
    ls = rbf_lengthscales(k, 3)
    assert np.allclose(ls, [1.0, 2.0, 3.0])


# ── end-to-end on the mock ────────────────────────────────────────────────────

def test_recovers_probability_and_boundary():
    mock = MockScenario(d=3, c=1.0)   # ground-truth P(fail | uniform) = 0.5
    res = run_active_boundary(
        mock, seed=0, n_seed=40, batch=10, n_iter=3,
        weight_by_odd=True, pool_size=1500, odd_samples=3000, verbose=False,
    )
    # (a) P(failure) close to the analytic 0.5
    assert abs(res.p_fail - 0.5) < 0.12
    # CI is a proper interval that brackets the estimate
    lo, hi = res.p_fail_ci
    assert lo <= res.p_fail <= hi
    # (b) the two relevant params dominate the ARD importance
    imp = res.param_importance
    assert imp[0] + imp[1] > imp[2]
    # budget accounting
    assert res.n_evaluations == len(res.margins) == len(res.labels)
    assert res.n_evaluations >= 40


def test_summary_fields_present():
    mock = MockScenario(d=3, c=1.0)
    res = run_active_boundary(mock, seed=1, n_seed=30, batch=8, n_iter=2,
                              pool_size=1000, odd_samples=1500)
    s = res.boundary_summary
    for key in ("weighting", "p_fail", "p_fail_ci", "top_params", "length_scales"):
        assert key in s
    assert len(s["top_params"]) <= 5
