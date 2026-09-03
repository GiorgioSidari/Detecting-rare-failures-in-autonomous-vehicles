"""
Tests for the LHS-vs-random-search feature: samplers, the two sampler-parameterised
algorithms, the failure-region comparison and the QoI worst-case optimisers.

Everything runs on MOCK scenarios with analytically known answers — no Docker, no
simulator, no scenario registry. Each mock is a plain duck-typed object exposing
the BaseScenario interface, exactly as tests/test_active_boundary.py does.

Run:  pytest tests/test_random_search_comparison.py -q
"""
import os

import numpy as np
import pytest
from scipy import stats

from pipeline.active_boundary_random import (
    ActiveBoundaryRunner, LHSActiveBoundary, RandomSearchActiveBoundary,
)
from pipeline.failure_regions import extract_failure_regions
from pipeline.region_comparison import compare_failure_regions
from pipeline.model_comparison import ModelComparison, PlainSamplingBaseline
from pipeline.qoi_optimizer import BayesianQoIOptimizer, CMAESQoIOptimizer, optimize_qoi
from pipeline.rare_event_random import (
    LHSCrossEntropy, RandomSearchCrossEntropy,
)
from pipeline.samplers import (
    LHSSampler, RandomSampler, discrepancy, get_sampler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mocks
# ─────────────────────────────────────────────────────────────────────────────
class LinearMock:

    def __init__(self, d: int = 3, c: float = 1.0):
        """Fail iff p0 + p1 > c. Uniform ODD on the unit cube => P(fail) = 0.5 at c = 1."""
        self.d, self.c = d, c
        self._valid_mask = None

    def param_bounds(self):
        return {"names": [f"p{j}" for j in range(self.d)],
                "lower": np.zeros(self.d), "upper": np.ones(self.d)}

    def param_distributions(self, lower=None, upper=None):
        lo = np.zeros(self.d) if lower is None else np.asarray(lower, float)
        hi = np.ones(self.d) if upper is None else np.asarray(upper, float)
        return [stats.uniform(loc=lo[j], scale=max(hi[j] - lo[j], 1e-9))
                for j in range(self.d)]

    def failure_threshold(self):
        return 0.0

    def run_simulation(self, theta, verbose=False):
        return np.zeros((len(np.asarray(theta)), 2, 4), dtype=float)

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        self._valid_mask = np.ones(len(theta), dtype=bool)
        return self.c - (theta[:, 0] + theta[:, 1])


class TwoRegionMock(LinearMock):

    def __init__(self, d: int = 4):
        """
        Two disjoint failure regions with very different volumes:

          A (broad) : p0 > 0.80 and p1 > 0.80            volume 0.04
          B (rare)  : |p2 - 0.42| < 0.02 and p3 > 0.90   volume 0.004

        Ground truth for the region-comparison tests: any method that explores the
        space should find A; B is the one that separates the designs.
        """
        super().__init__(d=d)

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        self._valid_mask = np.ones(len(theta), dtype=bool)
        mA = np.maximum(0.80 - theta[:, 0], 0.80 - theta[:, 1])
        mB = np.maximum(np.abs(theta[:, 2] - 0.42) - 0.02, 0.90 - theta[:, 3])
        return np.minimum(mA, mB)


class BowlMock(LinearMock):

    def __init__(self, target=(0.8, 0.2, 0.5)):
        """Smooth bowl with a known minimum at `target`: margin = ||theta - target||^2 - 0.5."""
        super().__init__(d=len(target))
        self.target = np.asarray(target, float)

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        self._valid_mask = np.ones(len(theta), dtype=bool)
        return np.sum((theta - self.target) ** 2, axis=1) - 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Samplers
# ─────────────────────────────────────────────────────────────────────────────
def test_sampler_registry_and_shapes():
    for name, cls in (("lhs", LHSSampler), ("random", RandomSampler)):
        s = get_sampler(name)
        assert isinstance(s, cls) and s.name == name
        u = s.unit(32, 5, seed=0)
        assert u.shape == (32, 5)
        assert u.min() >= 0.0 and u.max() < 1.0
    with pytest.raises(KeyError):
        get_sampler("sobol_not_registered")


def test_sampler_is_reproducible_and_seed_sensitive():
    s = get_sampler("random")
    assert np.allclose(s.unit(16, 3, seed=7), s.unit(16, 3, seed=7))
    assert not np.allclose(s.unit(16, 3, seed=7), s.unit(16, 3, seed=8))


def test_lhs_covers_every_stratum_but_random_does_not():
    """The property the whole comparison rests on: LHS fills every 1-D stratum."""
    n, d = 40, 4
    u_lhs = get_sampler("lhs").unit(n, d, seed=0)
    for j in range(d):
        strata = np.floor(u_lhs[:, j] * n).astype(int)
        assert len(np.unique(strata)) == n          # one point per stratum, exactly

    # Random search leaves holes: with n points in n strata the expected number
    # of empty strata is n/e ~ 37%. Assert it misses at least one.
    u_rnd = get_sampler("random").unit(n, d, seed=0)
    empty = [n - len(np.unique(np.floor(u_rnd[:, j] * n).astype(int))) for j in range(d)]
    assert max(empty) > 0


def test_lhs_has_lower_discrepancy_than_random():
    u_lhs = get_sampler("lhs").unit(64, 4, seed=1)
    u_rnd = get_sampler("random").unit(64, 4, seed=1)
    assert discrepancy(u_lhs) < discrepancy(u_rnd)


def test_from_dists_matches_the_target_distribution():
    dists = [stats.norm(loc=3.0, scale=2.0)]
    x = get_sampler("lhs").from_dists(4000, dists, seed=0)
    assert abs(float(x.mean()) - 3.0) < 0.15
    assert abs(float(x.std()) - 2.0) < 0.15
    assert np.isfinite(x).all()                      # ppf(0)/ppf(1) are clipped away


# ─────────────────────────────────────────────────────────────────────────────
# Active boundary with a pluggable sampler
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cls", [LHSActiveBoundary, RandomSearchActiveBoundary])
def test_active_boundary_arms_recover_the_probability(cls):
    """Both designs must recover P(fail) = 0.5 on the linear mock: same algorithm."""
    runner = cls(LinearMock(d=3), n_seed=40, batch=10, n_iter=3,
                 pool_size=1500, odd_samples=1500)
    res = runner.run(seed=0)
    assert abs(res.p_fail - 0.5) < 0.12
    lo, hi = res.p_fail_ci
    assert lo <= res.p_fail <= hi
    imp = res.param_importance
    assert imp[0] + imp[1] > imp[2]                  # the two relevant params dominate
    assert res.n_evaluations == len(res.margins) == len(res.labels)


def test_active_boundary_budget_is_matched_across_samplers():
    kw = dict(n_seed=20, batch=6, n_iter=2, pool_size=600, odd_samples=800)
    a = LHSActiveBoundary(LinearMock(d=3), **kw)
    b = RandomSearchActiveBoundary(LinearMock(d=3), **kw)
    assert a.budget == b.budget == 20 + 6 * 2
    ra, rb = a.run(seed=0), b.run(seed=0)
    assert ra.n_evaluations == rb.n_evaluations == a.budget
    assert ra.boundary_summary["sampler"] == "lhs"
    assert rb.boundary_summary["sampler"] == "random"


def test_active_boundary_matches_the_legacy_function():
    """
    The LHS arm must reproduce the existing run_active_boundary() behaviour.

    The seed design is compared exactly; the probability only within tolerance,
    because the new class pins the GP's random_state (see build_seeded_gp) while
    the legacy function leaves the hyper-parameter restarts to numpy's global
    RNG. Same algorithm, one fewer source of irreproducibility.
    """
    from pipeline.active_boundary import run_active_boundary
    kw = dict(n_seed=30, batch=8, n_iter=2, pool_size=800, odd_samples=1200)
    legacy = run_active_boundary(LinearMock(d=3), seed=3, **kw)
    new = LHSActiveBoundary(LinearMock(d=3), **kw).run(seed=3)
    assert np.allclose(legacy.theta_evaluated[:30], new.theta_evaluated[:30])
    assert abs(legacy.p_fail - new.p_fail) < 0.1


def test_active_boundary_is_reproducible_across_runs():
    """Same seed, same result — the premise of every multi-seed comparison."""
    kw = dict(n_seed=20, batch=6, n_iter=2, pool_size=600, odd_samples=600)
    a = LHSActiveBoundary(LinearMock(d=3), **kw).run(seed=5)
    b = LHSActiveBoundary(LinearMock(d=3), **kw).run(seed=5)
    assert np.allclose(a.theta_evaluated, b.theta_evaluated)
    assert a.p_fail == pytest.approx(b.p_fail)


def test_random_acquisition_is_a_pure_search_baseline():
    runner = ActiveBoundaryRunner(LinearMock(d=3), sampler="random",
                                  acquisition="random", n_seed=20, batch=6,
                                  n_iter=2, pool_size=500, odd_samples=800)
    res = runner.run(seed=0)
    assert "randacq" in runner.label
    assert res.n_evaluations == 20 + 6 * 2


def test_active_boundary_rejects_a_bad_acquisition():
    with pytest.raises(ValueError):
        ActiveBoundaryRunner(LinearMock(d=2), acquisition="magic")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-entropy with a pluggable sampler
# ─────────────────────────────────────────────────────────────────────────────
class TailMock(LinearMock):

    def __init__(self):
        """Fail iff p0 > 0.9 under a uniform ODD => P(fail) = 0.10 exactly."""
        super().__init__(d=2)

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        self._valid_mask = np.ones(len(theta), dtype=bool)
        return 0.9 - theta[:, 0]


@pytest.mark.parametrize("cls", [LHSCrossEntropy, RandomSearchCrossEntropy])
def test_cross_entropy_arms_recover_a_known_tail_probability(cls):
    runner = cls(TailMock(), samples_per_iter=200, max_iter=6, final_samples=1500)
    res = runner.run(seed=0)
    assert abs(res.p_fail - 0.10) < 0.04             # ground truth 0.10
    lo, hi = res.ci
    assert lo <= res.p_fail <= hi
    # The evaluated cloud is exported for the region analysis.
    assert res.theta_evaluated.shape[0] == len(res.margins) == len(res.labels)
    assert res.n_evaluations >= res.theta_evaluated.shape[0]


def test_cross_entropy_random_arm_agrees_with_the_legacy_function():
    """sampler='random' is the current rare_event behaviour, up to the RNG stream."""
    from pipeline.rare_event import estimate_failure_probability, scenario_margin_fn
    sc = TailMock()
    b = sc.param_bounds()
    kw = dict(samples_per_iter=200, max_iter=6, final_samples=1500)
    legacy = estimate_failure_probability(
        scenario_margin_fn(sc), sc.param_distributions(b["lower"], b["upper"]),
        b["lower"], b["upper"], threshold=0.0, seed=0, **kw)
    new = RandomSearchCrossEntropy(sc, **kw).run(seed=0)
    assert abs(legacy.p_fail - new.p_fail) < 0.03    # same estimator, different stream


def test_cross_entropy_needs_an_operational_distribution():
    class NoODD(TailMock):
        """A scenario with no operational distribution: CE has nothing to tilt."""
        param_distributions = None

        def __getattribute__(self, item):
            if item == "param_distributions":
                raise AttributeError(item)
            return super().__getattribute__(item)

    with pytest.raises(ValueError):
        RandomSearchCrossEntropy(NoODD())


# ─────────────────────────────────────────────────────────────────────────────
# Failure regions
# ─────────────────────────────────────────────────────────────────────────────
def test_extract_regions_separates_two_known_clusters():
    theta = np.vstack([
        np.random.default_rng(0).uniform([0.85, 0.85], [1.0, 1.0], size=(20, 2)),
        np.random.default_rng(1).uniform([0.0, 0.0], [0.08, 0.08], size=(15, 2)),
    ])
    margins = -np.ones(len(theta))                    # all failing
    regions = extract_failure_regions(theta, margins, [0, 0], [1, 1],
                                      eps=0.2, min_samples=2)
    assert len(regions) == 2
    assert sum(r.n_points for r in regions) == len(theta)
    # Regions come back sorted rarest-first and every probability is a probability.
    assert all(0.0 <= r.p_hit_uniform <= 1.0 for r in regions)
    assert regions[0].p_hit_odd <= regions[-1].p_hit_odd


def test_isolated_failure_becomes_its_own_region():
    """A lone rare failure must never be discarded as clustering noise."""
    theta = np.vstack([np.full((10, 2), 0.9) + 0.01, [[0.05, 0.05]]])
    margins = -np.ones(len(theta))
    regions = extract_failure_regions(theta, margins, [0, 0], [1, 1],
                                      eps=0.2, min_samples=3)
    singles = [r for r in regions if r.is_singleton]
    assert len(singles) == 1
    assert np.allclose(singles[0].centroid, [0.05, 0.05])


def test_single_point_region_has_no_estimable_rarity():
    """One draw measures no span: report n/a, never a fabricated 1e-15."""
    regions = extract_failure_regions(np.array([[0.5, 0.5]]), np.array([-1.0]),
                                      [0, 0], [1, 1], pad=0.01)
    r = regions[0]
    assert r.is_singleton and not r.is_estimable
    assert np.isnan(r.p_hit_uniform) and np.isnan(r.n_for_50pct("uniform"))
    assert r.constraining_axes == []


def test_hit_probability_economics_are_consistent():
    """A tight cluster on one axis, loose on the other: only the tight axis counts."""
    rng = np.random.default_rng(0)
    n = 30
    theta = np.column_stack([rng.uniform(0.40, 0.44, n),     # constrained
                             rng.uniform(0.0, 1.0, n)])      # free
    regions = extract_failure_regions(theta, -np.ones(n), [0, 0], [1, 1],
                                      eps=1.0, pad=0.01)
    r = regions[0]
    assert r.n_points == n
    assert r.constraining_axes == [0]                 # axis 1 is not a constraint
    assert 0.02 < r.p_hit_uniform < 0.12              # ~the width of the slab
    assert r.p_hit_in_n(1) == pytest.approx(r.p_hit_uniform, rel=1e-6)
    assert r.p_hit_in_n(10_000) > r.p_hit_in_n(100)
    n50 = r.n_for_50pct("uniform")
    assert r.p_hit_in_n(int(round(n50)), "uniform") == pytest.approx(0.5, abs=0.01)


def test_unlocalised_group_is_not_called_rare():
    """Points scattered over the ODD constrain nothing: P(hit) = 1, not 1e-9."""
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 1, size=(40, 5))
    regions = extract_failure_regions(theta, -np.ones(40), np.zeros(5), np.ones(5),
                                      eps=5.0)
    assert len(regions) == 1
    assert regions[0].constraining_axes == []
    assert regions[0].p_hit_odd == 1.0
    assert "no axis constrained" in regions[0].describe_constraints()


def test_two_point_group_does_not_invent_constraints():
    """
    The failure mode that made the 9-D report meaningless: two neighbouring
    points look tightly bounded on most axes, and reporting that as a region
    turns it into P(hit) ~ 1e-9. Groups below min_points get no bounds at all.
    """
    rng = np.random.default_rng(3)
    spurious = 0
    for _ in range(40):
        theta = rng.uniform(0, 1, size=(2, 9))
        regions = extract_failure_regions(theta, -np.ones(2), np.zeros(9),
                                          np.ones(9), eps=5.0)
        spurious += len(regions[0].constraining_axes)
        assert not regions[0].is_estimable
    assert spurious == 0


def test_eps_is_estimated_and_scales_with_dimension():
    """A radius tuned in 2-D isolates every point in 9-D; the estimate must not."""
    from pipeline.failure_regions import estimate_eps
    rng = np.random.default_rng(0)
    eps2 = estimate_eps(rng.uniform(0, 1, size=(200, 2)))
    eps9 = estimate_eps(rng.uniform(0, 1, size=(200, 9)))
    assert eps9 > 2 * eps2                            # grows with the dimension
    # And with the estimate the 9-D cloud does NOT shatter into singletons: most
    # POINTS end up in a real cluster (counting regions would be misleading, since
    # a handful of stragglers each contribute a region of their own).
    theta = rng.uniform(0, 1, size=(120, 9))
    regions = extract_failure_regions(theta, -np.ones(120), np.zeros(9), np.ones(9))
    isolated = sum(r.n_points for r in regions if r.is_singleton)
    assert isolated < 0.5 * len(theta)

    # The old fixed radius is what broke: in 9-D it isolates essentially everything.
    shattered = extract_failure_regions(theta, -np.ones(120), np.zeros(9),
                                        np.ones(9), eps=0.25)
    assert sum(r.is_singleton for r in shattered) == len(theta)


def test_no_failures_yields_no_regions():
    assert extract_failure_regions(np.random.rand(10, 3), np.ones(10),
                                   [0, 0, 0], [1, 1, 1]) == []


def test_compare_detects_disjoint_exploration():
    """Two methods that explore different corners must show Jaccard = 0."""
    rng = np.random.default_rng(0)
    a_theta = rng.uniform([0.85, 0.85], [1.0, 1.0], size=(12, 2))
    b_theta = rng.uniform([0.0, 0.0], [0.08, 0.08], size=(12, 2))
    runs = {"A": (a_theta, -np.ones(12)), "B": (b_theta, -np.ones(12))}
    cmp_ = compare_failure_regions(runs, [0, 0], [1, 1], eps=0.15,
                                   param_names=["x", "y"])
    pw = cmp_.pairwise[("A", "B")]
    assert pw["jaccard"] == 0.0
    assert pw["coverage_a_in_b"] == 0.0
    assert len(cmp_.exclusive_regions("A")) == len(cmp_.discovery["A"]) >= 1
    assert "FAILURE REGIONS" in cmp_.report()
    assert cmp_.to_dict()["regions"][0]["p_hit_uniform"] >= 0.0


def test_compare_detects_identical_exploration():
    rng = np.random.default_rng(2)
    pts = rng.uniform([0.9, 0.9], [1.0, 1.0], size=(15, 2))
    runs = {"A": (pts, -np.ones(15)), "B": (pts.copy(), -np.ones(15))}
    cmp_ = compare_failure_regions(runs, [0, 0], [1, 1], eps=0.2)
    pw = cmp_.pairwise[("A", "B")]
    assert pw["jaccard"] == 1.0
    assert pw["coverage_a_in_b"] == 1.0
    assert cmp_.exclusive_regions("A") == []


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison harness
# ─────────────────────────────────────────────────────────────────────────────
def test_model_comparison_runs_all_arms_and_reports():
    sc = TwoRegionMock()
    arms = [
        LHSActiveBoundary(sc, n_seed=25, batch=8, n_iter=2,
                          pool_size=800, odd_samples=800),
        RandomSearchActiveBoundary(sc, n_seed=25, batch=8, n_iter=2,
                                   pool_size=800, odd_samples=800),
        PlainSamplingBaseline(sc, "lhs", n_samples=41),
        PlainSamplingBaseline(sc, "random", n_samples=41),
    ]
    res = ModelComparison(sc, arms, seeds=[0, 1], budget=41, verbose=False).run()

    assert len(res.arm_labels) == 4
    for lab in res.arm_labels:
        assert res.per_arm[lab]["n_seeds"] == 2
        assert res.per_arm[lab]["mean_evaluations"] > 0
    assert "MODEL COMPARISON" in res.report()
    d = res.to_dict()
    assert set(d["arms"]) == set(res.arm_labels)
    assert "regions" in d


def test_model_comparison_survives_a_failing_arm():
    class Exploding:
        label = "exploding_arm"
        budget = 10

        def run(self, seed=0):
            raise RuntimeError("simulator unreachable")

    sc = LinearMock(d=3)
    arms = [PlainSamplingBaseline(sc, "lhs", n_samples=30), Exploding()]
    res = ModelComparison(sc, arms, seeds=[0], budget=30, verbose=False).run()
    assert res.arm_labels == ["plain_sampling[lhs]"]
    assert res.failed_runs and res.failed_runs[0][0] == "exploding_arm"


def test_model_comparison_saves_json_and_csv(tmp_path):
    sc = TwoRegionMock()
    arms = [PlainSamplingBaseline(sc, "lhs", n_samples=60),
            PlainSamplingBaseline(sc, "random", n_samples=60)]
    res = ModelComparison(sc, arms, seeds=[0], budget=60, verbose=False).run()
    files = res.save(str(tmp_path / "cmp"))
    assert len(files) == 4                      # raw npz + json + 2 csv
    for f in files:
        assert os.path.getsize(f) > 0

    # The raw dump must round-trip: the simulations cost hours, the analysis on
    # top of them is cheap and will be redone with different settings.
    import numpy as _np
    z = _np.load(files[0], allow_pickle=True)
    labels = [str(x) for x in z["labels"]]
    assert labels == res.arm_labels
    for i, lab in enumerate(labels):
        th, mg = res.clouds[lab]
        assert _np.allclose(z[f"theta_{i}"], th)
        assert _np.allclose(z[f"margins_{i}"], mg, equal_nan=True)


def test_plain_baseline_recovers_the_uniform_failure_rate():
    res = PlainSamplingBaseline(LinearMock(d=3), "lhs", n_samples=800).run(seed=0)
    assert abs(res.p_fail - 0.5) < 0.06
    lo, hi = res.p_fail_ci
    assert lo <= res.p_fail <= hi


# ─────────────────────────────────────────────────────────────────────────────
# QoI worst-case optimisers
# ─────────────────────────────────────────────────────────────────────────────
def test_cmaes_finds_the_known_minimum():
    target = np.array([0.8, 0.2, 0.5])
    res = CMAESQoIOptimizer(BowlMock(tuple(target)), budget=300, sigma0=0.3).run(seed=0)
    assert res.best_margin < -0.45                    # true minimum is -0.5
    assert np.linalg.norm(res.best_theta - target) < 0.2
    assert res.n_evaluations <= 300
    assert res.history == sorted(res.history, reverse=True)   # best-so-far only improves


def test_bayesopt_finds_the_known_minimum_and_ranks_parameters():
    target = np.array([0.8, 0.2, 0.5])
    res = BayesianQoIOptimizer(BowlMock(tuple(target)), budget=70, batch=5,
                               n_init=25).run(seed=0)
    assert res.best_margin < -0.35
    assert res.n_evaluations <= 70
    assert res.param_importance is not None
    assert abs(float(res.param_importance.sum()) - 1.0) < 1e-9
    assert "WORST-CASE" in res.report()


def test_optimizer_finds_the_worst_failure_of_a_two_region_scenario():
    """The optimiser must land INSIDE a failure region, not merely near one."""
    res = optimize_qoi(TwoRegionMock(), method="cmaes", budget=250, seed=1)
    assert res.best_margin < 0.0
    assert res.labels.sum() >= 1


def test_optimizer_plugs_into_the_comparison_harness():
    sc = TwoRegionMock()
    arms = [CMAESQoIOptimizer(sc, budget=60),
            PlainSamplingBaseline(sc, "random", n_samples=60)]
    res = ModelComparison(sc, arms, seeds=[0], budget=60, verbose=False).run()
    assert "qoi_cmaes" in res.arm_labels


def test_unknown_optimizer_is_rejected():
    with pytest.raises(KeyError):
        optimize_qoi(BowlMock(), method="gradient_descent")


# ─────────────────────────────────────────────────────────────────────────────
# The verdict: the one line the whole report exists to produce
# ─────────────────────────────────────────────────────────────────────────────
def _cloud(rng, n, lo, hi, d=4):
    """n failing points inside the box [lo, hi] on the first 2 axes, free elsewhere."""
    X = rng.uniform(0, 1, size=(n, d))
    X[:, 0] = rng.uniform(lo, hi, n)
    X[:, 1] = rng.uniform(lo, hi, n)
    return X


def test_verdict_says_not_answerable_when_failures_are_everywhere():
    """The situation on the real simulator: failures spread over the whole ODD."""
    rng = np.random.default_rng(0)
    runs = {"m[lhs]": (rng.uniform(0, 1, (80, 4)), -np.ones(80)),
            "m[random]": (rng.uniform(0, 1, (80, 4)), -np.ones(80))}
    cmp_ = compare_failure_regions(runs, np.zeros(4), np.ones(4))
    assert not cmp_.is_rare_regime()
    # The uncircular signal: the real clustering groups no more tightly than
    # the same data with its axes shuffled.
    assert cmp_.structure["z"] < 2.0
    assert not cmp_.structure["has_structure"]
    assert "NOT ANSWERABLE" in cmp_.report()


def test_verdict_credits_lhs_when_it_reaches_regions_random_does_not():
    rng = np.random.default_rng(1)
    # LHS reaches three tight, well-separated regions; random only the first.
    lhs = np.vstack([_cloud(rng, 12, 0.00, 0.06), _cloud(rng, 12, 0.47, 0.53),
                     _cloud(rng, 12, 0.94, 1.00)])
    rnd = _cloud(rng, 12, 0.00, 0.06)
    runs = {"m[lhs]": (lhs, -np.ones(len(lhs))), "m[random]": (rnd, -np.ones(len(rnd)))}
    cmp_ = compare_failure_regions(runs, np.zeros(4), np.ones(4), eps=0.35)
    assert cmp_.is_rare_regime() and cmp_.structure["z"] > 2.0
    v = cmp_.sampler_verdict()["m"]
    assert v["verdict"] == "lhs" and v["decisive"]
    assert len(v["only_lhs"]) >= 2 and len(v["only_random"]) == 0
    assert "ANSWER:  YES" in cmp_.report()


@pytest.mark.parametrize("only_a,only_b,pool,expected", [
    ({1}, set(), 2, "tie"),            # one region of difference: noise
    ({1, 2}, set(), 3, "lhs"),         # two out of three: decisive
    (set(), {1, 2, 3}, 4, "random"),
    ({1, 2}, {3, 4}, 8, "tie"),        # both found their own: no winner
])
def test_verdict_threshold_is_not_fooled_by_a_single_region(only_a, only_b,
                                                            pool, expected):
    """
    The decision rule itself: a winner needs a gap of at least 2 regions AND at
    least 30% of what the family found. One region of difference is noise and
    the report must not sell it as a result.
    """
    gap = len(only_a) - len(only_b)
    decisive = abs(gap) >= 2 and abs(gap) / pool >= 0.3
    verdict = "tie" if not decisive else "lhs" if gap > 0 else "random"
    assert verdict == expected


def test_report_is_short_by_default_and_detailed_on_request():
    rng = np.random.default_rng(3)
    runs = {"m[lhs]": (rng.uniform(0, 1, (60, 4)), -np.ones(60)),
            "m[random]": (rng.uniform(0, 1, (60, 4)), -np.ones(60))}
    cmp_ = compare_failure_regions(runs, np.zeros(4), np.ones(4))
    short, long = cmp_.report(), cmp_.report(verbose=True)
    assert len(short.splitlines()) < len(long.splitlines())
    assert "Jaccard" not in short and "Jaccard" in long
    assert "ANSWER" in short                       # the verdict is never hidden


# ─────────────────────────────────────────────────────────────────────────────
# Paired seed-by-seed test — the sensitive instrument
# ─────────────────────────────────────────────────────────────────────────────
def _result_with(per_seed):
    from pipeline.model_comparison import ComparisonResult
    return ComparisonResult(
        arm_labels=list(per_seed), per_arm={}, per_seed=per_seed, clouds={},
        regions=None, seeds=[r["seed"] for r in next(iter(per_seed.values()))],
        budget=100, param_names=["p0"])


def _rows(fails, worst=None, base_seed=0):
    worst = worst or [-0.5] * len(fails)
    return [{"seed": i, "n_failures": f, "worst_margin": w, "p_fail": f / 100.0,
             "n_evaluations": 100}
            for i, (f, w) in enumerate(zip(fails, worst))]


def test_paired_test_detects_a_consistent_advantage():
    """LHS ahead in every seed: the test must reach its resolution floor."""
    res = _result_with({"m[lhs]": _rows([12, 15, 11, 14, 13, 16]),
                        "m[random]": _rows([9, 11, 8, 10, 10, 12])})
    pt = res.paired_test()["m"]
    assert pt["n_failures"]["lhs_wins"] == 6
    assert pt["n_failures"]["p_value"] == pytest.approx(pt["min_attainable_p"],
                                                       abs=1e-9)
    assert "**" in res._paired_block()


def test_paired_test_reports_the_resolution_floor():
    """
    With few seeds the Wilcoxon p cannot go below 2/2^n. The report must say so,
    otherwise p=0.06 on 6 seeds reads as 'no effect' when it means 'add seeds'.
    """
    # The real campaign's active_boundary numbers: 5 wins out of 6, p = 0.062.
    res = _result_with({"m[lhs]": _rows([31, 45, 38, 38, 35, 39]),
                        "m[random]": _rows([27, 36, 32, 31, 36, 22])})
    pt = res.paired_test()["m"]
    assert pt["min_attainable_p"] == pytest.approx(2 / 2 ** 6)
    assert pt["n_failures"]["lhs_wins"] == 5
    assert 0.05 < pt["n_failures"]["p_value"] < 0.10
    block = res._paired_block()
    assert "add seeds" in block and "suggestive" in block


def test_paired_test_finds_nothing_when_there_is_nothing():
    res = _result_with({"m[lhs]": _rows([10, 8, 12, 9, 11, 10]),
                        "m[random]": _rows([11, 9, 10, 10, 10, 11])})
    pt = res.paired_test()["m"]
    assert pt["n_failures"]["p_value"] > 0.10
    assert "No design difference" in res._paired_block()


def test_paired_test_reads_worst_margin_in_the_right_direction():
    """Lower margin = worse crash = better search. The win count must reflect it."""
    res = _result_with({
        "m[lhs]": _rows([5] * 6, worst=[-0.9, -0.8, -0.7, -0.85, -0.75, -0.95]),
        "m[random]": _rows([5] * 6, worst=[-0.4, -0.3, -0.5, -0.35, -0.45, -0.2])})
    assert res.paired_test()["m"]["worst_margin"]["lhs_wins"] == 6


def test_paired_test_ignores_unpaired_arms():
    res = _result_with({"m[lhs]": _rows([5] * 3), "other[lhs]": _rows([5] * 3)})
    assert res.paired_test() == {}


# ─────────────────────────────────────────────────────────────────────────────
# Execution order — the confound that measures the clock, not the design
# ─────────────────────────────────────────────────────────────────────────────
class DriftingMock(LinearMock):

    def __init__(self, d: int = 3, drift: float = 0.01):
        """
        A scenario that gets easier over time, like the real simulator warming up.
        Every call to compute_qoi shifts the margin up a little.
        """
        super().__init__(d=d)
        self.drift = drift
        self.calls = 0

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        self._valid_mask = np.ones(len(theta), dtype=bool)
        self.calls += 1
        return self.c - (theta[:, 0] + theta[:, 1]) + self.drift * self.calls


def _order_of(monkeypatch, order):
    """Record the (label, seed) sequence a ModelComparison actually executes."""
    sc = LinearMock(d=3)
    seen = []

    class Spy(PlainSamplingBaseline):
        def run(self, seed=0):
            seen.append((self.label, seed))
            return super().run(seed=seed)

    arms = [Spy(sc, "lhs", n_samples=12), Spy(sc, "random", n_samples=12)]
    ModelComparison(sc, arms, seeds=[0, 1, 2], budget=12, order=order,
                    verbose=False).run()
    return seen


def test_sequential_order_runs_one_arm_then_the_other():
    """The old behaviour, kept only to reproduce past campaigns."""
    seen = _order_of(None, "sequential")
    labels = [lab for lab, _ in seen]
    assert labels == ["plain_sampling[lhs]"] * 3 + ["plain_sampling[random]"] * 3


def test_interleaved_order_alternates_the_arms():
    """Seed-major: every seed runs both arms back to back, so drift is shared."""
    seen = _order_of(None, "interleaved")
    assert [s for _, s in seen] == [0, 0, 1, 1, 2, 2]
    for i in range(0, 6, 2):
        assert seen[i][0] != seen[i + 1][0]


def test_shuffled_order_is_mixed_and_reproducible():
    a = _order_of(None, "shuffled")
    b = _order_of(None, "shuffled")
    assert a == b                                   # same order_seed -> same order
    labels = [lab for lab, _ in a]
    # Not arm-major: the two arms are not in two contiguous blocks.
    blocks = 1 + sum(1 for x, y in zip(labels, labels[1:]) if x != y)
    assert blocks > 2
    assert sorted(a) == sorted(_order_of(None, "sequential"))   # same work, new order


def test_rejects_an_unknown_order():
    with pytest.raises(ValueError):
        ModelComparison(LinearMock(d=2), [PlainSamplingBaseline(LinearMock(d=2))],
                        order="alphabetical")


def test_drift_is_detected_and_flagged_when_arms_ran_in_blocks():
    """
    The situation that invalidated the first campaign: the machine drifts AND the
    arms ran one after the other, so 'LHS found more' and 'LHS ran first' are the
    same statement. The report must refuse to endorse the comparison.
    """
    sc = DriftingMock(d=3, drift=0.05)
    arms = [PlainSamplingBaseline(sc, "lhs", n_samples=20),
            PlainSamplingBaseline(sc, "random", n_samples=20)]
    res = ModelComparison(sc, arms, seeds=[0, 1, 2, 3, 4, 5], budget=20,
                          order="sequential", verbose=False).run()
    dr = res.drift_diagnostic()
    assert dr["drift_detected"] and not dr["interleaved"]
    block = res._paired_block()
    assert "ARM-MAJOR" in block and "confounded with execution order" in block


def test_drift_with_mixed_order_is_reported_as_noise_not_bias():
    sc = DriftingMock(d=3, drift=0.05)
    arms = [PlainSamplingBaseline(sc, "lhs", n_samples=20),
            PlainSamplingBaseline(sc, "random", n_samples=20)]
    res = ModelComparison(sc, arms, seeds=[0, 1, 2, 3, 4, 5], budget=20,
                          order="shuffled", verbose=False).run()
    dr = res.drift_diagnostic()
    assert dr["interleaved"]
    assert "does not bias the comparison" in res._paired_block()


def test_run_records_when_each_run_happened():
    sc = LinearMock(d=3)
    arms = [PlainSamplingBaseline(sc, "lhs", n_samples=15)]
    res = ModelComparison(sc, arms, seeds=[0, 1], budget=15, verbose=False).run()
    rows = res.per_seed["plain_sampling[lhs]"]
    assert [r["seed"] for r in rows] == [0, 1]          # rows stay seed-ordered
    assert all("run_index" in r and "started_at_s" in r for r in rows)


def test_margin_percentile_is_recorded_and_paired():
    """
    The robust metric. The failure LABEL flips on ~20% of the points from
    simulator noise; the margin does not, so the 5th percentile of the margins an
    arm reached is the statistic to compare designs on.
    """
    sc = TwoRegionMock()
    arms = [PlainSamplingBaseline(sc, "lhs", n_samples=60),
            PlainSamplingBaseline(sc, "random", n_samples=60)]
    res = ModelComparison(sc, arms, seeds=[0, 1, 2], budget=60, verbose=False).run()

    rows = res.per_seed["plain_sampling[lhs]"]
    assert all(np.isfinite(r["margin_q05"]) for r in rows)
    # It sits between the worst margin and the median, by construction.
    for r in rows:
        assert r["worst_margin"] <= r["margin_q05"]

    pt = res.paired_test()["plain_sampling"]
    assert "margin_q05" in pt and pt["margin_q05"]["n"] == 3
    block = res._paired_block()
    assert "5th pct margin" in block
    assert "Trust '5th pct margin'" in block


def test_paired_test_skips_metrics_an_old_campaign_lacks():
    """A campaign saved before margin_q05 existed must still produce a report."""
    from pipeline.model_comparison import ComparisonResult
    per_seed = {
        "m[lhs]": [{"seed": i, "n_failures": 10 + i, "worst_margin": -0.5,
                    "p_fail": 0.1, "n_evaluations": 100} for i in range(4)],
        "m[random]": [{"seed": i, "n_failures": 8 + i, "worst_margin": -0.4,
                       "p_fail": 0.08, "n_evaluations": 100} for i in range(4)],
    }
    res = ComparisonResult(arm_labels=list(per_seed), per_arm={}, per_seed=per_seed,
                           clouds={}, regions=None, seeds=[0, 1, 2, 3], budget=100,
                           param_names=["p0"])
    pt = res.paired_test()["m"]
    assert "margin_q05" not in pt and "n_failures" in pt
    assert "5th pct margin" not in res._paired_block()


def test_small_groups_do_not_get_spuriously_constrained_axes():
    """
    The over-fitting the 12-seed campaign exposed: three random points span half
    an axis on average, so a flat 'span < 0.5' rule reported them as constrained
    on six dimensions out of nine, with a P(hit) of 1e-8 that meant nothing. The
    threshold has to scale with the group size.
    """
    rng = np.random.default_rng(0)
    spurious = []
    for _ in range(60):
        theta = rng.uniform(0, 1, size=(4, 9))       # 4 unstructured points
        regions = extract_failure_regions(theta, -np.ones(4), np.zeros(9),
                                          np.ones(9), eps=5.0, min_points=4)
        spurious.append(len(regions[0].constraining_axes))
    # On unstructured data a 4-point group should almost never look constrained
    # on the majority of a 9-dimensional space.
    assert np.mean(spurious) < 2.0
    assert max(spurious) <= 5


def test_a_genuinely_tight_group_is_still_detected():
    """The stricter rule must not blind the analysis to real structure."""
    rng = np.random.default_rng(1)
    n = 12
    theta = rng.uniform(0, 1, size=(n, 9))
    theta[:, 3] = rng.uniform(0.40, 0.46, n)         # genuinely narrow
    regions = extract_failure_regions(theta, -np.ones(n), np.zeros(9), np.ones(9),
                                      eps=5.0)
    assert 3 in regions[0].constraining_axes
    assert len(regions[0].constraining_axes) <= 3    # and not much else
