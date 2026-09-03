"""
Tests for `pipeline.arm_ranking`: the rare-failure ranking, the paired tests and
the pre-registered sequence.

Groups of tests:

  * defensive budget -- `MIN_DEFENSIVE_SAMPLES` is the number of draws from the
    nominal distribution (`alpha * final_samples`), the warning fires below that
    floor and is silent above it, and an estimate flagged unusable is carried
    through the harness and printed without a number. With
    `final_samples = 45` and `alpha = 0.2` the estimator is quantised at
    multiples of `(1 / alpha) / final_samples` and its effective sample size
    collapses; the validated budget does not quantise on the same problem.
  * rarity -- the cut selects the declared fraction of the ODD, it depends on
    where a failure sits and not on which arm found it, and rankings computed on
    different ODDs refuse to be compared.
  * pairing -- runs are paired by seed when the provenance is present, and the
    ranking reports the absence of provenance instead of testing on a guess.
  * pre-registered sequence -- a clean sweep rejects with enough seeds and
    cannot at three, the sequence stops at the first non-rejection and marks the
    rest untested, an effect in the wrong direction is not a win, a missing arm
    stops the sequence, the shipped plan is valid and a malformed one is
    rejected.
  * degenerate campaigns -- a campaign with no failures is not a ranking, the
    paired block does not claim "no difference" when nothing failed, and an
    undefined drift correlation is not reported as absence of drift.
"""
import warnings

import numpy as np
import pytest
from scipy import stats

from pipeline.arm_ranking import (
    RARITY_Q,
    compare_rankings,
    odd_log_density,
    rank_arms,
    rarity_reference,
)
from pipeline.rare_event import (
    check_defensive_budget,
    defensive_sample_count,
    effective_sample_size,
)
from pipeline.rare_event_random import LHSCrossEntropy, RandomSearchCrossEntropy


# ─────────────────────────────────────────────────────────────────────────────
# A 2-D scenario whose tail probability is known exactly
# ─────────────────────────────────────────────────────────────────────────────
class TailScenario:

    def param_bounds(self):
        """Fails iff p0 > 0.9 under a uniform ODD on [0,1]^2 => P(fail) = 0.10."""
        return {"names": ["p0", "p1"], "lower": np.zeros(2), "upper": np.ones(2)}

    def param_distributions(self, lower=None, upper=None):
        return [stats.uniform(0, 1), stats.uniform(0, 1)]

    def failure_threshold(self):
        return 0.0

    def run_simulation(self, theta):
        return np.asarray(theta, float)

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        self._valid_mask = np.ones(len(theta), dtype=bool)
        return 0.9 - theta[:, 0]


CAMPAIGN_KW = dict(samples_per_iter=15, max_iter=5, final_samples=45)
VALIDATED_KW = dict(samples_per_iter=200, max_iter=6, final_samples=1500)


# ─────────────────────────────────────────────────────────────────────────────
# The guard
# ─────────────────────────────────────────────────────────────────────────────
def test_defensive_sample_count_is_the_number_that_matters():
    assert defensive_sample_count(45, 0.2) == 9        # the campaign budget
    assert defensive_sample_count(1500, 0.2) == 300    # what the tests used


def test_check_defensive_budget_warns_below_the_floor():
    with pytest.warns(RuntimeWarning, match="only 9 points from f"):
        ok = check_defensive_budget(45, 0.2, label="cross_entropy[lhs]")
    assert ok is False


def test_check_defensive_budget_is_silent_when_the_budget_is_adequate():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # any warning fails here
        assert check_defensive_budget(1500, 0.2) is True


@pytest.mark.parametrize("cls", [LHSCrossEntropy, RandomSearchCrossEntropy])
def test_campaign_budget_flags_p_fail_as_unusable(cls):
    with pytest.warns(RuntimeWarning):
        runner = cls(TailScenario(), **CAMPAIGN_KW)
    res = runner.run(seed=0)
    assert res.p_fail_usable is False
    assert res.n_defensive == 9
    # ...and the arm still did its real job: it evaluated points and found
    # failures. Only the probability estimate is unusable.
    assert res.theta_evaluated.shape[0] > 0
    assert res.margins.shape[0] == res.theta_evaluated.shape[0]


@pytest.mark.parametrize("cls", [LHSCrossEntropy, RandomSearchCrossEntropy])
def test_validated_budget_stays_usable_and_accurate(cls):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        runner = cls(TailScenario(), **VALIDATED_KW)
    res = runner.run(seed=0)
    assert res.p_fail_usable is True
    assert abs(res.p_fail - 0.10) < 0.04               # ground truth 0.10


def _lane_keeping_like_odd():
    """
    The 9-D ODD the campaigns actually ran on: road angles as half-normals
    pinned at the lower bound, speeds tilted low, the rest uniform. The
    degeneracy needs this shape -- on a 2-D uniform ODD the proposal never walks
    far enough from f for the defensive weights to vanish, which is exactly why
    the existing 2-D tests could not see it.
    """
    lo = np.array([0, 0, 0, 0, 0, 5, 10, 10, 150.0])
    hi = np.array([85, 85, 85, 85, 85, 15, 30, 40, 350.0])
    dists = []
    for j in range(9):
        a, c = float(lo[j]), float(hi[j])
        rng = c - a
        if j < 5:
            mu, s = a, rng * 0.5
        elif j in (5, 6):
            mu, s = a + 0.4 * rng, rng * 0.3
        else:
            dists.append(stats.uniform(loc=a, scale=rng))
            continue
        dists.append(stats.truncnorm((a - mu) / s, (c - mu) / s, loc=mu, scale=s))
    return lo, hi, dists


def _multimodal_margin(X):
    """Union of three disjoint failure modes — what the region analysis of the
    real campaigns says the failure set looks like."""
    return np.min(np.column_stack([70 - X[:, 0], 27.5 - X[:, 6], 75 - X[:, 4]]),
                  axis=1)


def test_the_degenerate_estimator_is_quantised():
    """
    The signature that gives the collapse away: at the campaign budget the
    estimate can only land on multiples of (1/alpha)/final_samples, because it
    is counting failures among nine defensive draws. This is the exact shape
    seen in the saved campaigns (0.11111 = 1/9, 0.33333 = 3/9, and 0).
    """
    lo, hi, dists = _lane_keeping_like_odd()
    step = (1.0 / 0.2) / 45                            # = 1/9
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for seed in range(12):
            seen.append(LHSCrossEntropy(
                margin_fn=_multimodal_margin, f_dists=dists, lower=lo, upper=hi,
                threshold=0.0, **CAMPAIGN_KW).run(seed=seed).p_fail)
    seen = np.array(seen)

    dust = seen < 1e-3                                 # only the q branch left
    on_grid = np.abs(seen / step - np.round(seen / step)) < 0.02
    assert np.all(dust | on_grid), seen                # nothing in between
    assert dust.sum() >= 1, seen                       # some seeds return zero
    assert (~dust).sum() >= 1, seen                    # others a multiple of 1/9


def test_the_validated_budget_does_not_quantise_on_the_same_problem():
    """Same target, same estimator, a budget the defensive mixture can support:
    the grid disappears and the estimates cluster. Pins the cause to the budget
    rather than to the shape of the failure set."""
    lo, hi, dists = _lane_keeping_like_odd()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        runners = [LHSCrossEntropy(margin_fn=_multimodal_margin, f_dists=dists,
                                   lower=lo, upper=hi, threshold=0.0,
                                   samples_per_iter=200, max_iter=4,
                                   final_samples=1500) for _ in range(3)]
    est = np.array([r.run(seed=s).p_fail for s, r in enumerate(runners)])
    assert np.all(est > 1e-3)
    assert est.std() / est.mean() < 0.25               # no coarse grid


def test_effective_sample_size_collapses_with_degenerate_weights():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    w = np.concatenate([np.full(5, 1.0), np.full(45, 1e-9)])
    assert effective_sample_size(w) < 6.0


# ─────────────────────────────────────────────────────────────────────────────
# Rarity
# ─────────────────────────────────────────────────────────────────────────────
def _odd():
    """Angles peaked at the lower bound, speed uniform — same shape as the real
    lane-keeping ODD, small enough to reason about."""
    return [stats.truncnorm(0.0, 2.0, loc=0.0, scale=1.0), stats.uniform(0, 10)]


def test_rarity_cut_selects_the_declared_fraction_of_the_odd():
    dists = _odd()
    ref = rarity_reference(dists, q=RARITY_Q, n=50_000, seed=1)
    rng = np.random.default_rng(7)
    sample = np.column_stack([d.ppf(rng.random(20_000)) for d in dists])
    frac = (odd_log_density(sample, dists) <= ref["log_f_cut"]).mean()
    assert abs(frac - RARITY_Q) < 0.01


def test_rarity_is_about_where_the_failure_is_not_who_found_it():
    """
    Two arms, the same number of failures. One works in the likely part of the
    domain, the other out in the tail. The tail arm must win, and the count of
    plain failures must NOT separate them.
    """
    dists = _odd()
    lower, upper = np.array([0.0, 0.0]), np.array([2.0, 10.0])
    n = 200
    rng = np.random.default_rng(0)

    common = np.column_stack([rng.uniform(0.0, 0.2, n), rng.uniform(0, 10, n)])
    tail = np.column_stack([rng.uniform(1.8, 2.0, n), rng.uniform(0, 10, n)])
    margins = np.full(n, -1.0)                          # every point fails

    clouds = {"ordinary": (common, margins), "tail": (tail, margins)}
    rk = rank_arms(clouds, lower, upper, dists, threshold=0.0)

    assert rk.best().label == "tail"
    assert rk.best().n_rare > 0
    scores = {s.label: s for s in rk.scores}
    assert scores["ordinary"].n_rare < scores["tail"].n_rare
    assert scores["ordinary"].n_failures == scores["tail"].n_failures
    assert scores["ordinary"].median_odd_pct > scores["tail"].median_odd_pct


def test_ranking_pairs_by_seed_when_provenance_is_present():
    dists = _odd()
    lower, upper = np.array([0.0, 0.0]), np.array([2.0, 10.0])
    rng = np.random.default_rng(3)
    n_seeds, per = 8, 40
    seeds = np.repeat(np.arange(n_seeds), per)

    tail = np.column_stack([rng.uniform(1.8, 2.0, n_seeds * per),
                            rng.uniform(0, 10, n_seeds * per)])
    common = np.column_stack([rng.uniform(0.0, 0.2, n_seeds * per),
                              rng.uniform(0, 10, n_seeds * per)])
    m = np.full(n_seeds * per, -1.0)
    rk = rank_arms({"tail": (tail, m), "ordinary": (common, m)},
                   lower, upper, dists, threshold=0.0,
                   cloud_seeds={"tail": seeds, "ordinary": seeds})

    assert rk.n_seeds == n_seeds
    assert rk.min_attainable_p == pytest.approx(2.0 / 2 ** n_seeds)
    key = "tail vs ordinary"
    assert key in rk.pairwise
    assert rk.pairwise[key]["wins"] == n_seeds          # a clean sweep
    assert rk.pairwise[key]["p_holm"] < 0.05
    assert key in rk.significant_pairs()


def test_ranking_without_provenance_says_so_instead_of_testing():
    dists = _odd()
    rng = np.random.default_rng(5)
    th = np.column_stack([rng.uniform(0, 2, 50), rng.uniform(0, 10, 50)])
    rk = rank_arms({"a": (th, np.full(50, -1.0))},
                   [0.0, 0.0], [2.0, 10.0], dists, threshold=0.0)
    assert rk.pairwise == {}
    assert any("no paired test" in n for n in rk.notes)


def test_rankings_from_different_odds_refuse_to_be_compared():
    """
    The mistake this exists to stop: campaign A ran on angles [0, 8] and B on
    [0, 85]. Rarity is defined against the ODD, so their rare-failure counts
    are not the same quantity and must never be tabled together.
    """
    dists = _odd()
    th = np.array([[1.9, 5.0], [1.95, 6.0]])
    m = np.array([-1.0, -1.0])
    a = rank_arms({"x": (th, m)}, [0.0, 0.0], [2.0, 10.0], dists, threshold=0.0)
    b = rank_arms({"x": (th, m)}, [0.0, 0.0], [20.0, 10.0], dists, threshold=0.0)
    with pytest.raises(ValueError, match="different ODDs"):
        compare_rankings(a, b)
    assert compare_rankings(a, a).count("x") >= 1


def test_odd_log_density_rejects_a_mismatched_parameter_space():
    with pytest.raises(ValueError, match="different parameter spaces"):
        odd_log_density(np.zeros((3, 5)), _odd())


# ─────────────────────────────────────────────────────────────────────────────
# The harness end to end: the flag has to survive all the way to the report
# ─────────────────────────────────────────────────────────────────────────────
class ThreeDScenario(TailScenario):

    def param_bounds(self):
        """Uniform ODD on [0,1]^3, fails iff p0 > 0.85."""
        return {"names": ["a", "b", "c"], "lower": np.zeros(3),
                "upper": np.ones(3)}

    def param_distributions(self, lower=None, upper=None):
        return [stats.uniform(0, 1)] * 3

    def compute_qoi(self, traj, theta):
        theta = np.asarray(theta, float)
        self._valid_mask = np.ones(len(theta), dtype=bool)
        return 0.85 - theta[:, 0]


def _campaign(tmp_path):
    from pipeline.model_comparison import ModelComparison, PlainSamplingBaseline

    sc = ThreeDScenario()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arms = [LHSCrossEntropy(sc, **CAMPAIGN_KW),
                RandomSearchCrossEntropy(sc, **CAMPAIGN_KW),
                PlainSamplingBaseline(sc, "lhs", n_samples=120),
                PlainSamplingBaseline(sc, "random", n_samples=120)]
    return ModelComparison(sc, arms, seeds=[0, 1, 2], budget=120,
                           verbose=False).run()


def test_harness_carries_the_usability_flag_into_the_results(tmp_path):
    res = _campaign(tmp_path)
    row = res.per_seed["cross_entropy[lhs]"][0]
    for key in ("p_fail_usable", "n_defensive", "ess", "n_fail_effective"):
        assert key in row, f"{key} was dropped by the harness"
    assert res.per_arm["cross_entropy[lhs]"]["p_fail_usable"] is False
    assert res.per_arm["plain_sampling[lhs]"]["p_fail_usable"] is True


def test_paired_test_refuses_to_compare_two_degenerate_estimates(tmp_path):
    pt = _campaign(tmp_path).paired_test()
    assert "p_fail" not in pt["cross_entropy"]        # the whole point
    assert pt["cross_entropy"]["p_fail_usable"] is False
    assert "n_failures" in pt["cross_entropy"]        # the real metrics survive
    assert "p_fail" in pt["plain_sampling"]           # ...and stay where valid


def test_report_prints_no_number_for_an_unusable_estimate(tmp_path):
    text = _campaign(tmp_path).report()
    line = next(l for l in text.splitlines()
                if l.strip().startswith("cross_entropy[lhs]"))
    assert "n/a" in line
    assert "P(fail) reads n/a for" in text


def test_saved_campaign_records_per_point_seed_provenance(tmp_path):
    res = _campaign(tmp_path)
    prefix = str(tmp_path / "camp")
    res.save(prefix)
    z = np.load(prefix + "_raw.npz", allow_pickle=True)
    labels = [str(x) for x in z["labels"]]
    for i, lab in enumerate(labels):
        assert f"seeds_{i}" in z, f"{lab} saved no seed provenance"
        assert len(z[f"seeds_{i}"]) == len(z[f"margins_{i}"])
        assert set(np.unique(z[f"seeds_{i}"])) == {0, 1, 2}


def test_ranking_runs_on_a_real_campaign_and_can_be_tested(tmp_path):
    """The end the whole chain exists for: a leaderboard with a paired test."""
    res = _campaign(tmp_path)
    sc = ThreeDScenario()
    b = sc.param_bounds()
    rk = rank_arms(res.clouds, b["lower"], b["upper"],
                   sc.param_distributions(), threshold=0.0,
                   cloud_seeds=res.cloud_seeds, regions=res.regions,
                   per_arm=res.per_arm)
    assert len(rk.scores) == 4
    assert rk.n_seeds == 3
    assert rk.pairwise, "per-seed provenance was present, a paired test was not run"
    assert any("P(fail) is not usable" in n for n in rk.notes)
    assert "WHICH ARM PRODUCED THE MOST RARE FAILURES" in rk.report()


# ─────────────────────────────────────────────────────────────────────────────
# The pre-registered sequence
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.arm_ranking import (  # noqa: E402
    ArmRanking,
    fixed_sequence_test,
    load_plan,
)


def _ranking_with(per_seed_metric, n_seeds):
    return ArmRanking(scores=[], metric="rare_per_100", rarity={"q": 0.1},
                      bounds={}, per_seed_metric=per_seed_metric,
                      n_seeds=n_seeds, min_attainable_p=2.0 / 2 ** n_seeds)


def _plan(*hyps, alpha=0.05):
    return {"alpha": alpha, "metric": "rare_per_100", "hypotheses": list(hyps)}


def test_a_clean_sweep_rejects_when_there_are_enough_seeds():
    n = 12
    rk = _ranking_with({"strong": {s: 40.0 + s for s in range(n)},
                        "weak": {s: 1.0 + s for s in range(n)}}, n)
    res = fixed_sequence_test(
        rk, _plan({"id": "H1", "a": "strong", "b": "weak", "direction": "a>b"}))
    assert res["results"][0]["wins_a"] == n
    assert res["results"][0]["p_value"] < 0.05
    assert res["stopped_at"] is None


def test_the_same_sweep_cannot_reject_at_three_seeds():
    """The floor, not the effect, is what decides at small n -- the reason the
    campaign has to be re-run rather than re-analysed."""
    rk = _ranking_with({"strong": {s: 40.0 + s for s in range(3)},
                        "weak": {s: 1.0 + s for s in range(3)}}, 3)
    res = fixed_sequence_test(
        rk, _plan({"id": "H1", "a": "strong", "b": "weak", "direction": "a>b"}))
    assert res["results"][0]["wins_a"] == 3          # a perfect sweep...
    assert res["results"][0]["p_value"] > 0.05       # ...and still not enough
    assert res["stopped_at"] == "H1"


def test_the_sequence_stops_and_marks_the_rest_untested():
    n = 12
    rk = _ranking_with({"a": {s: 10.0 for s in range(n)},
                        "b": {s: 10.0 + (s % 2) for s in range(n)},
                        "c": {s: 1.0 for s in range(n)}}, n)
    res = fixed_sequence_test(rk, _plan(
        {"id": "H1", "a": "a", "b": "b", "direction": "a>b"},     # will fail
        {"id": "H2", "a": "a", "b": "c", "direction": "a>b"}))    # would pass
    assert res["stopped_at"] == "H1"
    assert "not tested" in res["results"][1]["outcome"]
    assert "p_value" not in res["results"][1]


def test_an_effect_in_the_wrong_direction_is_a_failure_not_a_win():
    n = 12
    rk = _ranking_with({"a": {s: 1.0 + s for s in range(n)},
                        "b": {s: 40.0 + s for s in range(n)}}, n)
    res = fixed_sequence_test(
        rk, _plan({"id": "H1", "a": "a", "b": "b", "direction": "a>b"}))
    row = res["results"][0]
    assert row["direction_observed_as_declared"] is False
    assert row["p_value"] == 1.0
    assert "OTHER way" in row["outcome"]


def test_a_missing_arm_stops_the_sequence_instead_of_skipping_it():
    rk = _ranking_with({"a": {s: 1.0 for s in range(12)}}, 12)
    res = fixed_sequence_test(
        rk, _plan({"id": "H1", "a": "a", "b": "absent", "direction": "a>b"}))
    assert res["stopped_at"] == "H1"


def test_the_shipped_preregistration_is_a_valid_plan():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    plan = load_plan(str(root / "docs" / "preregistrazione_classifica.json"))
    ids = [h["id"] for h in plan["hypotheses"]]
    assert ids == sorted(ids), "the declared order must be the tested order"
    # The hypothesis expected to tie sits last: anything behind an expected
    # null is unreachable in a fixed sequence.
    assert plan["hypotheses"][-1]["direction"] == "two-sided"
    assert plan["alpha"] == 0.05


@pytest.mark.parametrize("bad", [
    {"alpha": 0.05, "metric": "x"},                                   # no hypotheses
    {"alpha": 0.05, "metric": "x", "hypotheses": [{"id": "H1"}]},      # incomplete
    {"alpha": 0.05, "metric": "x",
     "hypotheses": [{"id": "H1", "a": "p", "b": "q", "direction": "up"}]},
])
def test_a_malformed_plan_is_rejected(bad, tmp_path):
    import json

    f = tmp_path / "plan.json"
    f.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(str(f))


# ─────────────────────────────────────────────────────────────────────────────
# A campaign that found nothing must say so, not rank zeros
# ─────────────────────────────────────────────────────────────────────────────
def test_a_campaign_with_no_failures_is_not_a_ranking():
    """
    The 12-seed narrow-ODD run produced zero failures in 8198 simulations. The
    ranking printed six identical rows of 0.00 and a column of nan, and the
    pre-registered sequence read it as "the effect goes the OTHER way" -- a
    campaign with no data in it, reported as evidence against the hypothesis.
    """
    dists = _odd()
    lower, upper = np.array([0.0, 0.0]), np.array([2.0, 10.0])
    rng = np.random.default_rng(11)
    n_seeds, per = 12, 30
    seeds = np.repeat(np.arange(n_seeds), per)
    th = np.column_stack([rng.uniform(0, 2, n_seeds * per),
                          rng.uniform(0, 10, n_seeds * per)])
    safe = np.full(n_seeds * per, +0.9)                # nothing ever fails

    rk = rank_arms({"a": (th, safe), "b": (th, safe)}, lower, upper, dists,
                   threshold=0.0, cloud_seeds={"a": seeds, "b": seeds})
    assert all(s.n_failures == 0 for s in rk.scores)
    assert any("NO ARM PRODUCED A SINGLE FAILURE" in n for n in rk.notes)
    assert "NOTHING TO RANK" in rk.report()

    res = fixed_sequence_test(
        rk, _plan({"id": "H1", "a": "a", "b": "b", "direction": "a>b"}))
    assert res.get("void") is True
    assert "VOID" in res["results"][0]["outcome"]
    # the failure mode this pins down: it must NOT read as a refutation
    assert "OTHER way" not in res["results"][0]["outcome"]


def test_paired_block_does_not_claim_no_difference_when_nothing_failed():
    """The verdict used to read n_failures alone; with zero failures every p is
    1.0 and it printed 'no design difference' under rows starred at p=0.001."""
    from pipeline.model_comparison import ComparisonResult

    rows = {d: [{"seed": s, "n_failures": 0, "p_fail": 0.0,
                 "worst_margin": 0.93 + 0.001 * (d == "lhs"),
                 "margin_q05": 0.95 + 0.001 * (d == "lhs"),
                 "n_evaluations": 100, "run_index": s * 2 + (d == "random")}
                for s in range(12)]
            for d in ("lhs", "random")}
    res = ComparisonResult(
        arm_labels=["f[lhs]", "f[random]"], per_arm={}, clouds={},
        per_seed={"f[lhs]": rows["lhs"], "f[random]": rows["random"]},
        regions=None, seeds=list(range(12)), budget=120)
    text = res._paired_block()
    assert "NOT A COMPARISON OF THE DESIGNS" in text
    assert "No design difference in any family" not in text


def test_undefined_drift_correlation_is_not_reported_as_no_drift():
    from pipeline.model_comparison import ComparisonResult

    rows = [{"seed": s, "n_failures": 0, "run_index": s} for s in range(12)]
    res = ComparisonResult(
        arm_labels=["f[lhs]"], per_arm={}, clouds={},
        per_seed={"f[lhs]": rows}, regions=None, seeds=list(range(12)),
        budget=120)
    dr = res.drift_diagnostic()
    assert dr["computable"] is False
    assert dr["drift_detected"] is False        # absent, but for the right reason
