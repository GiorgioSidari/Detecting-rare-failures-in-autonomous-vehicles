"""
Rank the search arms by rare-failure yield, and test the ranking.

What "rare" means here
----------------------
A failure at theta is rare when theta itself is unlikely under the ODD:

  1. a large reference sample is drawn from the operational distribution f;
  2. the q-quantile (default 10%) of its log-density is ``log_f_cut``;
  3. a failure counts as RARE when ``log f(theta) <= log_f_cut``.

The cut is a property of the ODD, not of the arm that found the failure. It
follows that campaigns run on different ODDs are not comparable on this metric:
``rank_arms`` records the bounds it used and ``compare_rankings`` refuses to
merge two rankings whose bounds differ.

Rarity is independent of severity. A failure can be deep (very negative margin)
and ordinary, or shallow and extremely unlikely; both are reported.

Ranking and the statistics
--------------------------
Arms are ranked on rare failures per 100 simulations, which normalises the
budget each arm actually spent -- the CE descent stops early, so realised
budgets differ. Every pair of arms is then compared seed by seed with a Wilcoxon
signed-rank test (a sign test when Wilcoxon has no power), and the p-values are
corrected with Holm over all pairs.

With n seeds the smallest attainable two-sided p is 2/2^n: 0.25 at 3 seeds,
0.00049 at 12. ``ArmRanking`` carries that floor in ``min_attainable_p`` and
``report`` prints it above the pairwise table.

Usage
-----
    from pipeline.arm_ranking import rank_arms

    rk = rank_arms(clouds, lower, upper, dists, threshold=0.0,
                   cloud_seeds=cloud_seeds, regions=result.regions)
    print(rk.report())
    rk.to_dict()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A failure is "rare" when its ODD log-density sits in the bottom decile of the
# operational distribution. 0.10 is a convention, not a discovery: report it
# next to the numbers and vary it if a reviewer asks.
RARITY_Q = 0.10

# Reference draws used to calibrate the rarity cut. Free -- no simulation.
RARITY_REFERENCE_N = 200_000


def odd_log_density(theta: np.ndarray, dists: list) -> np.ndarray:
    """
    Sum of per-dimension log-pdfs at each row of ``theta`` -- the log-density of
    the product ODD. Points outside the support give -inf, which correctly reads
    as "maximally unlikely".
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2:
        raise ValueError(f"theta must be (M, d), got {theta.shape}")
    if theta.shape[1] != len(dists):
        raise ValueError(
            f"theta has {theta.shape[1]} columns but {len(dists)} ODD marginals "
            "were supplied -- these are different parameter spaces")
    lp = np.zeros(theta.shape[0], dtype=float)
    for j, dist in enumerate(dists):
        with np.errstate(divide="ignore"):
            lp += dist.logpdf(theta[:, j])
    return lp


def rarity_reference(dists: list, *, q: float = RARITY_Q, n: int = RARITY_REFERENCE_N,
                     seed: int = 12345) -> dict:
    """
    Calibrate the rarity cut on the ODD.

    Draws `n` points from the marginals, computes their log-densities and returns
    the q-quantile as ``log_f_cut`` together with the sorted densities, which
    :func:`odd_percentile` uses to place a point on the operational-likelihood
    scale.
    """
    rng = np.random.default_rng(seed)
    ref = np.column_stack([d.ppf(np.clip(rng.random(n), 1e-12, 1 - 1e-12))
                           for d in dists])
    logf = np.sort(odd_log_density(ref, dists))
    finite = logf[np.isfinite(logf)]
    if finite.size == 0:
        raise ValueError("the ODD reference sample has no finite density -- "
                         "check that the marginals match the bounds")
    return {"q": float(q), "n": int(n), "seed": int(seed),
            "log_f_cut": float(np.quantile(finite, q)),
            "_sorted_logf": finite}


def odd_percentile(logf: np.ndarray, reference: dict) -> np.ndarray:
    """Percentile of each log-density within the ODD reference (0 = rarest)."""
    ref = reference["_sorted_logf"]
    return np.searchsorted(ref, np.asarray(logf, float)) / len(ref) * 100.0


@dataclass
class ArmScore:
    """One arm's yield. Everything here is counted, none of it is modelled."""

    label: str
    n_points: int                 # valid simulations this arm actually spent
    n_failures: int
    n_rare: int                   # failures in the rarest q of the ODD
    failures_per_100: float
    rare_per_100: float
    worst_margin: float
    margin_q05: float
    median_odd_pct: float         # median operational percentile of its failures
    n_regions: int = 0
    n_exclusive_regions: int = 0
    p_fail: float = float("nan")
    p_fail_usable: bool = True

    def as_row(self) -> dict:
        d = dict(self.__dict__)
        return d


@dataclass
class ArmRanking:
    """The leaderboard plus the evidence that it means something."""

    scores: list                          # list[ArmScore], best first
    metric: str
    rarity: dict                          # q, cut, reference size
    bounds: dict                          # the ODD this ranking is valid for
    pairwise: dict = field(default_factory=dict)
    per_seed_metric: dict = field(default_factory=dict)   # label -> {seed: value}
    n_seeds: int = 0
    min_attainable_p: float = float("nan")
    notes: list = field(default_factory=list)

    # ── helpers ────────────────────────────────────────────────────────────
    def best(self) -> ArmScore:
        return self.scores[0]

    def significant_pairs(self, alpha: float = 0.05) -> list:
        """Pairs whose Holm-adjusted p clears alpha -- the defensible part."""
        return [k for k, v in self.pairwise.items()
                if np.isfinite(v.get("p_holm", np.nan)) and v["p_holm"] < alpha]

    def to_dict(self) -> dict:
        r = {k: v for k, v in self.rarity.items() if not k.startswith("_")}
        return {"metric": self.metric, "rarity": r, "bounds": self.bounds,
                "per_seed_metric": {k: {str(s): v for s, v in d.items()}
                                    for k, d in self.per_seed_metric.items()},
                "n_seeds": self.n_seeds,
                "min_attainable_p": self.min_attainable_p,
                "ranking": [s.as_row() for s in self.scores],
                "pairwise": self.pairwise, "notes": list(self.notes)}

    # ── the report ─────────────────────────────────────────────────────────
    _WIDTH = 78

    def _header_lines(self) -> list:
        """Title, what "rare" means here, and the no-failures warning."""
        q = self.rarity["q"]
        L = ["=" * self._WIDTH,
             " WHICH ARM PRODUCED THE MOST RARE FAILURES",
             "=" * self._WIDTH,
             f"   rare = failure in the least likely {q * 100:.0f}% of the ODD",
             f"          (log-density cut {self.rarity['log_f_cut']:.3f}, "
             f"calibrated on {self.rarity['n']:,} ODD draws)",
             f"   ranked on: {self.metric}",
             ""]
        if self.scores and not any(s.n_failures for s in self.scores):
            L += ["   *** NO FAILURES FOUND — NOTHING TO RANK ***",
                  f"   {sum(s.n_points for s in self.scores)} valid simulations, "
                  "zero failures in every arm.",
                  "   The table below is a list of zeros; its order is arbitrary",
                  "   and the paired tests are undefined. Check the operating",
                  "   point and the ODD before reading anything into it.",
                  ""]
        return L

    def _table_lines(self) -> list:
        """The leaderboard itself, plus the legend of its columns."""
        L = [f" {'#':<3}{'arm':<26}{'sims':>6}{'fails':>7}{'rare':>6}"
             f"{'rare/100':>10}{'worst':>8}{'ODD pct':>9}{'excl':>6}"]
        for i, s in enumerate(self.scores, 1):
            L.append(f" {i:<3}{s.label:<26}{s.n_points:>6}{s.n_failures:>7}"
                     f"{s.n_rare:>6}{s.rare_per_100:>10.2f}"
                     f"{s.worst_margin:>8.3f}{s.median_odd_pct:>9.1f}"
                     f"{s.n_exclusive_regions:>6}")
        L += ["",
              "   sims     valid simulations the arm actually spent",
              "   rare     failures in the rarest decile of the ODD",
              "   ODD pct  median operational percentile of the failures found",
              "            (lower = the arm works further out in the tail)",
              "   excl     failure regions no other arm reached",
              ""]
        return L

    def _power_lines(self) -> list:
        """
        The number of seeds and the smallest p they can attain.

        Printed above the pairwise table, and replaced by an explanation when no
        paired test could run at all.
        """
        L = []
        if not self.pairwise and self.n_seeds < 2:
            L.append(f"   only {self.n_seeds} seed(s): a paired test needs at")
            L.append("   least 2. The per-seed provenance IS recorded -- add")
            L.append("   seeds, this campaign does not need re-running.")
        elif not self.pairwise:
            L.append("   no per-seed provenance in this campaign: the ranking")
            L.append("   above is a description of the pooled clouds and CANNOT")
            L.append("   be tested. Re-run to record it (clouds now carry seeds).")
        elif self.n_seeds < 6:
            L.append(f"   {self.n_seeds} seeds. Smallest attainable two-sided p: "
                     f"{self.min_attainable_p:.3f}")
            L.append("   Nothing here can reach significance at that floor.")
            L.append("   Read the table as a description and add seeds before")
            L.append("   claiming an ordering.")
        else:
            L.append(f"   {self.n_seeds} seeds. Smallest attainable two-sided p: "
                     f"{self.min_attainable_p:.3f}")
        return L

    def _pairwise_lines(self) -> list:
        """Every pair, raw p and Holm-corrected p, marking the ones that hold."""
        if not self.pairwise:
            return []
        sig = self.significant_pairs()
        L = ["", f"   {'pair':<48}{'wins':>7}{'p':>9}{'p Holm':>9}"]
        for k, v in sorted(self.pairwise.items(),
                           key=lambda kv: kv[1].get("p_holm", 1.0)):
            mark = "  *" if k in sig else ""
            name = k if len(k) <= 48 else k[:45] + "..."
            wins = f"{v['wins']}/{v['n']}"
            L.append(f"   {name:<48}{wins:>7}"
                     f"{v['p_value']:>9.3f}{v['p_holm']:>9.3f}{mark}")
        L.append("")
        if sig:
            L.append("   * = survives the correction. These are the only")
            L.append("   orderings you can defend.")
        else:
            L.append("   No pair survives the correction: the arms are not")
            L.append("   distinguishable at this number of seeds. That is a")
            L.append("   statement about the power, not about the arms.")
        return L

    def report(self) -> str:
        """The whole ranking as printable text: leaderboard, then evidence."""
        L = self._header_lines() + self._table_lines()
        L.append("-" * self._WIDTH)
        L.append(" IS THE ORDER REAL? paired seed by seed, Holm-corrected")
        L.append("-" * self._WIDTH)
        L += self._power_lines()
        L += self._pairwise_lines()
        for n in self.notes:
            L.append("")
            L.append("   NOTE: " + n)
        L.append("=" * self._WIDTH)
        return "\n".join(L)


def _paired_pvalue(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Two-sided paired test on (a - b): Wilcoxon when it has any power, sign test
    otherwise. Returns (wins_for_a, n_pairs, p_value).
    """
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(len(a))
    wins = int((a > b).sum())
    diff = a - b
    nz = int((diff != 0).sum())
    if n < 2 or nz == 0:
        return wins, n, float("nan")
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        p = float(stats.binomtest(wins, nz).pvalue) if nz else float("nan")
    return wins, n, p


def _holm(pairs: dict) -> None:
    """Holm-Bonferroni over the pairwise p-values, in place."""
    items = [(k, v["p_value"]) for k, v in pairs.items()
             if np.isfinite(v["p_value"])]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # Holm's monotonicity step
        pairs[k]["p_holm"] = float(running)
    for k, v in pairs.items():
        v.setdefault("p_holm", float("nan"))


def _score_one_arm(label: str, theta: np.ndarray, margins: np.ndarray, *,
                   threshold: float, dists: list,
                   ref: dict, cut: float, regions, per_arm: dict | None):
    """
    One arm's ArmScore, plus the masks the per-seed counts need.

    Returns ``(score, finite_mask, rare_mask)``: the caller needs the masks to
    split the rare failures per seed without recomputing the densities.
    """
    theta = np.asarray(theta, float)
    margins = np.asarray(margins, float)
    finite = np.isfinite(margins)
    fail = finite & (margins < threshold)
    n_points = int(finite.sum())

    logf = odd_log_density(theta, dists)
    rare = fail & (logf <= cut)
    pct = odd_percentile(logf[fail], ref) if fail.any() else np.array([])

    stats_arm = (per_arm or {}).get(label, {})
    score = ArmScore(
        label=label,
        n_points=n_points,
        n_failures=int(fail.sum()),
        n_rare=int(rare.sum()),
        failures_per_100=float(100.0 * fail.sum() / n_points) if n_points else 0.0,
        rare_per_100=float(100.0 * rare.sum() / n_points) if n_points else 0.0,
        worst_margin=float(np.nanmin(margins)) if finite.any() else float("nan"),
        margin_q05=(float(np.nanpercentile(margins[finite], 5))
                    if finite.any() else float("nan")),
        median_odd_pct=float(np.median(pct)) if pct.size else float("nan"),
        n_regions=len(regions.discovery.get(label, ())) if regions else 0,
        n_exclusive_regions=(len(regions.exclusive_regions(label))
                             if regions else 0),
        p_fail=float(stats_arm.get("mean_p_fail", float("nan"))),
        p_fail_usable=bool(stats_arm.get("p_fail_usable", True)),
    )
    return score, finite, rare


def _rare_rate_per_seed(finite, rare, seeds) -> dict:
    """`rare_per_100` restricted to each seed: the unit of the paired test."""
    seeds = np.asarray(seeds)
    return {
        int(s): float(100.0 * (rare & (seeds == s)).sum()
                      / max(int((finite & (seeds == s)).sum()), 1))
        for s in np.unique(seeds)}


def _all_pairs_paired_test(scores: list, per_seed_metric: dict) -> tuple:
    """
    Every pair of arms compared seed by seed, then Holm-corrected.

    Returns ``(pairwise, n_seeds)``. `n_seeds` counts the seeds the arms have in
    common: pairing on seeds only one arm ran would compare different designs.
    """
    if len(per_seed_metric) < 2:
        return {}, 0

    common = set.intersection(*[set(v) for v in per_seed_metric.values()])
    n_seeds = len(common)
    if n_seeds < 2:
        return {}, n_seeds

    order = sorted(common)
    labels = [s.label for s in scores if s.label in per_seed_metric]
    pairwise: dict = {}
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            a = np.array([per_seed_metric[la][s] for s in order])
            b = np.array([per_seed_metric[lb][s] for s in order])
            wins, n, p = _paired_pvalue(a, b)
            pairwise[f"{la} vs {lb}"] = {
                "wins": wins, "n": n, "p_value": p,
                "mean_a": float(a.mean()), "mean_b": float(b.mean())}
    _holm(pairwise)
    return pairwise, n_seeds


def _ranking_notes(scores: list, pairwise: dict, n_seeds: int) -> list:
    """
    The notes attached to the ranking.

    Covers three cases: no paired test was possible (distinguishing "one seed"
    from "no per-seed provenance in the file"), no arm produced any failure at
    all, and arms whose P(fail) estimate is out of its validity regime.
    """
    notes: list = []

    if not pairwise:
        if n_seeds < 2:
            notes.append(
                f"no paired test was possible: this campaign has {n_seeds} "
                "seed(s) and pairing needs at least 2. The provenance IS "
                "recorded; add seeds, do not re-run for provenance. The ranking "
                "is a description of the pooled clouds.")
        else:
            notes.append(
                "no paired test was possible: this campaign has no per-point seed "
                "provenance (campaigns saved before seeds_<i> was added to the .npz). "
                "The ranking is a description of the pooled clouds.")

    # No failures anywhere: the order of the table is arbitrary and every
    # p-value is undefined, so the note goes first.
    if not any(s.n_failures for s in scores):
        notes.insert(0,
            "NO ARM PRODUCED A SINGLE FAILURE in this campaign "
            f"({sum(s.n_points for s in scores)} valid simulations). There is "
            "nothing to rank: the order above is arbitrary and every p-value is "
            "undefined. This is a finding about the system under test and its "
            "operating point in this ODD, not about the search algorithms.")

    unusable = [s.label for s in scores if not s.p_fail_usable]
    if unusable:
        notes.append(
            "P(fail) is not usable for " + ", ".join(unusable) +
            " at this budget; those arms are ranked on what they FOUND, which is "
            "unaffected, and their probability estimates are excluded.")

    return notes


def rank_arms(clouds: dict, lower: np.ndarray, upper: np.ndarray, dists: list,
              *, threshold: float = 0.0,
              cloud_seeds: dict | None = None, regions=None,
              per_arm: dict | None = None, q: float = RARITY_Q,
              reference_n: int = RARITY_REFERENCE_N, reference_seed: int = 12345,
              metric: str = "rare_per_100") -> ArmRanking:
    """
    Score and rank every arm on rare-failure yield.

    Parameters
    ----------
    clouds       : {label: (theta (M, d), margins (M,))} -- the pooled evaluated
                   points, exactly what ``ComparisonResult.clouds`` holds.
    lower, upper : the ODD bounds the campaign ran on, recorded in the result.
    dists        : the ODD marginals (``scenario.param_distributions(lo, hi)``).
    threshold    : failure threshold (margin < threshold).
    cloud_seeds  : {label: (M,) seed per point}. Without it no paired test runs
                   and the ranking carries a note saying so.
    regions      : optional RegionComparison, for the region columns.
    per_arm      : optional per-arm stats dict, for p_fail and its usability.
    metric       : the field of ArmScore to rank on.
    """
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    ref = rarity_reference(dists, q=q, n=reference_n, seed=reference_seed)
    cut = ref["log_f_cut"]

    scores: list = []
    per_seed_rare: dict = {}
    for label, (theta, margins) in clouds.items():
        score, finite, rare = _score_one_arm(
            label, theta, margins, threshold=threshold, dists=dists, ref=ref,
            cut=cut, regions=regions, per_arm=per_arm)
        scores.append(score)

        seeds = (cloud_seeds or {}).get(label)
        if seeds is not None and len(seeds) == len(np.asarray(margins, float)):
            per_seed_rare[label] = _rare_rate_per_seed(finite, rare, seeds)

    scores.sort(key=lambda s: getattr(s, metric), reverse=True)
    pairwise, n_seeds = _all_pairs_paired_test(scores, per_seed_rare)

    return ArmRanking(
        scores=scores, metric=metric, rarity=ref,
        bounds={"lower": lower.tolist(), "upper": upper.tolist()},
        pairwise=pairwise, per_seed_metric=per_seed_rare, n_seeds=n_seeds,
        min_attainable_p=(2.0 / 2 ** n_seeds) if n_seeds else float("nan"),
        notes=_ranking_notes(scores, pairwise, n_seeds))


def compare_rankings(a: ArmRanking, b: ArmRanking, *, name_a: str = "A",
                     name_b: str = "B") -> str:
    """
    Put two campaigns' leaderboards side by side.

    Raises when the two rankings were calibrated on different ODD bounds, since
    the rarity cut is defined relative to the operational distribution.
    """
    if a.bounds != b.bounds:
        raise ValueError(
            "these rankings were computed on different ODDs, so their rare-failure "
            "counts are not comparable.\n"
            f"  {name_a}: lower={a.bounds['lower']}\n"
            f"           upper={a.bounds['upper']}\n"
            f"  {name_b}: lower={b.bounds['lower']}\n"
            f"           upper={b.bounds['upper']}\n"
            "Re-run one campaign on the other's bounds before comparing.")
    w = 78
    L = ["=" * w, f" {name_a} vs {name_b} — same ODD, same rarity cut", "=" * w,
         f" {'arm':<26}{name_a + ' rare/100':>16}{name_b + ' rare/100':>16}"]
    sa = {s.label: s for s in a.scores}
    sb = {s.label: s for s in b.scores}
    for lab in sorted(set(sa) | set(sb)):
        va = f"{sa[lab].rare_per_100:.2f}" if lab in sa else "-"
        vb = f"{sb[lab].rare_per_100:.2f}" if lab in sb else "-"
        L.append(f" {lab:<26}{va:>16}{vb:>16}")
    L.append("=" * w)
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════════════════
# Pre-registered testing
# ═════════════════════════════════════════════════════════════════════════════
"""
Why a fixed sequence rather than "test everything and correct"

Six arms make fifteen pairs. Holm over fifteen compares the smallest p against
0.05/15 = 0.0033, and a paired test over n seeds cannot go below 2/2^n even on a
perfect sweep -- so an all-pairs analysis needs TEN seeds before its best result
can clear the correction, however large the effect is. That is a property of the
design, not of the data, and it is worth knowing before buying the simulator
time rather than after.

A fixed sequence buys the same protection for less. Order the hypotheses in
advance, test each at the full alpha, and stop at the first one that fails:
the family-wise error rate is still alpha, with no correction at all, because
each test is only reached when every earlier one has already rejected. The price
is that the order is a commitment -- put a hypothesis you expect to fail early
and everything behind it is unreachable, no matter how strong.

So the order encodes what you believe, and it has to be written down before the
data exists. That is what ``load_plan`` reads and what ``fixed_sequence_test``
executes; neither of them can see the data when the plan is written.
"""


def load_plan(path: str) -> dict:
    """Read a pre-registration file and check it is a usable plan."""
    import json

    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    for key in ("alpha", "metric", "hypotheses"):
        if key not in plan:
            raise ValueError(f"the plan is missing '{key}'")
    if not plan["hypotheses"]:
        raise ValueError("the plan declares no hypotheses")
    for i, h in enumerate(plan["hypotheses"], 1):
        for key in ("id", "a", "b", "direction"):
            if key not in h:
                raise ValueError(f"hypothesis {i} is missing '{key}'")
        if h["direction"] not in ("a>b", "b>a", "two-sided"):
            raise ValueError(
                f"{h['id']}: direction must be 'a>b', 'b>a' or 'two-sided', "
                f"got {h['direction']!r}")
    return plan


def _one_sided_p(p_two: float, wins: int, n: int, direction: str) -> tuple:
    """
    The p for one hypothesis, and whether the effect went in the declared direction.

    A two-sided hypothesis keeps the two-sided p. A one-sided one halves it when
    the observed direction matches the declaration, and returns 1.0 otherwise.
    """
    if direction == "two-sided":
        return float(p_two), True
    correct_way = (wins > n - wins) if direction == "a>b" else (wins < n - wins)
    return (float(p_two / 2.0) if correct_way else 1.0), bool(correct_way)


def _test_one_hypothesis(h: dict, per_seed: dict, alpha: float) -> dict:
    """
    One hypothesis of the pre-registered plan, tested paired by seed.

    The returned row always carries an ``outcome``; ``stop`` says whether the
    sequence must halt here.
    """
    row = {"id": h["id"], "a": h["a"], "b": h["b"],
           "direction": h["direction"],
           "rationale": h.get("rationale", "")}

    if h["a"] not in per_seed or h["b"] not in per_seed:
        row["outcome"] = "not tested (an arm has no per-seed values)"
        return {"row": row, "stop": True}

    seeds = sorted(set(per_seed[h["a"]]) & set(per_seed[h["b"]]))
    a = np.array([per_seed[h["a"]][s] for s in seeds], float)
    b = np.array([per_seed[h["b"]][s] for s in seeds], float)

    if not np.any(a) and not np.any(b):
        # Both arms identically zero on this metric: there is nothing to test,
        # so the hypothesis is marked VOID and the sequence stops.
        row["outcome"] = ("VOID — both arms are identically zero on this "
                          "metric; the campaign produced nothing to test")
        row.update({"n": len(seeds), "wins_a": 0, "mean_a": 0.0, "mean_b": 0.0})
        return {"row": row, "stop": True, "void": True}

    wins, n, p_two = _paired_pvalue(a, b)
    row.update({"n": n, "wins_a": wins,
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "p_two_sided": p_two})

    p, correct_way = _one_sided_p(p_two, wins, n, h["direction"])
    row["p_value"] = float(p)
    row["direction_observed_as_declared"] = correct_way

    if np.isfinite(p) and p < alpha:
        row["outcome"] = "REJECTED the null — the hypothesis holds"
        return {"row": row, "stop": False}

    row["outcome"] = ("failed — the sequence stops here" if correct_way
                      else "failed: the effect goes the OTHER way")
    return {"row": row, "stop": True}


def fixed_sequence_test(ranking: ArmRanking, plan: dict) -> dict:
    """
    Execute a pre-registered ordered sequence of pairwise hypotheses.

    Each hypothesis is tested at the full ``alpha``. The sequence stops at the
    first one that does not reject, and the remaining entries are reported with
    an outcome of "not tested".
    """
    alpha = float(plan["alpha"])
    per_seed = ranking.per_seed_metric
    out = {"alpha": alpha, "metric": ranking.metric,
           "n_seeds": ranking.n_seeds,
           "min_attainable_p": ranking.min_attainable_p,
           "results": [], "stopped_at": None}

    stopped = False
    for h in plan["hypotheses"]:
        if stopped:
            out["results"].append(
                {"id": h["id"], "a": h["a"], "b": h["b"],
                 "direction": h["direction"],
                 "rationale": h.get("rationale", ""),
                 "outcome": "not tested (the sequence stopped earlier)"})
            continue

        res = _test_one_hypothesis(h, per_seed, alpha)
        out["results"].append(res["row"])
        if res.get("void"):
            out["void"] = True
        if res["stop"]:
            stopped = True
            out["stopped_at"] = h["id"]
    return out


def fixed_sequence_report(res: dict) -> str:
    w = 78
    L = ["=" * w, " PRE-REGISTERED SEQUENCE — tested in the declared order",
         "=" * w,
         f"   alpha = {res['alpha']} per hypothesis, no correction: a fixed",
         "   sequence holds the family-wise rate on its own, as long as the",
         "   order was fixed before the data existed.",
         f"   {res['n_seeds']} seeds. Smallest attainable two-sided p: "
         f"{res['min_attainable_p']:.5f}",
         ""]
    for r in res["results"]:
        L.append(f"   {r['id']}: {r['a']}  vs  {r['b']}   [{r['direction']}]")
        if "p_value" in r:
            L.append(f"       {r['mean_a']:.2f} vs {r['mean_b']:.2f}   "
                     f"wins {r['wins_a']}/{r['n']}   p = {r['p_value']:.5f}")
        L.append(f"       -> {r['outcome']}")
        if r.get("rationale"):
            L.append(f"       ({r['rationale']})")
        L.append("")
    if res.get("void"):
        L.append("   THE SEQUENCE IS VOID. The campaign produced no failures at")
        L.append("   all, so no hypothesis about which arm finds more of them")
        L.append("   could be tested. Nothing here counts for or against the")
        L.append("   pre-registered claims — the plan is simply unspent, and can")
        L.append("   be applied unchanged to a campaign that produces data.")
        L.append("=" * w)
        return "\n".join(L)
    if res["stopped_at"]:
        L.append(f"   The sequence stopped at {res['stopped_at']}. Everything")
        L.append("   after it is UNTESTED, not disproved — reporting it as a")
        L.append("   null result would be the whole point of the plan, undone.")
    else:
        L.append("   Every pre-registered hypothesis held.")
    L.append("=" * w)
    return "\n".join(L)
