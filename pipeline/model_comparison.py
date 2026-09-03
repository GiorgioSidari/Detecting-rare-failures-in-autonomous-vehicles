"""
Harness that runs several search arms at a matched budget and compares them.

The classes in :mod:`pipeline.active_boundary_random` and
:mod:`pipeline.rare_event_random` each produce a cloud of evaluated points and a
P(failure) estimate. This module runs them over a list of seeds and assembles:

  * per-arm statistics across the seeds (P(failure), failures found, worst
    margin, 5th-percentile margin);
  * one shared map of failure regions built from the union of every arm's
    failures (:mod:`pipeline.failure_regions`), so the arms are scored against
    the same regions;
  * for each region, the probability that a blind draw lands in it and the
    number of blind draws needed for even odds;
  * a paired LHS-vs-random test per family, and a diagnostic for drift over the
    session.

Every arm receives the same simulation budget.

Usage
-----
    from pipeline.model_comparison import ModelComparison

    cmp = ModelComparison.default(scenario, budget=200, seeds=[0, 1, 2])
    result = cmp.run()
    print(result.report())
    result.save("results/comparison")        # .json + .csv

To compare a subset, pass the arms explicitly:

    cmp = ModelComparison(scenario, arms=[LHSActiveBoundary(scenario, n_seed=40),
                                          RandomSearchActiveBoundary(scenario, n_seed=40)])
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from pipeline.active_boundary_random import (
    LHSActiveBoundary,
    RandomSearchActiveBoundary,
)
from pipeline.region_comparison import compare_failure_regions
from pipeline.rare_event_random import LHSCrossEntropy, RandomSearchCrossEntropy
from pipeline.samplers import get_sampler


# ─────────────────────────────────────────────────────────────────────────────
# A plain-sampling baseline, so "does the clever method help?" has an answer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BaselineResult:
    """Same shape as the other arms' results, so the harness treats them alike."""

    p_fail: float
    p_fail_ci: tuple
    n_evaluations: int
    theta_evaluated: np.ndarray
    margins: np.ndarray
    labels: np.ndarray
    boundary_summary: dict = field(default_factory=dict)


class PlainSamplingBaseline:

    def __init__(self, scenario, sampler="random", *, n_samples: int = 200,
                 weight_by_odd: bool = True, param_lower=None, param_upper=None,
                 verbose: bool = False):
        """
        One-shot sampling of the ODD: draw n points, simulate them, count failures.

        No model and no adaptation: the points are chosen before any result is seen.
        ``sampler="lhs"`` stratifies the draw, ``sampler="random"`` draws i.i.d.
        """
        if isinstance(scenario, str):
            from scenarios import SCENARIOS
            scenario = SCENARIOS[scenario]
        self.scenario = scenario
        self.sampler = get_sampler(sampler)
        self.n_samples = int(n_samples)
        self.weight_by_odd = weight_by_odd
        self.param_lower = param_lower
        self.param_upper = param_upper
        self.verbose = verbose

    @property
    def label(self) -> str:
        return f"plain_sampling[{self.sampler.name}]"

    @property
    def budget(self) -> int:
        return self.n_samples

    def run(self, seed: int = 0) -> BaselineResult:
        sc = self.scenario
        b = sc.param_bounds()
        lower = np.asarray(self.param_lower if self.param_lower is not None
                           else b["lower"], float)
        upper = np.asarray(self.param_upper if self.param_upper is not None
                           else b["upper"], float)
        thr = float(sc.failure_threshold())

        if self.weight_by_odd and hasattr(sc, "param_distributions"):
            dists = sc.param_distributions(lower, upper)
            theta = self.sampler.from_dists(self.n_samples, dists, seed=seed)
        else:
            theta = self.sampler.bounded(self.n_samples, lower, upper, seed=seed)

        traj = sc.run_simulation(theta)
        margins = np.asarray(sc.compute_qoi(traj, theta), float)
        valid = getattr(sc, "_valid_mask", None)
        if valid is None or np.asarray(valid).shape[0] != margins.shape[0]:
            valid = np.isfinite(margins)
        valid = np.asarray(valid, dtype=bool)

        theta, margins = theta[valid], margins[valid]
        labels = (margins < thr).astype(int)
        k, n = int(labels.sum()), len(labels)
        p = float(k / n) if n else 0.0
        # Wilson interval: stays inside [0, 1] and does not collapse at p ~ 0.
        if n:
            z = 1.96
            den = 1.0 + z * z / n
            centre = (p + z * z / (2 * n)) / den
            half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            ci = (max(0.0, centre - half), min(1.0, centre + half))
        else:
            ci = (float("nan"), float("nan"))

        return BaselineResult(
            p_fail=p, p_fail_ci=ci, n_evaluations=n,
            theta_evaluated=theta, margins=margins, labels=labels,
            boundary_summary={
                "label": self.label, "sampler": self.sampler.name, "seed": int(seed),
                "n_evaluations": n, "n_failures_found": k,
                "worst_margin": float(margins.min()) if n else float("nan"),
                "p_fail": p, "p_fail_ci": ci,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Comparison result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ComparisonResult:
    """Everything the harness produced: per-arm stats, raw clouds, region map."""

    arm_labels: list
    per_arm: dict                       # label -> aggregated stats over seeds
    per_seed: dict                      # label -> [per-seed summary dicts]
    clouds: dict                        # label -> (theta, margins) concatenated
    regions: object                     # RegionComparison
    seeds: list
    budget: int
    param_names: list = field(default_factory=list)
    failed_runs: list = field(default_factory=list)
    cloud_seeds: dict = field(default_factory=dict)
    param_lower: object = None
    param_upper: object = None
    odd_dists: object = None
    threshold: float = 0.0
    #: Operating point and provenance of the campaign. Without this a result
    #: file does not say which system it refers to, and two campaigns with
    #: different speed_scale are indistinguishable after the fact.
    metadata: dict = field(default_factory=dict)


    # ── the sensitive test: pair the two designs seed by seed ───────────────
    def paired_test(self) -> dict:
        """
        Compare each family's LHS arm against its random arm, seed by seed.

        The two arms of a family are compared within each seed, on three metrics:
        failures found, worst margin and P(failure). The test on the per-seed
        differences is a Wilcoxon signed-rank, which assumes nothing about their
        distribution.

        With n seeds the smallest two-sided p-value the test can return is 2/2^n
        (0.031 at 6 seeds, 0.002 at 10); `min_attainable_p` reports it alongside
        each result.

        Returns {family: {metric: {...}}} for n_failures, worst_margin, p_fail.
        """
        from scipy import stats

        families: dict = {}
        for lab in self.per_seed:
            if "[" not in lab:
                continue
            fam, design = lab.split("[", 1)
            design = design.rstrip("]").split("+")[0]
            if design in ("lhs", "random"):
                families.setdefault(fam, {})[design] = {
                    r["seed"]: r for r in self.per_seed[lab]}

        # Lower is better for the margin metrics; higher for the failure count.
        metrics = {"n_failures": +1, "margin_q05": -1, "worst_margin": -1,
                   "p_fail": 0}
        out: dict = {}
        for fam, arms in families.items():
            if set(arms) != {"lhs", "random"}:
                continue
            seeds = sorted(set(arms["lhs"]) & set(arms["random"]))
            if len(seeds) < 2:
                continue
            entry: dict = {"n_seeds": len(seeds),
                           "min_attainable_p": 2.0 / 2 ** len(seeds)}
            # p_fail is only a metric when both arms actually estimated a
            # probability. When the estimator degenerates (too few defensive
            # draws) the "comparison" is between two quantised few-sample Monte
            # Carlos and a tie there means nothing at all.
            p_usable = all(arms[d][s].get("p_fail_usable", True)
                           for d in ("lhs", "random") for s in seeds)
            entry["p_fail_usable"] = bool(p_usable)
            for metric, direction in metrics.items():
                if metric == "p_fail" and not p_usable:
                    continue
                if any(metric not in arms[d][s] for d in ("lhs", "random")
                       for s in seeds):
                    continue          # campaign predates this metric
                a = np.array([arms["lhs"][s][metric] for s in seeds], float)
                b = np.array([arms["random"][s][metric] for s in seeds], float)
                ok = np.isfinite(a) & np.isfinite(b)
                a, b = a[ok], b[ok]
                wins = int(((a > b) if direction >= 0 else (a < b)).sum())
                try:
                    pval = float(stats.wilcoxon(a, b).pvalue)
                except Exception:      # all differences zero, or too few pairs
                    pval = float("nan")
                entry[metric] = {
                    "lhs_mean": float(a.mean()), "lhs_std": float(a.std(ddof=1)),
                    "random_mean": float(b.mean()), "random_std": float(b.std(ddof=1)),
                    "lhs_wins": wins, "n": int(len(a)), "p_value": pval,
                    "informative": direction != 0,
                }
            out[fam] = entry
        return out



    def drift_diagnostic(self) -> dict:
        """
        Spearman correlation between execution order and failures found.

        A negative correlation means later runs found fewer failures than earlier
        ones. The diagnostic needs `run_index`, recorded per run by :meth:`run`, and
        reports `interleaved` so the reader knows whether the arms were mixed: with
        arm-major execution the correlation cannot be separated from the arm.

        Returns {} when the campaign carries no run order, and sets
        ``computable=False`` when the failure counts have no variance to correlate.
        """
        from scipy import stats

        idx, fails, labels = [], [], []
        for lab, rows in self.per_seed.items():
            for r in rows:
                if "run_index" not in r:
                    continue
                idx.append(r["run_index"])
                fails.append(r["n_failures"])
                labels.append(lab)
        if len(idx) < 5:
            return {}

        idx_a, fails_a = np.asarray(idx, float), np.asarray(fails, float)
        rho, pval = stats.spearmanr(idx_a, fails_a)
        # Was the order mixed, or did each arm run in one block? If arm labels
        # come in contiguous blocks the diagnostic cannot separate the two.
        order = [lab for _, lab in sorted(zip(idx, labels))]
        blocks = 1 + sum(1 for a, b in zip(order, order[1:]) if a != b)
        interleaved = blocks > len(set(labels))

        # Spearman is undefined when the failure counts have zero variance
        # (every run found the same number), which `computable` reports.
        computable = bool(np.isfinite(rho) and np.isfinite(pval))
        return {"spearman_rho": float(rho), "p_value": float(pval),
                "n_runs": len(idx), "interleaved": bool(interleaved),
                "computable": computable,
                "drift_detected": bool(computable and pval < 0.05)}

    def ranking(self, **kw):
        """
        Rank the arms by rare-failure yield (see :mod:`pipeline.arm_ranking`).

        Returns None when the campaign carries no operational distribution:
        rarity is defined against it.
        """
        if self.odd_dists is None or self.param_lower is None:
            return None
        cached = getattr(self, "_ranking_cache", None)
        if cached is not None and not kw:
            return cached
        from pipeline.arm_ranking import rank_arms
        kw.setdefault("threshold", float(self.threshold))
        rk = rank_arms(self.clouds, self.param_lower, self.param_upper,
                       self.odd_dists, cloud_seeds=self.cloud_seeds,
                       regions=self.regions, per_arm=self.per_arm, **kw)
        if not kw:
            self._ranking_cache = rk
        return rk

    @staticmethod
    def _paired_table_lines(rows: list) -> list:
        """The per-family table: LHS mean, random mean, wins and p."""
        width = max([len(r[0]) for r in rows] + [16]) + 2
        L = [f"   {'family / metric':<{width}}{'LHS':>10}{'random':>10}"
             f"{'LHS wins':>10}{'p':>8}"]
        prev_fam = None
        for name, m, metric in rows:
            fam = name.split(" / ")[0]
            if prev_fam is not None and fam != prev_fam:
                L.append("")
            prev_fam = fam
            fmt = "{:.1f}" if metric == "n_failures" else "{:+.3f}"
            star = ("  **" if m["p_value"] < 0.05 else
                    "  *" if m["p_value"] < 0.10 else "")
            L.append(f"   {name:<{width}}"
                     f"{fmt.format(m['lhs_mean']):>10}"
                     f"{fmt.format(m['random_mean']):>10}"
                     f"{str(m['lhs_wins']) + '/' + str(m['n']):>10}"
                     f"{m['p_value']:>8.3f}{star}")
        L.append("")
        return L

    def _drift_lines(self) -> list:
        """
        Whether the session drifted, and whether that invalidates the block.

        Drift only biases the comparison when the arms ran in blocks; with a
        mixed order it inflates the noise instead, which is why the execution
        order is reported next to the correlation.
        """
        dr = self.drift_diagnostic()
        if not dr:
            return []
        tag = ("mixed" if dr["interleaved"] else
               "ARM-MAJOR — the check below cannot be trusted")
        L = ["", "   SESSION DRIFT CHECK", f"   execution order: {tag}"]
        if not dr.get("computable", True):
            L.append("   failures vs run order: NOT COMPUTABLE — the failure")
            L.append("   counts have no variance across runs (every run found")
            L.append("   the same number). Drift is unmeasured here, not absent.")
            return L
        L.append(f"   failures vs run order: rho={dr['spearman_rho']:+.2f} "
                 f"(p={dr['p_value']:.3f}, {dr['n_runs']} runs)")
        if dr["drift_detected"] and not dr["interleaved"]:
            L.append("   The machine drifted AND the arms ran in blocks: the")
            L.append("   comparison above is confounded with execution order.")
            L.append("   Re-run with order='shuffled' before believing it.")
        elif dr["drift_detected"]:
            L.append("   The machine drifted, but the arms were mixed, so the")
            L.append("   drift hits both designs equally — it inflates the noise,")
            L.append("   it does not bias the comparison.")
        else:
            L.append("   No detectable drift over the session.")
        return L

    def _paired_block(self) -> str:
        """The seed-by-seed LHS vs random table, with its own verdict."""
        pt = self.paired_test()
        w = 78
        L = ["=" * w,
             " LHS vs RANDOM — PAIRED SEED BY SEED (same budget, same seeds)",
             "=" * w]
        if not pt:
            L.append("   no paired lhs/random arms in this run.")
            L.append("=" * w)
            return "\n".join(L)

        n = next(iter(pt.values()))["n_seeds"]
        floor = next(iter(pt.values()))["min_attainable_p"]
        L.append(f"   {n} seeds. Smallest p this many seeds can produce: {floor:.3f}")
        L.append("   (a p near that floor means 'add seeds', not 'no effect')")
        L.append("")
        rows = [(f"{fam} / {label}", e[metric], metric)
                for fam, e in pt.items()
                for metric, label in (("margin_q05", "5th pct margin"),
                                      ("n_failures", "failures found"),
                                      ("worst_margin", "worst margin"))
                if metric in e]
        L += self._paired_table_lines(rows)

        # The verdict used to read n_failures alone. With zero failures anywhere
        # that metric is constant, every p is 1.0, and the block printed "no
        # design difference" directly underneath rows starred at p=0.001 -- and
        # directly above its own advice to trust the margin over the count.
        # When the failure count carries no information, say so and fall back to
        # the metric this report already tells the reader to prefer.
        total_fail = sum(e["n_failures"]["lhs_mean"] + e["n_failures"]["random_mean"]
                         for e in pt.values() if "n_failures" in e)
        counts_informative = total_fail > 0
        L.append("   READING:")
        if not counts_informative:
            L.append("   NOT A COMPARISON OF THE DESIGNS: not one arm produced a")
            L.append("   single failure, in any seed. There is nothing to find, so")
            L.append("   'LHS vs random' has no content here — whatever the margin")
            L.append("   rows above show, they are comparing how far from failure")
            L.append("   each design stayed, not which one reaches failures.")
            L.append("   The operating point or the ODD is wrong for this system,")
            L.append("   not the sampling design.")
            L.append("=" * w)
            return "\n".join(L)

        best = min((e["n_failures"]["p_value"], f) for f, e in pt.items()
                   if np.isfinite(e["n_failures"]["p_value"]))
        p_best, fam_best = best
        if p_best < 0.05:
            L.append(f"   Stratification finds significantly more failures in "
                     f"{fam_best} (p={p_best:.3f}).")
        elif p_best < 0.10:
            e = pt[fam_best]["n_failures"]
            need = int(np.ceil(np.log2(2 / 0.05)))
            L.append(f"   {fam_best}: LHS wins {e['lhs_wins']}/{e['n']} seeds, p={p_best:.3f}")
            L.append("   — suggestive, not conclusive. The direction is consistent but")
            L.append(f"   the sample is too small to settle it; {max(need, n + 4)} seeds "
                     "would give the")
            L.append("   test room to reach p<0.05. Re-run those two arms only.")
        else:
            L.append("   No design difference in any family at this budget.")
        L.append("   ** p<0.05   * p<0.10   |   margins: lower is better")
        if any(m == "margin_q05" for _, _, m in rows):
            L.append("   Trust '5th pct margin' over 'failures found': the failure")
            L.append("   label flips on ~20% of the points from simulator noise")
            L.append("   alone, while the margin is a continuous measurement.")

        L += self._drift_lines()
        L.append("=" * w)
        return "\n".join(L)

    def report(self) -> str:
        w = 78
        L = ["=" * w, " MODEL COMPARISON — same budget, same scenario", "=" * w,
             f" seeds        : {self.seeds}",
             f" target budget: {self.budget} simulations per arm per seed", ""]
        L.append(f" {'arm':<28}{'evals':>7}{'fails':>7}{'P(fail)':>12}"
                 f"{'+/-':>9}{'worst':>9}")
        unusable: list = []
        for lab in self.arm_labels:
            s = self.per_arm[lab]
            # An unusable estimate gets no number: printing one invites it to
            # be quoted, and that is exactly how it ended up in a report.
            if s.get("p_fail_usable", True):
                pf = f"{s['mean_p_fail']:>12.4g}{s['std_p_fail']:>9.2g}"
            else:
                pf = f"{'n/a':>12}{'':>9}"
                unusable.append(lab)
            L.append(f" {lab:<28}{s['mean_evaluations']:>7.0f}"
                     f"{s['mean_failures']:>7.1f}{pf}{s['worst_margin']:>9.3f}")
        if unusable:
            L.append("")
            L.append("   P(fail) reads n/a for: " + ", ".join(unusable))
            L.append("   Their estimator has too few defensive draws at this")
            L.append("   budget to BE a probability estimate (see per_seed.ess and")
            L.append("   n_defensive). The failures and regions they found are")
            L.append("   unaffected; read P(fail) off a plain_sampling arm.")
        if self.failed_runs:
            L.append("")
            L.append(" runs that errored out:")
            for lab, sd, err in self.failed_runs:
                L.append(f"   {lab} seed={sd}: {err}")
        L.append("")
        L.append(self._paired_block())
        L.append("")
        rk = self.ranking()
        if rk is not None:
            L.append(rk.report())
            L.append("")
        L.append(self.regions.report())
        L.append("")
        L.append(" reading this: two arms with a similar P(fail) but a low Jaccard")
        L.append(" are exploring different parts of the failure set. The regions with")
        L.append(" the smallest P(hit) are the ones a blind design misses.")
        return "\n".join(L)

    def to_dict(self) -> dict:
        return {
            "metadata": dict(self.metadata),
            "seeds": list(self.seeds),
            "budget": self.budget,
            "param_names": list(self.param_names),
            "arms": list(self.arm_labels),
            "per_arm": self.per_arm,
            "per_seed": self.per_seed,
            "paired_test": self.paired_test(),
            "drift_diagnostic": self.drift_diagnostic(),
            "ranking": (self.ranking().to_dict()
                        if self.ranking() is not None else None),
            "failed_runs": [{"arm": a, "seed": s, "error": e}
                            for a, s, e in self.failed_runs],
            "regions": self.regions.to_dict(),
        }

    def save(self, path_prefix: str) -> list:
        """
        Write ``<prefix>.json``, ``<prefix>_arms.csv``, ``<prefix>_regions.csv``
        and ``<prefix>_raw.npz``.

        The .npz holds every evaluated point, its margin and the seed that produced
        it, per arm, which is what the analysis scripts re-read to recompute the
        tables without simulating.
        """
        d = os.path.dirname(os.path.abspath(path_prefix))
        if d:
            os.makedirs(d, exist_ok=True)
        written = []

        rf_raw = f"{path_prefix}_raw.npz"
        payload = {}
        for i, (label, (th, mg)) in enumerate(self.clouds.items()):
            payload[f"theta_{i}"] = np.asarray(th, float)
            payload[f"margins_{i}"] = np.asarray(mg, float)
            # Which seed produced each point. Campaigns saved before this field
            # existed cannot be split back into runs; readers must treat a
            # missing seeds_<i> as "pooled only" rather than guessing.
            sd = self.cloud_seeds.get(label)
            if sd is not None and len(sd) == len(mg):
                payload[f"seeds_{i}"] = np.asarray(sd, int)
        payload["labels"] = np.array(list(self.clouds.keys()), dtype=object)
        payload["param_names"] = np.array(self.param_names, dtype=object)
        np.savez_compressed(rf_raw, **payload)
        written.append(rf_raw)

        jf = f"{path_prefix}.json"
        with open(jf, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=float)
        written.append(jf)

        af = f"{path_prefix}_arms.csv"
        with open(af, "w", encoding="utf-8") as fh:
            keys = ["mean_evaluations", "mean_failures", "mean_p_fail", "std_p_fail",
                    "worst_margin", "n_regions", "n_exclusive_regions"]
            fh.write("arm," + ",".join(keys) + "\n")
            for lab in self.arm_labels:
                s = self.per_arm[lab]
                fh.write(lab + "," + ",".join(str(s.get(k, "")) for k in keys) + "\n")
        written.append(af)

        rf = f"{path_prefix}_regions.csv"
        names = self.param_names or [f"p{j}" for j in
                                     range(len(self.regions.regions[0].centroid))] \
            if self.regions.regions else []
        with open(rf, "w", encoding="utf-8") as fh:
            head = (["region_id", "n_points", "worst_margin", "p_hit_uniform",
                     "p_hit_odd", "n_for_50pct", "is_singleton",
                     "n_constraining_axes", "constraints", "found_by"]
                    + [f"centroid_{n}" for n in names])
            fh.write(",".join(head) + "\n")
            for r in self.regions.regions:
                row = [r.region_id, r.n_points, f"{r.worst_margin:.6f}",
                       f"{r.p_hit_uniform:.6e}", f"{r.p_hit_odd:.6e}",
                       f"{r.n_for_50pct(self.regions.weighting):.1f}",
                       int(r.is_singleton), len(r.constraining_axes),
                       '"' + r.describe_constraints().replace('"', "'") + '"',
                       "|".join(sorted(r.found_by))]
                row += [f"{v:.6f}" for v in r.centroid]
                fh.write(",".join(str(x) for x in row) + "\n")
        written.append(rf)
        return written


# ─────────────────────────────────────────────────────────────────────────────
# The harness
# ─────────────────────────────────────────────────────────────────────────────
class ModelComparison:

    def __init__(self, scenario, arms: list, *, seeds=(0,), budget: int = 0,
                 param_lower=None, param_upper=None, eps: float | None = None,
                 min_samples: int = 2, order: str = "shuffled",
                 order_seed: int = 0, verbose: bool = True):
        # eps=None means "estimate the clustering radius from the data". A fixed
        # value silently breaks when the dimension changes: in a 9-parameter
        # scenario two random points sit ~1.2 apart, so any small eps declares
        # every failure isolated and the region comparison reports nothing.
        """
        Run a set of arms over a set of seeds and compare their failure regions.

        An "arm" is any object exposing ``label``, ``budget`` and
        ``run(seed) -> result``, where the result carries ``theta_evaluated``,
        ``margins``, ``p_fail`` and ``n_evaluations``.
        """
        if isinstance(scenario, str):
            from scenarios import SCENARIOS
            scenario = SCENARIOS[scenario]
        self.scenario = scenario
        self.arms = list(arms)
        self.seeds = list(seeds)
        self.budget = int(budget) or int(getattr(arms[0], "budget", 0))
        self.param_lower = param_lower
        self.param_upper = param_upper
        self.eps = None if eps is None else float(eps)
        self.min_samples = int(min_samples)
        if order not in ("shuffled", "interleaved", "sequential"):
            raise ValueError("order must be 'shuffled', 'interleaved' or 'sequential'")
        self.order = order
        self.order_seed = int(order_seed)
        self.verbose = verbose

    # ── ready-made configurations ──────────────────────────────────────────
    @classmethod
    def default(cls, scenario, *, budget: int = 200, seeds=(0, 1, 2),
                include_baseline: bool = True, n_iter: int = 5,
                ce_max_iter: int = 5, pool_size: int = 4000,
                odd_samples: int = 8000, param_lower=None, param_upper=None,
                verbose: bool = True) -> "ModelComparison":
        """
        The standard arm set at a matched budget:

            active_boundary[lhs]   vs  active_boundary[random]
            cross_entropy[lhs]     vs  cross_entropy[random]

        plus, when ``include_plain``, plain LHS and plain random sampling.

        Budget split for active boundary: a third on the seed design, the rest over
        ``n_iter`` active batches. The cross-entropy descent can stop early, so its
        realised budget is <= the target and the report prints the realised count.

        ``pool_size`` and ``odd_samples`` cost no simulations, but the credible
        interval draws 200 GP posterior samples at ``odd_samples`` points, an
        O(odd_samples^3) Cholesky.
        """
        if isinstance(scenario, str):
            from scenarios import SCENARIOS
            scenario = SCENARIOS[scenario]

        n_seed = max(8, budget // 3)
        batch = max(1, (budget - n_seed) // max(1, n_iter))
        ab_kw = dict(n_seed=n_seed, batch=batch, n_iter=n_iter,
                     pool_size=pool_size, odd_samples=odd_samples,
                     param_lower=param_lower, param_upper=param_upper,
                     verbose=verbose)

        spi = max(8, budget // (ce_max_iter + 3))
        final = max(8, budget - spi * ce_max_iter)
        ce_kw = dict(samples_per_iter=spi, max_iter=ce_max_iter,
                     final_samples=final, lower=param_lower, upper=param_upper,
                     verbose=verbose)
        # The CE arms will warn on construction when this split leaves too few
        # defensive draws; say what the split IS here, so the budget that caused
        # it is visible next to the number that has to change.
        from pipeline.rare_event import (MIN_DEFENSIVE_SAMPLES,
                                         defensive_sample_count)
        n_def = defensive_sample_count(final, 0.2)
        if n_def < MIN_DEFENSIVE_SAMPLES and verbose:
            need = int(np.ceil(MIN_DEFENSIVE_SAMPLES / 0.2)) + spi * ce_max_iter
            print(f"[budget] cross_entropy: samples_per_iter={spi}, "
                  f"final_samples={final} -> only {n_def} defensive draws. "
                  f"p_fail from the CE arms is NOT usable below "
                  f"{MIN_DEFENSIVE_SAMPLES}; a budget of ~{need} would fix it. "
                  f"The failures and regions the arms find stay valid.",
                  flush=True)

        arms = [
            LHSActiveBoundary(scenario, **ab_kw),
            RandomSearchActiveBoundary(scenario, **ab_kw),
        ]
        if hasattr(scenario, "param_distributions"):
            arms += [
                LHSCrossEntropy(scenario, **ce_kw),
                RandomSearchCrossEntropy(scenario, **ce_kw),
            ]
        if include_baseline:
            arms += [
                PlainSamplingBaseline(scenario, "lhs", n_samples=budget,
                                      param_lower=param_lower, param_upper=param_upper),
                PlainSamplingBaseline(scenario, "random", n_samples=budget,
                                      param_lower=param_lower, param_upper=param_upper),
            ]
        return cls(scenario, arms, seeds=seeds, budget=budget,
                   param_lower=param_lower, param_upper=param_upper, verbose=verbose)

    # ── execution ──────────────────────────────────────────────────────────
    def _execution_order(self) -> list:
        """
        The (arm, seed) pairs in the order they will be simulated.

        "shuffled" permutes them with `order_seed`, "interleaved" runs one seed at a
        time across the arms, and "sequential" keeps the definition order, i.e. all
        of one arm before the next.
        """
        tasks = [(i, arm, sd) for i, arm in enumerate(self.arms) for sd in self.seeds]
        if self.order == "shuffled":
            np.random.default_rng(self.order_seed).shuffle(tasks)
        elif self.order == "interleaved":
            tasks.sort(key=lambda t: (self.seeds.index(t[2]), t[0]))
        return tasks

    @staticmethod
    def _seed_row(sd, res, mg, fails: int, run_index: int, started: float) -> dict:
        """
        What one (arm, seed) run contributes to the campaign record.

        `margin_q05` is the 5th percentile of the run's margins, a continuous
        measure of how far into the failure region it got; `worst_margin` is the
        single deepest one.

        `p_fail_usable` is False when the arm's estimator is out of its validity
        regime at this budget (see rare_event.MIN_DEFENSIVE_SAMPLES): `p_fail` still
        carries a number, and the ranking excludes it.

        `run_index` and `started_at_s` are what :meth:`drift_diagnostic` reads.
        """
        return {
            "seed": int(sd),
            "p_fail": float(res.p_fail),
            "p_fail_usable": bool(getattr(res, "p_fail_usable", True)),
            "n_defensive": int(getattr(res, "n_defensive", 0)),
            "ess": float(getattr(res, "ess", float("nan"))),
            "n_fail_effective": int(getattr(res, "n_fail_effective", 0)),
            "n_evaluations": int(res.n_evaluations),
            "n_failures": fails,
            "worst_margin": float(np.nanmin(mg)) if mg.size else float("nan"),
            "margin_q05": (float(np.nanpercentile(mg, 5))
                           if np.isfinite(mg).any() else float("nan")),
            # Bookkeeping for the drift diagnostic: when this run happened.
            "run_index": int(run_index),
            "started_at_s": float(started),
        }

    @staticmethod
    def _arm_stats(rows: list, regions, label: str) -> dict:
        """One arm's aggregates over its seeds."""
        p = np.array([r["p_fail"] for r in rows], float)
        f = np.array([r["n_failures"] for r in rows], float)
        e = np.array([r["n_evaluations"] for r in rows], float)
        wm = np.array([r["worst_margin"] for r in rows], float)
        return {
            "n_seeds": len(rows),
            "mean_p_fail": float(p.mean()),
            "std_p_fail": float(p.std(ddof=1)) if len(p) > 1 else 0.0,
            "mean_failures": float(f.mean()),
            "std_failures": float(f.std(ddof=1)) if len(f) > 1 else 0.0,
            "mean_evaluations": float(e.mean()),
            "worst_margin": float(np.nanmin(wm)) if wm.size else float("nan"),
            "n_regions": len(regions.discovery.get(label, ())),
            "n_exclusive_regions": len(regions.exclusive_regions(label)),
            # False as soon as one seed produced an unusable estimate.
            "p_fail_usable": bool(all(r.get("p_fail_usable", True) for r in rows)),
        }

    def _run_all_tasks(self, tasks: list, thr: float) -> tuple:
        """
        Simulate every (arm, seed) pair.

        An arm that raises loses that seed only: the exception is recorded in
        `failed` and the remaining runs continue.

        Returns ``(collected, failed)``.
        """
        collected: dict = {}
        failed: list = []
        t_start = time.time()
        for run_index, (_, arm, sd) in enumerate(tasks):
            label = arm.label
            if self.verbose:
                print(f"\n=== {label}  seed={sd}   "
                      f"[{run_index + 1}/{len(tasks)}] ===", flush=True)
            started = time.time() - t_start
            try:
                res = arm.run(seed=sd)
            except Exception as exc:
                failed.append((label, sd, f"{type(exc).__name__}: {exc}"))
                if self.verbose:
                    print(f"  !! {type(exc).__name__}: {exc}", flush=True)
                continue
            th = np.asarray(res.theta_evaluated, float)
            mg = np.asarray(res.margins, float)
            fails = int((mg[np.isfinite(mg)] < thr).sum())
            collected.setdefault(label, {"theta": [], "margins": [],
                                         "seeds": [], "rows": []})
            collected[label]["theta"].append(th)
            collected[label]["margins"].append(mg)
            collected[label]["seeds"].append(np.full(len(mg), int(sd), dtype=int))
            collected[label]["rows"].append(
                self._seed_row(sd, res, mg, fails, run_index, started))
        return collected, failed

    def run(self) -> ComparisonResult:
        """
        Run the whole campaign: every arm on every seed, at a matched budget.

        The result carries the pooled clouds AND the per-seed provenance, which
        is what makes the paired tests downstream possible.
        """
        b = self.scenario.param_bounds()
        lower = np.asarray(self.param_lower if self.param_lower is not None
                           else b["lower"], float)
        upper = np.asarray(self.param_upper if self.param_upper is not None
                           else b["upper"], float)
        names = list(b["names"])
        thr = float(self.scenario.failure_threshold())
        dists = (self.scenario.param_distributions(lower, upper)
                 if hasattr(self.scenario, "param_distributions") else None)

        clouds: dict = {}
        cloud_seeds: dict = {}
        per_seed: dict = {}

        tasks = self._execution_order()
        collected, failed = self._run_all_tasks(tasks, thr)

        for label in [a.label for a in self.arms]:
            c = collected.get(label)
            if not c or not c["theta"]:
                continue
            clouds[label] = (np.vstack(c["theta"]), np.concatenate(c["margins"]))
            cloud_seeds[label] = np.concatenate(c["seeds"])
            per_seed[label] = sorted(c["rows"], key=lambda r: r["seed"])

        if not clouds:
            raise RuntimeError("every arm failed — nothing to compare "
                               f"({failed})")

        regions = compare_failure_regions(
            clouds, lower, upper, threshold=thr, param_names=names,
            dists=dists, eps=self.eps, min_samples=self.min_samples,
        )

        per_arm = {label: self._arm_stats(rows, regions, label)
                   for label, rows in per_seed.items()}

        return ComparisonResult(
            arm_labels=list(clouds.keys()),
            per_arm=per_arm,
            per_seed=per_seed,
            clouds=clouds,
            regions=regions,
            seeds=self.seeds,
            budget=self.budget,
            param_names=names,
            failed_runs=failed,
            cloud_seeds=cloud_seeds,
            param_lower=lower,
            param_upper=upper,
            odd_dists=dists,
            threshold=thr,
        )
