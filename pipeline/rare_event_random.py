"""
Cross-entropy rare-event estimation with a pluggable sampling design.

Class form of :func:`pipeline.rare_event.estimate_failure_probability`: the
proposal is walked toward the failure region by refitting it on the elite
fraction of each batch with f/q weights, and the final probability is estimated
by importance sampling under the defensive mixture ``alpha*f + (1-alpha)*q``,
with a bootstrap confidence interval. Every draw goes through a
:class:`~pipeline.samplers.BaseSampler`.

The two designs differ in how each batch is drawn from the current proposal:

  * ``sampler="random"``  draws i.i.d., which is what
    :func:`pipeline.rare_event.sample_product` does;
  * ``sampler="lhs"``     pushes a Latin Hypercube through the proposal's
    inverse CDF, so each batch covers every stratum of every axis once.

The draw feeds the gamma quantile that sets the elite set, so it affects where
the descent goes, not only its variance.

Usage
-----
    from pipeline.rare_event_random import CrossEntropyRunner, RandomSearchCrossEntropy

    res_rnd = RandomSearchCrossEntropy(scenario).run(seed=0)
    res_lhs = CrossEntropyRunner(scenario, sampler="lhs").run(seed=0)

Both return a :class:`SampledRareEventResult`, a
:class:`~pipeline.rare_event.RareEventResult` extended with every evaluated
point and margin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.rare_event import (
    MIN_DEFENSIVE_SAMPLES,
    RareEventResult,
    bootstrap_ci,
    build_proposal,
    logpdf_product,
    check_defensive_budget,
    defensive_sample_count,
    effective_sample_size,
    scenario_margin_fn,
)
from pipeline.samplers import get_sampler


@dataclass
class SampledRareEventResult(RareEventResult):
    """RareEventResult + the evaluated design, needed for failure-region analysis."""

    theta_evaluated: np.ndarray = None      # (M, d) every point actually simulated
    margins: np.ndarray = None              # (M,) their margins (finite only)
    labels: np.ndarray = None               # (M,) 1 = failure
    sampler: str = ""
    summary: dict = field(default_factory=dict)


class CrossEntropyRunner:

    def __init__(
        self,
        scenario=None,
        sampler="lhs",
        *,
        margin_fn=None,
        f_dists=None,
        lower=None,
        upper=None,
        threshold: float | None = None,
        samples_per_iter: int = 60,
        rho: float = 0.2,
        max_iter: int = 10,
        final_samples: int = 300,
        alpha: float = 0.2,
        scale_floor: float = 0.03,
        verbose: bool = False,
    ):
        """
        Cross-Entropy + defensive-mixture importance sampling, sampler-parameterised.

        Parameters
        ----------
        scenario : registry key (str), a BaseScenario-like object, or None when
                   ``margin_fn`` / ``f_dists`` / ``lower`` / ``upper`` are given
                   explicitly (the synthetic-validation path).
        sampler  : "lhs" | "random" | BaseSampler.
        samples_per_iter : simulations per CE iteration.
        rho      : elite fraction (the margin quantile that sets gamma).
        max_iter : cap on CE iterations.
        final_samples : simulations spent on the final IS estimate.
        alpha    : defensive-mixture fraction drawn from f; bounds the weights by 1/alpha.
        scale_floor : minimum proposal scale as a fraction of the range (anti-collapse).
        threshold : failure threshold; taken from the scenario when omitted.

        The simulation budget is ``samples_per_iter * iterations + final_samples``;
        since ``iterations`` is data-dependent (the descent can stop early), the
        comparison harness reports the realised ``n_evaluations`` alongside the
        result rather than assuming equal budgets.
        """
        if isinstance(scenario, str):
            from scenarios import SCENARIOS
            scenario = SCENARIOS[scenario]

        if scenario is None and (margin_fn is None or f_dists is None):
            raise ValueError(
                "Provide either a scenario, or margin_fn + f_dists + lower + upper.")

        if scenario is not None:
            b = scenario.param_bounds()
            lower = b["lower"] if lower is None else lower
            upper = b["upper"] if upper is None else upper
            if not hasattr(scenario, "param_distributions"):
                raise ValueError(
                    "The Cross-Entropy method needs an operational distribution: "
                    "this scenario does not expose param_distributions().")
            f_dists = f_dists or scenario.param_distributions(
                np.asarray(lower, float), np.asarray(upper, float))
            margin_fn = margin_fn or scenario_margin_fn(scenario)
            threshold = (float(scenario.failure_threshold())
                         if threshold is None else float(threshold))
            self.param_names = list(b["names"])
        else:
            threshold = 0.0 if threshold is None else float(threshold)
            self.param_names = [f"p{j}" for j in range(len(np.asarray(lower, float)))]

        self.scenario = scenario
        self.sampler = get_sampler(sampler)
        self.margin_fn = margin_fn
        self.f_dists = f_dists
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.threshold = float(threshold)
        self.samples_per_iter = int(samples_per_iter)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.final_samples = int(final_samples)
        self.alpha = float(alpha)
        self.scale_floor = float(scale_floor)
        self.verbose = verbose


        self.n_defensive = defensive_sample_count(self.final_samples, self.alpha)
        self.p_fail_usable = check_defensive_budget(
            self.final_samples, self.alpha, label=self.label, stacklevel=4)

    @property
    def label(self) -> str:
        return f"cross_entropy[{self.sampler.name}]"

    def _ce_descent(self, loc: np.ndarray, scale: np.ndarray,
                    scale_min: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                    thr: float, seed: int,
                    seen_X: list, seen_m: list, n_eval: int) -> tuple:
        """
        Walk the proposal distribution toward the failure region.

        Each iteration samples from the current proposal, keeps the worst `rho`
        fraction as the elite set, and re-fits the proposal on it with
        importance weights (in log space, stabilised). Stops when the elite
        quantile reaches the failure threshold, or when there are too few valid
        points left to fit anything.

        Returns the updated ``(loc, scale, gamma_history, n_eval)``; the
        evaluated points are appended to `seen_X` / `seen_m` in place.
        """
        gamma_hist: list = []
        for it in range(self.max_iter):
            q = build_proposal(lo, hi, loc, scale)
            X = self.sampler.from_dists(self.samples_per_iter, q, seed=seed + 1000 + it)
            m = np.asarray(self.margin_fn(X), dtype=float)
            n_eval += self.samples_per_iter
            ok = np.isfinite(m)
            Xv, mv = X[ok], m[ok]
            seen_X.append(Xv)
            seen_m.append(mv)
            if mv.size < 2:
                break
            gamma = max(float(np.quantile(mv, self.rho)), thr)
            gamma_hist.append(gamma)
            Xe = Xv[mv <= gamma]
            if Xe.shape[0] < 2:
                break
            logw = logpdf_product(self.f_dists, Xe) - logpdf_product(q, Xe)
            w = np.exp(logw - logw.max())
            w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
            loc = np.clip((w[:, None] * Xe).sum(axis=0), lo, hi)
            scale = np.maximum(
                np.sqrt((w[:, None] * (Xe - loc) ** 2).sum(axis=0)), scale_min)
            if self.verbose:
                print(f"  [{self.label} CE] gamma={gamma:+.4f}  "
                      f"elite={Xe.shape[0]}  eval={n_eval}", flush=True)
            if gamma <= thr:
                break
        return loc, scale, gamma_hist, n_eval

    def _final_estimate(self, q, thr: float, seed: int,
                        rng: np.random.Generator, seen_X: list, seen_m: list,
                        n_eval: int) -> tuple:
        """
        Importance sampling under the defensive mixture `alpha*f + (1-alpha)*q`.

        Weights are `f/d = 1 / (alpha + (1-alpha) * q/f)`, computed in log space and
        bounded above by `1/alpha`.

        Returns ``(p_hat, ci, ess, fail_flags, n_eval)``.
        """
        n_f = int(self.alpha * self.final_samples)
        X = np.vstack([
            self.sampler.from_dists(n_f, self.f_dists, seed=seed + 9001),
            self.sampler.from_dists(self.final_samples - n_f, q, seed=seed + 9002),
        ])
        m = np.asarray(self.margin_fn(X), dtype=float)
        n_eval += self.final_samples
        ok = np.isfinite(m)
        Xv, mv = X[ok], m[ok]
        seen_X.append(Xv)
        seen_m.append(mv)

        log_qf = logpdf_product(q, Xv) - logpdf_product(self.f_dists, Xv)
        w = 1.0 / (self.alpha + (1.0 - self.alpha) * np.exp(log_qf))
        fail = (mv < thr).astype(float)
        h = fail * w
        p_hat = float(h.mean()) if h.size else 0.0
        return p_hat, bootstrap_ci(h, rng), effective_sample_size(w), fail, n_eval

    def _summary(self, seed: int, n_eval, gamma_hist, p_hat, ci, ess, fail,
                 labels_all, margins_all, loc: np.ndarray, scale: np.ndarray) -> dict:
        """Everything the campaign records about this run, as a flat dict."""
        return {
            "label": self.label,
            "sampler": self.sampler.name,
            "seed": int(seed),
            "n_evaluations": int(n_eval),
            "iterations": len(gamma_hist),
            "p_fail": p_hat,
            "p_fail_ci": ci,
            "p_fail_usable": bool(self.p_fail_usable),
            "n_defensive": int(self.n_defensive),
            "min_defensive_samples": int(MIN_DEFENSIVE_SAMPLES),
            "ess": float(ess),
            "n_fail_effective": int(fail.sum()),
            "n_failures_found": int(labels_all.sum()),
            "worst_margin": float(np.min(margins_all)) if margins_all.size else float("nan"),
            "gamma_history": [float(g) for g in gamma_hist],
            "q_loc": {n: float(v) for n, v in zip(self.param_names, loc)},
            "q_scale": {n: float(v) for n, v in zip(self.param_names, scale)},
        }

    def run(self, seed: int = 0) -> SampledRareEventResult:
        """Execute the CE descent and the final IS estimate."""
        rng = np.random.default_rng(seed)
        lo, hi = self.lower, self.upper
        d = len(lo)
        thr = self.threshold

        loc = np.array([float(fd.mean()) for fd in self.f_dists])
        scale = np.array([float(fd.std()) for fd in self.f_dists])
        scale_min = self.scale_floor * (hi - lo)

        n_eval = 0
        gamma_hist: list = []
        seen_X: list = []
        seen_m: list = []

        loc, scale, gamma_hist, n_eval = self._ce_descent(
            loc, scale, scale_min, lo, hi, thr, seed, seen_X, seen_m, n_eval)

        q = build_proposal(lo, hi, loc, scale)
        p_hat, ci, ess, fail, n_eval = self._final_estimate(
            q, thr, seed, rng, seen_X, seen_m, n_eval)

        theta_all = np.vstack([x for x in seen_X if len(x)]) if seen_X else np.empty((0, d))
        margins_all = np.concatenate([y for y in seen_m if len(y)]) if seen_m else np.empty(0)
        labels_all = (margins_all < thr).astype(int)

        summary = self._summary(seed, n_eval, gamma_hist, p_hat, ci, ess,
                                fail, labels_all, margins_all, loc, scale)

        return SampledRareEventResult(
            p_fail=p_hat,
            ci=ci,
            n_evaluations=n_eval,
            iterations=len(gamma_hist),
            q_loc=loc,
            q_scale=scale,
            gamma_history=gamma_hist,
            n_fail_effective=int(fail.sum()),
            n_defensive=int(self.n_defensive),
            ess=float(ess),
            p_fail_usable=bool(self.p_fail_usable),
            theta_evaluated=theta_all,
            margins=margins_all,
            labels=labels_all,
            sampler=self.sampler.name,
            summary=summary,
        )


class LHSCrossEntropy(CrossEntropyRunner):

    def __init__(self, scenario=None, **kw):
        """Cross-Entropy whose per-iteration batches are stratified with a Latin Hypercube."""
        kw.pop("sampler", None)
        super().__init__(scenario, sampler="lhs", **kw)


class RandomSearchCrossEntropy(CrossEntropyRunner):

    def __init__(self, scenario=None, **kw):
        """
        Cross-Entropy with plain i.i.d. random draws — statistically identical to the
        existing :func:`pipeline.rare_event.estimate_failure_probability`, and the
        control arm for the stratified version.
        """
        kw.pop("sampler", None)
        super().__init__(scenario, sampler="random", **kw)
