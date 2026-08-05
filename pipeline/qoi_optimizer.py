"""
Worst-case search: find the parameters that MINIMISE the QoI safety margin.

The rest of the pipeline asks "how often does it fail?" (rarity) and "where is
the fail/safe boundary?" (active boundary). This module asks the third question:
**how bad can it get, and with which parameters?** Formally

    theta* = argmin_theta  margin(theta),    theta in [lower, upper]

The margin is the scenario's QoI (for lane keeping the composite margin of
``scenarios/lane_keeping/qoi.py``: XTE margin, steering peak, early approach).
It is expensive (one Unity run per evaluation), noisy (the control loop is not
deterministic) and has no gradient — which rules out anything gradient-based and
argues for the two optimisers implemented here:

* :class:`BayesianQoIOptimizer` — GP surrogate + Expected Improvement. The most
  sample-efficient option, and the right default at ~10 s per evaluation. It
  also leaves behind a fitted GP that says which parameters drive the worst case.
* :class:`CMAESQoIOptimizer` — self-contained CMA-ES (no extra dependency). It
  evaluates a whole generation at once, so it maps directly onto the parallel
  simulator pool, and it copes better when the landscape is rugged or the
  dimension is high enough that a GP starts to struggle.

Both are budgeted in simulations, both return every point they evaluated, and
both expose the ``label`` / ``budget`` / ``run(seed)`` interface, so they drop
straight into :class:`pipeline.model_comparison.ModelComparison` as an extra arm
and their findings feed the failure-region analysis like any other method.

A word of warning on interpretation: the minimiser is the WORST case, not the
most likely failure. A theta* that sits in a corner of the ODD with probability
1e-9 is a valid stress test and a poor risk estimate. Read this module together
with the rare-event probability, never instead of it.

Usage
-----
    from pipeline.qoi_optimizer import BayesianQoIOptimizer

    opt = BayesianQoIOptimizer(scenario, budget=120)
    res = opt.run(seed=0)
    print(res.report())
    res.best_theta, res.best_margin
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.active_boundary import _evaluate, _greedy_diverse, _rbf_lengthscales
from pipeline.active_boundary_random import build_seeded_gp
from pipeline.samplers import get_sampler


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QoIOptimizationResult:
    """Outcome of a worst-case search, shaped like the other pipeline results."""

    best_theta: np.ndarray              # (d,) physical units
    best_margin: float                  # the minimised QoI
    theta_evaluated: np.ndarray         # (M, d) everything simulated
    margins: np.ndarray                 # (M,)
    labels: np.ndarray                  # (M,) 1 = failure
    n_evaluations: int
    history: list = field(default_factory=list)   # best-so-far per iteration
    param_names: list = field(default_factory=list)
    param_importance: np.ndarray = None           # ARD, only for the GP optimiser
    summary: dict = field(default_factory=dict)
    model: object = None

    # ModelComparison compatibility. NOTE: this is the empirical failure fraction
    # of the points the optimiser CHOSE to evaluate — a biased, deliberately
    # failure-seeking sample. It is not an estimate of P(failure) under the ODD.
    @property
    def p_fail(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0

    @property
    def p_fail_ci(self) -> tuple:
        return (float("nan"), float("nan"))

    def report(self) -> str:
        names = self.param_names or [f"p{j}" for j in range(len(self.best_theta))]
        w = 78
        L = ["=" * w, " QoI WORST-CASE OPTIMISATION", "=" * w,
             f" optimiser        : {self.summary.get('label', '?')}",
             f" evaluations      : {self.n_evaluations}",
             f" best (worst) QoI : {self.best_margin:+.4f}",
             f" failures found   : {int(self.labels.sum())}/{len(self.labels)}",
             " worst-case parameters:"]
        for n, v in zip(names, self.best_theta):
            L.append(f"    {n:<22}{v:>10.3f}")
        if self.param_importance is not None:
            L.append(" parameters driving the worst case (ARD importance):")
            for j in np.argsort(-self.param_importance)[:5]:
                imp = float(self.param_importance[j])
                L.append(f"    {names[j]:<22}{imp * 100:5.1f}%  {'#' * int(round(imp * 40))}")
        if self.history:
            L.append(" best-so-far per iteration:")
            L.append("    " + "  ".join(f"{h:+.3f}" for h in self.history))
        L.append(" caveat: the minimiser is the worst case, not the likeliest failure —")
        L.append(" pair it with the rare-event probability before drawing conclusions.")
        L.append("=" * w)
        return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Shared plumbing
# ─────────────────────────────────────────────────────────────────────────────
class _BaseQoIOptimizer:
    """Bounds handling, unit-cube mapping and evaluation bookkeeping."""

    def __init__(self, scenario, *, budget: int = 120, n_init: int = 0,
                 sampler="lhs", param_lower=None, param_upper=None,
                 verbose: bool = False):
        if isinstance(scenario, str):
            from scenarios import SCENARIOS
            scenario = SCENARIOS[scenario]
        self.scenario = scenario
        self.budget = int(budget)
        self.sampler = get_sampler(sampler)
        self.verbose = verbose

        b = scenario.param_bounds()
        self.lower = np.asarray(param_lower if param_lower is not None
                                else b["lower"], float)
        self.upper = np.asarray(param_upper if param_upper is not None
                                else b["upper"], float)
        self.param_names = list(b["names"])
        self.d = len(self.lower)
        self.span = np.where((self.upper - self.lower) > 0,
                             self.upper - self.lower, 1.0)
        self.threshold = float(scenario.failure_threshold())
        self.n_init = int(n_init) if n_init else max(2 * self.d, self.budget // 4)

    def to_unit(self, theta):
        return (np.asarray(theta, float) - self.lower) / self.span

    def from_unit(self, u):
        return np.asarray(u, float) * self.span + self.lower

    def _simulate(self, unit_pts: np.ndarray):
        """Evaluate unit-cube points; returns (theta, margins) for the valid ones."""
        theta = self.from_unit(np.clip(unit_pts, 0.0, 1.0))
        margins, valid = _evaluate(self.scenario, theta)
        return theta[valid], margins[valid]

    def _finish(self, theta_all, margin_all, history, model=None,
                importance=None, extra=None) -> QoIOptimizationResult:
        theta_all = np.asarray(theta_all, float)
        margin_all = np.asarray(margin_all, float)
        if len(margin_all) == 0:
            raise RuntimeError(f"{self.label}: no valid evaluation — check the simulator.")
        i_best = int(np.argmin(margin_all))
        labels = (margin_all < self.threshold).astype(int)
        summary = {
            "label": self.label,
            "sampler": self.sampler.name,
            "budget": self.budget,
            "n_evaluations": int(len(margin_all)),
            "best_margin": float(margin_all[i_best]),
            "best_theta": {n: float(v) for n, v in
                           zip(self.param_names, theta_all[i_best])},
            "n_failures_found": int(labels.sum()),
            "empirical_failure_rate": float(labels.mean()),
        }
        summary.update(extra or {})
        return QoIOptimizationResult(
            best_theta=theta_all[i_best],
            best_margin=float(margin_all[i_best]),
            theta_evaluated=theta_all,
            margins=margin_all,
            labels=labels,
            n_evaluations=int(len(margin_all)),
            history=history,
            param_names=self.param_names,
            param_importance=importance,
            summary=summary,
            model=model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bayesian optimisation (GP + Expected Improvement)
# ─────────────────────────────────────────────────────────────────────────────
def _expected_improvement(mu, sigma, best, xi: float = 0.01) -> np.ndarray:
    """
    EI for MINIMISATION: E[max(best - f(x) - xi, 0)] under the GP posterior.

    It balances exploitation (low predicted margin) against exploration (high
    posterior variance), which is what makes it sample-efficient on an expensive
    black box. ``xi`` is a small exploration bonus that keeps it from collapsing
    onto the incumbent once the surrogate is confident.
    """
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


class BayesianQoIOptimizer(_BaseQoIOptimizer):
    """
    Sample-efficient worst-case search with a GP surrogate and EI acquisition.

    Parameters
    ----------
    budget  : total simulations (initial design included).
    n_init  : size of the initial design; defaults to max(2d, budget/4).
    batch   : evaluations per iteration — set it to the number of simulator
              workers so each round fills the pool.
    pool_size : candidate points scored by EI per iteration (model-only, free).
    xi      : EI exploration bonus.
    """

    def __init__(self, scenario, *, budget: int = 120, n_init: int = 0,
                 batch: int = 4, pool_size: int = 4000, xi: float = 0.01,
                 sampler="lhs", param_lower=None, param_upper=None,
                 verbose: bool = False):
        super().__init__(scenario, budget=budget, n_init=n_init, sampler=sampler,
                         param_lower=param_lower, param_upper=param_upper,
                         verbose=verbose)
        self.batch = max(1, int(batch))
        self.pool_size = int(pool_size)
        self.xi = float(xi)

    @property
    def label(self) -> str:
        return f"qoi_bayesopt[{self.sampler.name}]"

    def run(self, seed: int = 0) -> QoIOptimizationResult:
        n_init = min(self.n_init, self.budget)
        u0 = self.sampler.unit(n_init, self.d, seed=seed)
        theta_all, margin_all = self._simulate(u0)
        if len(margin_all) < 2:
            raise RuntimeError(f"{self.label}: initial design produced "
                               f"{len(margin_all)} valid runs — increase n_init.")
        history = [float(margin_all.min())]
        if self.verbose:
            print(f"[{self.label}] init: {len(margin_all)} valid, "
                  f"best={margin_all.min():+.4f}", flush=True)

        gp = None
        spent = n_init
        it = 0
        while spent < self.budget:
            k = min(self.batch, self.budget - spent)
            gp = build_seeded_gp(self.d, seed + it)
            gp.fit(self.to_unit(theta_all), margin_all)

            pool = self.sampler.unit(self.pool_size, self.d, seed=seed + 500 + it)
            mu, sd = gp.predict(pool, return_std=True)
            ei = _expected_improvement(mu, sd, float(margin_all.min()), self.xi)
            # Diversity filter: without it a batch collapses onto one EI peak and
            # the k parallel simulations return k copies of the same information.
            picks = _greedy_diverse(ei, pool, k, min_dist=0.3 / self.d ** 0.5)

            th_new, m_new = self._simulate(pool[picks])
            spent += k
            it += 1
            if len(m_new):
                theta_all = np.vstack([theta_all, th_new])
                margin_all = np.concatenate([margin_all, m_new])
            history.append(float(margin_all.min()))
            if self.verbose:
                print(f"[{self.label}] iter {it}: spent={spent}/{self.budget}  "
                      f"best={margin_all.min():+.4f}", flush=True)

        gp = build_seeded_gp(self.d, seed)
        gp.fit(self.to_unit(theta_all), margin_all)
        ls = _rbf_lengthscales(gp.kernel_, self.d)
        inv = 1.0 / np.maximum(ls, 1e-9)
        importance = inv / inv.sum()

        return self._finish(theta_all, margin_all, history, model=gp,
                            importance=importance,
                            extra={"iterations": it, "n_init": n_init,
                                   "length_scales": {n: float(v) for n, v
                                                     in zip(self.param_names, ls)}})


# ─────────────────────────────────────────────────────────────────────────────
# 2. CMA-ES
# ─────────────────────────────────────────────────────────────────────────────
class CMAESQoIOptimizer(_BaseQoIOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy, implemented in numpy.

    CMA-ES samples a generation from N(m, sigma^2 C), keeps the best mu, and
    adapts m, sigma and C so the search distribution stretches along the
    directions that improved the objective. It needs no gradient, tolerates
    noise, and — the practical reason it is here — evaluates a full generation
    per step, which the parallel simulator pool absorbs for free.

    Box constraints are handled by clipping the samples to [lower, upper] and
    updating from the CLIPPED points, so the distribution learns to stay inside
    the ODD instead of wasting samples outside it.

    Parameters
    ----------
    budget   : total simulations.
    popsize  : generation size lambda; defaults to 4 + floor(3 ln d).
    sigma0   : initial step size, as a fraction of each parameter's range.
    x0       : starting point in physical units (defaults to the ODD centre).
    """

    def __init__(self, scenario, *, budget: int = 120, popsize: int = 0,
                 sigma0: float = 0.3, x0=None, sampler="lhs",
                 param_lower=None, param_upper=None, verbose: bool = False):
        super().__init__(scenario, budget=budget, n_init=1, sampler=sampler,
                         param_lower=param_lower, param_upper=param_upper,
                         verbose=verbose)
        self.popsize = int(popsize) if popsize else 4 + int(3 * np.log(max(self.d, 2)))
        self.sigma0 = float(sigma0)
        self.x0 = x0

    @property
    def label(self) -> str:
        return "qoi_cmaes"

    def run(self, seed: int = 0) -> QoIOptimizationResult:
        rng = np.random.default_rng(seed)
        N = self.d
        lam = self.popsize
        mu = lam // 2

        # Recombination weights: log-decreasing, so the best of the generation
        # pulls the mean hardest. mueff is the variance-effective sample size.
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Learning rates (Hansen's defaults; they scale with N and mueff).
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)      # cumulation for C
        cs = (mueff + 2) / (N + mueff + 5)                  # cumulation for sigma
        c1 = 2 / ((N + 1.3) ** 2 + mueff)                   # rank-one learning rate
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
        chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        xmean = (np.full(N, 0.5) if self.x0 is None
                 else np.clip(self.to_unit(np.asarray(self.x0, float)).ravel(), 0, 1))
        sigma = self.sigma0
        pc = np.zeros(N)
        ps = np.zeros(N)
        B = np.eye(N)
        D = np.ones(N)
        C = np.eye(N)
        invsqrtC = np.eye(N)
        eigeneval = 0

        theta_all = np.empty((0, N))
        margin_all = np.empty(0)
        history: list = []
        counteval = 0
        gen = 0

        while counteval < self.budget:
            k = min(lam, self.budget - counteval)
            z = rng.standard_normal((k, N))
            y = z @ (B * D).T                       # y ~ N(0, C)
            x = np.clip(xmean + sigma * y, 0.0, 1.0)

            th, m = self._simulate(x)
            counteval += k
            gen += 1
            if len(m) == 0:                          # whole generation invalid
                sigma *= 0.7                         # shrink and retry
                if self.verbose:
                    print(f"[{self.label}] gen {gen}: no valid run, sigma->{sigma:.3f}",
                          flush=True)
                continue
            theta_all = np.vstack([theta_all, th])
            margin_all = np.concatenate([margin_all, m])

            # Rank by margin (minimisation). Only the valid points take part in
            # the update, so an aborted simulation cannot steer the search.
            xv = self.to_unit(th)
            order = np.argsort(m)
            n_use = max(1, min(mu, len(order)))
            sel = xv[order[:n_use]]
            ww = w[:n_use] / w[:n_use].sum()

            xold = xmean
            xmean = ww @ sel

            ps = ((1 - cs) * ps
                  + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ ((xmean - xold) / sigma))
            hsig = (np.linalg.norm(ps)
                    / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN
                    < 1.4 + 2 / (N + 1))
            pc = ((1 - cc) * pc
                  + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma)

            artmp = (sel - xold) / sigma
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
                 + cmu * (artmp.T * ww) @ artmp)
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-4, 1.0))

            # Eigendecomposition is O(N^3): refresh it only every so often.
            if counteval - eigeneval > lam / (c1 + cmu) / N / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T          # enforce symmetry
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-20))
                invsqrtC = B @ np.diag(1.0 / D) @ B.T

            history.append(float(margin_all.min()))
            if self.verbose:
                print(f"[{self.label}] gen {gen}: spent={counteval}/{self.budget}  "
                      f"best={margin_all.min():+.4f}  sigma={sigma:.3f}", flush=True)

        return self._finish(theta_all, margin_all, history,
                            extra={"generations": gen, "popsize": lam,
                                   "final_sigma": float(sigma)})


# ─────────────────────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZERS = {
    "bayes": BayesianQoIOptimizer,
    "cmaes": CMAESQoIOptimizer,
}


def optimize_qoi(scenario, method: str = "bayes", *, budget: int = 120,
                 seed: int = 0, **kw) -> QoIOptimizationResult:
    """
    Minimise the scenario's QoI margin and return the worst case found.

    method : "bayes" (GP + EI, most sample-efficient) or "cmaes" (evolutionary,
             generation-parallel, more robust on rugged landscapes).
    """
    if method not in OPTIMIZERS:
        raise KeyError(f"Unknown optimiser '{method}'. Available: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[method](scenario, budget=budget, **kw).run(seed=seed)
