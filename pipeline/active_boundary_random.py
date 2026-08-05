"""
Active-learning of the failure boundary, with a pluggable sampling design.

This is the class form of :func:`pipeline.active_boundary.run_active_boundary`.
The algorithm is unchanged — GP regressor on the safety margin, entropy
acquisition at P(fail) ~ 0.5, diversity-penalised batches, Monte-Carlo
integration of P(fail | theta) against the ODD — but every place the original
hard-codes a Latin Hypercube is routed through a :class:`~pipeline.samplers.BaseSampler`.

Why
---
LHS is a *stratified* design: it guarantees each axis is probed in every
stratum. The hypothesis behind the pipeline is that this is what lets a small
simulation budget reach the narrow parameter combinations that fail. Swapping in
plain uniform random search and re-running the identical algorithm turns that
hypothesis into a measurement: if the random arm systematically finds fewer
failures, a less negative worst margin, or a different (smaller) failure region,
the stratification is doing real work.

Three sampling decisions exist inside the algorithm, and they are kept separate
on purpose:

  1. **seed design**   - the initial batch of real simulations. This is the one
     that matters: it is the only unguided look at the space.
  2. **candidate pool** - where the acquisition function shops for the next
     batch. Model-only, so it costs nothing, but a poorly-covered pool cannot
     propose points in a region it never generated.
  3. **ODD integration** - the final Monte-Carlo estimate of P(failure). Also
     model-only. Kept LHS in BOTH arms by default (``odd_sampler="lhs"``) so
     that integration noise does not contaminate the comparison: we want to
     measure the effect of the SEARCH design, not of the quadrature.

Usage
-----
    from pipeline.active_boundary_random import (
        ActiveBoundaryRunner, RandomSearchActiveBoundary, LHSActiveBoundary,
    )

    res_lhs = LHSActiveBoundary(scenario).run(seed=0)
    res_rnd = RandomSearchActiveBoundary(scenario).run(seed=0)

Both return the same :class:`~pipeline.active_boundary.ActiveBoundaryResult` the
existing code already knows how to consume, so scripts, the API and the
failure-region comparison work on either arm unchanged.
"""
from __future__ import annotations

import numpy as np

from pipeline.active_boundary import (
    ActiveBoundaryResult,
    _build_gp,
    _evaluate,
    _greedy_diverse,
    _p_fail,
    _rbf_lengthscales,
)
from pipeline.samplers import get_sampler


def build_seeded_gp(d: int, seed: int):
    """
    The pipeline's GP, made reproducible.

    ``_build_gp`` leaves ``random_state`` unset, so the four hyper-parameter
    restarts draw from numpy's global RNG: two runs with the same ``seed`` fit
    slightly different kernels and therefore acquire different points. That is
    harmless when you run the learner once, and fatal for a comparison whose
    entire premise is "same seed, same budget, only the design differs".
    Pinning the GP's random_state removes that confound.
    """
    gp = _build_gp(d)
    gp.random_state = int(seed)
    return gp


class ActiveBoundaryRunner:
    """
    Active-learning failure-boundary learner with a configurable design.

    Parameters
    ----------
    scenario : registry key (str) or any object with the BaseScenario interface
               (param_bounds, run_simulation, compute_qoi, failure_threshold,
               and optionally param_distributions).
    sampler  : "lhs" | "random" | BaseSampler — used for the seed design and the
               acquisition candidate pool.
    odd_sampler : design used only for the final P(failure) integration. Defaults
               to "lhs" for every arm so the comparison isolates the search.
    n_seed   : initial evaluations (the unguided look at the space).
    batch    : real simulations added per active iteration.
    n_iter   : number of active-learning iterations.
    acquisition : "entropy" reproduces the published algorithm (classification
               entropy, maximal on the boundary). "random" disables the active
               step and draws each batch from the pool at random — a *pure*
               search baseline, useful to separate "stratification helps" from
               "active learning helps".
    weight_by_odd : integrate P(fail|theta) against param_distributions instead
               of a uniform ODD.
    pool_size / odd_samples : model-only sample counts (cheap, no simulation).

    Total simulation budget = ``n_seed + batch * n_iter``, identical across
    samplers by construction — a fair comparison needs equal budgets.
    """

    def __init__(
        self,
        scenario,
        sampler="lhs",
        *,
        odd_sampler="lhs",
        n_seed: int = 40,
        batch: int = 16,
        n_iter: int = 8,
        acquisition: str = "entropy",
        weight_by_odd: bool = True,
        pool_size: int = 4000,
        odd_samples: int = 8000,
        param_lower=None,
        param_upper=None,
        verbose: bool = False,
    ):
        if isinstance(scenario, str):
            from scenarios import SCENARIOS
            scenario = SCENARIOS[scenario]
        if acquisition not in ("entropy", "random"):
            raise ValueError("acquisition must be 'entropy' or 'random'")

        self.scenario = scenario
        self.sampler = get_sampler(sampler)
        self.odd_sampler = get_sampler(odd_sampler)
        self.n_seed = int(n_seed)
        self.batch = int(batch)
        self.n_iter = int(n_iter)
        self.acquisition = acquisition
        self.weight_by_odd = weight_by_odd
        self.pool_size = int(pool_size)
        self.odd_samples = int(odd_samples)
        self.param_lower = param_lower
        self.param_upper = param_upper
        self.verbose = verbose

    # ── naming / bookkeeping ────────────────────────────────────────────────
    @property
    def label(self) -> str:
        """Short identifier used as the arm name in comparison reports."""
        suffix = "" if self.acquisition == "entropy" else "+randacq"
        return f"active_boundary[{self.sampler.name}{suffix}]"

    @property
    def budget(self) -> int:
        """Number of real simulations this configuration will spend."""
        return self.n_seed + self.batch * self.n_iter

    # ── main entry point ────────────────────────────────────────────────────
    def run(self, seed: int = 0) -> ActiveBoundaryResult:
        """Execute the full loop and return an ActiveBoundaryResult."""
        sc = self.scenario
        bounds = sc.param_bounds()
        lower = np.asarray(
            self.param_lower if self.param_lower is not None else bounds["lower"], float)
        upper = np.asarray(
            self.param_upper if self.param_upper is not None else bounds["upper"], float)
        d = len(lower)
        thr = float(sc.failure_threshold())
        names = list(bounds["names"])
        span = np.where((upper - lower) > 0, upper - lower, 1.0)

        def to_unit(theta):
            return (theta - lower) / span

        def from_unit(u):
            return u * span + lower

        rng = np.random.default_rng(seed)

        # ── 1. Seed design (the only unguided look at the space) ────────────
        seed_unit = self.sampler.unit(self.n_seed, d, seed=seed)
        theta = from_unit(seed_unit)
        margins, valid = _evaluate(sc, theta)

        X_all = theta[valid]
        y_all = margins[valid]
        if self.verbose:
            fr = float((y_all < thr).mean()) if len(y_all) else float("nan")
            print(f"[{self.label} seed] {len(y_all)} valid / {self.n_seed}   "
                  f"failure rate={fr:.2%}", flush=True)

        if len(y_all) < 2:
            raise RuntimeError(
                f"{self.label}: only {len(y_all)} valid seed evaluations — cannot fit a GP. "
                "Increase n_seed or check the simulator.")

        gp = None
        # ── 2. Active-learning iterations ──────────────────────────────────
        for it in range(self.n_iter):
            gp = build_seeded_gp(d, seed + it)
            gp.fit(to_unit(X_all), y_all)

            pool_u = self.sampler.unit(self.pool_size, d, seed=seed + 100 + it)
            if self.acquisition == "entropy":
                mu, sigma = gp.predict(pool_u, return_std=True)
                p = _p_fail(mu, sigma, thr)
                eps = 1e-9
                score = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
                picks = _greedy_diverse(score, pool_u, self.batch,
                                        min_dist=0.5 / d ** 0.5)
            else:
                # Pure search baseline: ignore the model, take the batch at random.
                picks = rng.choice(self.pool_size, size=self.batch, replace=False)

            new_theta = from_unit(pool_u[picks])
            m_new, v_new = _evaluate(sc, new_theta)
            X_all = np.vstack([X_all, new_theta[v_new]])
            y_all = np.concatenate([y_all, m_new[v_new]])
            if self.verbose:
                fr = float((y_all < thr).mean())
                print(f"[{self.label} iter {it + 1}/{self.n_iter}] "
                      f"+{int(v_new.sum())} valid  total={len(y_all)}  "
                      f"cumulative failure rate={fr:.2%}", flush=True)

        # ── 3. Final fit ───────────────────────────────────────────────────
        gp = build_seeded_gp(d, seed)
        gp.fit(to_unit(X_all), y_all)

        # ── 4. P(failure) under the ODD, with a credible interval ──────────
        if self.weight_by_odd and hasattr(sc, "param_distributions"):
            dists = sc.param_distributions(lower, upper)
            odd_theta = self.odd_sampler.from_dists(self.odd_samples, dists,
                                                    seed=seed + 777)
            weighting = "ODD (param_distributions)"
        else:
            odd_theta = self.odd_sampler.bounded(self.odd_samples, lower, upper,
                                                 seed=seed + 777)
            weighting = "uniform"

        Xq = to_unit(odd_theta)
        mu_q, sig_q = gp.predict(Xq, return_std=True)
        p_point = _p_fail(mu_q, sig_q, thr)
        p_fail = float(np.mean(p_point))

        try:
            samples = gp.sample_y(Xq, n_samples=200, random_state=seed)
            p_per_sample = (samples < thr).mean(axis=0)
            lo, hi = np.percentile(p_per_sample, [2.5, 97.5])
            p_ci = (float(lo), float(hi))
        except Exception:
            se = float(np.std(p_point) / np.sqrt(len(p_point)))
            p_ci = (max(0.0, p_fail - 1.96 * se), min(1.0, p_fail + 1.96 * se))

        # ── 5. Feature importance (ARD length-scales) ──────────────────────
        ls = _rbf_lengthscales(gp.kernel_, d)
        inv = 1.0 / np.maximum(ls, 1e-9)
        importance = inv / inv.sum()

        labels = (y_all < thr).astype(int)
        order = np.argsort(-importance)
        summary = {
            "weighting": weighting,
            "sampler": self.sampler.name,
            "odd_sampler": self.odd_sampler.name,
            "acquisition": self.acquisition,
            "label": self.label,
            "seed": int(seed),
            "budget": self.budget,
            "n_evaluations": int(len(y_all)),
            "n_failures_found": int(labels.sum()),
            "worst_margin": float(np.min(y_all)) if len(y_all) else float("nan"),
            "empirical_failure_rate": float(labels.mean()),
            "p_fail": p_fail,
            "p_fail_ci": p_ci,
            "top_params": [(names[j], float(importance[j])) for j in order[:5]],
            "length_scales": {names[j]: float(ls[j]) for j in range(d)},
        }

        return ActiveBoundaryResult(
            p_fail=p_fail,
            p_fail_ci=p_ci,
            n_evaluations=int(len(y_all)),
            theta_evaluated=X_all,
            margins=y_all,
            labels=labels,
            param_names=names,
            param_importance=importance,
            boundary_summary=summary,
            model=gp,
        )


class LHSActiveBoundary(ActiveBoundaryRunner):
    """Active boundary with the stratified Latin Hypercube design (the current default)."""

    def __init__(self, scenario, **kw):
        kw.pop("sampler", None)
        super().__init__(scenario, sampler="lhs", **kw)


class RandomSearchActiveBoundary(ActiveBoundaryRunner):
    """
    Active boundary driven by plain uniform random search.

    Same GP, same acquisition, same budget — only the design changes. Use it as
    the control arm when arguing that LHS reaches failure regions that random
    search does not reliably see.
    """

    def __init__(self, scenario, **kw):
        kw.pop("sampler", None)
        super().__init__(scenario, sampler="random", **kw)
