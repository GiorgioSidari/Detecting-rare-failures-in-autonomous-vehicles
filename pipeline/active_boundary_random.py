"""
Active learning of the failure boundary, with a pluggable sampling design.

Class form of :func:`pipeline.active_boundary.run_active_boundary`: a Gaussian
process is fitted on the safety margin, batches are acquired where the predicted
probability of failure is closest to 0.5, and P(failure) is finally integrated
against the ODD. Every draw goes through a
:class:`~pipeline.samplers.BaseSampler`, so the design can be stratified (LHS) or
i.i.d. without touching the algorithm.

Three sampling decisions exist inside the algorithm and are configured
separately:

  1. **seed design** - the initial batch of real simulations, the only points
     chosen without the model. Controlled by ``sampler``.
  2. **candidate pool** - the points the acquisition function scores at each
     iteration. Model-only, no simulations. Also controlled by ``sampler``.
  3. **ODD integration** - the final estimate of P(failure). Model-only, and
     controlled by ``odd_sampler``, which defaults to LHS in every arm.

``acquisition="random"`` replaces the entropy criterion with a uniform draw from
the pool, leaving everything else in place.

Usage
-----
    from pipeline.active_boundary_random import (
        ActiveBoundaryRunner, RandomSearchActiveBoundary, LHSActiveBoundary,
    )

    res_lhs = LHSActiveBoundary(scenario).run(seed=0)
    res_rnd = RandomSearchActiveBoundary(scenario).run(seed=0)

Both return an :class:`~pipeline.active_boundary.ActiveBoundaryResult`.
"""
from __future__ import annotations

import numpy as np

from pipeline.active_boundary import (
    ActiveBoundaryResult,
    build_gp,
    evaluate_batch,
    failure_probability,
    greedy_diverse,
    rbf_lengthscales,
)
from pipeline.samplers import get_sampler


def build_seeded_gp(d: int, seed: int):
    """
    The pipeline's GP, made reproducible.

    ``build_gp`` leaves ``random_state`` unset, so the four hyper-parameter
    restarts draw from numpy's global RNG: two runs with the same ``seed`` fit
    slightly different kernels and therefore acquire different points. That is
    harmless when you run the learner once, and fatal for a comparison whose
    entire premise is "same seed, same budget, only the design differs".
    Pinning the GP's random_state removes that confound.
    """
    gp = build_gp(d)
    gp.random_state = int(seed)
    return gp


class ActiveBoundaryRunner:

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
        if isinstance(scenario, str):
            from scenarios import SCENARIOS
            scenario = SCENARIOS[scenario]
        if acquisition not in ("entropy", "random"):
            raise ValueError("acquisition must be 'entropy' or 'random'")

        self.scenario = scenario
        self.sampler = get_sampler(sampler) #initial design
        self.odd_sampler = get_sampler(odd_sampler) # candidate pool
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
        return self.n_seed + self.batch * self.n_iter #default = 40 + 16 x 5 = 120

    # ── main entry point ────────────────────────────────────────────────────
    def _seed_design(self, sc, d: int, seed: int, from_unit, thr: float):
        """
        The initial design: the only points chosen without the model.

        This is where the sampling design (`lhs` or `random`) shows, and the GP
        needs at least two valid evaluations to be fitted at all.
        """
        seed_unit = self.sampler.unit(self.n_seed, d, seed=seed)
        theta = from_unit(seed_unit)
        margins, valid = evaluate_batch(sc, theta)

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
        return X_all, y_all

    def _pick_batch(self, gp, pool_u: np.ndarray, thr: float, d: int, rng: np.random.Generator):
        """
        Which candidates to simulate next.

        "entropy" scores each candidate by the binary entropy of its predicted
        failure probability, which peaks at p = 0.5, i.e. on the boundary the
        model is least sure about; `greedy_diverse` then keeps the batch spread
        out, since sixteen copies of the same question cost sixteen simulations
        and answer one. "random" ignores the model: it is the control arm that
        isolates how much of the yield comes from learning.
        """
        if self.acquisition != "entropy":
            return rng.choice(self.pool_size, size=self.batch, replace=False)
        mu, sigma = gp.predict(pool_u, return_std=True)
        p = failure_probability(mu, sigma, thr)
        eps = 1e-9
        score = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
        return greedy_diverse(score, pool_u, self.batch, min_dist=0.5 / d ** 0.5)

    def _estimate_p_fail(self, gp, sc, lower: np.ndarray, upper: np.ndarray,
                         thr: float, to_unit, seed: int):
        """
        P(failure) under the ODD from the surrogate, with a credible interval.

        Returns ``(p_fail, (lo, hi), weighting)``. The interval comes from
        posterior samples of the GP; if the draw fails, it falls back to the
        normal approximation on the pointwise probabilities.
        """
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
        p_point = failure_probability(mu_q, sig_q, thr)
        p_fail = float(np.mean(p_point))

        try:
            samples = gp.sample_y(Xq, n_samples=200, random_state=seed)
            p_per_sample = (samples < thr).mean(axis=0)
            lo, hi = np.percentile(p_per_sample, [2.5, 97.5])
            p_ci = (float(lo), float(hi))
        except Exception:
            se = float(np.std(p_point) / np.sqrt(len(p_point)))
            p_ci = (max(0.0, p_fail - 1.96 * se), min(1.0, p_fail + 1.96 * se))
        return p_fail, p_ci, weighting

    def _summary(self, names, d: int, ls, importance, labels, y_all: np.ndarray,
                 p_fail, p_ci, weighting, seed: int) -> dict:
        """Everything the campaign records about this run, as a flat dict."""
        order = np.argsort(-importance)
        return {
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

        def to_unit(theta: np.ndarray):
            return (theta - lower) / span

        def from_unit(u: np.ndarray):
            return u * span + lower

        rng = np.random.default_rng(seed)  #only for randacq

        # ── 1. Seed design (the only unguided look at the space) ────────────
        X_all, y_all = self._seed_design(sc, d, seed, from_unit, thr)

        gp = None
        # ── 2. Active-learning iterations ──────────────────────────────────
        for it in range(self.n_iter):
            gp = build_seeded_gp(d, seed + it)
            gp.fit(to_unit(X_all), y_all)

            pool_u = self.sampler.unit(self.pool_size, d, seed=seed + 100 + it)
            picks = self._pick_batch(gp, pool_u, thr, d, rng)

            new_theta = from_unit(pool_u[picks])
            m_new, v_new = evaluate_batch(sc, new_theta)
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
        p_fail, p_ci, weighting = self._estimate_p_fail(
            gp, sc, lower, upper, thr, to_unit, seed)

        # ── 5. Feature importance (ARD length-scales) ──────────────────────
        ls = rbf_lengthscales(gp.kernel_, d)
        inv = 1.0 / np.maximum(ls, 1e-9)
        importance = inv / inv.sum()

        labels = (y_all < thr).astype(int)
        summary = self._summary(names, d, ls, importance, labels, y_all,
                                p_fail, p_ci, weighting, seed)

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

    def __init__(self, scenario, **kw):
        """Active boundary with the stratified Latin Hypercube design (the current default)."""
        kw.pop("sampler", None)
        super().__init__(scenario, sampler="lhs", **kw)


class RandomSearchActiveBoundary(ActiveBoundaryRunner):

    def __init__(self, scenario, **kw):
        """
        Active boundary driven by plain uniform random search.

        Same GP, same acquisition, same budget — only the design changes. Use it as
        the control arm when arguing that LHS reaches failure regions that random
        search does not reliably see.
        """
        kw.pop("sampler", None)
        super().__init__(scenario, sampler="random", **kw)
