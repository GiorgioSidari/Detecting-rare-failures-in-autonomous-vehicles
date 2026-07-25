"""
Active-learning of the failure BOUNDARY (Step B).

Where run() (severity) and run_rare_event() (rarity via Cross-Entropy) give a
list of crashes and a probability, this module learns *where and why* a scenario
fails: a probabilistic model P(fail | theta) over the parameter space, refined by
sampling adaptively near the fail/safe boundary. It is the generalisation of the
hand-found meters-per-steer envelope into a method that discovers the boundary
in the full parameter space with few simulations.

Method
------
- Surrogate: a Gaussian Process REGRESSOR on the continuous safety margin
  g(theta) (more informative than the binary label). P(fail|theta) =
  Phi((thr - mu(theta)) / sigma(theta)), the probability that the margin is
  below the failure threshold.
- Active loop: seed with LHS, fit the GP, then pick the next batch where the
  classification is most uncertain (P ~ 0.5), with a diversity penalty so the
  batch spreads along the boundary. Evaluate on the real scenario, refit, repeat.
- P(failure) under the ODD: Monte-Carlo integrate P(fail|theta) against the
  operational distribution (scenario.param_distributions). A credible interval
  comes from GP posterior samples (captures model uncertainty), not just MC noise.
- Feature importance: ARD length-scales of the fitted RBF kernel — a short
  length-scale on a dimension means the failure is sensitive to it.

Backend-agnostic: uses only the BaseScenario interface (param_bounds,
param_distributions, run_simulation, compute_qoi, failure_threshold), so it runs
on the MetaDrive scenario (fast, practical) exactly as on any other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube


@dataclass
class ActiveBoundaryResult:
    p_fail: float                              # P(failure) under the ODD
    p_fail_ci: tuple                           # (lo, hi) credible interval
    n_evaluations: int                         # total simulations spent
    theta_evaluated: np.ndarray                # (M, d) all simulated points
    margins: np.ndarray                        # (M,) observed safety margins
    labels: np.ndarray                         # (M,) 1 = failure
    param_names: list = field(default_factory=list)
    param_importance: np.ndarray = None        # (d,) normalised, sums to 1
    boundary_summary: dict = field(default_factory=dict)
    model: object = None                       # fitted GP (for inspection)


def _build_gp(d: int):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    # Wide bounds so irrelevant dimensions can take a large length-scale (and
    # the constant/noise terms can settle) without hitting a bound -> fewer
    # ConvergenceWarnings and a cleaner ARD separation.
    kernel = (ConstantKernel(1.0, (1e-3, 1e4))
              * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e4))
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=4, alpha=1e-8)


def _rbf_lengthscales(kernel, d: int) -> np.ndarray:
    """Walk a composite kernel to find the RBF length_scale array."""
    from sklearn.gaussian_process.kernels import RBF
    stack = [kernel]
    while stack:
        k = stack.pop()
        if isinstance(k, RBF):
            ls = np.atleast_1d(k.length_scale).astype(float)
            return ls if ls.size == d else np.full(d, float(ls.ravel()[0]))
        for attr in ("k1", "k2"):
            if hasattr(k, attr):
                stack.append(getattr(k, attr))
    return np.ones(d)


def _p_fail(mu: np.ndarray, sigma: np.ndarray, thr: float) -> np.ndarray:
    """P(margin < thr) under the GP predictive Gaussian."""
    sigma = np.maximum(sigma, 1e-9)
    return norm.cdf((thr - mu) / sigma)


def _greedy_diverse(scores: np.ndarray, X: np.ndarray, k: int,
                    min_dist: float) -> list:
    """Pick k high-score points, penalising ones close to already-picked (in the
    unit cube) so the batch spreads along the boundary rather than clustering."""
    picked: list = []
    remaining = np.argsort(-scores)
    for idx in remaining:
        if len(picked) >= k:
            break
        if all(np.linalg.norm(X[idx] - X[j]) >= min_dist for j in picked):
            picked.append(idx)
    # top-up if diversity filter was too strict
    if len(picked) < k:
        for idx in remaining:
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= k:
                break
    return picked[:k]


def _evaluate(scenario, theta: np.ndarray):
    """Run the scenario on theta (M,d) and return (margins, valid_mask)."""
    traj = scenario.run_simulation(theta)
    margins = np.asarray(scenario.compute_qoi(traj, theta), dtype=float)
    valid = getattr(scenario, "_valid_mask", None)
    if valid is None or np.asarray(valid).shape[0] != margins.shape[0]:
        valid = ~np.isnan(margins)
    return margins, np.asarray(valid, dtype=bool)


def run_active_boundary(
    scenario,
    seed: int = 0,
    n_seed: int = 40,
    batch: int = 16,
    n_iter: int = 8,
    acquisition: str = "uncertainty",
    weight_by_odd: bool = True,
    pool_size: int = 4000,
    odd_samples: int = 8000,
    param_lower=None,
    param_upper=None,
    verbose: bool = False,
) -> ActiveBoundaryResult:
    """
    Learn the fail/safe boundary of `scenario_name` by active learning.

    Parameters
    ----------
    scenario : registry key (str) or a BaseScenario instance
    n_seed   : initial LHS evaluations
    batch    : points added per active-learning iteration
    n_iter   : number of active-learning iterations
    acquisition : "uncertainty" (max classification entropy, P~0.5) — the boundary
    weight_by_odd : if True, integrate P(failure) against param_distributions
    pool_size    : candidate pool size for acquisition (model-only, cheap)
    odd_samples  : Monte-Carlo sample count for the final P(failure) estimate

    Returns
    -------
    ActiveBoundaryResult
    """
    if isinstance(scenario, str):
        from scenarios import SCENARIOS
        scenario = SCENARIOS[scenario]
    bounds = scenario.param_bounds()
    lower = np.asarray(param_lower if param_lower is not None else bounds["lower"], float)
    upper = np.asarray(param_upper if param_upper is not None else bounds["upper"], float)
    d = len(lower)
    thr = float(scenario.failure_threshold())
    names = bounds["names"]
    span = np.where((upper - lower) > 0, upper - lower, 1.0)

    def to_unit(theta):
        return (theta - lower) / span

    def from_unit(u):
        return u * span + lower

    rng = np.random.default_rng(seed)

    # ── 1. Seed with LHS ──────────────────────────────────────────────────────
    seed_unit = LatinHypercube(d=d, seed=seed).random(n=n_seed)
    theta = from_unit(seed_unit)
    margins, valid = _evaluate(scenario, theta)

    X_all = theta[valid]
    y_all = margins[valid]
    if verbose:
        fr = float((y_all < thr).mean()) if len(y_all) else float("nan")
        print(f"[seed] {len(y_all)} valid / {n_seed}   failure rate={fr:.2%}", flush=True)

    gp = None
    # ── 2. Active-learning iterations ─────────────────────────────────────────
    for it in range(n_iter):
        gp = _build_gp(d)
        gp.fit(to_unit(X_all), y_all)

        # Candidate pool (cheap: model-only predictions).
        pool_u = LatinHypercube(d=d, seed=seed + 100 + it).random(n=pool_size)
        mu, sigma = gp.predict(pool_u, return_std=True)
        p = _p_fail(mu, sigma, thr)
        # Acquisition: classification entropy, maximal at the boundary (P~0.5).
        eps = 1e-9
        entropy = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
        picks = _greedy_diverse(entropy, pool_u, batch, min_dist=0.5 / d ** 0.5)

        new_theta = from_unit(pool_u[picks])
        m_new, v_new = _evaluate(scenario, new_theta)
        X_all = np.vstack([X_all, new_theta[v_new]])
        y_all = np.concatenate([y_all, m_new[v_new]])
        if verbose:
            fr = float((y_all < thr).mean())
            print(f"[iter {it+1}/{n_iter}] +{int(v_new.sum())} valid  "
                  f"total={len(y_all)}  cumulative failure rate={fr:.2%}", flush=True)

    # ── 3. Final fit ──────────────────────────────────────────────────────────
    gp = _build_gp(d)
    gp.fit(to_unit(X_all), y_all)

    # ── 4. P(failure) under the ODD (or uniform) with a credible interval ─────
    if weight_by_odd and hasattr(scenario, "param_distributions"):
        dists = scenario.param_distributions(lower, upper)
        odd_u = LatinHypercube(d=d, seed=seed + 777).random(n=odd_samples)
        odd_theta = np.empty_like(odd_u)
        for j, dist in enumerate(dists):
            odd_theta[:, j] = dist.ppf(odd_u[:, j])
        weighting = "ODD (param_distributions)"
    else:
        odd_theta = from_unit(LatinHypercube(d=d, seed=seed + 777).random(n=odd_samples))
        weighting = "uniform"

    Xq = to_unit(odd_theta)
    mu_q, sig_q = gp.predict(Xq, return_std=True)
    p_point = _p_fail(mu_q, sig_q, thr)
    p_fail = float(np.mean(p_point))

    # Credible interval from GP posterior function samples (model uncertainty).
    try:
        samples = gp.sample_y(Xq, n_samples=200, random_state=seed)   # (Nq, 200)
        p_per_sample = (samples < thr).mean(axis=0)                   # fraction failing
        lo, hi = np.percentile(p_per_sample, [2.5, 97.5])
        p_ci = (float(lo), float(hi))
    except Exception:
        se = float(np.std(p_point) / np.sqrt(len(p_point)))
        p_ci = (max(0.0, p_fail - 1.96 * se), min(1.0, p_fail + 1.96 * se))

    # ── 5. Feature importance (ARD length-scales) ─────────────────────────────
    ls = _rbf_lengthscales(gp.kernel_, d)
    inv = 1.0 / np.maximum(ls, 1e-9)
    importance = inv / inv.sum()

    labels = (y_all < thr).astype(int)
    order = np.argsort(-importance)
    summary = {
        "weighting": weighting,
        "n_evaluations": int(len(y_all)),
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
