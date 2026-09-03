from __future__ import annotations

"""
Efficient rare-event probability estimation via the Cross-Entropy method.

Goal: estimate P(failure) = P(QoI margin < threshold) under the realistic operational
distribution f when that probability is small. Plain Monte Carlo is inefficient: at a low
P it needs a huge number of runs to observe even a few failures.

Cross-Entropy + Importance Sampling:
  1. Start with a proposal q = f.
  2. Each iteration: sample from q, evaluate margins, keep the "elite" (closest to / inside
     failure, threshold gamma lowered toward 0), and refit q toward the failure region
     (CE update = weighted moments of the elite, weights = f/q).
  3. When gamma reaches the failure threshold, estimate P with importance sampling using the
     final proposal, so samples land where failures are and few runs suffice.

Robustness: the final estimate uses a defensive mixture  d = alpha*f + (1-alpha)*q, so the
weights f/d are bounded (<= 1/alpha) and support coverage is guaranteed (this avoids the
underestimation of a proposal that becomes too narrow). The CI is bootstrap-based (the IS
estimator is skewed at small P, where a normal CI would be optimistic).

Known limitation: the proposal is a product of independent unimodal distributions. On a
multimodal failure region (e.g. "at least one of N parameters extreme") this CE underestimates;
use a mixture proposal or Subset Simulation instead. It works well on single-mode regions.

This does not validate the simulator: it assumes the margin (via the QoI) is already reliable.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy import stats


# ── The floor below which the final estimate stops being an estimate ─────────
#
# The defensive mixture draws ``n_f = alpha * final_samples`` points from f and
# the rest from the tilted proposal q. Points drawn from q are re-weighted by
# f/d, which goes to zero once q has walked away from f's bulk -- exactly what
# the CE descent is designed to make happen. So the estimate effectively rests
# on the n_f points drawn from f, each carrying weight at most 1/alpha.
#
# With n_f = 9 (what a 120-simulation budget produces) the estimator can only
# return multiples of (1/alpha)/final_samples: it degenerates into a 9-sample
# Monte Carlo whose value is 0 whenever those 9 draws miss the failure region.
# That is not a small-sample inaccuracy, it is a different estimator, and it
# will happily report 0 or 1e-9 for a probability of 3e-2.
#
# 30 is the point below which the quantisation dominates everything else. It is
# a floor, not a target: a usable estimate of a probability p needs roughly
# (1-p)/(p * rel_err^2) defensive draws.
MIN_DEFENSIVE_SAMPLES = 30


def defensive_sample_count(final_samples: int, alpha: float) -> int:
    """Points the final estimate actually draws from f (the usable ones)."""
    return int(alpha * int(final_samples))


def check_defensive_budget(final_samples: int, alpha: float, *, label: str = "",
                           stacklevel: int = 3) -> bool:
    """
    Check whether the final IS estimate has enough defensive draws.

    Returns True when ``int(alpha * final_samples) >= MIN_DEFENSIVE_SAMPLES``, and
    otherwise emits a warning and returns False. The caller carries the result
    forward as ``p_fail_usable``.
    """
    n_f = defensive_sample_count(final_samples, alpha)
    if n_f >= MIN_DEFENSIVE_SAMPLES:
        return True
    who = f"{label}: " if label else ""
    warnings.warn(
        f"{who}the final importance-sampling estimate draws only {n_f} points "
        f"from f (alpha={alpha:g} x final_samples={int(final_samples)}); below "
        f"{MIN_DEFENSIVE_SAMPLES} the estimator degenerates into an {n_f}-sample "
        f"Monte Carlo quantised at multiples of {1.0 / max(n_f, 1):.4g}. "
        "p_fail is NOT a usable probability at this budget -- use the evaluated "
        "cloud (failures, margins, regions) and read p_fail from a plain-sampling "
        f"arm instead. Raise final_samples to at least "
        f"{int(np.ceil(MIN_DEFENSIVE_SAMPLES / max(alpha, 1e-9)))}.",
        RuntimeWarning, stacklevel=stacklevel)
    return False


def effective_sample_size(w: np.ndarray) -> float:
    """
    Kish ESS of the importance weights: (sum w)^2 / sum(w^2).

    The single number that says how many of the final samples are actually
    carrying the estimate. When it collapses toward alpha*final_samples the
    tilted half of the mixture is contributing nothing.
    """
    w = np.asarray(w, dtype=float)
    s2 = float((w ** 2).sum())
    return float((w.sum() ** 2) / s2) if s2 > 0 else 0.0


@dataclass
class RareEventResult:
    p_fail: float                       # estimate of P(failure) under f
    ci: tuple                           # (lo, hi) bootstrap 95% interval
    n_evaluations: int                  # total simulation runs used (budget)
    iterations: int                     # CE iterations performed
    q_loc: np.ndarray                   # final proposal location (per dimension)
    q_scale: np.ndarray                 # final proposal scale
    gamma_history: list = field(default_factory=list)   # per-iteration gamma thresholds
    n_fail_effective: int = 0           # failing samples in the final estimate (diagnostic)
    # ── estimate-quality diagnostics (see MIN_DEFENSIVE_SAMPLES) ────────────
    n_defensive: int = 0                # points drawn from f in the final estimate
    ess: float = 0.0                    # Kish ESS of the IS weights
    p_fail_usable: bool = True          # False => p_fail is not a probability estimate


def scenario_margin_fn(scenario):
    def _margin(params: np.ndarray) -> np.ndarray:
        """
        Build a margin(params) -> margins function from a scenario, running the simulator and the
        QoI (as the orchestrator does). params: (M, d) -> margins (M,). Invalid runs return NaN
        (handled by compute_qoi) and are filtered downstream.
        """
        traj = scenario.run_simulation(params)
        return np.asarray(scenario.compute_qoi(traj, params), dtype=float)
    return _margin


def build_proposal(lo: np.ndarray, hi: np.ndarray, loc: np.ndarray, scale: np.ndarray):
    """Proposal = product of truncated normals on [lo, hi] with (loc, scale) per dimension."""
    scale = np.maximum(scale, 1e-9)
    a = (lo - loc) / scale
    b = (hi - loc) / scale
    return [stats.truncnorm(a[j], b[j], loc=loc[j], scale=scale[j]) for j in range(len(lo))]


def logpdf_product(dists: list, X: np.ndarray):
    """Sum of per-dimension log-pdfs = log of the product density at X (M, d) -> (M,)."""
    lp = np.zeros(X.shape[0])
    for j, d in enumerate(dists):
        lp += d.logpdf(X[:, j])
    return lp


def sample_product(dists: list, n: int, rng: np.random.Generator, d: int):
    """Draw n points from the product of distributions (shared rng -> independent dims)."""
    X = np.empty((n, d))
    for j in range(d):
        X[:, j] = dists[j].rvs(size=n, random_state=rng)
    return X


def bootstrap_ci(h: np.ndarray, rng: np.random.Generator, n_boot: int = 4000):
    """95% percentile bootstrap CI for the mean of h (per-sample IS contributions)."""
    n = len(h)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = h[idx].mean(axis=1)
    return (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


@dataclass(frozen=True)
class _CEProblem:
    """
    The parts of the estimation that do not change between iterations.

    `margin_fn` evaluates a batch of points, `f_dists` are the nominal
    marginals, `lo`/`hi` the ODD bounds, `scale_min` the floor on the proposal
    scale, `threshold` the failure threshold, `rng` the generator and `d` the
    number of parameters.
    """

    margin_fn: object
    f_dists: object
    lo: np.ndarray
    hi: np.ndarray
    scale_min: np.ndarray
    threshold: float
    rng: object
    d: int


def _ce_descent(pb: _CEProblem, loc: np.ndarray, scale: np.ndarray, *, samples_per_iter: int,
                rho: float, max_iter: int, n_eval: int,
                verbose: bool) -> tuple:
    """
    Move the proposal `q` toward the failure region, one elite set at a time.

    Each iteration keeps the worst `rho` fraction of the sampled margins and
    re-fits `q` on it with importance weights (in log space, stabilised). It
    stops when the elite quantile reaches the failure threshold, or when too few
    valid points are left to fit anything.

    Returns ``(loc, scale, gamma_history, n_eval)``.
    """
    gamma_hist: list = []
    for _ in range(max_iter):
        q = build_proposal(pb.lo, pb.hi, loc, scale)   # current proposal
        X = sample_product(q, samples_per_iter, pb.rng, pb.d)
        m = np.asarray(pb.margin_fn(X), dtype=float)
        n_eval += samples_per_iter
        ok = np.isfinite(m)
        Xv, mv = X[ok], m[ok]
        if mv.size < 2:
            break
        gamma = max(float(np.quantile(mv, rho)), pb.threshold)
        gamma_hist.append(gamma)
        Xe = Xv[mv <= gamma]
        if Xe.shape[0] < 2:
            break
        # Elite importance weights: the real density over the proposal's.
        logw = logpdf_product(pb.f_dists, Xe) - logpdf_product(q, Xe)
        w = np.exp(logw - logw.max())
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        loc = np.clip((w[:, None] * Xe).sum(axis=0), pb.lo, pb.hi)
        scale = np.maximum(
            np.sqrt((w[:, None] * (Xe - loc) ** 2).sum(axis=0)), pb.scale_min)
        if verbose:
            print(f"  [CE] gamma={gamma:+.4f}  elite={Xe.shape[0]}  "
                  f"eval={n_eval}", flush=True)
        if gamma <= pb.threshold:
            break   # failure threshold reached
    return loc, scale, gamma_hist, n_eval


def _defensive_is_estimate(pb: _CEProblem, q, *, alpha: float,
                           final_samples: int, n_eval: int) -> tuple:
    """
    Importance sampling under the defensive mixture `alpha*f + (1-alpha)*q`.

    The mixture bounds the weights by `1/alpha`, which is what keeps the
    estimator finite once the proposal has drifted away from the nominal
    distribution. Weights are `f/d = 1 / (alpha + (1-alpha) * q/f)`, computed
    stably in log space.

    Returns ``(p_hat, ci, weights, fail_flags, n_eval)``.
    """
    n_f = int(alpha * final_samples)
    X = np.vstack([sample_product(pb.f_dists, n_f, pb.rng, pb.d),
                   sample_product(q, final_samples - n_f, pb.rng, pb.d)])
    m = np.asarray(pb.margin_fn(X), dtype=float)
    n_eval += final_samples
    ok = np.isfinite(m)
    Xv, mv = X[ok], m[ok]

    log_qf = logpdf_product(q, Xv) - logpdf_product(pb.f_dists, Xv)
    w = 1.0 / (alpha + (1.0 - alpha) * np.exp(log_qf))
    fail = (mv < pb.threshold).astype(float)
    h = fail * w                       # weighted failures
    p_hat = float(h.mean()) if h.size else 0.0
    return p_hat, bootstrap_ci(h, pb.rng), w, fail, n_eval


def estimate_failure_probability(
    margin_fn,
    f_dists: list,
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float = 0.0,
    samples_per_iter: int = 500,
    rho: float = 0.2,
    max_iter: int = 30,
    final_samples: int = 8000,
    alpha: float = 0.2,
    scale_floor: float = 0.03,
    seed: int = 0,
    verbose: bool = False,
) -> RareEventResult:
    """
    Estimate P(margin < threshold) under distribution f (a list of frozen scipy
    distributions, one per dimension), via Cross-Entropy + defensive-mixture IS.

    Parameters
    ----------
    margin_fn        : callable((M, d)) -> (M,) margins. NaN = invalid run (filtered out).
    f_dists          : realistic operational distribution (e.g. scenario.param_distributions()).
    lower, upper     : bounds used to truncate the proposal.
    threshold        : failure threshold (default 0.0: margin < 0).
    samples_per_iter : samples per CE iteration.
    rho              : elite fraction (margin quantile used to lower gamma).
    final_samples    : samples of the final estimate (defensive mixture).
    alpha            : mixture fraction drawn from f (0<alpha<1): bounds the weights to 1/alpha.
    scale_floor      : minimum proposal scale (fraction of the range) to avoid collapse.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    d = len(lo)
    usable = check_defensive_budget(final_samples, alpha,
                                    label="estimate_failure_probability")

    # Init the proposal from f's moments (q0 matches f in shape).
    loc = np.array([float(fd.mean()) for fd in f_dists])
    scale = np.array([float(fd.std()) for fd in f_dists])
    scale_min = scale_floor * (hi - lo)

    n_eval = 0
    gamma_hist: list = []

    problem = _CEProblem(margin_fn=margin_fn, f_dists=f_dists, lo=lo, hi=hi,
                         scale_min=scale_min, threshold=threshold, rng=rng, d=d)

    loc, scale, gamma_hist, n_eval = _ce_descent(
        problem, loc, scale, samples_per_iter=samples_per_iter, rho=rho,
        max_iter=max_iter, n_eval=n_eval, verbose=verbose)

    q = build_proposal(lo, hi, loc, scale)
    p_hat, ci, w, fail, n_eval = _defensive_is_estimate(
        problem, q, alpha=alpha, final_samples=final_samples, n_eval=n_eval)

    return RareEventResult(
        p_fail=p_hat,
        ci=ci,
        n_evaluations=n_eval,
        iterations=len(gamma_hist),
        q_loc=loc,
        q_scale=scale,
        gamma_history=gamma_hist,
        n_fail_effective=int(fail.sum()),
        n_defensive=defensive_sample_count(final_samples, alpha),
        ess=effective_sample_size(w),
        p_fail_usable=bool(usable),
    )
