from __future__ import annotations

import inspect

"""
Pipeline orchestrator — scenario-agnostic.

Given a scenario name and sampling config, runs the full pipeline:
    LHS -> simulation -> QoI -> POD embedding -> rare failure detection

Returns a PipelineResult that the API serialises and the frontend renders.
"""
import numpy as np
from dataclasses import dataclass, field
from scipy.stats.qmc import LatinHypercube, scale

from scenarios import SCENARIOS
from scenarios.base_scenario import BaseScenario
from embedder.pod import EmbedderPOD
from pipeline.severity import find_severe_failures
from pipeline.rare_event import estimate_failure_probability, scenario_margin_fn


def run_rare_event(
    scenario_name: str,
    seed: int = 0,
    param_lower: list[float] | None = None,
    param_upper: list[float] | None = None,
    samples_per_iter: int = 80,
    final_samples: int = 400,
    rho: float = 0.2,
    max_iter: int = 15,
    alpha: float = 0.2,
    verbose: bool = False,
):
    """
    Efficient P(failure) estimate via the Cross-Entropy method.

    Unlike run() (which samples once and counts), this samples ADAPTIVELY toward the failure
    region and estimates the probability with importance sampling: useful when P is low and
    plain Monte Carlo would be inefficient.

    Budget: each iteration runs `samples_per_iter` simulations plus `final_samples` at the end.
    Defaults here are small (suited to the real ~10 s/run simulator). Raise them for more
    precision if you can afford more runs (the synthetic validation, where runs are free, uses
    much larger values -- see scripts/validate_rare_event.py).

    Returns a RareEventResult (p_fail, bootstrap ci, budget, final proposal, ...).
    Requires the scenario to expose `param_distributions` (the operational distribution).
    """
    scenario: BaseScenario = SCENARIOS[scenario_name]
    bounds = scenario.param_bounds()
    if param_lower is not None:
        bounds["lower"] = np.array(param_lower, dtype=float)
    if param_upper is not None:
        bounds["upper"] = np.array(param_upper, dtype=float)

    if not hasattr(scenario, "param_distributions"):
        raise ValueError(
            f"Scenario '{scenario_name}' does not expose param_distributions(): "
            "the Cross-Entropy method requires an operational distribution."
        )

    f_dists = scenario.param_distributions(bounds["lower"], bounds["upper"])
    margin_fn = scenario_margin_fn(scenario)
    return estimate_failure_probability(
        margin_fn, f_dists, bounds["lower"], bounds["upper"],
        threshold=scenario.failure_threshold(),
        samples_per_iter=samples_per_iter, final_samples=final_samples,
        rho=rho, max_iter=max_iter, alpha=alpha, seed=seed, verbose=verbose,
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson confidence interval (95% with z=1.96) for a proportion k/n.

    Preferred over Wald because it stays within [0, 1] and does not collapse when p is near 0 or
    1 (the typical rare-event case). Used to attach uncertainty to the P(failure) estimate.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class PipelineResult:
    scenario_name:   str
    n_samples:       int
    params:          np.ndarray          # (N, d)
    trajectories:    np.ndarray          # (N, T, D)
    safety_margins:  np.ndarray          # (N,)
    failures:        np.ndarray          # (N,) binary
    rare_failure_idx: np.ndarray         # indices into N of rare failures
    pod_codes:       np.ndarray          # (N, k) compressed trajectories
    pod_n_modes:     int
    failure_rate:    float
    rare_failure_rate: float
    # Rarity (probability), distinct from severity (rare_failure_* = bottom-k%):
    failure_probability:    float = 0.0            # P(failure) under the sampling used
    failure_probability_ci: tuple | None = None    # (lo, hi) Wilson 95% interval
    sampling:               str = "uniform"        # "uniform" | "realistic"
    param_names:     list[str] = field(default_factory=list)
    n_degenerate:    int = 0             # degenerate/aborted runs (subset of the invalid)
    n_invalid:       int = 0             # invalid scenarios, excluded from rates and rare failures
    valid_mask:      np.ndarray | None = None   # (N,) True = valid scenario
    n_low_fidelity:  int = 0             # under-sampled runs (loop too slow), excluded
    control_hz:      np.ndarray | None = None   # (N,) control-loop rate per run
    meters_per_step: np.ndarray | None = None   # (N,) metres travelled between two decisions
    infer_ms_per_step: np.ndarray | None = None # (N,) ms/step in inference (agent.predict)
    wait_ms_per_step:  np.ndarray | None = None # (N,) ms/step waiting for the Unity frame (env.step)

    # Derived convenience views
    @property
    def rare_params(self) -> np.ndarray:
        """Parameter rows for rare failure samples."""
        return self.params[self.rare_failure_idx]

    @property
    def rare_trajectories(self) -> np.ndarray:
        """Trajectory array for rare failure samples."""
        return self.trajectories[self.rare_failure_idx]

    @property
    def safe_trajectories(self) -> np.ndarray:
        return self.trajectories[self.failures == 0]

    @property
    def nominal_trajectory(self) -> np.ndarray:
        """Single trajectory at mid-point params (used as green reference)."""
        return self.trajectories[0:1]


def run(
    scenario_name: str,
    n_samples: int = 500,
    seed: int = 42,
    rare_fraction: float = 0.05,
    pod_variance_threshold: float = 0.99,
    param_lower: list[float] | None = None,
    param_upper: list[float] | None = None,
    sampling: str = "uniform",
    verbose: bool = False,
) -> PipelineResult:
    """
    Run the full pipeline for a given scenario.

    Parameters
    ----------
    scenario_name         : key in SCENARIOS registry
    n_samples             : number of LHS samples
    seed                  : random seed for reproducibility
    rare_fraction         : bottom-k fraction of failures treated as 'rare'
    pod_variance_threshold: variance threshold for POD mode selection
    param_lower           : optional custom lower bounds (overrides scenario defaults)
    param_upper           : optional custom upper bounds (overrides scenario defaults)
    sampling              : "uniform" (uniform LHS over the bounds, backward-compatible default)
                            or "realistic" (LHS transformed with the ppf of the scenario's
                            operational distributions: the failure fraction becomes an estimate
                            of P(failure) under the ODD).

    Returns
    -------
    PipelineResult with everything the API/frontend needs.
    """
    scenario: BaseScenario = SCENARIOS[scenario_name]
    bounds = scenario.param_bounds()

    # Apply custom bounds if provided (frontend sliders override scenario defaults).
    if param_lower is not None:
        bounds["lower"] = np.array(param_lower, dtype=float)
    if param_upper is not None:
        bounds["upper"] = np.array(param_upper, dtype=float)

    # 1. Latin Hypercube Sampling (stratified, good coverage). Mapping to parameter space depends on
    #    the sampling mode: "uniform" = linear to the bounds (equiprobable -> SEVERITY); "realistic"
    #    = ppf of the operational distributions (weighted by real-driving likelihood -> P(failure)).
    d = len(bounds["lower"])
    sampler = LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    use_realistic = (sampling == "realistic"
                     and hasattr(scenario, "param_distributions"))
    if use_realistic:
        dists = scenario.param_distributions(bounds["lower"], bounds["upper"])
        params = np.empty_like(unit_samples)
        for j, dist in enumerate(dists):
            params[:, j] = dist.ppf(unit_samples[:, j])                # inverse CDF
    else:
        if sampling == "realistic":
            # requested but the scenario has no distributions: transparent fallback
            sampling = "uniform"
        params = scale(unit_samples, bounds["lower"], bounds["upper"])   # (N, d)

    # 2. Add a nominal (mid-point) sample as the first row for the reference trajectory. Uses the
    #    EFFECTIVE bounds (including any --preset/frontend override), not scenario.nominal_params().
    nominal = ((bounds["lower"] + bounds["upper"]) / 2.0).reshape(1, -1)  # (1, d)
    params_with_nominal = np.vstack([nominal, params])               # (N+1, d)

    # 3. Simulate. verbose is forwarded only if the scenario supports it (lane_keeping), so the
    #    other scenarios stay compatible.
    if verbose and 'verbose' in inspect.signature(scenario.run_simulation).parameters:
        trajectories = scenario.run_simulation(params_with_nominal, verbose=True)  # (N+1, T, D)
    else:
        trajectories = scenario.run_simulation(params_with_nominal)      # (N+1, T, D)
    trajectories = trajectories[1:]                                  # (N, T, D), drop nominal
    params = params_with_nominal[1:]                                 # (N, d)

    # 4. QoI
    safety_margins = scenario.compute_qoi(trajectories, params)      # (N,)
    failures = scenario.is_failure(safety_margins)                   # (N,)

    # Extra info produced by compute_qoi (if the scenario exposes it):
    #  - survival: steps per sample, used as the rare-failure tie-breaker
    #  - n_degenerate: aborted runs (too short), already counted as failures
    survival     = getattr(scenario, "_last_survival", None)
    n_degenerate = int(getattr(scenario, "_n_degenerate", 0))

    # Validity mask: invalid scenarios (aborted sims or incoherent params) are EXCLUDED from rates
    # and rare failures, not counted as failures.
    valid = getattr(scenario, "_valid_mask", None)
    if valid is None:
        valid = ~np.isnan(safety_margins)
    valid = np.asarray(valid, dtype=bool)
    n_valid   = int(valid.sum())
    n_invalid = int(getattr(scenario, "_n_invalid", int((~valid).sum())))

    # Control-loop fidelity (if the scenario exposes it): under-sampled runs (too many workers
    # contending for CPU) are already included in _valid_mask as invalid.
    n_low_fidelity  = int(getattr(scenario, "_n_low_fidelity", 0))
    control_hz      = getattr(scenario, "_last_control_hz", None)
    meters_per_step = getattr(scenario, "_last_meters_per_step", None)
    infer_ms        = getattr(scenario, "_last_infer_ms", None)
    wait_ms         = getattr(scenario, "_last_wait_ms", None)

    # 5. POD embedding — use x, XTE and steering (channels 0, 2, 3); drop y (x is monotone along
    #    the road and carries the dominant temporal structure; y is redundant with XTE).
    traj_pod = trajectories[:, :, [0, 2, 3]]                       # (N, T, 3): x + xte + steering
    pod = EmbedderPOD(variance_threshold=pod_variance_threshold)
    pod_codes = pod.fit_transform(traj_pod)                         # (N, k)

    # 6. SEVERITY: the worst k% failures by margin.
    rare_idx = find_severe_failures(
        safety_margins=safety_margins,
        failures=failures * valid.astype(float),   # valid failures only
        fraction=rare_fraction,
        tiebreak=survival,
    )

    # 6b. RARITY: P(failure) estimate. Under "realistic" sampling the failure fraction over valid
    #     runs estimates P(failure) under the ODD (under "uniform" it's a uniform-ODD baseline). A
    #     Wilson CI (robust for p near 0/1 and small n) attaches uncertainty. Rarity and severity
    #     (bottom-k%) stay DISTINCT.
    k_fail = int(failures[valid].sum()) if n_valid > 0 else 0
    failure_probability = float(k_fail / n_valid) if n_valid > 0 else 0.0
    failure_probability_ci = _wilson_ci(k_fail, n_valid) if n_valid > 0 else None #normal distribution

    return PipelineResult(
        scenario_name=scenario_name,
        n_samples=n_samples,
        params=params,
        trajectories=trajectories,
        safety_margins=safety_margins,
        failures=failures,
        rare_failure_idx=rare_idx,
        pod_codes=pod_codes,
        pod_n_modes=pod.nModes,
        failure_rate=float(failures[valid].mean()) if n_valid > 0 else 0.0,
        rare_failure_rate=float(len(rare_idx) / n_valid) if n_valid > 0 else 0.0,
        failure_probability=failure_probability,
        failure_probability_ci=failure_probability_ci,
        sampling=sampling,
        param_names=bounds["names"],
        n_degenerate=n_degenerate,
        n_invalid=n_invalid,
        valid_mask=valid,
        n_low_fidelity=n_low_fidelity,
        control_hz=control_hz,
        meters_per_step=meters_per_step,
        infer_ms_per_step=infer_ms,
        wait_ms_per_step=wait_ms,
    )
