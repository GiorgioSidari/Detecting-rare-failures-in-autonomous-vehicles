"""
Pipeline orchestrator — scenario-agnostic.

Given a scenario name and sampling config, runs the full pipeline:
    LHS → simulation → QoI → POD embedding → rare failure detection

Returns a PipelineResult that the API serialises and the frontend renders.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.stats.qmc import LatinHypercube, scale

from scenarios import SCENARIOS
from scenarios.base_scenario import BaseScenario
from embedder.pod import EmbedderPOD
from pipeline.rare_failures import find_rare_failures


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
    param_names:     list[str] = field(default_factory=list)

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
        return self.trajectories[0:1]   # set during run() — see orchestrator


def run(
    scenario_name: str,
    n_samples: int = 500,
    seed: int = 42,
    rare_fraction: float = 0.05,
    pod_variance_threshold: float = 0.99,
    param_lower: list[float] | None = None,
    param_upper: list[float] | None = None,
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

    Returns
    -------
    PipelineResult with everything the API/frontend needs.
    """
    scenario: BaseScenario = SCENARIOS[scenario_name]
    bounds = scenario.param_bounds()

    # Apply custom bounds if provided (frontend sliders override scenario defaults)
    if param_lower is not None:
        bounds["lower"] = np.array(param_lower, dtype=float)
    if param_upper is not None:
        bounds["upper"] = np.array(param_upper, dtype=float)

    # 1. Latin Hypercube Sampling
    d = len(bounds["lower"])
    sampler = LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    params = scale(unit_samples, bounds["lower"], bounds["upper"])   # (N, d)

    # 2. Add nominal (mid-point) sample as first row for reference trajectory
    nominal = scenario.nominal_params()                               # (1, d)
    params_with_nominal = np.vstack([nominal, params])               # (N+1, d)

    # 3. Simulate
    trajectories = scenario.run_simulation(params_with_nominal)      # (N+1, T, D)
    nominal_traj = trajectories[0:1]
    trajectories = trajectories[1:]                                  # (N, T, D)
    params = params_with_nominal[1:]                                 # (N, d)

    # 4. QoI
    safety_margins = scenario.compute_qoi(trajectories, params)      # (N,)
    failures = scenario.is_failure(safety_margins)                   # (N,)

    # 5. POD embedding
    pod = EmbedderPOD(variance_threshold=pod_variance_threshold)
    pod_codes = pod.fit_transform(trajectories)                      # (N, k)

    # 6. Rare failure detection
    rare_idx = find_rare_failures(
        safety_margins=safety_margins,
        failures=failures,
        fraction=rare_fraction,
    )

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
        failure_rate=float(failures.mean()),
        rare_failure_rate=float(len(rare_idx) / n_samples),
        param_names=bounds["names"],
    )
