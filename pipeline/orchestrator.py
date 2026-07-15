"""
Pipeline orchestrator — scenario-agnostic.

Given a scenario name and sampling config, runs the full pipeline:
    LHS → simulation → QoI → POD embedding → rare failure detection

Returns a PipelineResult that the API serialises and the frontend renders.
"""
from __future__ import annotations

import inspect
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
    n_degenerate:    int = 0             # run degeneri/abortiti (sottoinsieme dei non validi)
    n_invalid:       int = 0             # scenari non validi, esclusi da tassi e rare failure
    valid_mask:      np.ndarray | None = None   # (N,) True = scenario valido
    nominal_trajectory: np.ndarray | None = None  # (1, T, D) — mid-point params reference run

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


def run(
    scenario_name: str,
    n_samples: int = 500,
    seed: int = 42,
    rare_fraction: float = 0.05,
    pod_variance_threshold: float = 0.99,
    param_lower: list[float] | None = None,
    param_upper: list[float] | None = None,
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
    #    Usa i bounds EFFETTIVI (inclusi eventuali override --preset/frontend),
    #    non scenario.nominal_params() che ricalcola sempre dai default.
    nominal = ((bounds["lower"] + bounds["upper"]) / 2.0).reshape(1, -1)  # (1, d)
    params_with_nominal = np.vstack([nominal, params])               # (N+1, d)

    # 3. Simulate
    #    verbose viene inoltrato solo se lo scenario lo supporta (lane_keeping),
    #    cosi' gli altri scenari restano compatibili.
    if verbose and 'verbose' in inspect.signature(scenario.run_simulation).parameters:
        trajectories = scenario.run_simulation(params_with_nominal, verbose=True)  # (N+1, T, D)
    else:
        trajectories = scenario.run_simulation(params_with_nominal)      # (N+1, T, D)
    nominal_traj = trajectories[0:1]
    trajectories = trajectories[1:]                                  # (N, T, D)
    params = params_with_nominal[1:]                                 # (N, d)

    # 4. QoI
    safety_margins = scenario.compute_qoi(trajectories, params)      # (N,)
    failures = scenario.is_failure(safety_margins)                   # (N,)

    # Info extra prodotte da compute_qoi (se lo scenario le espone):
    #  - survival: n. step per campione, usato come spareggio dei rare failure
    #  - n_degenerate: run abortiti (troppo corti), gia' contati come failure
    survival     = getattr(scenario, "_last_survival", None)
    n_degenerate = int(getattr(scenario, "_n_degenerate", 0))

    # Maschera di validita': scenari non validi (sim abortite o parametri incoerenti)
    # sono ESCLUSI da tassi e rare failure, non contati come fallimenti.
    valid = getattr(scenario, "_valid_mask", None)
    if valid is None:
        valid = ~np.isnan(safety_margins)
    valid = np.asarray(valid, dtype=bool)
    n_valid   = int(valid.sum())
    n_invalid = int(getattr(scenario, "_n_invalid", int((~valid).sum())))

    # 5. POD embedding — canali dichiarati dallo scenario (default: tutti).
    #    lane_keeping seleziona [0,2,3] (x + xte + steering, esclude y ridondante);
    #    scenari a 2 canali come cut_in/emergency_braking usano tutto lo stato.
    channels = scenario.pod_channels()
    traj_pod = trajectories if channels is None else trajectories[:, :, channels]
    pod = EmbedderPOD(variance_threshold=pod_variance_threshold)
    pod_codes = pod.fit_transform(traj_pod)                         # (N, k)

    # 6. Rare failure detection
    rare_idx = find_rare_failures(
        safety_margins=safety_margins,
        failures=failures * valid.astype(float),   # solo fallimenti VALIDI
        fraction=rare_fraction,
        tiebreak=survival,
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
        failure_rate=float(failures[valid].mean()) if n_valid > 0 else 0.0,
        rare_failure_rate=float(len(rare_idx) / n_valid) if n_valid > 0 else 0.0,
        param_names=bounds["names"],
        n_degenerate=n_degenerate,
        n_invalid=n_invalid,
        valid_mask=valid,
        nominal_trajectory=nominal_traj,
    )
