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
    Stima efficiente di P(fallimento) con la Cross-Entropy (livello massimo, M4).

    A differenza di `run()` (che campiona una volta e conta), qui si campiona in modo
    ADATTIVO verso la regione di fallimento e si stima la probabilita' con importance
    sampling: utile quando P e' bassa e il Monte Carlo ingenuo sarebbe inefficiente.

    Budget: ogni iterazione esegue `samples_per_iter` simulazioni + `final_samples`
    alla fine. I default qui sono CONTENUTI (adatti al simulatore vero, ~10 s/run):
    ~ samples_per_iter*iterazioni + final_samples run. Alzali per piu' precisione se
    puoi permetterti piu' run (nella validazione sintetica, dove i run sono gratis, si
    usano valori molto piu' alti — vedi scripts/validate_rare_event.py).

    Ritorna un `RareEventResult` (p_fail, ci bootstrap, budget, proposta finale, ecc.).
    Richiede che lo scenario esponga `param_distributions` (la distribuzione operativa).
    """
    scenario: BaseScenario = SCENARIOS[scenario_name]
    bounds = scenario.param_bounds()
    if param_lower is not None:
        bounds["lower"] = np.array(param_lower, dtype=float)
    if param_upper is not None:
        bounds["upper"] = np.array(param_upper, dtype=float)

    if not hasattr(scenario, "param_distributions"):
        raise ValueError(
            f"Lo scenario '{scenario_name}' non espone param_distributions(): "
            "il metodo Cross-Entropy richiede una distribuzione operativa."
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
    Intervallo di confidenza di Wilson (95% con z=1.96) per una proporzione k/n.

    Preferito al Wald perche' resta dentro [0,1] e non collassa quando p e' vicino a
    0 o 1 (il caso tipico di un evento raro). Usato per dare un'incertezza alla stima
    di P(fallimento) invece di riportare un numero nudo.
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
    # Rarita' (probabilita'), distinta dalla severita' (rare_failure_* = bottom-k%):
    failure_probability:    float = 0.0            # P(fallimento) sotto il sampling usato
    failure_probability_ci: tuple | None = None    # (lo, hi) intervallo di Wilson 95%
    sampling:               str = "uniform"        # "uniform" | "realistic"
    param_names:     list[str] = field(default_factory=list)
    n_degenerate:    int = 0             # run degeneri/abortiti (sottoinsieme dei non validi)
    n_invalid:       int = 0             # scenari non validi, esclusi da tassi e rare failure
    valid_mask:      np.ndarray | None = None   # (N,) True = scenario valido
    nominal_trajectory: np.ndarray | None = None  # (1, T, D) — mid-point params reference run
    n_low_fidelity:  int = 0             # run sotto-campionati (loop troppo lento), esclusi
    control_hz:      np.ndarray | None = None   # (N,) frequenza loop di controllo per run
    meters_per_step: np.ndarray | None = None   # (N,) metri percorsi tra due decisioni
    infer_ms_per_step: np.ndarray | None = None # (N,) ms/step in inferenza (agent.predict)
    wait_ms_per_step:  np.ndarray | None = None # (N,) ms/step in attesa frame Unity (env.step)

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
    sampling              : "uniform" (LHS uniforme sui bound, default retro-compatibile)
                            oppure "realistic" (LHS trasformato con la ppf delle
                            distribuzioni operative dello scenario: la frazione di
                            fallimenti diventa una stima di P(fallimento) sotto l'ODD).

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
    #    unit_samples in [0,1]^d: stratificati (buona copertura). Come li si mappa
    #    sullo spazio dei parametri dipende dal modo di campionamento:
    #      - "uniform"   : scale lineare sui bound (equiprobabilita' -> misura SEVERITA')
    #      - "realistic" : trasformazione con la ppf delle distribuzioni operative
    #                      (pesa gli scenari per quanto sono probabili nella guida reale
    #                       -> la frazione di fallimenti stima P(fallimento) sull'ODD)
    d = len(bounds["lower"])
    sampler = LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    use_realistic = (sampling == "realistic"
                     and hasattr(scenario, "param_distributions"))
    if use_realistic:
        dists = scenario.param_distributions(bounds["lower"], bounds["upper"])
        params = np.empty_like(unit_samples)
        for j, dist in enumerate(dists):
            params[:, j] = dist.ppf(unit_samples[:, j])                # inversa CDF
    else:
        if sampling == "realistic":
            # richiesto ma lo scenario non espone distribuzioni: fallback trasparente
            sampling = "uniform"
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

    # Fedelta' del loop di controllo (se lo scenario la espone): run sotto-campionati
    # (troppi worker in contesa CPU) sono gia' inclusi in _valid_mask come non validi.
    n_low_fidelity  = int(getattr(scenario, "_n_low_fidelity", 0))
    control_hz      = getattr(scenario, "_last_control_hz", None)
    meters_per_step = getattr(scenario, "_last_meters_per_step", None)
    infer_ms        = getattr(scenario, "_last_infer_ms", None)
    wait_ms         = getattr(scenario, "_last_wait_ms", None)

    # 5. POD embedding — canali dichiarati dallo scenario (default: tutti).
    #    lane_keeping seleziona [0,2,3] (x + xte + steering, esclude y ridondante);
    #    scenari a 2 canali come cut_in/emergency_braking usano tutto lo stato.
    channels = scenario.pod_channels()
    traj_pod = trajectories if channels is None else trajectories[:, :, channels]
    pod = EmbedderPOD(variance_threshold=pod_variance_threshold)
    pod_codes = pod.fit_transform(traj_pod)                         # (N, k)

    # 6. Rare failure detection (SEVERITA': i k% peggiori per margine)
    rare_idx = find_rare_failures(
        safety_margins=safety_margins,
        failures=failures * valid.astype(float),   # solo fallimenti VALIDI
        fraction=rare_fraction,
        tiebreak=survival,
    )

    # 6-bis. RARITA': stima della probabilita' di fallimento.
    #   Sotto sampling "realistic" i campioni sono estratti (via ppf) dalla
    #   distribuzione operativa, quindi la frazione di fallimenti sui run validi e'
    #   una stima di P(fallimento) sotto l'ODD. Sotto "uniform" e' P(fallimento) su
    #   ODD uniforme (baseline). Aggiungiamo un intervallo di confidenza di Wilson
    #   (robusto anche per p vicino a 0/1 e n piccolo) per non riportare un numero
    #   nudo. Rarita' (probabilita') e severita' (bottom-k%) restano DISTINTE.
    k_fail = int(failures[valid].sum()) if n_valid > 0 else 0
    failure_probability = float(k_fail / n_valid) if n_valid > 0 else 0.0
    failure_probability_ci = _wilson_ci(k_fail, n_valid) if n_valid > 0 else None

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
        nominal_trajectory=nominal_traj,
        n_low_fidelity=n_low_fidelity,
        control_hz=control_hz,
        meters_per_step=meters_per_step,
        infer_ms_per_step=infer_ms,
        wait_ms_per_step=wait_ms,
    )
