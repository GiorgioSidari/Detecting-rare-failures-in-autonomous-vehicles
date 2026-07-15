from __future__ import annotations

"""
Lane-Keeping scenario configuration.

Uses the existing Udacity/Unity simulator (opensbt-core) via Docker.
The DNN autopilot (SupervisedAgent / AutopilotModel) drives the vehicle;
we test how robust it is across a wider parameter space than the original
5-angle configuration.

Extended parameter space (Step C)
----------------------------------
Original : 5 road angles in [0°, 85°]
Extended : 5 road angles + min_speed + max_speed + segment_length + map_size

QoI (Step B) — composite safety metric:
  M1 (0.6): XTE margin          MAX_XTE - max(|xte|)              positive = in lane
  M2 (0.2): Steering peak dev   -(max|s| - mean|s|)/STEER_RANGE   penalises sudden jerks
  M3 (0.2): Early violation     -(T - first_near_boundary)/T      penalises early approach

Failure  : composite QoI < 0.0

Multi-model (Step D): instantiate with simulator_url and name_suffix to run two
  autopilots in parallel on separate Docker ports and compare failure profiles.

NOTE: requires Docker (opensbt-core) to be running:
      docker compose up --build  (in opensbt-core/)
"""

import numpy as np

from scenarios.base_scenario import BaseScenario
from simulators.common.http_worker_pool import (
    build_pool_from_env,
    healthy_workers,
    run_job_pool,
    DEFAULT_JOB_TIMEOUT,
)

# ── Simulator connection ──────────────────────────────────────────────────────
SIMULATOR_URL   = "http://localhost:8000"   # default SimulatorServer FastAPI base URL
DEFAULT_TIMEOUT = DEFAULT_JOB_TIMEOUT       # seconds to wait for a SINGLE simulation job (per-job, not global)

# Default number of parallel simulator containers (worker pool).
# Override at runtime with NUM_WORKERS, or pass explicit SIMULATOR_URLS.
DEFAULT_NUM_WORKERS = 4


def build_simulator_pool() -> list[str]:
    """
    Resolve the pool of SimulatorServer endpoints used for parallel execution
    (see simulators/common/http_worker_pool.build_pool_from_env for priority
    order: SIMULATOR_URLS > NUM_WORKERS/SIMULATOR_BASE_PORT > default).

    Il numero di worker DEVE combaciare con i container avviati da
    docker-compose.parallel.yml (vedi opensbt-core/gen_parallel_compose.py).
    Worker non raggiungibili vengono comunque scartati a runtime dall'health-check.
    """
    return build_pool_from_env(default_num_workers=DEFAULT_NUM_WORKERS)

# ── Physical constants ────────────────────────────────────────────────────────
MAX_XTE         = 2.5   # metres — matches opensbt-core/Simulator/lanekeeping/config.py
STEER_RANGE_NORM = 0.4  # normalisation for steering peak deviation: max|s| - mean|s|
                        # rettilineo: range≈0.04 → M2≈-0.10 (quasi 0)
                        # zigzag:     range≈0.65 → M2=-1.00 (saturato)
                        # soglia 0.4 lascia M2 non saturato per la maggior parte dei run safe
EARLY_FRAC      = 0.7   # fraction of MAX_XTE that triggers the "approaching boundary" flag
MIN_VALID_STEPS = 3     # run piu' corti di cosi' sono degeneri/abortiti
                        # (la sim si e' interrotta subito): non sono guida sicura


class LaneKeepingScenario(BaseScenario):
    """
    Lane-keeping scenario driven by the Udacity DNN autopilot.

    Parameters
    ----------
    simulator_url : base URL of the SimulatorServer FastAPI instance.
                    Override per-instance to target a different Docker port
                    (Step D multi-model comparison).
    name_suffix   : appended to the scenario name for multi-model registry keys
                    (e.g. '_chauffeur', '_ch2').
    """

    name = "lane_keeping"
    description = (
        "The Udacity DNN autopilot drives on a procedurally generated road. "
        "Tests whether the neural network can stay within lane boundaries "
        "across varying road geometries and speed settings."
    )

    def __init__(
        self,
        simulator_url: str | None = None,
        name_suffix: str = "",
        simulator_urls: list[str] | None = None,
    ):
        # Pool resolution (retro-compatibile):
        #   • simulator_urls passato  → pool esplicito (parallelo, N container)
        #   • simulator_url passato   → pool a singolo container (comportamento Step D)
        #   • nessuno dei due         → pool da env (NUM_WORKERS/SIMULATOR_URLS,
        #                               default DEFAULT_NUM_WORKERS worker)
        if simulator_urls is not None:
            self.simulator_urls = [u.rstrip("/") for u in simulator_urls]
        elif simulator_url is not None:
            self.simulator_urls = [simulator_url.rstrip("/")]
        else:
            self.simulator_urls = build_simulator_pool()
        # Preserved for legacy references / logging (primo endpoint del pool).
        self.simulator_url = self.simulator_urls[0]
        self.name = f"lane_keeping{name_suffix}"
        # run_simulation() stores actual run lengths here so compute_qoi()
        # can mask out zero-padded timesteps when computing M2 and M3.
        self._run_lengths: list[int] | None = None
        # Popolati da compute_qoi: sopravvivenza (n. step) e n. run degeneri.
        self._last_survival: np.ndarray | None = None
        self._valid_mask: np.ndarray | None = None
        self._n_degenerate: int = 0
        self._n_invalid: int = 0

    # ── Worker pool helpers ───────────────────────────────────────────────────

    def _build_payload(self, row: np.ndarray, ncols: int) -> dict:
        """Costruisce il payload JSON POST /simulate da una riga di parametri."""
        map_size = float(row[8]) if ncols > 8 else 250.0
        return {
            "angles":    [int(round(a)) for a in row[:5]],
            "minSpeed":  int(round(row[5])),
            "maxSpeed":  int(round(row[6])),
            "segLength": int(round(row[7])),
            "map_size":  int(round(map_size)),
            "maxTime":   30,
            "maxXTE":    MAX_XTE,
        }

    def _healthy_workers(self, verbose: bool = False) -> list[str]:
        """Ritorna solo i worker che rispondono a GET /health."""
        return healthy_workers(self.simulator_urls, verbose=verbose)

    # ── Parameter space ───────────────────────────────────────────────────────

    def param_bounds(self) -> dict:
        """
        Nine controllable parameters:
          angles 1-5  [0°, 85°]   — road geometry (absolute bearing per segment)
          min_speed   [5, 15] m/s — lower speed limit for the agent
          max_speed   [10, 30] m/s — upper speed limit for the agent
          seg_length  [10, 40] m  — length of each road segment
          map_size    [150, 350] m — side length of the simulation map (Step C)
        """
        return {
            "names": [
                "angle_1 (°)", "angle_2 (°)", "angle_3 (°)",
                "angle_4 (°)", "angle_5 (°)",
                "min_speed (m/s)", "max_speed (m/s)",
                "segment_length (m)",
                "map_size (m)",
            ],
            "lower": np.array([0,   0,   0,   0,   0,   5.0,  10.0, 10.0, 150.0]),
            "upper": np.array([85,  85,  85,  85,  85,  15.0, 30.0, 40.0, 350.0]),
        }

    # ── Simulation (Step A) ───────────────────────────────────────────────────

    def run_simulation(self, params: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Submit N simulation jobs to a POOL of SimulatorServer containers and
        collect results.

        Ogni container esegue le sue simulazioni in modo sequenziale (una sola
        istanza Unity per container). Il parallelismo si ottiene distribuendo i
        job su più container: con W worker sani il throughput è ~W× rispetto al
        singolo container. I job vengono presi da una coda condivisa (bilanciamento
        dinamico: i container più veloci ne processano di più).

        Ogni job ha un timeout INDIVIDUALE (DEFAULT_TIMEOUT), non più un deadline
        globale N×timeout: un container che si blocca su Unity fa fallire solo il
        suo job in ~90s, invece di far sembrare congelato l'intero batch per un'ora.

        Returns
        -------
        trajectories : (N, T_max, 4)  float32, zero-padded to the longest run.
            Channel 0 — x position (m)
            Channel 1 — y position (m)
            Channel 2 — cross-track error XTE (m)
            Channel 3 — steering angle (normalised)

        Side-effect: stores actual run lengths in self._run_lengths so that
        compute_qoi() can ignore zero-padded timesteps.
        """
        N     = params.shape[0]
        ncols = params.shape[1]
        payloads = [self._build_payload(row, ncols) for row in params]

        # ── Pool: tieni solo i worker che rispondono a /health ────────────────
        workers = self._healthy_workers(verbose=verbose)
        if not workers:
            raise RuntimeError(
                "Nessun simulatore raggiungibile. Avvia opensbt-core, es.:\n"
                "  cd opensbt-core && "
                "docker compose -f docker-compose.parallel.yml up --build\n"
                f"URL tentati: {self.simulator_urls}\n"
                "(imposta NUM_WORKERS o SIMULATOR_URLS per cambiare il pool)."
            )
        if verbose:
            ports = ", ".join(u.split(":")[-1] for u in workers)
            print(f"[pool] {len(workers)} worker attivi (porte: {ports})", flush=True)

        # ── Coda di lavoro condivisa: un thread per worker (simulators/common) ─
        # Ogni thread possiede UN container e cicla: prende un indice, POSTa il
        # job su quel container, attende il risultato (poll), passa al successivo.
        # Così ogni container ha al più 1 job in coda: niente serializzazione
        # nascosta lato server. Timeout PER-JOB (DEFAULT_TIMEOUT), non un
        # deadline globale N×timeout.
        def _on_progress(completed: int, total: int, idx: int, url: str, result: dict) -> None:
            if not verbose:
                return
            out     = result["output"]
            L       = len(out["xtes"])
            max_xte = max((abs(x) for x in out["xtes"]), default=0.0)
            row     = params[idx]
            print(
                f"  [{completed:2d}/{total}] (sample #{idx+1:2d} @ "
                f"porta {url.split(':')[-1]})"
                f"  angoli=[{','.join(f'{int(round(a)):2d}' for a in row[:5])}]"
                f"  →  {L} step,  XTE max={max_xte:.3f}m",
                flush=True,
            )

        results = run_job_pool(
            payloads, workers,
            job_timeout=DEFAULT_TIMEOUT,
            on_progress=_on_progress,
        )

        # ── Build all_stats from parallel results ────────────────────────────
        # The SimulatorServer returns parallel arrays, not a list of per-step dicts.
        # Actual output structure (verified against live Docker response):
        #   output.positions  : list of [x, y, z]  (one entry per iteration)
        #   output.xtes       : list of float       (cross-track error per step)
        #   output.steerings  : list of float       (steering angle per step)
        all_stats = []
        for result in results:
            out = result["output"]
            all_stats.append({
                "positions": out["positions"],   # [[x,y,z], ...]
                "xtes":      out["xtes"],        # [float, ...]
                "steerings": out["steerings"],   # [float, ...]
            })

        # ── Build zero-padded trajectory tensor (N, T_max, 4) ────────────────
        #    Zero-padding is safe for the POD embedder (SVD handles zeros).
        #    self._run_lengths lets compute_qoi() mask out padded timesteps
        #    so M2/M3 statistics are computed only on real simulation steps.
        run_lengths = [len(s["xtes"]) for s in all_stats]
        self._run_lengths = run_lengths
        T_max = max(run_lengths)

        traj = np.zeros((N, T_max, 4), dtype=np.float32)
        for i, stats in enumerate(all_stats):
            n = run_lengths[i]
            traj[i, :n, 0] = [p[0] for p in stats["positions"]]   # x
            traj[i, :n, 1] = [p[1] for p in stats["positions"]]   # y
            traj[i, :n, 2] = stats["xtes"]
            traj[i, :n, 3] = stats["steerings"]

        return traj

    # ── Quality of Interest (Step B) ─────────────────────────────────────────

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Composite safety metric combining three signals:

        M1 — XTE margin (weight 0.6)
            MAX_XTE - max(|xte|) over valid steps.
            Positive → car stayed well within the lane.
            Negative → car left the road (failure).

        M2 — Steering peak deviation (weight 0.2)
            -(max(|steering|) - mean(|steering|)) / STEER_RANGE_NORM, clipped to [-1, 0].
            High peak deviation = sudden jerks / unstable control = early failure signal.
            More informative than std: std saturates at -1 for almost all runs with curves,
            while peak deviation retains gradation across safe scenarios.

        M3 — Early boundary approach (weight 0.2)
            -(valid_steps - first_step_near_boundary) / valid_steps.
            0 if the car never approaches within EARLY_FRAC * MAX_XTE of the edge.
            -1 if it approaches on the very first step.

        Zero-padded timesteps are excluded from M2 and M3 via self._run_lengths
        (set by run_simulation). If called standalone (e.g. in tests), the full
        trajectory length is used, which may slightly underestimate M2/M3 for
        short runs with long zero padding.
        """
        N, T, _ = trajectories.shape
        xte      = trajectories[:, :, 2].astype(np.float64)   # (N, T)
        steering = trajectories[:, :, 3].astype(np.float64)   # (N, T)

        # Build validity mask — True for real simulation steps, False for padding
        run_lengths = self._run_lengths if self._run_lengths is not None else [T] * N
        # self._run_lengths puo' includere righe iniziali extra (es. il campione
        # 'nominale' simulato insieme al batch ma poi scartato dall'orchestrator):
        # allinea alla coda cosi' da corrispondere alle N traiettorie ricevute.
        if len(run_lengths) != N:
            run_lengths = list(run_lengths)[-N:]
        valid = np.zeros((N, T), dtype=bool)
        for i, L in enumerate(run_lengths):
            valid[i, :min(L, T)] = True
        valid_count = valid.sum(axis=1).astype(np.float64)    # (N,) — at least 1
        valid_count = np.where(valid_count > 0, valid_count, 1.0)

        # Replace padded timesteps with NaN so nan-aware functions ignore them
        xte_m      = np.where(valid, xte,      np.nan)
        steering_m = np.where(valid, steering, np.nan)

        # M1 — safety margin
        m1 = MAX_XTE - np.nanmax(np.abs(xte_m), axis=1)      # (N,)

        # M2 — steering peak deviation (max|s| - mean|s|)
        steer_abs  = np.abs(steering_m)
        steer_peak = np.nanmax(steer_abs, axis=1) - np.nanmean(steer_abs, axis=1)  # (N,)
        m2 = -np.clip(steer_peak / STEER_RANGE_NORM, 0.0, 1.0)                     # (N,) in [-1, 0]

        # M3 — time of first near-boundary approach
        near_boundary = np.abs(xte_m) > EARLY_FRAC * MAX_XTE  # (N, T), NaN→False
        near_boundary = np.where(np.isnan(xte_m), False, near_boundary)
        any_near      = near_boundary.any(axis=1)              # (N,)
        first_idx     = np.where(
            any_near,
            near_boundary.argmax(axis=1).astype(np.float64),
            valid_count,                                       # never triggered → use T
        )
        m3 = -(valid_count - first_idx) / valid_count         # (N,) in [-1, 0]

        qoi = 0.6 * m1 + 0.2 * m2 + 0.2 * m3                  # composite safety metric

        # ── Run degeneri/abortiti ─────────────────────────────────────────────
        # Una simulazione con pochissimi step (es. 1) non e' guida "sicura": si e'
        # interrotta subito e la QoI la premierebbe (XTE bassa -> M1 alto). La
        # trattiamo come fallimento, ma con margine appena < 0, cosi' conta come
        # failure senza diventare uno spurio 'rare failure' (che dev'essere un crash
        # vero). Salviamo la sopravvivenza (n. step) per dare risoluzione temporale
        # ai rare failure a valle (vedi find_rare_failures tiebreak).
        survival = np.asarray(run_lengths, dtype=float)      # (N,)
        degenerate = survival < MIN_VALID_STEPS              # (N,) bool, sim abortite

        # Parametri incoerenti: min_speed > max_speed (colonne 5 e 6) non e' uno
        # scenario reale ma un input mal campionato.
        if params.shape[1] > 6:
            bad_params = np.asarray(params)[:, 5] > np.asarray(params)[:, 6]
        else:
            bad_params = np.zeros(len(survival), dtype=bool)

        # I run NON validi sono misurazioni da escludere dall'analisi: margine = NaN
        # cosi' non contano ne' come safe ne' come failure. L'orchestrator calcola
        # i tassi e i rare failure solo sugli scenari validi.
        invalid = degenerate | bad_params
        qoi = np.where(invalid, np.nan, qoi)

        self._last_survival = survival
        self._valid_mask = ~invalid
        self._n_degenerate = int(degenerate.sum())
        self._n_invalid = int(invalid.sum())

        return qoi

    def failure_threshold(self) -> float:
        return 0.0

    def pod_channels(self) -> list[int] | None:
        # x (temporal structure) + xte + steering; y is redundant with xte.
        return [0, 2, 3]
