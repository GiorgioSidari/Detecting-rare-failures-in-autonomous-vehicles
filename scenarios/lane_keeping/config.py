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

import os
import time
import queue
import threading
import concurrent.futures
import numpy as np
import requests

from scenarios.base_scenario import BaseScenario

# ── Simulator connection ──────────────────────────────────────────────────────
SIMULATOR_URL   = "http://localhost:8000"   # default SimulatorServer FastAPI base URL
DEFAULT_TIMEOUT = 90      # seconds to wait for a SINGLE simulation job (per-job, not global)
POLL_INTERVAL   = 0.5     # seconds between GET polling requests
HEALTH_TIMEOUT  = 5       # seconds for the /health probe of each worker

# Default number of parallel simulator containers (worker pool).
# Override at runtime with NUM_WORKERS, or pass explicit SIMULATOR_URLS.
DEFAULT_NUM_WORKERS = 4


def build_simulator_pool() -> list[str]:
    """
    Resolve the pool of SimulatorServer endpoints used for parallel execution.

    Priority:
      1. SIMULATOR_URLS  — comma-separated explicit URLs
                           (e.g. "http://localhost:8000,http://localhost:8001").
      2. NUM_WORKERS     — builds localhost:BASE .. BASE+N-1
                           (BASE = SIMULATOR_BASE_PORT, default 8000).

    Il numero di worker DEVE combaciare con i container avviati da
    docker-compose.parallel.yml (vedi opensbt-core/gen_parallel_compose.py).
    Worker non raggiungibili vengono comunque scartati a runtime dall'health-check.
    """
    explicit = os.getenv("SIMULATOR_URLS")
    if explicit:
        return [u.strip().rstrip("/") for u in explicit.split(",") if u.strip()]
    n    = int(os.getenv("NUM_WORKERS", str(DEFAULT_NUM_WORKERS)))
    base = int(os.getenv("SIMULATOR_BASE_PORT", "8000"))
    return [f"http://localhost:{base + i}" for i in range(max(1, n))]

# ── Physical constants ────────────────────────────────────────────────────────
MAX_XTE         = 2.5   # metres — matches opensbt-core/Simulator/lanekeeping/config.py
STEER_RANGE_NORM = 0.4  # normalisation for steering peak deviation: max|s| - mean|s|
                        # rettilineo: range≈0.04 → M2≈-0.10 (quasi 0)
                        # zigzag:     range≈0.65 → M2=-1.00 (saturato)
                        # soglia 0.4 lascia M2 non saturato per la maggior parte dei run safe
EARLY_FRAC      = 0.7   # fraction of MAX_XTE that triggers the "approaching boundary" flag
MIN_VALID_STEPS = 3     # run piu' corti di cosi' sono degeneri/abortiti
                        # (la sim si e' interrotta subito): non sono guida sicura

# ── Fedelta' del loop di controllo (parallelismo senza perdita di accuratezza) ──
# Il loop chiude a una certa frequenza: DNN.predict -> env.step -> nuovo frame Unity.
# Con troppi worker paralleli i container Unity competono per la CPU: il real-time
# factor cala, il loop gira piu' lento e l'auto percorre PIU' METRI tra due decisioni
# di sterzo. Il DNN, addestrato a cadenza ~real-time, si ritrova a sterzare su
# osservazioni vecchie e distanti -> oscillazione/instabilita' e fallimenti che sono
# ARTEFATTI del carico macchina, non difetti del modello.
#
# Misuriamo due indicatori per ogni run (dai dati che il server GIA' restituisce:
# elapsedTime, iterations, speeds):
#   control_hz      = iterations / elapsedTime          (frequenza del loop, Hz)
#   meters_per_step = mean_speed_mps * elapsedTime/iter (risoluzione spaziale, m/step)
# Un run con control_hz troppo basso o meters_per_step troppo alto e' sotto-campionato:
# lo marchiamo NON valido (escluso da tassi e rare failure, come i degeneri), cosi' il
# parallelismo resta ma i risultati NON dipendono da quanti worker giravano.
#
# Default = 0 (gate DISATTIVATO, nessun cambio di comportamento): gli indicatori sono
# comunque calcolati e stampati dal runner, cosi' puoi prima MISURARE la cadenza al
# variare di --workers e poi scegliere le soglie. Attiva/tara via env:
#   LK_MIN_CONTROL_HZ=5   LK_MAX_METERS_PER_STEP=2   python scripts/run_lanekeeping.py ...
MIN_CONTROL_HZ      = float(os.getenv("LK_MIN_CONTROL_HZ", "0"))       # Hz; 0 = off
MAX_METERS_PER_STEP = float(os.getenv("LK_MAX_METERS_PER_STEP", "0"))  # m/step; 0 = off


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
        # Fedelta' del loop di controllo, popolata da run_simulation (una voce per
        # run, allineata a self._run_lengths). None se la sim non le fornisce.
        self._control_hz: np.ndarray | None = None       # Hz per run
        self._meters_per_step: np.ndarray | None = None  # m percorsi per decisione
        self._infer_ms: np.ndarray | None = None         # ms/step in inferenza
        self._wait_ms: np.ndarray | None = None          # ms/step in attesa Unity
        # Viste finali (allineate alle N traiettorie) e conteggio, da compute_qoi.
        self._last_control_hz: np.ndarray | None = None
        self._last_meters_per_step: np.ndarray | None = None
        self._last_infer_ms: np.ndarray | None = None
        self._last_wait_ms: np.ndarray | None = None
        self._n_low_fidelity: int = 0

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
        healthy: list[str] = []
        for url in self.simulator_urls:
            try:
                r = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
                if r.ok:
                    healthy.append(url)
                elif verbose:
                    print(f"[pool] {url} risponde ma non healthy "
                          f"({r.status_code}) — ignorato", flush=True)
            except requests.RequestException:
                if verbose:
                    print(f"[pool] {url} non raggiungibile — ignorato", flush=True)
        return healthy

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

    def param_distributions(self, lower=None, upper=None) -> list:
        """
        Distribuzione operativa REALISTICA per ciascun parametro, troncata ai bound
        effettivi (lower/upper, che possono venire da un preset o dallo sweep).

        Ritorna una lista di distribuzioni scipy.stats CONGELATE, una per dimensione,
        con supporto in [lower_j, upper_j]. Serve al campionamento distribution-aware
        (modo 'realistic' in pipeline.orchestrator): trasformando i campioni LHS con
        dist.ppf() si ottengono scenari pesati secondo quanto sono probabili nella
        guida reale, cosi' la frazione di fallimenti diventa una stima di P(fallimento)
        sotto l'ODD -- non piu' una frazione su campionamento uniforme.

        NOTA METODOLOGICA: le forme qui sono un punto di partenza PLAUSIBILE, da
        calibrare su dati reali/letteratura prima di trarne numeri definitivi.
          - angoli (0-4): piu' massa sulle curve dolci (half-normal troncata, moda al
            bordo basso): le curve strette sono rare nella guida reale.
          - velocita' (5,6): normale troncata centrata su un valore di crociera (~40%
            del range): si guida piu' spesso a velocita' intermedie che agli estremi.
          - seg_length (7), map_size (8): uniforme (nessun prior forte).
        """
        from scipy import stats
        b = self.param_bounds()
        lo = np.asarray(lower if lower is not None else b["lower"], dtype=float)
        hi = np.asarray(upper if upper is not None else b["upper"], dtype=float)

        def _uniform(a, c):
            return stats.uniform(loc=a, scale=max(c - a, 1e-9))

        def _truncnorm(a, c, mu, sigma):
            if sigma <= 0 or c <= a:
                return _uniform(a, c)
            return stats.truncnorm((a - mu) / sigma, (c - mu) / sigma,
                                   loc=mu, scale=sigma)

        dists = []
        for j in range(len(lo)):
            a, c = float(lo[j]), float(hi[j])
            rng = c - a
            if j < 5:                       # angoli: moda sulle curve dolci
                dists.append(_truncnorm(a, c, mu=a, sigma=max(rng * 0.5, 1e-6)))
            elif j in (5, 6):               # velocita': crociera ~40% del range
                dists.append(_truncnorm(a, c, mu=a + 0.4 * rng, sigma=max(rng * 0.3, 1e-6)))
            else:                           # seg_length, map_size: uniforme
                dists.append(_uniform(a, c))
        return dists

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

        # ── Coda di lavoro condivisa: un thread per worker ────────────────────
        # Ogni thread possiede UN container e cicla: prende un indice, POSTa il
        # job su quel container, attende il risultato (poll), passa al successivo.
        # Così ogni container ha al più 1 job in coda: niente serializzazione
        # nascosta lato server.
        task_q: "queue.Queue[int]" = queue.Queue()
        for i in range(N):
            task_q.put(i)

        results_ordered: list[dict | None] = [None] * N
        progress_lock = threading.Lock()
        completed = 0

        def _run_one(url: str, idx: int) -> dict:
            resp = requests.post(f"{url}/simulate", json=payloads[idx], timeout=10)
            if not resp.ok:
                raise RuntimeError(
                    f"POST /simulate fallita su {url} ({resp.status_code}).\n"
                    f"Payload: {payloads[idx]}\nRisposta: {resp.text}"
                )
            job_id = resp.json()["jobId"]
            deadline = time.time() + DEFAULT_TIMEOUT      # timeout PER-JOB
            while time.time() < deadline:
                poll = requests.get(f"{url}/simulate/{job_id}", timeout=10).json()
                status = poll.get("status")
                if status == "done":
                    return poll
                if status == "error":
                    raise RuntimeError(
                        f"Errore simulatore ({url}) job {job_id}: {poll.get('error')}"
                    )
                time.sleep(POLL_INTERVAL)
            raise TimeoutError(
                f"Job {job_id} su {url} non completato entro {DEFAULT_TIMEOUT}s "
                f"(sample #{idx + 1}). Il container potrebbe essere bloccato su Unity."
            )

        def _worker(url: str) -> None:
            nonlocal completed
            while True:
                try:
                    idx = task_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    result = _run_one(url, idx)
                    results_ordered[idx] = result
                    with progress_lock:
                        completed += 1
                        if verbose:
                            out     = result["output"]
                            L       = len(out["xtes"])
                            max_xte = max((abs(x) for x in out["xtes"]), default=0.0)
                            row     = params[idx]
                            print(
                                f"  [{completed:2d}/{N}] (sample #{idx+1:2d} @ "
                                f"porta {url.split(':')[-1]})"
                                f"  angoli=[{','.join(f'{int(round(a)):2d}' for a in row[:5])}]"
                                f"  →  {L} step,  XTE max={max_xte:.3f}m",
                                flush=True,
                            )
                finally:
                    task_q.task_done()

        errors: list[BaseException] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [executor.submit(_worker, url) for url in workers]
            for future in concurrent.futures.as_completed(futures):
                exc = future.exception()
                if exc is not None:
                    errors.append(exc)
        if errors:
            raise errors[0]     # fail-fast: propaga il primo errore/timeout

        results = results_ordered

        # ── Build all_stats from parallel results ────────────────────────────
        # The SimulatorServer returns parallel arrays, not a list of per-step dicts.
        # Actual output structure (verified against live Docker response):
        #   output.positions  : list of [x, y, z]  (one entry per iteration)
        #   output.xtes       : list of float       (cross-track error per step)
        #   output.steerings  : list of float       (steering angle per step)
        all_stats = []
        control_hz:      list[float] = []
        meters_per_step: list[float] = []
        infer_ms:        list[float] = []   # ms/step in inferenza (agent.predict)
        wait_ms:         list[float] = []   # ms/step in attesa frame Unity (env.step)
        for result in results:
            out = result["output"]
            all_stats.append({
                "positions": out["positions"],   # [[x,y,z], ...]
                "xtes":      out["xtes"],        # [float, ...]
                "steerings": out["steerings"],   # [float, ...]
            })

            # ── Fedelta' del loop: usa i campi che il server GIA' restituisce ──
            # elapsedTime = secondi di parete del loop di controllo
            # iterations  = numero di decisioni di sterzo
            # speeds      = velocita' per step (km/h: il telemetry fa m/s * 3.6)
            elapsed = float(out.get("elapsedTime", 0.0) or 0.0)
            iters   = int(out.get("iterations", 0) or 0)
            speeds  = out.get("speeds") or []
            if elapsed > 0.0 and iters > 0:
                hz = iters / elapsed
                sec_per_step = elapsed / iters
                mean_speed_mps = (float(np.mean(speeds)) / 3.6) if len(speeds) else 0.0
                mps = mean_speed_mps * sec_per_step
            else:
                # dati mancanti/degeneri: nessuna misura di fedelta' affidabile
                hz, mps = float("nan"), float("nan")
            control_hz.append(hz)
            meters_per_step.append(mps)

            # ── Split dei tempi per step: DOVE va il tempo del loop di controllo ──
            # predictSeconds/stepSeconds sono i totali per-run misurati nel loop del
            # simulatore. Divisi per gli step danno i ms medi/step in inferenza vs
            # attesa Unity: dice se il collo di bottiglia e' CPU o I/O.
            pS = out.get("predictSeconds", None)
            sS = out.get("stepSeconds", None)
            if iters > 0 and pS is not None and sS is not None and pS >= 0 and sS >= 0:
                infer_ms.append(float(pS) / iters * 1000.0)
                wait_ms.append(float(sS) / iters * 1000.0)
            else:
                infer_ms.append(float("nan"))
                wait_ms.append(float("nan"))

        # ── Build zero-padded trajectory tensor (N, T_max, 4) ────────────────
        #    Zero-padding is safe for the POD embedder (SVD handles zeros).
        #    self._run_lengths lets compute_qoi() mask out padded timesteps
        #    so M2/M3 statistics are computed only on real simulation steps.
        run_lengths = [len(s["xtes"]) for s in all_stats]
        self._run_lengths = run_lengths
        self._control_hz = np.asarray(control_hz, dtype=float)
        self._meters_per_step = np.asarray(meters_per_step, dtype=float)
        self._infer_ms = np.asarray(infer_ms, dtype=float)
        self._wait_ms = np.asarray(wait_ms, dtype=float)
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

        # ── Fedelta' del loop di controllo ────────────────────────────────────
        # Un run girato a frequenza troppo bassa (troppi worker in contesa CPU) e'
        # sotto-campionato: l'auto ha percorso troppi metri tra due decisioni di
        # sterzo. Non e' un test fedele del modello -> NON valido (come i degeneri),
        # cosi' i tassi restano indipendenti da quanti worker giravano.
        # Allinea alla coda come _run_lengths (il campione 'nominale' e' in testa).
        def _tail(arr):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            return arr[-N:] if arr.shape[0] >= N else np.full(N, np.nan)

        control_hz = _tail(self._control_hz)
        meters_per_step = _tail(self._meters_per_step)
        self._last_infer_ms = _tail(self._infer_ms)
        self._last_wait_ms = _tail(self._wait_ms)

        low_fidelity = np.zeros(len(survival), dtype=bool)
        if MIN_CONTROL_HZ > 0 and control_hz is not None:
            low_fidelity |= np.nan_to_num(control_hz, nan=np.inf) < MIN_CONTROL_HZ
        if MAX_METERS_PER_STEP > 0 and meters_per_step is not None:
            low_fidelity |= np.nan_to_num(meters_per_step, nan=0.0) > MAX_METERS_PER_STEP

        # I run NON validi sono misurazioni da escludere dall'analisi: margine = NaN
        # cosi' non contano ne' come safe ne' come failure. L'orchestrator calcola
        # i tassi e i rare failure solo sugli scenari validi.
        invalid = degenerate | bad_params | low_fidelity
        qoi = np.where(invalid, np.nan, qoi)

        self._last_survival = survival
        self._last_control_hz = control_hz
        self._last_meters_per_step = meters_per_step
        self._valid_mask = ~invalid
        self._n_degenerate = int(degenerate.sum())
        self._n_low_fidelity = int(low_fidelity.sum())
        self._n_invalid = int(invalid.sum())

        return qoi

    def failure_threshold(self) -> float:
        return 0.0
