from __future__ import annotations

"""
Lane-Keeping scenario configuration.

Uses the existing Udacity/Unity simulator (opensbt-core) via Docker.
The DNN autopilot (SupervisedAgent / AutopilotModel) drives the vehicle;
we test how robust it is across a wider parameter space than the original
5-angle configuration.

Extended parameter space
-------------------------
Original : 5 road angles in [0, 85] deg
Extended : 5 road angles + min_speed + max_speed + segment_length + map_size

QoI - composite safety metric:
  M1 (0.6): XTE margin          MAX_XTE - max(|xte|)              positive = in lane
  M2 (0.2): Steering peak dev   -(max|s| - mean|s|)/STEER_RANGE   penalises sudden jerks
  M3 (0.2): Early violation     -(T - first_near_boundary)/T      penalises early approach

Failure  : composite QoI < 0.0

Multi-model: instantiate with simulator_url and name_suffix to run two autopilots in parallel
on separate Docker ports and compare failure profiles.

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
      1. SIMULATOR_URLS  - comma-separated explicit URLs
                           (e.g. "http://localhost:8000,http://localhost:8001").
      2. NUM_WORKERS     - builds localhost:BASE .. BASE+N-1
                           (BASE = SIMULATOR_BASE_PORT, default 8000).

    The worker count MUST match the containers started by docker-compose.parallel.yml
    (see opensbt-core/gen_parallel_compose.py). Unreachable workers are dropped at runtime by
    the health check.
    """
    explicit = os.getenv("SIMULATOR_URLS")
    if explicit:
        return [u.strip().rstrip("/") for u in explicit.split(",") if u.strip()]
    n    = int(os.getenv("NUM_WORKERS", str(DEFAULT_NUM_WORKERS)))
    base = int(os.getenv("SIMULATOR_BASE_PORT", "8000"))
    return [f"http://localhost:{base + i}" for i in range(max(1, n))]

# ── Physical constants ────────────────────────────────────────────────────────
MAX_XTE         = 2.5   # metres - matches opensbt-core/Simulator/lanekeeping/config.py
STEER_RANGE_NORM = 0.4  # normalisation for steering peak deviation: max|s| - mean|s|
                        # straight: range ~0.04 -> M2 ~ -0.10 (near 0)
                        # zigzag:   range ~0.65 -> M2 = -1.00 (saturated)
                        # 0.4 keeps M2 unsaturated for most safe runs
EARLY_FRAC      = 0.7   # fraction of MAX_XTE that triggers the "approaching boundary" flag
MIN_VALID_STEPS = 3     # runs shorter than this are degenerate/aborted (the sim stopped at once):
                        # not safe driving

# ── Control-loop fidelity (parallelism without loss of accuracy) ──────────────
# The loop closes at some rate: DNN.predict -> env.step -> new Unity frame. With too many
# parallel workers the Unity containers contend for CPU: the real-time factor drops, the loop
# runs slower and the car covers MORE METRES between two steering decisions. The DNN, trained at
# ~real-time cadence, then steers on stale, distant observations -> oscillation/instability, and
# failures that are ARTIFACTS of machine load, not model faults.
#
# We measure two indicators per run (from data the server ALREADY returns: elapsedTime,
# iterations, speeds):
#   control_hz      = iterations / elapsedTime          (loop rate, Hz)
#   meters_per_step = mean_speed_mps * elapsedTime/iter (spatial resolution, m/step)
# A run with control_hz too low or meters_per_step too high is under-sampled: we mark it INVALID
# (excluded from rates and rare failures, like the degenerate ones), so parallelism stays but the
# results do NOT depend on how many workers were running.
#
# Default = 0 (gate OFF, no behaviour change): the indicators are still computed and printed by
# the runner, so you can first MEASURE the rate vs --workers and then choose thresholds. Enable/
# tune via env:
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
                    (multi-model comparison).
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
        # Pool resolution (backward-compatible):
        #   - simulator_urls given  -> explicit pool (parallel, N containers)
        #   - simulator_url given   -> single-container pool
        #   - neither               -> pool from env (NUM_WORKERS/SIMULATOR_URLS)
        if simulator_urls is not None:
            self.simulator_urls = [u.rstrip("/") for u in simulator_urls]
        elif simulator_url is not None:
            self.simulator_urls = [simulator_url.rstrip("/")]
        else:
            self.simulator_urls = build_simulator_pool()
        # Preserved for legacy references / logging (first pool endpoint).
        self.simulator_url = self.simulator_urls[0]
        self.name = f"lane_keeping{name_suffix}"
        # run_simulation() stores actual run lengths here so compute_qoi()
        # can mask out zero-padded timesteps when computing M2 and M3.
        self._run_lengths: list[int] | None = None
        # Populated by compute_qoi: survival (n. steps) and degenerate-run count.
        self._last_survival: np.ndarray | None = None
        self._valid_mask: np.ndarray | None = None
        self._n_degenerate: int = 0
        self._n_invalid: int = 0
        # Control-loop fidelity, populated by run_simulation (one entry per run, aligned to
        # self._run_lengths). None if the sim does not provide it.
        self._control_hz: np.ndarray | None = None       # Hz per run
        self._meters_per_step: np.ndarray | None = None  # metres travelled per decision
        self._infer_ms: np.ndarray | None = None         # ms/step in inference
        self._wait_ms: np.ndarray | None = None          # ms/step waiting for Unity
        # Final views (aligned to the N trajectories) and count, from compute_qoi.
        self._last_control_hz: np.ndarray | None = None
        self._last_meters_per_step: np.ndarray | None = None
        self._last_infer_ms: np.ndarray | None = None
        self._last_wait_ms: np.ndarray | None = None
        self._n_low_fidelity: int = 0

    # ── Worker pool helpers ───────────────────────────────────────────────────

    def _build_payload(self, row: np.ndarray, ncols: int) -> dict:
        """Build the POST /simulate JSON payload from a parameter row."""
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
        """Return only the workers that respond to GET /health."""
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
          angles 1-5  [0, 85] deg  - road geometry (absolute bearing per segment)
          min_speed   [5, 15] m/s  - lower speed limit for the agent
          max_speed   [10, 30] m/s - upper speed limit for the agent
          seg_length  [10, 40] m   - length of each road segment
          map_size    [150, 350] m - side length of the simulation map
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
        REALISTIC operational distribution for each parameter, truncated to the effective bounds
        (lower/upper, which may come from a preset or the sweep).

        Returns a list of FROZEN scipy.stats distributions, one per dimension, supported on
        [lower_j, upper_j]. Used for distribution-aware sampling ('realistic' mode in
        pipeline.orchestrator): transforming LHS samples with dist.ppf() weights scenarios by how
        likely they are in real driving, so the failure fraction becomes an estimate of P(failure)
        under the ODD instead of a fraction over uniform sampling.

        METHODOLOGICAL NOTE: the shapes here are a PLAUSIBLE starting point, to be calibrated from
        real data/literature before drawing final numbers.
          - angles (0-4): more mass on gentle curves (truncated half-normal, mode at the lower
            bound): sharp curves are rare in real driving.
          - speeds (5,6): truncated normal centred on a cruising value (~40% of the range): one
            drives more often at intermediate speeds than at the extremes.
          - seg_length (7), map_size (8): uniform (no strong prior).
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
            if j < 5:                       # angles: mode on gentle curves
                dists.append(_truncnorm(a, c, mu=a, sigma=max(rng * 0.5, 1e-6)))
            elif j in (5, 6):               # speeds: cruising ~40% of the range
                dists.append(_truncnorm(a, c, mu=a + 0.4 * rng, sigma=max(rng * 0.3, 1e-6)))
            else:                           # seg_length, map_size: uniform
                dists.append(_uniform(a, c))
        return dists

    # ── Simulation ────────────────────────────────────────────────────────────

    def run_simulation(self, params: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Submit N simulation jobs to a POOL of SimulatorServer containers and collect results.

        Each container runs its simulations sequentially (one Unity instance per container).
        Parallelism comes from distributing jobs across containers: with W healthy workers the
        throughput is ~Wx a single container. Jobs are pulled from a shared queue (dynamic load
        balancing: faster containers process more).

        Each job has an INDIVIDUAL timeout (DEFAULT_TIMEOUT), not a global N*timeout deadline: a
        container stuck on Unity fails only its own job in ~90s instead of freezing the whole batch.

        Returns
        -------
        trajectories : (N, T_max, 4)  float32, zero-padded to the longest run.
            Channel 0 - x position (m)
            Channel 1 - y position (m)
            Channel 2 - cross-track error XTE (m)
            Channel 3 - steering angle (normalised)

        Side-effect: stores actual run lengths in self._run_lengths so that compute_qoi() can
        ignore zero-padded timesteps.
        """
        N     = params.shape[0]
        ncols = params.shape[1]
        payloads = [self._build_payload(row, ncols) for row in params]

        # Pool: keep only the workers that respond to /health.
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

        # Shared work queue: one thread per worker. Each thread owns ONE container and loops:
        # take an index, POST the job to that container, wait for the result (poll), move on. So
        # each container has at most 1 queued job: no hidden server-side serialisation.
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
            deadline = time.time() + DEFAULT_TIMEOUT      # per-job timeout
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
            raise errors[0]     # fail-fast: propagate the first error/timeout

        results = results_ordered

        # Build all_stats from the parallel results. The SimulatorServer returns parallel arrays,
        # not a list of per-step dicts. Output structure (verified against a live Docker response):
        #   output.positions  : list of [x, y, z]  (one entry per iteration)
        #   output.xtes       : list of float       (cross-track error per step)
        #   output.steerings  : list of float       (steering angle per step)
        all_stats = []
        control_hz:      list[float] = []
        meters_per_step: list[float] = []
        infer_ms:        list[float] = []   # ms/step in inference (agent.predict)
        wait_ms:         list[float] = []   # ms/step waiting for the Unity frame (env.step)
        for result in results:
            out = result["output"]
            all_stats.append({
                "positions": out["positions"],   # [[x,y,z], ...]
                "xtes":      out["xtes"],        # [float, ...]
                "steerings": out["steerings"],   # [float, ...]
            })

            # Fidelity: use the fields the server ALREADY returns.
            #   elapsedTime = wall-clock seconds of the control loop
            #   iterations  = number of steering decisions
            #   speeds      = per-step speed (km/h: telemetry does m/s * 3.6)
            elapsed = float(out.get("elapsedTime", 0.0) or 0.0)
            iters   = int(out.get("iterations", 0) or 0)
            speeds  = out.get("speeds") or []
            if elapsed > 0.0 and iters > 0:
                hz = iters / elapsed
                sec_per_step = elapsed / iters
                mean_speed_mps = (float(np.mean(speeds)) / 3.6) if len(speeds) else 0.0
                mps = mean_speed_mps * sec_per_step
            else:
                # missing/degenerate data: no reliable fidelity measure
                hz, mps = float("nan"), float("nan")
            control_hz.append(hz)
            meters_per_step.append(mps)

            # Per-step time split: WHERE the control-loop time goes. predictSeconds/stepSeconds are
            # per-run totals measured in the simulator loop; divided by the steps they give mean
            # ms/step in inference vs waiting for Unity, i.e. whether the bottleneck is CPU or I/O.
            pS = out.get("predictSeconds", None)
            sS = out.get("stepSeconds", None)
            if iters > 0 and pS is not None and sS is not None and pS >= 0 and sS >= 0:
                infer_ms.append(float(pS) / iters * 1000.0)
                wait_ms.append(float(sS) / iters * 1000.0)
            else:
                infer_ms.append(float("nan"))
                wait_ms.append(float("nan"))

        # Build the zero-padded trajectory tensor (N, T_max, 4). Zero-padding is safe for the POD
        # embedder (SVD handles zeros). self._run_lengths lets compute_qoi() mask out padded
        # timesteps so M2/M3 statistics are computed only on real simulation steps.
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

    # ── Quality of Interest ───────────────────────────────────────────────────

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Composite safety metric combining three signals:

        M1 - XTE margin (weight 0.6)
            MAX_XTE - max(|xte|) over valid steps.
            Positive -> car stayed well within the lane.
            Negative -> car left the road (failure).

        M2 - Steering peak deviation (weight 0.2)
            -(max(|steering|) - mean(|steering|)) / STEER_RANGE_NORM, clipped to [-1, 0].
            High peak deviation = sudden jerks / unstable control = early failure signal.
            More informative than std: std saturates at -1 for almost all runs with curves,
            while peak deviation retains gradation across safe scenarios.

        M3 - Early boundary approach (weight 0.2)
            -(valid_steps - first_step_near_boundary) / valid_steps.
            0 if the car never approaches within EARLY_FRAC * MAX_XTE of the edge.
            -1 if it approaches on the very first step.

        Zero-padded timesteps are excluded from M2 and M3 via self._run_lengths (set by
        run_simulation). If called standalone (e.g. in tests), the full trajectory length is used,
        which may slightly underestimate M2/M3 for short runs with long zero padding.
        """
        N, T, _ = trajectories.shape
        xte      = trajectories[:, :, 2].astype(np.float64)   # (N, T)
        steering = trajectories[:, :, 3].astype(np.float64)   # (N, T)

        # Build validity mask - True for real simulation steps, False for padding.
        run_lengths = self._run_lengths if self._run_lengths is not None else [T] * N
        # self._run_lengths may include extra leading rows (e.g. the 'nominal' sample simulated
        # with the batch but then dropped by the orchestrator): align to the tail so it matches
        # the N trajectories received.
        if len(run_lengths) != N:
            run_lengths = list(run_lengths)[-N:]
        valid = np.zeros((N, T), dtype=bool)
        for i, L in enumerate(run_lengths):
            valid[i, :min(L, T)] = True
        valid_count = valid.sum(axis=1).astype(np.float64)    # (N,) - at least 1
        valid_count = np.where(valid_count > 0, valid_count, 1.0)

        # Replace padded timesteps with NaN so nan-aware functions ignore them.
        xte_m      = np.where(valid, xte,      np.nan)
        steering_m = np.where(valid, steering, np.nan)

        # M1 - safety margin
        m1 = MAX_XTE - np.nanmax(np.abs(xte_m), axis=1)      # (N,)

        # M2 - steering peak deviation (max|s| - mean|s|)
        steer_abs  = np.abs(steering_m)
        steer_peak = np.nanmax(steer_abs, axis=1) - np.nanmean(steer_abs, axis=1)  # (N,)
        m2 = -np.clip(steer_peak / STEER_RANGE_NORM, 0.0, 1.0)                     # (N,) in [-1, 0]

        # M3 - time of first near-boundary approach
        near_boundary = np.abs(xte_m) > EARLY_FRAC * MAX_XTE  # (N, T), NaN -> False
        near_boundary = np.where(np.isnan(xte_m), False, near_boundary)
        any_near      = near_boundary.any(axis=1)              # (N,)
        first_idx     = np.where(
            any_near,
            near_boundary.argmax(axis=1).astype(np.float64),
            valid_count,                                       # never triggered -> use T
        )
        m3 = -(valid_count - first_idx) / valid_count         # (N,) in [-1, 0]

        qoi = 0.6 * m1 + 0.2 * m2 + 0.2 * m3                  # composite safety metric

        # Degenerate/aborted runs. A run with very few steps (e.g. 1) is not "safe" driving: it
        # stopped at once and the QoI would reward it (low XTE -> high M1). We treat it as a
        # failure, but with margin just < 0, so it counts as a failure without becoming a spurious
        # 'rare failure' (which must be a real crash). We keep survival (n. steps) to give the
        # downstream rare failures temporal resolution (see find_rare_failures tiebreak).
        survival = np.asarray(run_lengths, dtype=float)      # (N,)
        degenerate = survival < MIN_VALID_STEPS              # (N,) bool, aborted sims

        # Incoherent params: min_speed > max_speed (columns 5 and 6) is not a real scenario but a
        # badly sampled input.
        if params.shape[1] > 6:
            bad_params = np.asarray(params)[:, 5] > np.asarray(params)[:, 6]
        else:
            bad_params = np.zeros(len(survival), dtype=bool)

        # Control-loop fidelity. A run at too low a rate (too many workers contending for CPU) is
        # under-sampled: the car covered too many metres between two steering decisions. It is not
        # a faithful test of the model -> INVALID (like the degenerate ones), so rates stay
        # independent of how many workers were running. Align to the tail like _run_lengths (the
        # 'nominal' sample is at the head).
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

        # Invalid runs are measurements to exclude from the analysis: margin = NaN so they count
        # neither as safe nor as failure. The orchestrator computes rates and rare failures over
        # the valid scenarios only.
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
