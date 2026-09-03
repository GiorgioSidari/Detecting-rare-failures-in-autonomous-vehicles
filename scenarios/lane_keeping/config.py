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
# QoI constants + the pure composite-margin function live in qoi.py; re-exported here so that
# `from scenarios.lane_keeping.config import MAX_XTE, ...` keeps working.
from scenarios.lane_keeping.qoi import (
    MAX_XTE, STEER_RANGE_NORM, EARLY_FRAC, MIN_VALID_STEPS, composite_lane_qoi,
)

# ── Simulator connection ──────────────────────────────────────────────────────
SIMULATOR_URL   = "http://localhost:8000"   # default SimulatorServer FastAPI base URL
DEFAULT_TIMEOUT = 90      # seconds to wait for a SINGLE simulation job (per-job, not global)
POLL_INTERVAL   = 0.5     # seconds between GET polling requests
HEALTH_TIMEOUT  = 5       # seconds for the /health probe of each worker

# ── Fault tolerance ───────────────────────────────────────────────────────────
# A container whose Unity instance hangs still answers GET /health, so the health
# probe cannot see it: it accepts jobs and never finishes them. Without retries a
# single such container aborts the whole batch, which on a multi-hour campaign
# throws away every simulation already paid for.
#   MAX_JOB_RETRIES     - how many times a failed sample is re-queued (on a
#                         different worker, since the failing one gets skipped).
#   MAX_WORKER_FAILURES - consecutive failures after which a worker is dropped
#                         from the pool (for the rest of the session, see
#                         QUARANTINE_PERSISTS). The last worker is never dropped.
# A sample that exhausts its retries is returned as an EMPTY run: the QoI marks
# it invalid (too few steps) and it is excluded from the rates, exactly like an
# aborted simulation. Set LK_MAX_JOB_RETRIES=0 to restore the old fail-fast.
MAX_JOB_RETRIES     = int(os.getenv("LK_MAX_JOB_RETRIES", "2"))
MAX_WORKER_FAILURES = int(os.getenv("LK_MAX_WORKER_FAILURES", "2"))
# A quarantined worker stays out for the whole SESSION, not just the batch that
# caught it. A dead container does not heal between batches, so re-admitting it
# every time costs MAX_WORKER_FAILURES x DEFAULT_TIMEOUT of pure waiting per
# batch — on a campaign with hundreds of batches that dwarfs the useful work.
# Set LK_QUARANTINE_PERSIST=0 to go back to per-batch quarantine.
QUARANTINE_PERSISTS = os.getenv("LK_QUARANTINE_PERSIST", "1") != "0"

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

# ── Control-loop fidelity ─────────────────────────────────────────────────────
# The loop closes at some rate (DNN.predict -> env.step -> new Unity frame). Too many parallel
# workers make the Unity containers contend for CPU: the loop slows, the car covers more metres
# between two steering decisions, and the DNN steers on stale observations -> failures that are
# artifacts of machine load, not model faults. We measure two indicators per run:
#   control_hz      = iterations / elapsedTime            (loop rate, Hz)
#   meters_per_step = mean_speed_mps * elapsedTime/iter   (spatial resolution, m/step)
# A run too slow / too coarse is under-sampled -> marked INVALID (excluded from rates), so the
# number of workers doesn't affect the results. Gate OFF by default (indicators still printed);
# tune via LK_MIN_CONTROL_HZ / LK_MAX_METERS_PER_STEP.
MIN_CONTROL_HZ      = float(os.getenv("LK_MIN_CONTROL_HZ", "0"))       # Hz; 0 = off
MAX_METERS_PER_STEP = float(os.getenv("LK_MAX_METERS_PER_STEP", "0"))  # m/step; 0 = off


def _post_and_poll(url: str, payload: dict, idx: int) -> dict:
    """
    Submit one simulation to a container and wait for its result.

    The timeout is per JOB, not per batch: a container stuck on Unity fails its
    own job in ~90 s instead of freezing everything behind it.
    """
    resp = requests.post(f"{url}/simulate", json=payload, timeout=10)
    if not resp.ok:
        raise RuntimeError(
            f"POST /simulate failed on {url} ({resp.status_code}).\n"
            f"Payload: {payload}\nResponse: {resp.text}"
        )
    job_id = resp.json()["jobId"]
    deadline = time.time() + DEFAULT_TIMEOUT
    while time.time() < deadline:
        poll = requests.get(f"{url}/simulate/{job_id}", timeout=10).json()
        status = poll.get("status")
        if status == "done":
            return poll
        if status == "error":
            raise RuntimeError(
                f"Simulator error ({url}) job {job_id}: {poll.get('error')}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Job {job_id} on {url} did not finish within {DEFAULT_TIMEOUT}s "
        f"(sample #{idx + 1}). The container may be stuck on Unity."
    )


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
        # Jobs lost to timeouts/errors after every retry (see MAX_JOB_RETRIES).
        # They are reported as invalid, never as failures.
        self._n_job_failures: int = 0
        self._last_job_errors: dict = {}
        # Workers dropped for the rest of the session (see QUARANTINE_PERSISTS).
        self._quarantined: set = set()
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

    def reset_quarantine(self) -> None:
        """Re-admit every worker dropped earlier (call after restarting a container)."""
        self._quarantined.clear()

    def _healthy_workers(self, verbose: bool = False) -> list[str]:
        """Workers that answer GET /health and are not under session quarantine."""
        healthy: list[str] = []
        for url in self.simulator_urls:
            if url in self._quarantined:
                if verbose:
                    print(f"[pool] {url} quarantined by an earlier batch -- skipped",
                          flush=True)
                continue
            try:
                r = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
                if r.ok:
                    healthy.append(url)
                elif verbose:
                    print(f"[pool] {url} answers but is not healthy "
                          f"({r.status_code}) -- skipped", flush=True)
            except requests.RequestException:
                if verbose:
                    print(f"[pool] {url} unreachable -- skipped", flush=True)
        return healthy

    def probe_workers(self, timeout: float = 60.0, prune: bool = True,
                      verbose: bool = True) -> dict:
        """
        Send one trivial simulation to each worker and report which ones complete it.

        `GET /health` answers from the FastAPI layer alone, so a container whose Unity
        instance is hung passes it and then accepts jobs without finishing them. This
        sends a real job instead.

        prune : remove the workers that fail from `self.simulator_urls`, so the rest of
                the session does not dispatch to them.

        Returns {url: "ok" | "<error>"}.
        """
        payload = {"angles": [0, 0, 0, 0, 0], "minSpeed": 6, "maxSpeed": 12,
                   "segLength": 25, "map_size": 250, "maxTime": 10, "maxXTE": MAX_XTE}
        status: dict = {}

        def _probe(url: str) -> tuple:
            try:
                resp = requests.post(f"{url}/simulate", json=payload, timeout=10)
                if not resp.ok:
                    return url, f"POST /simulate -> {resp.status_code}"
                job_id = resp.json()["jobId"]
                deadline = time.time() + timeout
                while time.time() < deadline:
                    poll = requests.get(f"{url}/simulate/{job_id}", timeout=10).json()
                    if poll.get("status") == "done":
                        return url, "ok"
                    if poll.get("status") == "error":
                        return url, f"simulator error: {poll.get('error')}"
                    time.sleep(POLL_INTERVAL)
                return url, f"no answer within {timeout:.0f}s (Unity stuck?)"
            except Exception as exc:
                return url, f"{type(exc).__name__}: {exc}"

        self.reset_quarantine()      # a fresh probe overrules earlier verdicts
        candidates = self._healthy_workers(verbose=verbose)
        if verbose:
            print(f"[preflight] trying one simulation on {len(candidates)} workers...",
                  flush=True)
        if candidates:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as ex:
                for url, msg in ex.map(_probe, candidates):
                    status[url] = msg
        for url in self.simulator_urls:
            status.setdefault(url, "unreachable (/health failed)")

        good = [u for u, m in status.items() if m == "ok"]
        if verbose:
            for url in self.simulator_urls:
                mark = "OK  " if status[url] == "ok" else "KO  "
                print(f"[preflight] {mark}{url}  {status[url]}", flush=True)
        if prune and good:
            self.simulator_urls = good
            self.simulator_url = good[0]
        return status

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
        Operational distribution of each parameter, truncated to the effective bounds
        (`lower` / `upper`, which may come from a preset or from the sweep).

        Returns a list of FROZEN scipy.stats distributions, one per dimension,
        supported on `[lower_j, upper_j]`. They are used by the distribution-aware
        sampling path ('realistic' mode in `pipeline.orchestrator`), which transforms
        LHS samples through `dist.ppf()`, and by every estimator that weights a
        scenario by its operational density.

        Shapes returned:
          - angles (0-4): truncated half-normal with its mode at the lower bound, so
            most of the mass sits on gentle curves;
          - speeds (5, 6): truncated normal centred at about 40% of the range;
          - seg_length (7), map_size (8): uniform.
        """
        from scipy import stats
        b = self.param_bounds()
        lo = np.asarray(lower if lower is not None else b["lower"], dtype=float)
        hi = np.asarray(upper if upper is not None else b["upper"], dtype=float)

        def _uniform(a, c):
            return stats.uniform(loc=a, scale=max(c - a, 1e-9))

        def _truncnorm(a, c, mu: np.ndarray, sigma: np.ndarray):
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

    def _dispatch_jobs(self, payloads: list, workers: list, params: np.ndarray,
                       verbose: bool) -> tuple:
        """
    Run every job across the worker pool, with retries and quarantine.

    One thread per container, all pulling from a shared queue, so a slower container
    takes fewer jobs. A job that exhausts its retries is recorded as lost instead of
    raising, and a container that keeps failing is quarantined for the rest of the
    batch.

    Returns ``(results_ordered, job_errors)``, both indexed by sample.
    """
        N = len(payloads)
        # Shared work queue: one thread per worker. Each thread owns ONE container and loops:
        # take an index, POST the job to that container, wait for the result (poll), move on. So
        # each container has at most 1 queued job: no hidden server-side serialisation.
        task_q: "queue.Queue[int]" = queue.Queue()
        for i in range(N):
            task_q.put(i)

        results_ordered: list[dict | None] = [None] * N
        progress_lock = threading.Lock()
        completed = 0
        attempts = [0] * N                      # retries already spent per sample
        job_errors: dict[int, str] = {}         # sample -> last error, once it gave up
        active_workers = set(workers)           # workers still trusted right now
        pool_lock = threading.Lock()

        def _run_one(url: str, idx: int) -> dict:
            return _post_and_poll(url, payloads[idx], idx)

        def _give_up_or_retry(url: str, idx: int, exc: BaseException) -> None:
            """Re-queue a failed sample, or record it as lost once retries run out."""
            with progress_lock:
                attempts[idx] += 1
                retry = attempts[idx] <= MAX_JOB_RETRIES
                if retry:
                    task_q.put(idx)
                else:
                    job_errors[idx] = f"{type(exc).__name__}: {exc}"
                tag = "retrying" if retry else "GIVEN UP"
                print(f"  [!] sample #{idx + 1} on port {url.split(':')[-1]}: "
                      f"{type(exc).__name__} -- {tag} "
                      f"(attempt {attempts[idx]}/{MAX_JOB_RETRIES + 1})", flush=True)

        def _quarantine(url: str) -> bool:
            """Drop a repeatedly failing worker, unless it is the last one standing."""
            with pool_lock:
                if len(active_workers) <= 1:
                    return False
                active_workers.discard(url)
                scope = "for the rest of the session" if QUARANTINE_PERSISTS \
                    else "for this batch"
                if QUARANTINE_PERSISTS:
                    self._quarantined.add(url)
                print(f"  [!] worker {url} dropped from the pool {scope} "
                      f"({MAX_WORKER_FAILURES} consecutive failures). "
                      f"{len(active_workers)} workers left.", flush=True)
                return True

        def _worker(url: str) -> None:
            nonlocal completed
            consecutive_failures = 0
            while True:
                with pool_lock:
                    if url not in active_workers:
                        return
                try:
                    idx = task_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    result = _run_one(url, idx)
                    consecutive_failures = 0
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
                                f"port {url.split(':')[-1]})"
                                f"  angles=[{','.join(f'{int(round(a)):2d}' for a in row[:5])}]"
                                f"  ->  {L} steps,  XTE max={max_xte:.3f}m",
                                flush=True,
                            )
                except (TimeoutError, RuntimeError, requests.RequestException) as exc:
                    consecutive_failures += 1
                    _give_up_or_retry(url, idx, exc)
                    if consecutive_failures >= MAX_WORKER_FAILURES:
                        # `finally` still runs on the way out, so task_done() is
                        # accounted for exactly once.
                        if _quarantine(url):
                            return
                finally:
                    task_q.task_done()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [executor.submit(_worker, url) for url in workers]
            for future in concurrent.futures.as_completed(futures):
                exc = future.exception()
                if exc is not None:      # a bug in the worker loop itself, not a job failure
                    raise exc

        # A sample re-queued just as the pool drained can be left behind: sweep the
        # remainder sequentially on whichever workers are still trusted.
        leftovers: list[int] = []
        while True:
            try:
                leftovers.append(task_q.get_nowait())
            except queue.Empty:
                break
        for idx in leftovers:
            for url in (sorted(active_workers) or workers):
                try:
                    results_ordered[idx] = _run_one(url, idx)
                    break
                except (TimeoutError, RuntimeError, requests.RequestException) as exc:
                    job_errors[idx] = f"{type(exc).__name__}: {exc}"

        return results_ordered, job_errors

    @staticmethod
    def _telemetry_from_results(results: list) -> tuple:
        """
        Turn the servers' replies into per-run trajectories and fidelity.

        The server returns parallel arrays -- positions [x,y,z], xtes, steerings,
        one entry per iteration -- not a list of per-step dicts.

        Returns ``(all_stats, control_hz, meters_per_step, infer_ms, wait_ms)``.
        """
        # Build all_stats from the parallel results. The server returns parallel arrays — positions
        # [x,y,z], xtes, steerings, one entry per iteration — not a list of per-step dicts.
        all_stats = []
        control_hz:      list[float] = []
        meters_per_step: list[float] = []
        infer_ms:        list[float] = []   # ms/step in inference (agent.predict)
        wait_ms:         list[float] = []   # ms/step waiting for the Unity frame (env.step)
        for result in results:
            if result is None:
                # Sample lost after every retry. An empty run has length 0, which
                # composite_lane_qoi treats as degenerate -> margin NaN -> excluded
                # from the rates. It is NOT counted as a failure: we simply have no
                # evidence either way for this parameter point.
                all_stats.append({"positions": [], "xtes": [], "steerings": []})
                control_hz.append(float("nan"))
                meters_per_step.append(float("nan"))
                infer_ms.append(float("nan"))
                wait_ms.append(float("nan"))
                continue
            out = result["output"]
            all_stats.append({
                "positions": out["positions"],   # [[x,y,z], ...]
                "xtes":      out["xtes"],        # [float, ...]
                "steerings": out["steerings"],   # [float, ...]
            })

            # Fidelity from fields the server already returns: elapsedTime (loop seconds),
            # iterations (steering decisions), speeds (per-step, km/h).
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

        return all_stats, control_hz, meters_per_step, infer_ms, wait_ms

    def run_simulation(self, params: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Submit N simulation jobs to a POOL of SimulatorServer containers and collect
        the results.

        Each container runs its simulations sequentially (one Unity instance per
        container); parallelism comes from distributing jobs across containers. Jobs
        are pulled from a shared queue, so a faster container processes more of them.

        Each job carries its own timeout (DEFAULT_TIMEOUT) rather than the batch
        sharing a single deadline, so a container stuck on Unity fails only its own
        job.

        Helpers: `_dispatch_jobs` runs the worker pool, `_post_and_poll` submits one
        job and polls it to completion, `_telemetry_from_results` assembles the arrays
        below.

        Returns
        -------
        trajectories : (N, T_max, 4) float32, zero-padded to the longest run.
            Channel 0 - x position (m)
            Channel 1 - y position (m)
            Channel 2 - cross-track error XTE (m)
            Channel 3 - steering angle (normalised)

        Side effect: stores the actual run lengths in `self._run_lengths`, which
        `compute_qoi()` uses to ignore the zero-padded timesteps.
        """
        N     = params.shape[0]
        ncols = params.shape[1]
        payloads = [self._build_payload(row, ncols) for row in params]

        # Pool: keep only the workers that respond to /health.
        workers = self._healthy_workers(verbose=verbose)
        if not workers:
            raise RuntimeError(
                "No simulator reachable. Start opensbt-core, e.g.:\n"
                "  cd opensbt-core && "
                "docker compose -f docker-compose.parallel.yml up --build\n"
                f"URLs tried: {self.simulator_urls}\n"
                "(set NUM_WORKERS or SIMULATOR_URLS to change the pool)."
            )
        if verbose:
            ports = ", ".join(u.split(":")[-1] for u in workers)
            print(f"[pool] {len(workers)} active workers (ports: {ports})", flush=True)

        results_ordered, job_errors = self._dispatch_jobs(
            payloads, workers, params, verbose)

        n_ok = sum(r is not None for r in results_ordered)
        if n_ok == 0:
            distinct = sorted(set(job_errors.values()))[:3]
            raise RuntimeError(
                f"None of the {N} simulations succeeded on {len(workers)} "
                f"workers. Typical errors:\n  "
                + "\n  ".join(distinct)
                + "\nCheck the containers (docker ps / docker logs): a stuck Unity "
                  "instance answers /health but never completes a job."
            )
        if job_errors:
            print(f"  [!] {len(job_errors)}/{N} simulations lost after "
                  f"{MAX_JOB_RETRIES} retries: marked invalid and excluded "
                  f"from the rates.", flush=True)

        self._n_job_failures = len(job_errors)
        self._last_job_errors = dict(job_errors)
        results = results_ordered

        (all_stats, control_hz, meters_per_step, infer_ms,
         wait_ms) = self._telemetry_from_results(results)

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
        Composite safety margin per scenario (failure when < 0). The maths lives in the pure
        `composite_lane_qoi` (see qoi.py); here we feed it the per-run state gathered by
        run_simulation and store the results the runner needs for its report.
        """
        res = composite_lane_qoi(
            trajectories,
            self._run_lengths,
            params,
            control_hz=self._control_hz,
            meters_per_step=self._meters_per_step,
            infer_ms=self._infer_ms,
            wait_ms=self._wait_ms,
            min_control_hz=MIN_CONTROL_HZ,
            max_meters_per_step=MAX_METERS_PER_STEP,
        )
        # Store per-run state used by the runner for its report.
        self._last_survival = res.survival
        self._last_control_hz = res.control_hz
        self._last_meters_per_step = res.meters_per_step
        self._last_infer_ms = res.infer_ms
        self._last_wait_ms = res.wait_ms
        self._valid_mask = res.valid_mask
        self._n_degenerate = res.n_degenerate
        self._n_low_fidelity = res.n_low_fidelity
        self._n_invalid = res.n_invalid
        return res.qoi

    def failure_threshold(self) -> float:
        return 0.0
