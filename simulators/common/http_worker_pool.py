from __future__ import annotations

"""
Generic HTTP job-queue worker pool, extracted from
scenarios/lane_keeping/config.py so the same "N Docker containers, each
sequential internally, job-queue distributed across them" pattern can be
reused by the CARLA-backed emergency_braking/cut_in services instead of
being duplicated per scenario.

Contract expected of each worker (matches opensbt-core's SimulatorServer.py):
    POST {url}{submit_endpoint}          -> {"jobId": "..."}
    GET  {url}{poll_endpoint}/{job_id}   -> {"status": "queued"|"simulating"|"done"|"error", ...}
    GET  {url}/health                    -> 200 OK if the worker is alive

See opensbt-core/PARALLELIZZAZIONE.md for the rationale (per-job timeout
instead of a global N*timeout deadline, health-checked workers).
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

import requests

DEFAULT_JOB_TIMEOUT = 90     # seconds — per-job, not global
DEFAULT_POLL_INTERVAL = 0.5  # seconds between GET polls
DEFAULT_HEALTH_TIMEOUT = 5   # seconds for the /health probe


def build_pool_from_env(
    default_num_workers: int = 4,
    default_base_port: int = 8000,
    urls_env: str = "SIMULATOR_URLS",
    num_workers_env: str = "NUM_WORKERS",
    base_port_env: str = "SIMULATOR_BASE_PORT",
) -> List[str]:
    """
    Resolve a pool of worker base URLs, in priority order:
      1. `urls_env`        — explicit comma-separated list.
      2. `num_workers_env` + `base_port_env` — localhost:base..base+N-1.
      3. `default_num_workers` localhost workers starting at `default_base_port`.
    """
    explicit = os.getenv(urls_env)
    if explicit:
        return [u.strip().rstrip("/") for u in explicit.split(",") if u.strip()]
    n = int(os.getenv(num_workers_env, str(default_num_workers)))
    base = int(os.getenv(base_port_env, str(default_base_port)))
    return [f"http://localhost:{base + i}" for i in range(max(1, n))]


def healthy_workers(
    urls: List[str],
    health_path: str = "/health",
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
    verbose: bool = False,
) -> List[str]:
    """Return only the URLs that respond OK to GET {url}{health_path}."""
    healthy: List[str] = []
    for url in urls:
        try:
            r = requests.get(f"{url}{health_path}", timeout=timeout)
            if r.ok:
                healthy.append(url)
            elif verbose:
                print(f"[pool] {url} risponde ma non healthy ({r.status_code}) — ignorato", flush=True)
        except requests.RequestException:
            if verbose:
                print(f"[pool] {url} non raggiungibile — ignorato", flush=True)
    return healthy


def submit_job(url: str, payload: dict, submit_endpoint: str = "/simulate", timeout: float = 10) -> str:
    resp = requests.post(f"{url}{submit_endpoint}", json=payload, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(
            f"POST {submit_endpoint} fallita su {url} ({resp.status_code}).\n"
            f"Payload: {payload}\nRisposta: {resp.text}"
        )
    return resp.json()["jobId"]


def poll_job(
    url: str,
    job_id: str,
    poll_endpoint: str = "/simulate",
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = 10,
) -> dict:
    """
    Returns the poll response once the job reaches a terminal state — "done"
    OR "error". A per-job application error (e.g. a CARLA spawn collision for
    that one sampled parameter combination) does NOT raise here: it's a
    well-formed, completed response, just one that failed. Raising would let
    a single unlucky sample abort results already collected for every other
    sample in the batch (see run_job_pool). Only a genuinely stuck job
    (timeout) raises.
    """
    deadline = time.time() + job_timeout
    while time.time() < deadline:
        poll = requests.get(f"{url}{poll_endpoint}/{job_id}", timeout=timeout).json()
        status = poll.get("status")
        if status in ("done", "error"):
            return poll
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} su {url} non completato entro {job_timeout}s.")


def run_job_pool(
    payloads: List[dict],
    workers: List[str],
    submit_endpoint: str = "/simulate",
    poll_endpoint: str = "/simulate",
    job_timeout: float = DEFAULT_JOB_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    on_progress: Optional[Callable[[int, int, int, str, dict], None]] = None,
) -> List[dict]:
    """
    Distribute len(payloads) jobs across `workers` via a shared task queue —
    one thread per worker, each processing at most one job at a time (no
    hidden server-side serialization). Returns results in payload order,
    ALWAYS one entry per payload — never raises because of an individual
    job's outcome.

    A per-job failure (application error reported by the worker, a submit
    that couldn't reach it, or a timeout) becomes a
    `{"status": "error", "error": "..."}` entry at that index instead of
    aborting the batch — a single unlucky sample (e.g. a rare CARLA spawn
    collision for one sampled parameter combination) must not discard every
    other sample's already-completed result. Callers that care (most
    scenarios' `compute_qoi`) turn these into an invalid/NaN margin for that
    sample, the same way degenerate runs are already excluded.

    on_progress(completed, total, payload_idx, worker_url, result) is called
    (under a lock) after each job finishes, letting the caller print
    scenario-specific progress lines without touching the pool logic.
    """
    N = len(payloads)
    results: List[Optional[dict]] = [None] * N
    task_q: "queue.Queue[int]" = queue.Queue()
    for i in range(N):
        task_q.put(i)

    lock = threading.Lock()
    completed = 0

    def _worker(url: str) -> None:
        nonlocal completed
        while True:
            try:
                idx = task_q.get_nowait()
            except queue.Empty:
                return
            try:
                job_id = submit_job(url, payloads[idx], submit_endpoint=submit_endpoint)
                result = poll_job(
                    url, job_id,
                    poll_endpoint=poll_endpoint,
                    job_timeout=job_timeout,
                    poll_interval=poll_interval,
                )
            except Exception as e:  # noqa: BLE001 — submit failure or timeout: a per-job outcome
                result = {"status": "error", "error": str(e)}
            results[idx] = result
            with lock:
                completed += 1
                if on_progress:
                    on_progress(completed, N, idx, url, result)
            task_q.task_done()

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(_worker, url) for url in workers]
        for future in as_completed(futures):
            future.result()  # re-raise only a genuine bug inside _worker itself

    return results
