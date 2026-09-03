"""
Tests for the fault tolerance of the lane-keeping worker pool.

A container whose Unity instance hangs keeps answering GET /health, so the health
probe cannot detect it: it accepts jobs and never finishes them. These tests
replace the `requests` module inside `scenarios.lane_keeping.config` with a fake
HTTP layer that reproduces exactly that behaviour — no Docker, no network — and
check that a batch survives it.

Run:  pytest tests/test_lane_keeping_resilience.py -q
"""
import numpy as np
import pytest

import scenarios.lane_keeping.config as lk


# ─────────────────────────────────────────────────────────────────────────────
# Fake simulator server
# ─────────────────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, payload, ok=True, status_code=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSimulatorPool:

    class RequestException(Exception):
        """
        Minimal stand-in for the SimulatorServer REST API.

        `stuck_urls` answer /health and accept POST /simulate, but their jobs stay
        "running" forever — the exact failure mode of a hung Unity container.
        """
        pass

    def __init__(self, stuck_urls=(), steps: int = 12):
        self.stuck = set(stuck_urls)
        self.steps = steps
        self.jobs: dict = {}
        self.submitted: dict = {}          # url -> number of jobs accepted
        self._next = 0

    # -- helpers ------------------------------------------------------------
    def _base(self, url: str) -> str:
        return url.split("/simulate")[0].split("/health")[0]

    def _output(self):
        n = self.steps
        return {
            "positions": [[float(i), 0.0, 0.0] for i in range(n)],
            "xtes": [0.1] * n,
            "steerings": [0.05] * n,
            "elapsedTime": 1.0,
            "iterations": n,
            "speeds": [36.0] * n,
            "predictSeconds": 0.1,
            "stepSeconds": 0.2,
        }

    # -- the two verbs the scenario uses ------------------------------------
    def get(self, url, timeout=None):
        if url.endswith("/health"):
            return _Resp({"status": "ok"})
        job_id = url.rsplit("/", 1)[-1]
        state = self.jobs.get(job_id)
        if state is None:
            return _Resp({"status": "error", "error": "unknown job"})
        if state["stuck"]:
            return _Resp({"status": "running"})       # hangs forever
        # A healthy job takes one poll interval to finish. Without this the fake
        # server is instantaneous and a single fast worker can drain the whole
        # queue before the stuck one is even scheduled, which makes the test
        # depend on thread timing.
        state["polls"] += 1
        if state["polls"] < 2:
            return _Resp({"status": "running"})
        return _Resp({"status": "done", "output": self._output()})

    def post(self, url, json=None, timeout=None):
        base = self._base(url)
        self.submitted[base] = self.submitted.get(base, 0) + 1
        self._next += 1
        job_id = f"job-{self._next}"
        self.jobs[job_id] = {"stuck": base in self.stuck, "polls": 0}
        return _Resp({"jobId": job_id})


@pytest.fixture
def fast_timeouts(monkeypatch):
    """Shrink the per-job timeout so a 'stuck' worker fails in a fraction of a second."""
    monkeypatch.setattr(lk, "DEFAULT_TIMEOUT", 0.3)
    monkeypatch.setattr(lk, "POLL_INTERVAL", 0.05)


def _params(n: int) -> np.ndarray:
    """n identical, coherent parameter rows (min_speed < max_speed)."""
    row = np.array([10, 10, 10, 10, 10, 6.0, 12.0, 25.0, 250.0], dtype=float)
    return np.tile(row, (n, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_batch_survives_one_stuck_container(monkeypatch, fast_timeouts):
    """The scenario that broke the campaign: one hung worker out of four."""
    urls = [f"http://localhost:{p}" for p in (8000, 8001, 8002, 8003)]
    fake = FakeSimulatorPool(stuck_urls=["http://localhost:8002"])
    monkeypatch.setattr(lk, "requests", fake)

    sc = lk.LaneKeepingScenario(simulator_urls=urls)
    traj = sc.run_simulation(_params(8))

    assert traj.shape[0] == 8
    # Every sample completed: the ones that hit 8002 were retried elsewhere.
    assert sc._n_job_failures == 0
    assert all(L > 0 for L in sc._run_lengths)
    # And the bad worker stopped receiving work after its consecutive failures.
    assert fake.submitted.get("http://localhost:8002", 0) <= lk.MAX_WORKER_FAILURES


def test_lost_samples_are_invalid_not_failures(monkeypatch, fast_timeouts):
    """
    With retries disabled a lost sample must be excluded from the rates, never
    counted as a crash — we have no evidence either way for that parameter point.
    """
    monkeypatch.setattr(lk, "MAX_JOB_RETRIES", 0)
    monkeypatch.setattr(lk, "MAX_WORKER_FAILURES", 99)      # keep the bad worker in
    urls = [f"http://localhost:{p}" for p in (8000, 8001)]
    monkeypatch.setattr(lk, "requests",
                        FakeSimulatorPool(stuck_urls=["http://localhost:8001"]))

    sc = lk.LaneKeepingScenario(simulator_urls=urls)
    params = _params(6)
    traj = sc.run_simulation(params)
    margins = sc.compute_qoi(traj, params)

    assert sc._n_job_failures > 0
    lost = np.array([L == 0 for L in sc._run_lengths])
    assert np.isnan(margins[lost]).all()            # NaN -> excluded from the rates
    assert (~sc._valid_mask[lost]).all()
    assert sc._valid_mask[~lost].all()              # the good runs are untouched


def test_all_workers_stuck_raises_a_useful_error(monkeypatch, fast_timeouts):
    """If nothing works at all, fail loudly — a silent batch of NaN helps nobody."""
    urls = [f"http://localhost:{p}" for p in (8000, 8001)]
    monkeypatch.setattr(lk, "requests", FakeSimulatorPool(stuck_urls=urls))

    sc = lk.LaneKeepingScenario(simulator_urls=urls)
    with pytest.raises(RuntimeError, match="None of the"):
        sc.run_simulation(_params(4))


def test_healthy_pool_is_unaffected(monkeypatch, fast_timeouts):
    """No regression: with every container healthy nothing changes."""
    urls = [f"http://localhost:{p}" for p in (8000, 8001)]
    monkeypatch.setattr(lk, "requests", FakeSimulatorPool())

    sc = lk.LaneKeepingScenario(simulator_urls=urls)
    params = _params(6)
    traj = sc.run_simulation(params)
    margins = sc.compute_qoi(traj, params)

    assert traj.shape == (6, 12, 4)
    assert sc._n_job_failures == 0
    assert np.isfinite(margins).all()
    assert sc._valid_mask.all()


def test_quarantine_persists_across_batches(monkeypatch, fast_timeouts):
    """
    A dead container does not heal between batches. Re-admitting it every time
    costs two timeouts per batch, which on a campaign of hundreds of batches
    dwarfs the useful work — so the quarantine has to outlive the batch.
    """
    urls = [f"http://localhost:{p}" for p in (8000, 8001, 8002)]
    fake = FakeSimulatorPool(stuck_urls=["http://localhost:8000"])
    monkeypatch.setattr(lk, "requests", fake)

    sc = lk.LaneKeepingScenario(simulator_urls=urls)
    sc.run_simulation(_params(6))
    submitted_after_first = fake.submitted.get("http://localhost:8000", 0)
    assert submitted_after_first > 0            # it did get work, and failed

    for _ in range(3):                          # three more batches
        sc.run_simulation(_params(6))
    # The bad worker never receives another job.
    assert fake.submitted["http://localhost:8000"] == submitted_after_first
    assert "http://localhost:8000" in sc._quarantined
    assert sc._n_job_failures == 0               # the good workers absorbed everything


def test_quarantine_can_be_reset(monkeypatch, fast_timeouts):
    """After restarting a container the user must be able to re-admit it."""
    urls = [f"http://localhost:{p}" for p in (8000, 8001)]
    fake = FakeSimulatorPool(stuck_urls=["http://localhost:8000"])
    monkeypatch.setattr(lk, "requests", fake)

    sc = lk.LaneKeepingScenario(simulator_urls=urls)
    sc.run_simulation(_params(4))
    assert sc._quarantined
    sc.reset_quarantine()
    assert not sc._quarantined


def test_per_batch_quarantine_still_available(monkeypatch, fast_timeouts):
    """LK_QUARANTINE_PERSIST=0 restores the old behaviour."""
    monkeypatch.setattr(lk, "QUARANTINE_PERSISTS", False)
    urls = [f"http://localhost:{p}" for p in (8000, 8001, 8002)]
    fake = FakeSimulatorPool(stuck_urls=["http://localhost:8000"])
    monkeypatch.setattr(lk, "requests", fake)

    sc = lk.LaneKeepingScenario(simulator_urls=urls)
    sc.run_simulation(_params(6))
    first = fake.submitted.get("http://localhost:8000", 0)
    sc.run_simulation(_params(6))
    assert fake.submitted["http://localhost:8000"] > first    # re-admitted
    assert not sc._quarantined
