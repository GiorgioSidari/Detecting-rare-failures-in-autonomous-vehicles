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

import time
import concurrent.futures
import numpy as np
import requests

from scenarios.base_scenario import BaseScenario

# ── Simulator connection ──────────────────────────────────────────────────────
SIMULATOR_URL   = "http://localhost:8000"   # SimulatorServer FastAPI base URL
DEFAULT_TIMEOUT = 90      # seconds to wait for a single simulation job
POLL_INTERVAL   = 0.5     # seconds between GET polling requests

# ── Physical constants ────────────────────────────────────────────────────────
MAX_XTE         = 2.5   # metres — matches opensbt-core/Simulator/lanekeeping/config.py
STEER_RANGE_NORM = 0.4  # normalisation for steering peak deviation: max|s| - mean|s|
                        # rettilineo: range≈0.04 → M2≈-0.10 (quasi 0)
                        # zigzag:     range≈0.65 → M2=-1.00 (saturato)
                        # soglia 0.4 lascia M2 non saturato per la maggior parte dei run safe
EARLY_FRAC      = 0.7   # fraction of MAX_XTE that triggers the "approaching boundary" flag


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

    def __init__(self, simulator_url: str = SIMULATOR_URL, name_suffix: str = ""):
        self.simulator_url = simulator_url
        self.name = f"lane_keeping{name_suffix}"
        # run_simulation() stores actual run lengths here so compute_qoi()
        # can mask out zero-padded timesteps when computing M2 and M3.
        self._run_lengths: list[int] | None = None

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
        Submit N simulation jobs to SimulatorServer and collect results.

        Each row of params is sent as a POST /simulate JSON payload.
        Results are polled via GET /simulate/{job_id} until done or timeout.

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
        N = params.shape[0]

        # ── Phase 1: submit all jobs (sequential, fast) ───────────────────────
        # Each POST returns immediately with a job_id — no waiting for execution.
        job_ids: list[str] = []
        for row in params:
            map_size = float(row[8]) if params.shape[1] > 8 else 250.0
            config_payload = {
                "angles":    [int(round(a)) for a in row[:5]],
                "minSpeed":  int(round(row[5])),
                "maxSpeed":  int(round(row[6])),
                "segLength": int(round(row[7])),
                "map_size":  int(round(map_size)),
                "maxTime":   30,
                "maxXTE":    MAX_XTE,
            }
            resp = requests.post(
                f"{self.simulator_url}/simulate",
                json=config_payload,
                timeout=10,
            )
            if not resp.ok:
                raise RuntimeError(
                    f"POST /simulate failed ({resp.status_code}).\n"
                    f"Payload inviato: {config_payload}\n"
                    f"Risposta server: {resp.text}"
                )
            job_ids.append(resp.json()["jobId"])

        # ── Phase 2: poll all jobs in parallel ────────────────────────────────
        # Il server esegue i job in coda sequenziale (singolo processo Unity),
        # quindi l'ultimo job attende N × sim_time prima di partire.
        # Usiamo un deadline GLOBALE condiviso tra tutti i thread:
        #   global_deadline = now + N × DEFAULT_TIMEOUT
        # così il job in fondo alla coda ha comunque tempo sufficiente.
        global_deadline = time.time() + N * DEFAULT_TIMEOUT

        def _poll(job_id: str) -> dict:
            while time.time() < global_deadline:
                poll = requests.get(
                    f"{self.simulator_url}/simulate/{job_id}",
                    timeout=10,
                ).json()
                if poll["status"] == "done":
                    return poll
                if poll["status"] == "error":
                    raise RuntimeError(
                        f"Simulator error for job {job_id}: {poll.get('error')}"
                    )
                time.sleep(POLL_INTERVAL)
            raise TimeoutError(
                f"Job {job_id} did not finish within {N * DEFAULT_TIMEOUT}s "
                f"(N={N} × {DEFAULT_TIMEOUT}s). "
                "Check that opensbt-core Docker is running and healthy."
            )

        # as_completed permette di stampare il progresso non appena ogni job finisce,
        # indipendentemente dall'ordine — il contatore mostra quanti sono stati completati.
        results_ordered = [None] * N
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            future_to_idx = {executor.submit(_poll, jid): i
                             for i, jid in enumerate(job_ids)}
            for future in concurrent.futures.as_completed(future_to_idx):
                i      = future_to_idx[future]
                result = future.result()          # ri-lancia eventuali eccezioni
                results_ordered[i] = result
                completed += 1
                if verbose:
                    out     = result["output"]
                    L       = len(out["xtes"])
                    max_xte = max(abs(x) for x in out["xtes"]) if out["xtes"] else 0.0
                    row     = params[i]
                    print(
                        f"  [{completed:2d}/{N}] (sample #{i+1:2d})"
                        f"  angoli=[{','.join(f'{int(round(a)):2d}' for a in row[:5])}]"
                        f"  speed=[{int(round(row[5])):d},{int(round(row[6])):d}]"
                        f"  seg={int(round(row[7]))}m  map={int(round(row[8]))}m"
                        f"  →  {L} step,  XTE max={max_xte:.3f}m",
                        flush=True,
                    )
        results = results_ordered

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
        valid = np.zeros((N, T), dtype=bool)
        for i, L in enumerate(run_lengths):
            valid[i, :L] = True
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

        return 0.6 * m1 + 0.2 * m2 + 0.2 * m3

    def failure_threshold(self) -> float:
        return 0.0
