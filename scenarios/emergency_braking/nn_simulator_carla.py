from __future__ import annotations

"""
Emergency Braking simulator driven by the CARLA-trained camera CNN (Fase 2
del piano video-CNN) — the video counterpart of nn_simulator.py's scalar
BrakingMLP.

Talks to the CARLA job-queue service
(opensbt-core/Simulator/emergency_braking/SimulatorServer.py) exactly like
scenarios/lane_keeping/config.py talks to the Udacity one, reusing
simulators/common/http_worker_pool.py instead of a third copy of the
pool/health-check/timeout pattern.
"""

import numpy as np

from simulators.base_simulator import BaseSimulator
from simulators.common.http_worker_pool import (
    DEFAULT_JOB_TIMEOUT,
    build_pool_from_env,
    healthy_workers,
    run_job_pool,
)

DEFAULT_NUM_WORKERS = 1
DEFAULT_BASE_PORT = 8100   # separate port range from lanekeeping's 8000+


class EmergencyBrakingCarlaNNSimulator(BaseSimulator):
    """
    Drop-in replacement for EmergencyBrakingSimulator/EmergencyBrakingNNSimulator
    that runs the scenario inside CARLA with the trained CNN driving the ego.

    Parameters
    ----------
    simulator_urls : explicit pool of SimulatorServer endpoints. If omitted,
                     resolved from env (EB_SIMULATOR_URLS > EB_NUM_WORKERS +
                     EB_SIMULATOR_BASE_PORT > single worker on port 8100).
    job_timeout    : per-job timeout in seconds (a CARLA episode is capped at
                     ~10s of sim time, but real-time factor can be << 1).
    """

    def __init__(
        self,
        simulator_urls: list[str] | None = None,
        job_timeout: float = DEFAULT_JOB_TIMEOUT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.simulator_urls = simulator_urls or build_pool_from_env(
            default_num_workers=DEFAULT_NUM_WORKERS,
            default_base_port=DEFAULT_BASE_PORT,
            num_workers_env="EB_NUM_WORKERS",
            base_port_env="EB_SIMULATOR_BASE_PORT",
            urls_env="EB_SIMULATOR_URLS",
        )
        self.job_timeout = job_timeout

    def run(self, controllable_parameters: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        controllable_parameters : (N, 4)
            [initial_speed (m/s), friction_coefficient (-), detection_distance (m), nominal_delay (s)]
            (nominal_delay now controls the lead vehicle's braking onset — see
            opensbt-core/Simulator/emergency_braking/carla_episode.py)

        Returns trajectories (N, T, 2): [position (m), velocity (m/s)], zero-length
        runs padded by freezing the final state (same convention as the physics sim).
        """
        N = controllable_parameters.shape[0]
        payloads = [
            {
                "initial_speed": float(row[0]),
                "friction_coefficient": float(row[1]),
                "detection_distance": float(row[2]),
                "nominal_delay": float(row[3]),
            }
            for row in controllable_parameters
        ]

        workers = healthy_workers(self.simulator_urls, verbose=verbose)
        if not workers:
            raise RuntimeError(
                "Nessun servizio CARLA raggiungibile. Avvia CARLA e poi, es.:\n"
                "  uvicorn Simulator.emergency_braking.SimulatorServer:app --port 8100\n"
                f"URL tentati: {self.simulator_urls}\n"
                "(imposta EB_NUM_WORKERS o EB_SIMULATOR_URLS per cambiare il pool)."
            )
        if verbose:
            ports = ", ".join(u.split(":")[-1] for u in workers)
            print(f"[pool] {len(workers)} worker attivi (porte: {ports})", flush=True)

        def _on_progress(completed: int, total: int, idx: int, url: str, result: dict) -> None:
            if not verbose:
                return
            row = controllable_parameters[idx]
            if result.get("status") != "done":
                print(
                    f"  [{completed:3d}/{total}] (sample #{idx + 1:3d} @ porta {url.split(':')[-1]})"
                    f"  speed={row[0]:.1f}m/s friction={row[1]:.2f} detect={row[2]:.0f}m delay={row[3]:.2f}s"
                    f"  ->  FALLITO: {result.get('error')} — escluso come non valido",
                    flush=True,
                )
                return
            out = result["output"]
            print(
                f"  [{completed:3d}/{total}] (sample #{idx + 1:3d} @ porta {url.split(':')[-1]})"
                f"  speed={row[0]:.1f}m/s friction={row[1]:.2f} detect={row[2]:.0f}m delay={row[3]:.2f}s"
                f"  ->  {out['iterations']} step  collided={out['collided']}",
                flush=True,
            )

        results = run_job_pool(
            payloads, workers, job_timeout=self.job_timeout, on_progress=_on_progress
        )

        # Jobs that came back with status != "done" (rare application-level failure,
        # e.g. a CARLA spawn collision for that sampled combination, or a timeout)
        # are treated as invalid samples — a 0-length trajectory here, turned into a
        # NaN safety margin by EmergencyBrakingScenario.compute_qoi, not counted as
        # either safe or a crash.
        failed = np.array([r.get("status") != "done" for r in results], dtype=bool)
        run_lengths = [
            len(r["output"]["positions"]) if r.get("status") == "done" else 0
            for r in results
        ]
        T_max = max(run_lengths) if any(l > 0 for l in run_lengths) else 1
        trajectories = np.zeros((N, T_max, 2), dtype=np.float32)
        for i, r in enumerate(results):
            n = run_lengths[i]
            if n == 0:
                continue
            out = r["output"]
            trajectories[i, :n, 0] = out["positions"]
            trajectories[i, :n, 1] = out["velocities"]
            if n < T_max:   # freeze final state — same convention as the physics simulator
                trajectories[i, n:, 0] = out["positions"][-1]
                trajectories[i, n:, 1] = out["velocities"][-1]

        self._invalid_mask = failed
        return trajectories

    def ParamBounds(self) -> dict:
        return {
            "lower": np.array([5.0, 0.3, 10.0, 0.05]),
            "upper": np.array([50.0, 1.0, 100.0, 0.50]),
        }
