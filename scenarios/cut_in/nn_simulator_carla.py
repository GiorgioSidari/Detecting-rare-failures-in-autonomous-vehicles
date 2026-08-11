from __future__ import annotations

"""
Cut-In simulator driven by the CARLA-trained camera CNN (Fase 3 del piano
video-CNN) — mirrors scenarios/emergency_braking/nn_simulator_carla.py.

Talks to the CARLA job-queue service
(opensbt-core/Simulator/cut_in/SimulatorServer.py) via
simulators/common/http_worker_pool.py.
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
DEFAULT_BASE_PORT = 8200   # separate range from lanekeeping (8000+) / emergency_braking (8100+)


class CutInCarlaNNSimulator(BaseSimulator):
    """
    Drop-in replacement for CutInSimulator that runs the scenario inside CARLA
    with the trained CNN driving the ego (braking-only response, same scope
    as the analytic baseline in scenarios/cut_in/simulator.py — no evasive
    steering).
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
            num_workers_env="CI_NUM_WORKERS",
            base_port_env="CI_SIMULATOR_BASE_PORT",
            urls_env="CI_SIMULATOR_URLS",
        )
        self.job_timeout = job_timeout

    def run(self, controllable_parameters: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        controllable_parameters : (N, 4)
            [ego_speed (m/s), cutter_speed (m/s), lateral_gap (m), reaction_delay (s)]
            (reaction_delay now shifts the cutter's merge onset — see
            opensbt-core/Simulator/cut_in/carla_episode.py)

        Returns trajectories (N, T, 2): [longitudinal_gap (m), lateral_gap (m)],
        matching scenarios/cut_in/qoi.py::compute_min_gap's expected format.
        """
        N = controllable_parameters.shape[0]
        payloads = [
            {
                "ego_speed": float(row[0]),
                "cutter_speed": float(row[1]),
                "lateral_gap": float(row[2]),
                "reaction_delay": float(row[3]),
            }
            for row in controllable_parameters
        ]

        workers = healthy_workers(self.simulator_urls, verbose=verbose)
        if not workers:
            raise RuntimeError(
                "Nessun servizio CARLA raggiungibile. Avvia CARLA e poi, es.:\n"
                "  uvicorn Simulator.cut_in.SimulatorServer:app --port 8200\n"
                f"URL tentati: {self.simulator_urls}\n"
                "(imposta CI_NUM_WORKERS o CI_SIMULATOR_URLS per cambiare il pool)."
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
                    f"  ego={row[0]:.1f}m/s cutter={row[1]:.1f}m/s gap0={row[2]:.1f}m delay={row[3]:.2f}s"
                    f"  ->  FALLITO: {result.get('error')} — escluso come non valido",
                    flush=True,
                )
                return
            out = result["output"]
            print(
                f"  [{completed:3d}/{total}] (sample #{idx + 1:3d} @ porta {url.split(':')[-1]})"
                f"  ego={row[0]:.1f}m/s cutter={row[1]:.1f}m/s gap0={row[2]:.1f}m delay={row[3]:.2f}s"
                f"  ->  {out['iterations']} step  collided={out['collided']}",
                flush=True,
            )

        results = run_job_pool(
            payloads, workers, job_timeout=self.job_timeout, on_progress=_on_progress
        )

        # See scenarios/emergency_braking/nn_simulator_carla.py for why failed jobs
        # become an invalid (NaN-margin) sample instead of aborting the whole batch.
        failed = np.array([r.get("status") != "done" for r in results], dtype=bool)
        run_lengths = [
            len(r["output"]["longitudinal_gaps"]) if r.get("status") == "done" else 0
            for r in results
        ]
        T_max = max(run_lengths) if any(l > 0 for l in run_lengths) else 1
        trajectories = np.zeros((N, T_max, 2), dtype=np.float32)
        for i, r in enumerate(results):
            n = run_lengths[i]
            if n == 0:
                continue
            out = r["output"]
            trajectories[i, :n, 0] = out["longitudinal_gaps"]
            trajectories[i, :n, 1] = out["lateral_gaps"]
            if n < T_max:
                trajectories[i, n:, 0] = out["longitudinal_gaps"][-1]
                trajectories[i, n:, 1] = out["lateral_gaps"][-1]

        self._invalid_mask = failed
        return trajectories

    def ParamBounds(self) -> dict:
        return {
            "lower": np.array([10.0, 10.0, 1.0, 0.05]),
            "upper": np.array([40.0, 40.0, 4.0, 0.80]),
        }
