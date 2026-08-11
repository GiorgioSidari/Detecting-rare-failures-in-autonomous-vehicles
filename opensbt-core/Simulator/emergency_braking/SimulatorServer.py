from __future__ import annotations

"""
Job-queue FastAPI service for the emergency_braking CARLA scenario — same
POST /simulate + GET /simulate/{id} contract as
opensbt-core/Simulator/SimulatorServer.py (lanekeeping/Udacity), so
scenarios/emergency_braking/nn_simulator.py can reuse
simulators/common/http_worker_pool.py unchanged.

One process = one CARLA connection = one simulation at a time (CARLA, like
the Unity Udacity binary, is not thread-safe); parallelism is achieved by
running several CARLA server + this service pairs on different ports (Fase
4 — Dockerization, mirrors opensbt-core/gen_parallel_compose.py).

Run (after training a model with scenarios/emergency_braking/train_cnn.py):
    CARLA_HOST=localhost CARLA_PORT=2000 \
    EB_MODEL_PATH=scenarios/emergency_braking/models/emergency_braking_cnn.h5 \
    uvicorn Simulator.emergency_braking.SimulatorServer:app --host 0.0.0.0 --port 8100
"""

import os
import uuid
from queue import Queue
from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .carla_episode import EmergencyBrakingCarlaEpisode
from .cnn_agent import CnnAgent

CARLA_HOST = os.environ.get("CARLA_HOST", "localhost")
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
MODEL_PATH = os.environ.get(
    "EB_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "scenarios", "emergency_braking", "models",
        "emergency_braking_cnn.h5"),
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobQueue: Queue = Queue()
results: dict = {}


class EmergencyBrakingConfig(BaseModel):
    initial_speed: float
    friction_coefficient: float
    detection_distance: float
    nominal_delay: float


def _connect_with_retry(episode: EmergencyBrakingCarlaEpisode, attempts: int = 6, delay: float = 5.0) -> None:
    """
    CARLA can take longer than one client-side timeout to finish booting/loading
    the map. If this (daemon) thread dies here, the service keeps answering
    /health OK forever while every queued job silently times out — so retry
    instead of raising on the first attempt.
    """
    import time as _time
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            episode.connect()
            return
        except Exception as e:  # noqa: BLE001
            last_exc = e
            print(f"[emergency_braking] connessione a CARLA fallita (tentativo {attempt}/{attempts}): {e}",
                  flush=True)
            _time.sleep(delay)
    raise RuntimeError(f"Impossibile connettersi a CARLA dopo {attempts} tentativi") from last_exc


def simulationThread() -> None:
    episode = EmergencyBrakingCarlaEpisode(host=CARLA_HOST, port=CARLA_PORT)
    _connect_with_retry(episode)
    agent = CnnAgent(MODEL_PATH)

    while True:
        jobId, params = jobQueue.get(block=True)
        try:
            results[jobId] = {"status": "simulating"}
            out = episode.run_episode(params, agent, record_frames=True)
            results[jobId] = {
                "status": "done",
                "output": {
                    "positions": out["positions"],
                    "velocities": out["velocities"],
                    "elapsedTime": out["elapsedTime"],
                    "iterations": out["iterations"],
                    "collided": out["collided"],
                },
            }
        except Exception as e:
            results[jobId] = {"status": "error", "error": str(e)}
        finally:
            jobQueue.task_done()


Thread(target=simulationThread, daemon=True).start()


@app.post("/simulate")
async def simulate(config: EmergencyBrakingConfig):
    jobId = str(uuid.uuid4())
    results[jobId] = {"status": "queued"}
    jobQueue.put((
        jobId,
        [config.initial_speed, config.friction_coefficient,
         config.detection_distance, config.nominal_delay],
    ))
    return {"jobId": jobId}


@app.get("/simulate/{job_id}")
def get_result(job_id: str):
    return results.get(job_id, {"status": "not_found"})


@app.get("/health")
def health_check():
    return {"status": "Up and running"}
