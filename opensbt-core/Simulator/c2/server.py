"""
FastAPI server of the C2 arm -- a copy of `SimulatorServer.py` with one change.

It is a copy and not a modification of the original because
`SimulatorServer.py` is pre-existing code of the Udacity pipeline: changing it
would also change the C1 arm and the work of whoever uses it.

The only difference is the simulator instantiated (`UdacitySimulatorC2` instead
of `UdacitySimulator`). Contract, queue, job handling and response format are
identical, so the client cannot tell the two arms apart.

It is selected in the compose file without touching the Dockerfile:

    command: uvicorn Simulator.c2.server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dataclasses import asdict
from fastapi.encoders import jsonable_encoder
import numpy as np
from queue import Queue
import uuid
from threading import Thread
from .udacity_simulation_c2 import UdacitySimulatorC2
from UdacitySimulatorIO import UdacitySimulatorConfig, UdacitySimulationOutput


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking
jobQueue = Queue()
results = {}


def simulationThread():
    simulator = UdacitySimulatorC2()

    while True:
        # Fetches a job from the queue
        jobId, config = jobQueue.get(block=True)

        try:
            results[jobId] = {'status' : 'simulating'}

            # Performs the simulation
            simOutput: UdacitySimulationOutput = simulator.simulate(
                simulator_config=config)

            # Parse the output into a json
            output = jsonable_encoder(simOutput, custom_encoder={
                np.float32: float,
                np.float64: float
            })

            # The timing fields (predictSeconds/stepSeconds) may not be DECLARED in the installed
            # version of UdacitySimulatorIO (if the package is pip-installed separately):
            # jsonable_encoder would drop them. Inject them here by reading them from the instance,
            # so they reach the JSON regardless of packaging.
            for _k in ("predictSeconds", "stepSeconds"):
                if _k not in output:
                    _v = getattr(simOutput, _k, None)
                    if _v is not None:
                        output[_k] = float(_v)

            # Add output to results
            results[jobId] = {"status": "done", "output": output}
        except Exception as e:
            # Add error to the results
            results[jobId] = {"status": "error", "error": str(e)}
        finally:
            jobQueue.task_done()


# Start simulator thread
Thread(target=simulationThread, daemon=True).start()


@app.post("/simulate")
async def simulate(config: UdacitySimulatorConfig):
    # Generate a unique jobId
    jobId = str(uuid.uuid4())

    # Set the status as queued
    results[jobId] = {"status": "queued"}

    # Enqueues the job
    jobQueue.put((jobId, config))

    # Returns the job id
    return {'jobId': jobId}


@app.get("/simulate/{job_id}")
def get_result(job_id: str):
    return results.get(job_id, {"status": "not_found"})


@app.get("/health")
def health_check():
    return {"status": "Up and running"}
