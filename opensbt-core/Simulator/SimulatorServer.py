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
from .lanekeeping import UdacitySimulator
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
    simulator = UdacitySimulator()

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
