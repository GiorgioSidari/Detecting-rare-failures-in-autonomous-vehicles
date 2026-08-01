"""
FastAPI server — unified entry point for all scenarios.

Endpoints
---------
GET  /scenarios              → list of available scenarios + their param metadata
POST /run/{scenario_name}    → launch pipeline, returns job_id immediately
GET  /status/{job_id}        → poll for results (queued | running | done | error)
POST /explain/{job_id}       → call LLM to summarise rare failures (returns text)

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import RunRequest, ScenarioInfo, StatusResponse, ExplainResponse
from scenarios import SCENARIOS
from pipeline.severity import summarise_rare_params
import pipeline.orchestrator as orchestrator

app = FastAPI(title="Rare Failure Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store  {job_id: {"status": ..., "result": ...}}
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# GET /scenarios
# ---------------------------------------------------------------------------

@app.get("/scenarios", response_model=list[ScenarioInfo])
def list_scenarios():
    """Return metadata for all registered scenarios."""
    out = []
    for key, scenario in SCENARIOS.items():
        bounds = scenario.param_bounds()
        out.append(ScenarioInfo(
            name=key,
            description=scenario.description,
            param_names=bounds["names"],
            param_lower=bounds["lower"].tolist(),
            param_upper=bounds["upper"].tolist(),
        ))
    return out


# ---------------------------------------------------------------------------
# POST /run/{scenario_name}
# ---------------------------------------------------------------------------

@app.post("/run/{scenario_name}")
def run_scenario(scenario_name: str, body: RunRequest):
    """
    Launch the pipeline asynchronously.
    Returns job_id immediately; poll /status/{job_id} for results.
    """
    if scenario_name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {scenario_name}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    def _run():
        _jobs[job_id]["status"] = "running"
        try:
            result = orchestrator.run(
                scenario_name=scenario_name,
                n_samples=body.n_samples,
                seed=body.seed,
                rare_fraction=body.rare_fraction,
                param_lower=body.param_lower,
                param_upper=body.param_upper,
            )
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
        except Exception as exc:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)

    _executor.submit(_run)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    """Poll pipeline status. When done, returns serialised results."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    status = job["status"]

    if status != "done":
        return StatusResponse(status=status, error=job.get("error"))

    result = job["result"]
    bounds = SCENARIOS[result.scenario_name].param_bounds()
    rare_summary = summarise_rare_params(result.rare_params, bounds["names"])

    return StatusResponse(
        status="done",
        scenario_name=result.scenario_name,
        n_samples=result.n_samples,
        failure_rate=result.failure_rate,
        rare_failure_rate=result.rare_failure_rate,
        pod_n_modes=result.pod_n_modes,
        param_names=bounds["names"],
        # Trajectories serialised as nested lists for JSON
        trajectories=result.trajectories.tolist(),
        safe_mask=(result.failures == 0).tolist(),
        rare_failure_idx=result.rare_failure_idx.tolist(),
        nominal_trajectory=result.nominal_trajectory.tolist(),
        rare_params_summary=rare_summary,
        safety_margins=result.safety_margins.tolist(),
    )


# ---------------------------------------------------------------------------
# POST /explain/{job_id}
# ---------------------------------------------------------------------------

@app.post("/explain/{job_id}", response_model=ExplainResponse)
def explain(job_id: str):
    """
    Call an LLM to generate a natural-language summary of the rare failures.

    TODO: implement LLM call (Anthropic / OpenAI API).
    The prompt should include:
      - scenario name and description
      - rare failure parameter table
      - failure rate
    """
    if job_id not in _jobs or _jobs[job_id]["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not done yet")

    result = _jobs[job_id]["result"]
    bounds = SCENARIOS[result.scenario_name].param_bounds()
    rare_summary = summarise_rare_params(result.rare_params, bounds["names"])

    # TODO: replace with real LLM call
    placeholder = (
        f"[LLM summary placeholder]\n"
        f"Scenario: {result.scenario_name}\n"
        f"Failure rate: {result.failure_rate:.1%}\n"
        f"Rare failures ({len(result.rare_failure_idx)} cases):\n"
        + "\n".join(str(r) for r in rare_summary)
    )

    return ExplainResponse(summary=placeholder)


@app.get("/health")
def health():
    return {"status": "ok"}
