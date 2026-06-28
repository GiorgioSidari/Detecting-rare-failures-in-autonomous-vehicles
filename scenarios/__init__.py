from __future__ import annotations

"""
Scenario registry.

Any module that needs to work with scenarios imports from here:

    from scenarios import SCENARIOS, get_scenario
    scenario = SCENARIOS["emergency_braking"]

Adding a new scenario = add one entry to SCENARIOS. Nothing else changes.

use_nn flag
-----------
The API server sets `use_nn=True` for emergency_braking once the model file
exists, so callers get the NN-driven simulator automatically.

Multi-model lane-keeping (Step D)
----------------------------------
Two LaneKeepingScenario instances target separate Docker containers on different
ports, each running a different autopilot model.  Start them with:

    docker compose up --build                  # port 8000 — chauffeur model
    docker compose -f docker-compose-ch2.yml up --build  # port 8001 — ch2 model

Then compare failure profiles:

    from pipeline.orchestrator import run
    r1 = run("lane_keeping_chauffeur", n_samples=200)
    r2 = run("lane_keeping_ch2",       n_samples=200)
    print(r1.failure_rate, r2.failure_rate)
"""

import os
from scenarios.emergency_braking import EmergencyBrakingScenario
from scenarios.cut_in import CutInScenario
from scenarios.lane_keeping import LaneKeepingScenario

_NN_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "emergency_braking", "models", "emergency_braking_mlp.keras"
)
_nn_trained = os.path.exists(_NN_MODEL_PATH)

SCENARIOS: dict = {
    # Use NN simulator if the model has been trained, otherwise fall back to physics
    "emergency_braking": EmergencyBrakingScenario(use_nn=_nn_trained),
    "cut_in":            CutInScenario(),

    # Single-model lane-keeping (default port 8000)
    "lane_keeping": LaneKeepingScenario(),

    # Multi-model comparison (Step D) — requires two Docker containers
    "lane_keeping_chauffeur": LaneKeepingScenario(
        simulator_url="http://localhost:8000",
        name_suffix="_chauffeur",
    ),
    "lane_keeping_ch2": LaneKeepingScenario(
        simulator_url="http://localhost:8001",
        name_suffix="_ch2",
    ),
}


def get_scenario(name: str, use_nn: bool | None = None):
    """
    Retrieve a scenario by name with optional use_nn override.

    Parameters
    ----------
    name   : scenario key (e.g. "emergency_braking")
    use_nn : if None, uses the registry default (auto-detected).
             if True/False, creates a fresh instance with that setting.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS)}")
    if use_nn is None:
        return SCENARIOS[name]
    if name == "emergency_braking":
        return EmergencyBrakingScenario(use_nn=use_nn)
    return SCENARIOS[name]
