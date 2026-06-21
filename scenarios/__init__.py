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
    "lane_keeping":      LaneKeepingScenario(),
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
