"""
MetaDrive map built from **Udacity's** centreline.

The problem: `map_builder` translated theta into MetaDrive PGBlocks, a different
parameterisation, and `config._apply_curve_geometry` collapsed the 5 angles into
their mean and left the turn direction to the seed. On the 60-scenario LHS
design the road lengths came out 202.7 m against 124.0 m, Spearman -0.026: at
the same theta the backends drove uncorrelated roads, and the cross-simulator
comparison was measuring that rather than the simulators.

The fix: MetaDrive 0.4.3 exposes `ScenarioOnlineEnv`, which takes an in-memory
ScenarioNet scenario whose lanes are explicit polylines -- so it can be handed
Udacity's Catmull-Rom centreline directly, with no arc approximation and no
random directions.

Measured parity, 400 points on the reconstructed lane, point-to-SEGMENT
distance: 0.021 cm mean, 1.88 cm worst, 202.55 m against 202.61 m. Head and tail
2 m excluded, where MetaDrive extrapolates (14 cm at s=0.5 m). Measuring
point-to-VERTEX instead inflates it to ~34 cm, because the vertices are 1.34 m
apart -- an easy mistake that makes a correct alignment look broken.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from scenarios.common.road_geometry import road_polyline

# Compact-saloon dimensions (m). They only populate the ego track, from which
# MetaDrive derives the route; the actual vehicle is MetaDrive's.
_LENGTH, _WIDTH, _HEIGHT = 4.6, 1.85, 1.37

# Nominal time step of the track: makes the declared speeds consistent with the
# positions. It does NOT govern the simulation.
_TS_DT = 0.1


"""
Udacity's Catmull-Rom centreline for `row`, as (M, 2). The single geometric
source of truth: the same function feeds both backends.
"""
def centerline(row: np.ndarray) -> np.ndarray:
    return np.asarray(road_polyline(np.asarray(row, dtype=float)), dtype=np.float64)


"""
theta -> a MetaDrive ScenarioDescription with ONE lane: the centreline.
tracks["ego"] is not replayed -- the policy is ours -- but MetaDrive derives
the ROUTE from its first and last point, so it must span the road; headings
come from np.gradient, which gives one for every point including the last.
MetaDrive recentres on the ego's start, so the map comes out translated by
-centerline[0], which the XTE measurement has to account for.
"""
def build_scenario_description(row: np.ndarray, *, scenario_id: str = "lk") -> Dict[str, Any]:
    poly = centerline(row)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(f"centerline degenere: shape {poly.shape}")

    n = len(poly)
    tang = np.gradient(poly, axis=0)
    heading = np.arctan2(tang[:, 1], tang[:, 0])

    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, :2] = poly

    ego = {
        "type": "VEHICLE",
        "state": {
            "position": pos,
            "heading": heading,
            "velocity": tang / _TS_DT,
            "valid": np.ones(n, dtype=bool),
            "length": np.full(n, _LENGTH),
            "width": np.full(n, _WIDTH),
            "height": np.full(n, _HEIGHT),
        },
        "metadata": {"type": "VEHICLE", "track_length": n, "object_id": "ego"},
    }

    return {
        "id": scenario_id,
        "version": "0.4.3",
        "length": n,
        "tracks": {"ego": ego},
        "dynamic_map_states": {},
        "map_features": {
            "lane_0": {"type": "LANE_SURFACE_STREET", "polyline": poly},
        },
        "metadata": {
            "sdc_id": "ego",
            "coordinate": "metadrive",
            "ts": np.arange(n, dtype=np.float64) * _TS_DT,
            "metadrive_processed": True,
            "dataset": "lanekeeping",
            "scenario_id": scenario_id,
            "source_file": "synthetic",
            "track_length": n,
            "object_summary": {},
            "number_summary": {},
        },
    }


"""
Builds and prepares a ScenarioOnlineEnv on the centreline of `row`: headless,
traffic-free, driven by our own policy rather than a replay one. MetaDrive is
imported lazily so this module and the geometry tests stay usable without it;
`data_directory` must exist, and `set_scenario` must precede `reset`.
"""
def make_online_env(row: np.ndarray, *, decision_repeat: int,
                    physics_world_step_size: float, max_steps: int,
                    seed: int = 0):
    from metadrive.engine.asset_loader import AssetLoader
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription

    sd = ScenarioDescription(build_scenario_description(row, scenario_id=f"lk_{seed}"))

    env = ScenarioOnlineEnv(dict(
        use_render=False,
        image_observation=False,
        agent_policy=EnvInputPolicy,
        data_directory=AssetLoader.file_path("nuscenes", unix_style=False),
        num_scenarios=1,
        start_scenario_index=0,
        sequential_seed=True,
        physics_world_step_size=physics_world_step_size,
        decision_repeat=decision_repeat,
        horizon=max_steps,
        no_traffic=True,
        no_light=True,
        no_static_vehicles=True,
    ))
    env.set_scenario(sd)
    return env


"""
The lane MetaDrive actually built, for the parity check. Scenario blocks do
not expose `.lanes` like PGBlocks, so access goes through EdgeRoadNetwork.
"""
def lane_of(env):
    net = env.current_map.road_network
    entries = list(net.graph.values())
    if not entries:
        raise RuntimeError("no lane in the road network")
    return getattr(entries[0], "lane", entries[0])
