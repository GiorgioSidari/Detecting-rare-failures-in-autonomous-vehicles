"""
Replay ONE simulation with rendering on, so it can be watched.

Campaigns run headless: of a single episode only the number survives, not the
footage. This script reopens a campaign's `.npz`, picks one arm's worst failure
(lowest margin) and replays THAT scenario alone in a MetaDrive window.

It is a demonstration, not a measurement. Rendering does not change the physics
(the control rate stays 1/(decision_repeat * physics_world_step_size)), but the
margin can differ in the last digit from the campaign if the installed MetaDrive
version is not the same.

Usage:
    python scripts/replay_worst.py results/cmp_md12_full_raw.npz
    python scripts/replay_worst.py results/cmp_md12_full_raw.npz --arm active_boundary[lhs]
    python scripts/replay_worst.py results/cmp_md12_full_raw.npz --list
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.common.episode_budget import budget_steps, polyline_length      # noqa: E402
from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario      # noqa: E402
from scenarios.lane_keeping_md.map_builder import target_speed                 # noqa: E402
from scenarios.lane_keeping_md.scenario_map import (                           # noqa: E402
    build_scenario_description, centerline)


def make_online_env_render(row, *, decision_repeat, physics_world_step_size,
                           max_steps, seed=0):
    """Same as scenario_map.make_online_env, but with the window open."""
    from metadrive.engine.asset_loader import AssetLoader
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription

    sd = ScenarioDescription(build_scenario_description(row, scenario_id=f"replay_{seed}"))
    env = ScenarioOnlineEnv(dict(
        use_render=True,                 # <-- the only difference
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


class RenderingScenario(LaneKeepingMetaDriveScenario):

    def _make_env(self, spec, seed, row=None):
        """Identical to the campaign scenario: only the environment differs."""
        if row is None:
            raise ValueError("`row` is required: the road comes from the parameters")
        poly = centerline(row)
        self._local_centerline = poly - poly[0]
        self._budget_steps = budget_steps(
            polyline_length(self._local_centerline),
            target_speed(spec) * self.speed_scale,
            self.control_hz_nominal)
        return make_online_env_render(
            row, decision_repeat=self.decision_repeat,
            physics_world_step_size=self.physics_world_step_size,
            max_steps=self._budget_steps, seed=seed)


def load(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(x) for x in z["labels"]]
    names = [str(x) for x in z["param_names"]] if "param_names" in z else None
    out = {}
    for i, lab in enumerate(labels):
        out[lab] = (z[f"theta_{i}"], z[f"margins_{i}"])
    return out, names


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--arm", default=None,
                    help="arm to pick from (default: the first one whose label\n"
                         "contains 'active_boundary')")
    ap.add_argument("--rank", type=int, default=0,
                    help="0 = the worst, 1 = the second worst, ...")
    ap.add_argument("--speed-scale", type=float, default=0.3625,
                    help="MUST match the campaign, otherwise this is a different\n"
                         "experiment")
    ap.add_argument("--list", action="store_true", help="list the arms and exit")
    return ap


def main():
    ap = _build_parser()
    args = ap.parse_args()

    clouds, names = load(args.npz)

    if args.list:
        print(f"{'arm':<40}{'points':>8}{'failures':>12}{'worst':>10}")
        for lab, (th, mg) in clouds.items():
            finite = np.isfinite(mg)
            print(f"{lab:<40}{int(finite.sum()):>8}{int((mg[finite] < 0).sum()):>12}"
                  f"{np.nanmin(mg):>10.3f}")
        return

    arm = args.arm
    if arm is None:
        arm = next((l for l in clouds if "active_boundary" in l), list(clouds)[0])
    if arm not in clouds:
        sys.exit(f"arm '{arm}' not in this file. Available:\n  " + "\n  ".join(clouds))

    theta, margins = clouds[arm]
    order = np.argsort(np.where(np.isfinite(margins), margins, np.inf))
    idx = int(order[args.rank])
    row = theta[idx]

    print(f"arm        : {arm}")
    print(f"scenario   : index {idx} in the cloud, margin {margins[idx]:.4f}")
    if names:
        print("parameters : " + ", ".join(f"{n}={v:.2f}" for n, v in zip(names, row)))
    else:
        print("parameters : " + np.array2string(row, precision=2))
    print(f"speed_scale: {args.speed_scale}")
    print("\nclose the window to stop.\n")

    sc = RenderingScenario(speed_scale=args.speed_scale, geometry="udacity", n_jobs=1)
    traj, fid = sc._simulate_one(np.asarray(row, float), ncols=len(row), seed=0, verbose=True)
    print(f"\nsteps={traj.shape[0] if hasattr(traj,'shape') else len(traj)}  "
          f"esito={fid.get('outcome')}  "
          f"control_hz={fid.get('control_hz'):.1f}  "
          f"m/step={fid.get('meters_per_step'):.3f}")


if __name__ == "__main__":
    main()
