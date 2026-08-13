"""
Rigira UNA simulazione con il rendering acceso, per mostrarla.

Le campagne girano headless: del singolo episodio resta il numero, non il
filmato. Questo script riapre il `.npz` di una campagna, sceglie il fallimento
peggiore (margine minimo) di una configurazione, e rigira QUEL solo scenario in
una finestra MetaDrive.

Nota: e' una dimostrazione, non una misura. Il rendering non cambia la fisica
(la frequenza di controllo resta 1/(decision_repeat * physics_world_step_size)),
ma il margine puo' differire nell'ultima cifra rispetto alla campagna se la
versione di MetaDrive installata non e' la stessa.

Uso:
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
    """Come scenario_map.make_online_env, ma con la finestra aperta."""
    from metadrive.engine.asset_loader import AssetLoader
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.scenario_description import ScenarioDescription

    sd = ScenarioDescription(build_scenario_description(row, scenario_id=f"replay_{seed}"))
    env = ScenarioOnlineEnv(dict(
        use_render=True,                 # <-- l'unica differenza
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
    """Identico allo scenario delle campagne: cambia solo l'ambiente costruito."""

    def _make_env(self, spec, seed, row=None):
        if row is None:
            raise ValueError("serve `row`: la strada nasce dai parametri")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--arm", default=None,
                    help="configurazione da cui pescare (default: la prima che contiene 'active_boundary')")
    ap.add_argument("--rank", type=int, default=0,
                    help="0 = il peggiore, 1 = il secondo peggiore, ...")
    ap.add_argument("--speed-scale", type=float, default=0.3625,
                    help="DEVE essere quello della campagna, altrimenti non e' lo stesso esperimento")
    ap.add_argument("--list", action="store_true", help="elenca le configurazioni e esci")
    args = ap.parse_args()

    clouds, names = load(args.npz)

    if args.list:
        print(f"{'configurazione':<40}{'punti':>8}{'fallimenti':>12}{'peggiore':>10}")
        for lab, (th, mg) in clouds.items():
            finite = np.isfinite(mg)
            print(f"{lab:<40}{int(finite.sum()):>8}{int((mg[finite] < 0).sum()):>12}"
                  f"{np.nanmin(mg):>10.3f}")
        return

    arm = args.arm
    if arm is None:
        arm = next((l for l in clouds if "active_boundary" in l), list(clouds)[0])
    if arm not in clouds:
        sys.exit(f"configurazione '{arm}' assente. Disponibili:\n  " + "\n  ".join(clouds))

    theta, margins = clouds[arm]
    order = np.argsort(np.where(np.isfinite(margins), margins, np.inf))
    idx = int(order[args.rank])
    row = theta[idx]

    print(f"configurazione : {arm}")
    print(f"scenario       : indice {idx} nella nuvola, margine {margins[idx]:.4f}")
    if names:
        print("parametri      : " + ", ".join(f"{n}={v:.2f}" for n, v in zip(names, row)))
    else:
        print("parametri      : " + np.array2string(row, precision=2))
    print(f"speed_scale    : {args.speed_scale}")
    print("\nchiudi la finestra per terminare.\n")

    sc = RenderingScenario(speed_scale=args.speed_scale, geometry="udacity", n_jobs=1)
    traj, fid = sc._simulate_one(np.asarray(row, float), ncols=len(row), seed=0, verbose=True)
    print(f"\npassi={traj.shape[0] if hasattr(traj,'shape') else len(traj)}  "
          f"esito={fid.get('outcome')}  "
          f"control_hz={fid.get('control_hz'):.1f}  "
          f"m/step={fid.get('meters_per_step'):.3f}")


if __name__ == "__main__":
    main()
