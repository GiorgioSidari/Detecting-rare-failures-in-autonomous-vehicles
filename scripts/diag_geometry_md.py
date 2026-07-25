"""
Diagnostic: do our 'angle' parameters actually change the MetaDrive road?

Symptom found by the active-boundary run: the 5 angle parameters have ~0
importance. From the code, `_build_md_config` only passes `"map": "CCCCC"` (block
TYPE), never a radius — so MetaDrive picks curve radii at random per seed and our
angle magnitudes never reach the road. This script confirms that (same seed,
gentle vs sharp angles -> compare the generated curves) and introspects the map
blocks to reveal the radius API, so the real fix can target the installed version.

Run (needs MetaDrive):
    python scripts/diag_geometry_md.py
"""
from __future__ import annotations

import numpy as np

from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario
from scenarios.lane_keeping_md.map_builder import build_scenario_spec


def _get_map(env):
    m = getattr(env, "current_map", None)
    if m is not None:
        return m
    eng = getattr(env, "engine", None)
    if eng is not None:
        m = getattr(eng, "current_map", None)
        if m is not None:
            return m
        mm = getattr(eng, "map_manager", None)
        if mm is not None:
            return getattr(mm, "current_map", None)
    return None


def introspect(seed: int, row: np.ndarray, label: str) -> None:
    sc = LaneKeepingMetaDriveScenario(max_steps=3)
    spec = build_scenario_spec(row, len(row))
    env = sc._make_env(spec, seed)
    try:
        env.reset()
        print(f"\n=== {label}   map_string={spec.block_string()} (seed={seed}) ===")
        try:
            print("  env map_config:", env.config.get("map_config"))
        except Exception as e:
            print("  (map_config read failed:", e, ")")
        m = _get_map(env)
        print("  map type:", type(m).__name__ if m is not None else None)
        blocks = getattr(m, "blocks", None)
        if blocks is None:
            print("  no .blocks; map attrs:",
                  [a for a in dir(m) if not a.startswith("_")][:40])
        else:
            for i, b in enumerate(blocks):
                cand = [a for a in dir(b)
                        if ("radi" in a.lower() or "angle" in a.lower()) and not a.startswith("__")]
                print(f"  block[{i}] {type(b).__name__}  radius={getattr(b, 'radius', None)}  "
                      f"cand_attrs={cand}")
                if i == 1:  # a curve block: show its parameter space / config
                    ps = getattr(b, "PARAMETER_SPACE", None)
                    print("    PARAMETER_SPACE:", ps)
                    print("    _config:", getattr(b, "_config", None))
        # Road length as a difficulty proxy.
        try:
            lanes = m.get_lanes() if hasattr(m, "get_lanes") else []
            total = sum(getattr(l, "length", 0.0) for l in lanes)
            print(f"  total lane length ~ {total:.1f} m over {len(lanes)} lanes")
        except Exception as e:
            print("  (lane length introspection failed:", e, ")")
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    gentle = np.array([5., 5, 5, 5, 5, 8, 10, 25, 250])
    sharp = np.array([80., 80, 80, 80, 80, 8, 10, 25, 250])
    introspect(0, gentle, "GENTLE angles=5")
    introspect(0, sharp, "SHARP  angles=80")
    print("\n>>> If blocks/radii/lengths are IDENTICAL between GENTLE and SHARP at the")
    print(">>> same seed, the angles are inert (confirmed). The 'radius'/PARAMETER_SPACE")
    print(">>> attributes show how to wire the real radius from the angles.")


if __name__ == "__main__":
    main()
