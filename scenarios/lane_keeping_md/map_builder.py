"""
Map / scenario parameter mapping for the MetaDrive lane-keeping backend.

Pure, dependency-light (numpy only) so it is unit-testable WITHOUT MetaDrive
installed. It translates the SAME 9-parameter vector used by the Unity scenario
(5 angles + min_speed + max_speed + segment_length + map_size) into a
backend-neutral description of the road and the speed band. The MetaDrive glue
(scenarios/lane_keeping_md/config.py) consumes this description at env-build
time and assembles the actual MetaDrive `map_config`.

Design goal: preserve the DISTRIBUTION OF DIFFICULTY (gentle curves frequent,
sharp curves rare), not to replicate the Unity geometry pixel-for-pixel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Curve radius clamps (metres). A near-zero angle -> (near) straight block, so we
# cap the radius at STRAIGHT_RADIUS; a sharp angle -> small radius, capped at
# MIN_RADIUS to stay within a physically drivable / MetaDrive-buildable range.
MIN_RADIUS      = 15.0     # m  — tightest curve
STRAIGHT_RADIUS = 1.0e4    # m  — treated as "straight" above STRAIGHT_ANGLE_EPS
STRAIGHT_ANGLE_EPS = 1.0   # deg — angles below this are straight segments


def angle_to_radius(angle_deg: float, seg_length: float) -> float:
    """
    Map a per-segment turn `angle_deg` over an arc of length `seg_length` to a
    curve radius. Arc geometry: arc_length = radius * theta(rad) => radius =
    seg_length / theta. Larger angle -> smaller radius (sharper curve).

    Returns STRAIGHT_RADIUS for (near) straight segments; clamps to
    [MIN_RADIUS, STRAIGHT_RADIUS].
    """
    a = abs(float(angle_deg))
    if a < STRAIGHT_ANGLE_EPS:
        return STRAIGHT_RADIUS
    theta = math.radians(a)
    radius = float(seg_length) / theta
    return float(np.clip(radius, MIN_RADIUS, STRAIGHT_RADIUS))


@dataclass
class RoadBlock:
    kind: str          # "S" (straight) | "C" (curve)
    radius: float      # m — STRAIGHT_RADIUS for straights
    angle: float       # deg — signed turn (sign = direction)
    direction: int     # +1 (left) / -1 (right) / 0 (straight)
    length: float      # m — segment / arc length


@dataclass
class ScenarioSpec:
    """Backend-neutral description consumed by the MetaDrive config builder."""
    blocks: list = field(default_factory=list)   # list[RoadBlock]
    min_speed: float = 5.0        # m/s — lower bound of the target-speed band
    max_speed: float = 10.0       # m/s — upper bound of the target-speed band
    map_size: float = 250.0       # m — map region side

    @property
    def sharpest_curve_radius(self) -> float:
        radii = [b.radius for b in self.blocks if b.kind == "C"]
        return min(radii) if radii else STRAIGHT_RADIUS

    def block_string(self) -> str:
        """Compact 'S'/'C' sequence (e.g. 'CSCCS'), handy for logging/summaries."""
        return "".join(b.kind for b in self.blocks)


def build_scenario_spec(row: np.ndarray, ncols: int | None = None) -> ScenarioSpec:
    """
    Build a ScenarioSpec from one parameter row.

    Layout (same as LaneKeepingScenario.param_bounds):
      row[0:5] : 5 road angles (deg)         -> 5 curve/straight blocks
      row[5]   : min_speed (m/s)
      row[6]   : max_speed (m/s)
      row[7]   : segment_length (m)          -> arc length per block
      row[8]   : map_size (m)  (optional)
    """
    row = np.asarray(row, dtype=float)
    n = int(ncols) if ncols is not None else row.shape[0]

    seg_length = float(row[7]) if n > 7 else 25.0
    map_size   = float(row[8]) if n > 8 else 250.0
    min_speed  = float(row[5]) if n > 5 else 5.0
    max_speed  = float(row[6]) if n > 6 else 10.0

    blocks: list[RoadBlock] = []
    for j in range(5):
        angle = float(row[j]) if n > j else 0.0
        radius = angle_to_radius(angle, seg_length)
        if radius >= STRAIGHT_RADIUS:
            blocks.append(RoadBlock("S", STRAIGHT_RADIUS, 0.0, 0, seg_length))
        else:
            # Deterministic alternating turn direction so the road actually
            # meanders instead of spiralling in one direction.
            direction = 1 if (j % 2 == 0) else -1
            blocks.append(RoadBlock("C", radius, direction * abs(angle),
                                    direction, seg_length))
    return ScenarioSpec(blocks=blocks, min_speed=min_speed,
                        max_speed=max_speed, map_size=map_size)


def target_speed(spec: ScenarioSpec) -> float:
    """Mid-band cruising target used by the throttle regulator (m/s)."""
    return 0.5 * (spec.min_speed + spec.max_speed)
