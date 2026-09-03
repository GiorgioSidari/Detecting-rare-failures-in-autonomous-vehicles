"""
Map and scenario parameter mapping for the MetaDrive lane-keeping backend.

Translates the 9-parameter vector used by the Unity scenario into a
backend-neutral `ScenarioSpec`: five `RoadBlock`s (straight or curve, with
radius, signed turn angle and length), the speed band and the map size. Pure
numpy, so it is testable without MetaDrive.

`angle_to_radius` converts a turn angle over an arc of `seg_length` into a curve
radius (`radius = seg_length / theta`), treating angles below
`STRAIGHT_ANGLE_EPS` as straight and clamping the result to
`[MIN_RADIUS, STRAIGHT_RADIUS]`. `target_speed` returns the midpoint of the
band.

`scenarios/lane_keeping_md/config.py` builds the road from the Udacity
centreline (`scenario_map.py`) and reads only the speed band from the spec; the
block sequence is used by the legacy PGBlock path and by `block_string()` for
logging.
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
    Turn angle over an arc of `seg_length` -> curve radius: arc = radius * theta,
    so radius = seg_length / theta. Larger angle, tighter curve. Near-straight
    below STRAIGHT_ANGLE_EPS; clamped to [MIN_RADIUS, STRAIGHT_RADIUS].
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
        """Radius of the tightest curve, or STRAIGHT_RADIUS if there is none."""
        radii = [b.radius for b in self.blocks if b.kind == "C"]
        return min(radii) if radii else STRAIGHT_RADIUS

    def block_string(self) -> str:
        """
        Compact 'S'/'C' sequence, e.g. "CSCCS". Used for logging and for the legacy
        map config.
        """
        return "".join(b.kind for b in self.blocks)


def build_scenario_spec(row: np.ndarray, ncols: int | None = None) -> ScenarioSpec:
    """
    One parameter row -> ScenarioSpec.

    row[0:5]  angles (deg)      row[7]  segment length (m)
    row[5:7]  speed band (m/s)  row[8]  map size (m)

    Turn direction alternates deterministically so the road meanders instead of
    spiralling in one direction.
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
    """Mid-band cruising target for the throttle regulator (m/s)."""
    return 0.5 * (spec.min_speed + spec.max_speed)
