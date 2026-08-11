from __future__ import annotations

"""
CARLA episode runner for the cut_in scenario (Fase 3 del piano video-CNN),
the counterpart of opensbt-core/Simulator/emergency_braking/carla_episode.py.

Design reference: srunner/scenarios/cut_in.py (scenario_runner v0.9.16) —
same idea (a vehicle cuts in front of the ego on a highway), scripted
directly against the CARLA API for the reasons explained in
emergency_braking/carla_episode.py's module docstring.

Hybrid physics: the ego is a real CARLA vehicle, driven by the controller
(expert or CNN) through brake/throttle — CARLA's physics engine handles its
motion. The cutter is driven KINEMATICALLY (its transform is set directly
every tick from closed-form equations) using the EXACT SAME formulas as
scenarios/cut_in/simulator.py (sigmoid lateral merge, constant longitudinal
speed): this keeps the (longitudinal_gap, lateral_gap) bookkeeping exactly
consistent with the analytic baseline and scenarios/cut_in/qoi.py, while
still rendering a real vehicle for the ego's camera to see merging in.

Parameter reinterpretation vs. scenarios/cut_in/simulator.py
----------------------------------------------------------------
ego_speed, cutter_speed, lateral_gap keep their meaning. `reaction_delay`
changes meaning exactly like emergency_braking's nominal_delay: the ego's
reaction is now emergent (from what the controller perceives), so
reaction_delay instead shifts WHEN the cutter starts its lateral merge
(effective_T0 = CUTIN_T0 + reaction_delay) — a property of the hazard, not
of the ego. The expert (see expert_controller) reacts near-instantly once
the cutter has actually entered the lane.
"""

import math
import numpy as np
import carla

TOWN = "Town04"
FIXED_DELTA = 0.05
T_MAX = 10.0
MAX_STEPS = int(T_MAX / FIXED_DELTA)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 160
STOP_SPEED = 0.3
CRUISE_THROTTLE = 0.6
SETTLE_TICKS = 25
SPIN_UP_MAX_TICKS = 300

# Same constants as scenarios/cut_in/simulator.py — kept in sync so the
# CARLA episode and the analytic baseline describe the identical maneuver.
IMPACT_THRESHOLD = 1.5
LANE_WIDTH = 3.5
INITIAL_LONG_GAP = 15.0
CUTIN_T0 = 1.5
CUTIN_STEEPNESS = 2.5


STEER_LOOKAHEAD = 6.0   # metres — waypoint lookahead for the lane-keeping steer below


def _speed(actor: "carla.Actor") -> float:
    v = actor.get_velocity()
    return (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5


def _lane_keeping_steer(carla_map: "carla.Map", vehicle: "carla.Actor", lookahead: float = STEER_LOOKAHEAD) -> float:
    """
    Minimal proportional heading controller that keeps the ego centred in its
    lane. Needed because the ego's own controller (expert or CNN) only ever
    decides braking_force — steer was previously always 0, which is only
    safe on a perfectly straight road. Town04's spawn_points[0] is followed
    by a highway curve; a few hundred metres of throttle-only driving with
    steer=0 runs straight into a static.guardrail there (confirmed live: the
    ego's speed gets pinned near 0 for the rest of the episode, producing a
    degenerate ~1-step trajectory). Lane-keeping is out of scope for what the
    CNN/expert is trained to decide (braking-only, see module docstring), so
    this stays a simple built-in controller, not something learned.
    """
    transform = vehicle.get_transform()
    loc = transform.location
    wp = carla_map.get_waypoint(loc)
    next_wps = wp.next(lookahead)
    target_loc = next_wps[0].transform.location if next_wps else wp.transform.location
    target_yaw = math.degrees(math.atan2(target_loc.y - loc.y, target_loc.x - loc.x))
    yaw_error = (target_yaw - transform.rotation.yaw + 180.0) % 360.0 - 180.0
    return max(-1.0, min(1.0, yaw_error / 45.0))


def _spawn_with_retry(world: "carla.World", blueprint, transform: "carla.Transform", attempts: int = 4):
    """
    world.spawn_actor() raises RuntimeError when the transform overlaps map
    geometry (e.g. the cutter's scripted spawn point, which depends on the
    sampled lateral_gap, occasionally lands on a barrier/sign in Town04).
    A bare retry with the SAME transform fails identically every time, so we
    nudge the spawn point (mostly upward, a little sideways) between attempts
    — this is what actually resolves it, not just retrying. Used for both
    actors: cheap insurance against the same class of failure for the ego too.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            t = carla.Transform(
                carla.Location(
                    x=transform.location.x + 0.15 * attempt,
                    y=transform.location.y,
                    z=transform.location.z + 0.5 * attempt,
                ),
                transform.rotation,
            )
            return world.spawn_actor(blueprint, t)
        except RuntimeError as e:
            last_exc = e
    raise last_exc


class CutInCarlaEpisode:
    def __init__(self, host: str = "localhost", port: int = 2000, town: str = TOWN):
        self.host = host
        self.port = port
        self.town = town
        self.client: carla.Client | None = None
        self.world: carla.World | None = None

    def connect(self, timeout: float = 20.0) -> None:
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        if self.world.get_map().name.split("/")[-1] != self.town:
            self.world = self.client.load_world(self.town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)

    def close(self) -> None:
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def run_episode(self, params, controller_fn, record_frames: bool = True) -> dict:
        """
        Parameters
        ----------
        params : [ego_speed, cutter_speed, lateral_gap, reaction_delay]
        controller_fn(frame, ego_speed_mps, elapsed_s, cutter_in_lane) -> braking_force [0,1]

        Returns
        -------
        dict with longitudinal_gaps/lateral_gaps (per control step), elapsedTime,
        iterations, collided — [longitudinal_gap, lateral_gap] matches
        scenarios/cut_in/qoi.py::compute_min_gap's expected trajectory format.
        """
        ego_speed_target, cutter_speed, lateral_gap0, reaction_delay = (float(p) for p in params)
        effective_t0 = CUTIN_T0 + reaction_delay

        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        carla_map = self.world.get_map()
        spawn_points = carla_map.get_spawn_points()
        ego_spawn = spawn_points[0]
        forward = ego_spawn.get_forward_vector()
        right = ego_spawn.get_right_vector()

        # All spawns happen inside the try so a failure partway through (e.g. the
        # cutter's scripted spawn point landing on map geometry) still destroys
        # whatever DID spawn — an orphaned ego left behind would permanently block
        # spawn_points[0] for every later episode in the same CARLA session.
        ego = None
        cutter = None
        camera = None
        frames: list = []
        try:
            ego = _spawn_with_retry(self.world, vehicle_bp, ego_spawn)

            # Cutter spawns INITIAL_LONG_GAP ahead, lateral_gap0 to the side (adjacent
            # lane); its transform is fully scripted (see module docstring), so physics
            # simulation is disabled for it.
            cutter_bp = bp_lib.filter("vehicle.audi.a2")[0]
            cutter_start = carla.Location(
                x=ego_spawn.location.x + forward.x * INITIAL_LONG_GAP + right.x * lateral_gap0,
                y=ego_spawn.location.y + forward.y * INITIAL_LONG_GAP + right.y * lateral_gap0,
                z=ego_spawn.location.z + 0.3,
            )
            cutter = _spawn_with_retry(
                self.world, cutter_bp, carla.Transform(cutter_start, ego_spawn.rotation)
            )
            cutter.set_simulate_physics(False)

            if record_frames:
                camera_bp = bp_lib.find("sensor.camera.rgb")
                camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
                camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
                camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
                camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
                camera.listen(lambda image: frames.append(image))

            for _ in range(SETTLE_TICKS):
                self.world.tick()

            for _ in range(SPIN_UP_MAX_TICKS):
                ego_speed = _speed(ego)
                if ego_speed >= ego_speed_target:
                    break
                steer = _lane_keeping_steer(carla_map, ego)
                ego.apply_control(carla.VehicleControl(throttle=1.0, steer=steer))
                self.world.tick()

            longitudinal_gaps: list[float] = []
            lateral_gaps: list[float] = []
            collided = False
            cutter_in_lane = False
            ego_start_loc = ego.get_transform().location

            ego.apply_control(carla.VehicleControl(throttle=CRUISE_THROTTLE))

            for step in range(MAX_STEPS):
                t = step * FIXED_DELTA

                ego_speed = _speed(ego)
                ego_loc = ego.get_transform().location
                ego_traveled = ego_start_loc.distance(ego_loc)

                # ── Cutter: scripted sigmoid lateral merge + constant longitudinal speed ──
                lateral_gap = lateral_gap0 / (1.0 + np.exp(CUTIN_STEEPNESS * (t - effective_t0)))
                cutter_long = INITIAL_LONG_GAP + cutter_speed * t
                cutter_loc = carla.Location(
                    x=ego_start_loc.x + forward.x * cutter_long + right.x * lateral_gap,
                    y=ego_start_loc.y + forward.y * cutter_long + right.y * lateral_gap,
                    z=ego_start_loc.z + 0.3,
                )
                cutter.set_transform(carla.Transform(cutter_loc, ego_spawn.rotation))

                longitudinal_gap = cutter_long - ego_traveled
                longitudinal_gaps.append(longitudinal_gap)
                lateral_gaps.append(lateral_gap)

                if lateral_gap < LANE_WIDTH / 2.0:
                    cutter_in_lane = True

                # ── Ego: ask the controller (expert or CNN) for a decision ──
                frame = frames[-1] if frames else None
                braking_force = float(np.clip(
                    controller_fn(frame, ego_speed, t, cutter_in_lane), 0.0, 1.0
                ))
                steer = _lane_keeping_steer(carla_map, ego)
                if braking_force > 0.0:
                    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=braking_force, steer=steer))
                else:
                    throttle = CRUISE_THROTTLE if ego_speed < ego_speed_target else 0.0
                    ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

                self.world.tick()

                if longitudinal_gap < IMPACT_THRESHOLD and cutter_in_lane:
                    collided = True
                    break
                if ego_speed < STOP_SPEED and cutter_in_lane:
                    break

            return {
                "longitudinal_gaps": longitudinal_gaps,
                "lateral_gaps": lateral_gaps,
                "elapsedTime": len(longitudinal_gaps) * FIXED_DELTA,
                "iterations": len(longitudinal_gaps),
                "collided": collided,
            }
        finally:
            if camera is not None:
                camera.stop()
                camera.destroy()
            if cutter is not None:
                cutter.destroy()
            if ego is not None:
                ego.destroy()


def expert_controller(frame, ego_speed: float, elapsed_s: float, cutter_in_lane: bool) -> float:
    """Near-instantaneous expert: brakes fully once the cutter has entered the lane."""
    return 1.0 if cutter_in_lane else 0.0
