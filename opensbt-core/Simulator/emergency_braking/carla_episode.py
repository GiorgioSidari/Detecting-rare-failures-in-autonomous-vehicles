from __future__ import annotations

"""
CARLA episode runner for the emergency_braking scenario (Fase 2 del piano
video-CNN). Design reference: srunner/scenarios/follow_leading_vehicle.py
(scenario_runner v0.9.16) — ego follows a lead vehicle that brakes hard;
we script the same behaviour directly against the CARLA API instead of
going through scenario_runner's BasicScenario/ScenarioManager machinery,
since we drive the ego ourselves (physics expert or trained CNN) rather
than evaluating an external agent against scenario_runner's route harness.

Parameter reinterpretation vs. simulators/emergency_braking.py
----------------------------------------------------------------
initial_speed, friction_coefficient, detection_distance keep their meaning
(cruise speed, ego braking capability via tire friction, initial gap to the
lead vehicle). `nominal_delay` changes meaning: it used to gate the EGO's
reaction directly; here the ego's reaction is whatever the controller
(expert or CNN) decides from what it perceives, so `nominal_delay` instead
controls WHEN THE LEAD VEHICLE starts braking — a property of the hazard,
not of the ego. Ego and lead cruise at the identical `initial_speed` before
that point, so the gap stays ~`detection_distance` until the hazard begins,
keeping evaluation/qoi.py's `detection_distance - final_position - buffer`
formula a reasonable (slightly conservative) approximation.
"""

import math
import numpy as np
import carla


def _speed(actor: "carla.Actor") -> float:
    v = actor.get_velocity()
    return (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5


STEER_LOOKAHEAD = 6.0   # metres — waypoint lookahead for the lane-keeping steer below


def _lane_keeping_steer(carla_map: "carla.Map", vehicle: "carla.Actor", lookahead: float = STEER_LOOKAHEAD) -> float:
    """
    Minimal proportional heading controller that keeps a vehicle centred in
    its lane. Needed because neither the ego (controller decides braking
    only) nor the lead (scripted cruise/brake) ever steer — steer was
    previously always 0, safe only on a perfectly straight road. Town04's
    spawn_points[0] is followed by a highway curve; confirmed live (via
    cut_in's identical setup) that enough throttle-only driving with steer=0
    runs straight into a static.guardrail, pinning speed near 0 for the rest
    of the episode. See the identical helper in cut_in/carla_episode.py.
    """
    transform = vehicle.get_transform()
    loc = transform.location
    wp = carla_map.get_waypoint(loc)
    next_wps = wp.next(lookahead)
    target_loc = next_wps[0].transform.location if next_wps else wp.transform.location
    target_yaw = math.degrees(math.atan2(target_loc.y - loc.y, target_loc.x - loc.x))
    yaw_error = (target_yaw - transform.rotation.yaw + 180.0) % 360.0 - 180.0
    return max(-1.0, min(1.0, yaw_error / 45.0))


TOWN = "Town04"            # has a long straight highway segment
FIXED_DELTA = 0.05         # 20 Hz control loop
T_MAX = 10.0                # seconds — matches simulators/emergency_braking.py
MAX_STEPS = int(T_MAX / FIXED_DELTA)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 160
STOP_SPEED = 0.3            # m/s — below this the ego is considered stopped
COLLISION_GAP = 3.0         # m — bumper-to-bumper proxy (vehicle bounding box ~2.5m)
CRUISE_THROTTLE = 0.6
SETTLE_TICKS = 25           # ~1.25s — time for spawn drop + suspension to settle before
                            # the episode starts; shorter settling leaves the vehicle
                            # still falling, so wheels have no ground contact and throttle
                            # silently does nothing (measured empirically at 20Hz).
SPIN_UP_MAX_TICKS = 300     # ~15s cap — CARLA's vehicle physics accelerates gradually,
                            # unlike the old analytic simulator's instant cruise speed.
                            # We spin both vehicles up to initial_speed BEFORE starting the
                            # episode clock, so t=0 / nominal_delay match the old semantics
                            # (already-cruising vehicles), instead of burning the hazard
                            # window on the acceleration ramp.


def _spawn_with_retry(world: "carla.World", blueprint, transform: "carla.Transform", attempts: int = 4):
    """
    world.spawn_actor() raises RuntimeError when the transform overlaps map
    geometry. A bare retry with the SAME transform fails identically every
    time, so we nudge the spawn point (mostly upward, a little sideways)
    between attempts — see the identical helper (and the failure it fixes in
    practice) in cut_in/carla_episode.py.
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


class EmergencyBrakingCarlaEpisode:
    """
    One connection to a CARLA server, reused across many episodes (spawning/
    destroying actors per episode is much cheaper than reconnecting/reloading
    the world). Mirrors the role of UdacitySimulator for lanekeeping.
    """

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
        current_map = self.world.get_map().name.split("/")[-1]
        if current_map != self.town:
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
        params : [initial_speed, friction_coefficient, detection_distance, nominal_delay]
        controller_fn(frame, ego_speed_mps, elapsed_s, lead_is_braking) -> braking_force [0,1]
            `frame` is the latest carla.Image (raw BGRA) or None when record_frames=False
            or no frame has arrived yet.
        record_frames : attach an RGB camera and pass frames to controller_fn.

        Returns
        -------
        dict with positions/velocities (ego, per control step), elapsedTime,
        iterations, collided — same shape of information SimulatorServer.py
        returns for lanekeeping, so the rest of the pipeline stays consistent.
        """
        initial_speed, friction, detection_distance, nominal_delay = (float(p) for p in params)

        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        carla_map = self.world.get_map()
        spawn_points = carla_map.get_spawn_points()
        ego_spawn = spawn_points[0]
        ego_wp = carla_map.get_waypoint(ego_spawn.location)

        next_wps = ego_wp.next(max(detection_distance, 5.0))
        lead_wp = next_wps[0] if next_wps else ego_wp
        lead_transform = lead_wp.transform
        lead_transform.location.z += 0.3

        # All spawns happen inside the try so a failure partway through still
        # destroys whatever DID spawn — an orphaned actor left behind would
        # permanently block spawn_points[0] for every later episode in the same
        # CARLA session (see the identical fix/comment in cut_in/carla_episode.py,
        # where this was caught in practice).
        ego = None
        lead = None
        camera = None
        frames: list = []
        try:
            ego = _spawn_with_retry(self.world, vehicle_bp, ego_spawn)
            lead = _spawn_with_retry(self.world, vehicle_bp, lead_transform)

            # Ego braking capability: tire friction sets how hard it can actually
            # decelerate once brake=1.0 is applied — CARLA's physics engine (not
            # our own integration, unlike the old analytic simulator) does the rest.
            physics = ego.get_physics_control()
            wheels = physics.wheels
            for wheel in wheels:
                wheel.tire_friction = friction
            physics.wheels = wheels
            ego.apply_physics_control(physics)
            self.world.tick()   # apply_physics_control only takes effect after a tick

            if record_frames:
                camera_bp = bp_lib.find("sensor.camera.rgb")
                camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
                camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
                camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
                camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
                camera.listen(lambda image: frames.append(image))

            for _ in range(SETTLE_TICKS):   # let spawn-drop physics settle before the episode starts
                self.world.tick()

            # ── Spin-up: bring both vehicles up to initial_speed before t=0 ──
            for _ in range(SPIN_UP_MAX_TICKS):
                ego_speed = _speed(ego)
                lead_speed = _speed(lead)
                if ego_speed >= initial_speed and lead_speed >= initial_speed:
                    break
                ego.apply_control(carla.VehicleControl(
                    throttle=1.0 if ego_speed < initial_speed else 0.0,
                    steer=_lane_keeping_steer(carla_map, ego),
                ))
                lead.apply_control(carla.VehicleControl(
                    throttle=1.0 if lead_speed < initial_speed else 0.0,
                    steer=_lane_keeping_steer(carla_map, lead),
                ))
                self.world.tick()

            positions: list[float] = []
            velocities: list[float] = []
            collided = False
            lead_braking = False
            ego_start_loc = ego.get_transform().location

            ego.apply_control(carla.VehicleControl(throttle=CRUISE_THROTTLE))
            lead.apply_control(carla.VehicleControl(throttle=CRUISE_THROTTLE))

            for step in range(MAX_STEPS):
                t = step * FIXED_DELTA

                ego_speed = _speed(ego)
                lead_speed = _speed(lead)

                ego_loc = ego.get_transform().location
                pos = ego_start_loc.distance(ego_loc)
                positions.append(pos)
                velocities.append(ego_speed)

                # ── Lead vehicle: cruise at initial_speed, then emergency-brake ──
                lead_steer = _lane_keeping_steer(carla_map, lead)
                if t >= nominal_delay:
                    lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=lead_steer))
                    lead_braking = True
                else:
                    throttle = CRUISE_THROTTLE if lead_speed < initial_speed else 0.0
                    lead.apply_control(carla.VehicleControl(throttle=throttle, steer=lead_steer))

                # ── Ego: ask the controller (expert or CNN) for a decision ──
                frame = frames[-1] if frames else None
                braking_force = float(np.clip(
                    controller_fn(frame, ego_speed, t, lead_braking), 0.0, 1.0
                ))
                ego_steer = _lane_keeping_steer(carla_map, ego)
                if braking_force > 0.0:
                    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=braking_force, steer=ego_steer))
                else:
                    throttle = CRUISE_THROTTLE if ego_speed < initial_speed else 0.0
                    ego.apply_control(carla.VehicleControl(throttle=throttle, steer=ego_steer))

                self.world.tick()

                gap = lead.get_transform().location.distance(ego.get_transform().location)
                if gap < COLLISION_GAP:
                    collided = True
                    break
                if ego_speed < STOP_SPEED and t > nominal_delay:
                    break

            return {
                "positions": positions,
                "velocities": velocities,
                "elapsedTime": len(positions) * FIXED_DELTA,
                "iterations": len(positions),
                "collided": collided,
            }
        finally:
            if camera is not None:
                camera.stop()
                camera.destroy()
            if lead is not None:
                lead.destroy()
            if ego is not None:
                ego.destroy()


def expert_controller(frame, ego_speed: float, elapsed_s: float, lead_is_braking: bool) -> float:
    """
    Near-instantaneous "perfect driver" used both as the behavioral-cloning
    label source and as a physics-only baseline. Brakes fully as soon as the
    lead vehicle is braking — this is exactly the visual cue (brake lights /
    deceleration) the CNN has to learn to react to from camera frames alone;
    its imperfect, delayed approximation of this rule is the source of the
    rare failures we're looking for.
    """
    return 1.0 if lead_is_braking else 0.0
