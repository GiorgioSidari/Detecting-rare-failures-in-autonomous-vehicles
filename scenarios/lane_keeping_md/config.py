"""
Lane-keeping scenario on the MetaDrive backend, driven from exact state.

Implements the same BaseScenario contract as the Unity/Docker
LaneKeepingScenario -- identical `param_bounds` and `param_distributions`, the
same (N, T, 4) trajectory layout [x, y, xte, steering], the same composite QoI
-- so the orchestrator, POD, rare_event and the runner scripts work against
either backend; selecting this one is a registry entry plus
`--scenario lane_keeping_md`.

MetaDrive advances the physics by `decision_repeat * physics_world_step_size`
per step, so the control rate is a configuration parameter:

    control_hz = 1 / (decision_repeat * physics_world_step_size)   # 5 * 0.02 -> 10 Hz

`meters_per_step = mean_speed / control_hz` follows exactly, and the fidelity
gate is left off (`min_control_hz=0`, `max_meters_per_step=0`).

The vehicle is driven by `LateralFeedbackDriver` on exact state, so the
environment runs headless with no rendering, no GPU and no Docker. Any object
implementing the same `Driver.act()` interface can be injected instead.

Imports are at the top except `metadrive` itself, imported inside `_make_env`
and `_apply_curve_geometry`: it requires Python < 3.12 and its own venv, and
this module stays importable without it for the geometry tests.
"""
from __future__ import annotations

import concurrent.futures
import math
import time
import warnings

import numpy as np
from scipy import stats

from scenarios.base_scenario import BaseScenario
from scenarios.common.episode_budget import budget_steps, polyline_length
from scenarios.common.road_frame import road_frame
from scenarios.lane_keeping.qoi import MAX_XTE, composite_lane_qoi
from scenarios.lane_keeping_md.driver import Driver, LateralFeedbackDriver
from scenarios.lane_keeping_md.map_builder import (
    ScenarioSpec, build_scenario_spec, target_speed,
)
from scenarios.lane_keeping_md.scenario_map import centerline, make_online_env


def _classify_outcome(last_info: dict, road_finished: bool, traj: np.ndarray) -> str:
    """
    Why the episode ended, as one label.

    The cascade is ordered by priority and the LAST match wins:
    ``max_step`` -> ``arrive_dest`` -> ``strada_completata`` -> ``out_of_road``
    -> ``crash`` -> the XTE threshold applied to the trajectory's final point.
    """
    outcome = "max_step"
    if last_info.get("arrive_dest") or last_info.get("arrive_destination"):
        outcome = "arrive_dest"
    if road_finished:
        outcome = "strada_completata"
    if last_info.get("out_of_road"):
        outcome = "out_of_road"
    if (last_info.get("crash") or last_info.get("crash_vehicle")
            or last_info.get("crash_object")):
        outcome = "crash"
    if len(traj) and abs(float(traj[-1, 2])) > MAX_XTE:
        outcome = "out_of_road"
    return outcome


def _metres_per_step(traj: np.ndarray) -> float:
    """Mean distance travelled between two control decisions, from the path."""
    if len(traj) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)
    return float(np.mean(d)) if d.size else 0.0


class LaneKeepingMetaDriveScenario(BaseScenario):
    name = "lane_keeping_md"
    description = (
        "Lane-keeping on the MetaDrive backend (in-process, headless, no Docker). "
        "The control rate is an exact config parameter, so meters-per-step is "
        "reproducible and can be swept directly."
    )

    def __init__(
        self,
        decision_repeat: int = 5,
        physics_world_step_size: float = 0.02,
        max_steps: int = 300,
        driver_factory=None,
        n_jobs: int = 1,
        speed_scale: float = 1.0,
        geometry: str = "udacity",
    ):
        # Control rate = 1 / (decision_repeat * physics_world_step_size).
        """
        Backend instance.

        `speed_scale` multiplies the target cruising speed and, together with the
        driver factory, defines the operating point.

        `geometry="udacity"` (default) builds the lane from the Catmull-Rom
        centreline, handed to ScenarioOnlineEnv as an explicit polyline.
        `geometry="pgblock"` uses MetaDrive's native blocks, which collapse the five
        angles into their mean.
        """
        self.decision_repeat = int(decision_repeat)
        self.physics_world_step_size = float(physics_world_step_size)
        self.max_steps = int(max_steps)
        # Scales the target cruising speed. Lower -> more control margin, so the
        # failure outcome depends on scenario difficulty (curvature x speed)
        # instead of sitting on a knife-edge -> a learnable boundary for Part B.
        self.speed_scale = float(speed_scale)
        self.n_jobs = int(n_jobs)

        if geometry not in ("udacity", "pgblock"):
            raise ValueError(
                f"geometry={geometry!r} is not valid: use 'udacity' (geometric "
                f"parity, default) or 'pgblock' (legacy, reproduction only)")
        self.geometry = geometry
        # Pluggable driver (A1 default). Callable returning a fresh Driver per run.
        self._driver_factory = driver_factory or (lambda: LateralFeedbackDriver())

        # Populated by run_simulation / compute_qoi (read by the orchestrator via getattr).
        self._run_lengths: list[int] | None = None
        self._control_hz: np.ndarray | None = None
        self._meters_per_step: np.ndarray | None = None
        self._infer_ms: np.ndarray | None = None
        self._wait_ms: np.ndarray | None = None
        self._md_outcome: list | None = None      # MetaDrive termination reason per run
        # Optional behavioral-cloning data collection: if set to a list, each step
        # appends (state_dict, teacher_steering) for training a LearnedDriver.
        self._record: list | None = None
        self._last_survival: np.ndarray | None = None
        self._valid_mask: np.ndarray | None = None
        self._n_degenerate: int = 0
        self._n_low_fidelity: int = 0
        self._n_invalid: int = 0
        self._last_control_hz: np.ndarray | None = None
        self._last_meters_per_step: np.ndarray | None = None
        self._last_infer_ms: np.ndarray | None = None
        self._last_wait_ms: np.ndarray | None = None
        # Centreline of the current episode in MetaDrive's frame (translated to
        # the ego's initial position). Filled by `_make_env` in "udacity" mode,
        # used by `_extract_state` to measure the XTE.
        self._local_centerline: np.ndarray | None = None
        # Orizzonte in steps dell'episodio corrente (modalita' "udacity").
        self._budget_steps: int = int(max_steps)

    @property
    def control_hz_nominal(self) -> float:
        """Exact by construction: 1 / (5 * 0.02) = 10 Hz. Not a measurement."""
        return 1.0 / (self.decision_repeat * self.physics_world_step_size)

    # ── Parameter space (identical to LaneKeepingScenario) ────────────────────

    def param_bounds(self) -> dict:
        """
        The ODD box: lower and upper bound of each of the nine parameters, the same
        values LaneKeepingScenario declares.
        """
        return {
            "names": [
                "angle_1 (°)", "angle_2 (°)", "angle_3 (°)",
                "angle_4 (°)", "angle_5 (°)",
                "min_speed (m/s)", "max_speed (m/s)",
                "segment_length (m)",
                "map_size (m)",
            ],
            "lower": np.array([0,  0,  0,  0,  0,  5.0,  10.0, 10.0, 150.0]),
            "upper": np.array([85, 85, 85, 85, 85, 15.0, 30.0, 40.0, 350.0]),
        }

    def param_distributions(self, lower=None, upper=None) -> list:
        """
        The operational marginals: how likely each value is inside the box.

        Angles peak on the lower bound, speeds at 40% of their band, geometry
        parameters are uniform. Rarity in `arm_ranking` is defined against these.
        """
        b = self.param_bounds()
        lo = np.asarray(lower if lower is not None else b["lower"], dtype=float)
        hi = np.asarray(upper if upper is not None else b["upper"], dtype=float)

        def _uniform(a, c):
            return stats.uniform(loc=a, scale=max(c - a, 1e-9))

        def _truncnorm(a, c, mu: np.ndarray, sigma: np.ndarray):
            if sigma <= 0 or c <= a:
                return _uniform(a, c)
            return stats.truncnorm((a - mu) / sigma, (c - mu) / sigma, loc=mu, scale=sigma)

        dists = []
        for j in range(len(lo)):
            a, c = float(lo[j]), float(hi[j])
            rng = c - a
            if j < 5:
                dists.append(_truncnorm(a, c, mu=a, sigma=max(rng * 0.5, 1e-6)))
            elif j in (5, 6):
                dists.append(_truncnorm(a, c, mu=a + 0.4 * rng, sigma=max(rng * 0.3, 1e-6)))
            else:
                dists.append(_uniform(a, c))
        return dists

    def failure_threshold(self) -> float:
        """The margin is built so that zero is the boundary. Not a tunable."""
        return 0.0

    # ── Simulation ────────────────────────────────────────────────────────────

    def run_simulation(self, params: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        N episodes -> zero-padded trajectories (N, T, 4): [x, y, xte, steering], the
        same layout Udacity produces. Seed = row index, so the pipeline is
        deterministic given (arm, seed).

        Real lengths are stored in `self._run_lengths`, which `compute_qoi` uses to
        mask the padding: a padded zero in the xte channel would otherwise read as
        dead-centre driving.
        """
        params = np.asarray(params, dtype=float)
        N, ncols = params.shape

        if self.n_jobs and self.n_jobs > 1:
            results = self._run_parallel(params, ncols, verbose)
        else:
            results = [self._simulate_one(params[i], ncols, seed=i, verbose=verbose)
                       for i in range(N)]

        run_lengths, control_hz, meters_per_step, infer_ms, wait_ms = [], [], [], [], []
        outcomes = []
        arrays = []
        for traj, fid in results:
            traj = np.asarray(traj, dtype=np.float32).reshape(-1, 4)
            arrays.append(traj)
            run_lengths.append(traj.shape[0])
            control_hz.append(fid.get("control_hz", np.nan))
            meters_per_step.append(fid.get("meters_per_step", np.nan))
            infer_ms.append(fid.get("infer_ms", np.nan))
            wait_ms.append(fid.get("wait_ms", np.nan))
            outcomes.append(fid.get("outcome", "unknown"))

        self._run_lengths = run_lengths
        self._md_outcome = outcomes
        self._control_hz = np.asarray(control_hz, dtype=float)
        self._meters_per_step = np.asarray(meters_per_step, dtype=float)
        self._infer_ms = np.asarray(infer_ms, dtype=float)
        self._wait_ms = np.asarray(wait_ms, dtype=float)

        T_max = max(1, max(run_lengths))
        traj_tensor = np.zeros((N, T_max, 4), dtype=np.float32)
        for i, a in enumerate(arrays):
            n = a.shape[0]
            if n:
                traj_tensor[i, :n, :] = a
        return traj_tensor

    def _run_parallel(self, params: np.ndarray, ncols: int, verbose: bool):
        """
        Run the episodes on a process pool and return them keyed by row index.

        Each episode builds its own physics engine, so the pool uses processes. The
        results are reordered by index before returning.
        """
        results = [None] * params.shape[0]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            futs = {ex.submit(self._simulate_one, params[i], ncols, i, False): i
                    for i in range(params.shape[0])}
            for fut in concurrent.futures.as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

    # ── QoI (shared composite metric) ─────────────────────────────────────────

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Safety margin per run, from the shared metric. Negative = failure.

        `composite_lane_qoi` is imported from the Udacity backend and used unchanged,
        with the fidelity gate off (`min_control_hz=0`, `max_meters_per_step=0`).

        When MetaDrive reports `out_of_road` or `crash` for a run, its margin is
        forced negative: the value is `-0.001 - (1 - fraction of the budget survived)`,
        so earlier failures score lower than late ones.
        """
        res = composite_lane_qoi(
            trajectories,
            self._run_lengths,
            params,
            control_hz=self._control_hz,
            meters_per_step=self._meters_per_step,
            infer_ms=self._infer_ms,
            wait_ms=self._wait_ms,
            min_control_hz=0.0,                 # fidelity gate off: see docstring
            max_meters_per_step=0.0,
        )
        qoi = np.asarray(res.qoi, dtype=float)
        N = trajectories.shape[0]

        def _tail(arr: np.ndarray):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            return arr[-N:] if arr.shape[0] >= N else np.full(N, np.nan)

        outcomes = self._md_outcome or []
        if len(outcomes) != N:
            outcomes = list(outcomes)[-N:]
        vm = np.asarray(res.valid_mask, dtype=bool)
        survival = np.asarray(res.survival, dtype=float)
        for i in range(N):
            oc = outcomes[i] if i < len(outcomes) else "unknown"
            if oc in ("out_of_road", "crash") and vm[i]:
                frac = float(survival[i]) / max(1, self.max_steps)
                forced = -0.001 - (1.0 - frac)
                cur = qoi[i]
                qoi[i] = forced if (np.isnan(cur) or forced < cur) else cur

        self._last_survival = res.survival
        self._valid_mask = res.valid_mask
        self._n_degenerate = res.n_degenerate
        self._n_low_fidelity = res.n_low_fidelity
        self._n_invalid = res.n_invalid
        self._last_control_hz = res.control_hz
        self._last_meters_per_step = res.meters_per_step
        self._last_infer_ms = _tail(self._infer_ms)
        self._last_wait_ms = _tail(self._wait_ms)
        return qoi

    # ── MetaDrive glue (lazy import; verify attr names vs installed version) ───

    def _make_env(self, spec: ScenarioSpec, seed: int, row=None):
        """
        Build the headless environment for one scenario.

        In "udacity" mode `row` is required: the road is generated from theta. The
        centreline is translated by `-poly[0]` because MetaDrive recentres the scene
        on the ego's start position.

        The episode budget is computed here and passed to MetaDrive as `horizon`,
        which is where the simulator applies its own truncation.
        """
        if self.geometry == "udacity":
            if row is None:
                raise ValueError(
                    "geometry='udacity' requires `row`: the road is generated "
                    "from the theta parameters, not from the block spec")
            poly = centerline(row)
            self._local_centerline = poly - poly[0]

            self._budget_steps = budget_steps(
                polyline_length(self._local_centerline),
                target_speed(spec) * self.speed_scale,
                self.control_hz_nominal)

            return make_online_env(
                row, decision_repeat=self.decision_repeat,
                physics_world_step_size=self.physics_world_step_size,
                max_steps=self._budget_steps, seed=seed)


        from metadrive.envs import MetaDriveEnv

        self._apply_curve_geometry(spec)      # make our angles actually shape the road
        return MetaDriveEnv(self._build_md_config(spec, seed))

    def _apply_curve_geometry(self, spec: ScenarioSpec) -> None:
        """
        Legacy "pgblock" path only. MetaDrive samples each Curve block's radius per
        seed, so our angle parameters were inert; this overrides
        Curve.PARAMETER_SPACE with a fixed radius/angle/length, leaving `dir` free so
        the road still winds both ways. Global per-episode override, safe
        sequentially and on a process pool.
        """
        radii = [b.radius for b in spec.blocks if b.kind == "C"]
        if not radii:
            return                                    # all-straight: nothing to shape
        radius = float(np.clip(np.mean(radii), 15.0, 500.0))
        seg = float(spec.blocks[0].length)
        angle_deg = float(np.clip(np.degrees(seg / max(radius, 1e-6)), 5.0, 160.0))
        try:
            from metadrive.component.pgblock.curve import Curve
            from metadrive.component.pg_space import (
                ParameterSpace, Parameter, ConstantSpace, BlockParameterSpace,
            )
            base = dict(BlockParameterSpace.CURVE)
            base[Parameter.radius] = ConstantSpace(radius)
            base[Parameter.angle] = ConstantSpace(angle_deg)
            base[Parameter.length] = ConstantSpace(seg)
            Curve.PARAMETER_SPACE = ParameterSpace(base)
        except Exception as e:
            # The scenario still runs, but with the angles ignored, so the
            # warning has to reach the caller even when nothing is verbose.
            warnings.warn(f"curve override failed ({e}); the angles of theta "
                          f"remain inert on this scenario", RuntimeWarning,
                          stacklevel=2)

    def _build_md_config(self, spec: ScenarioSpec, seed: int) -> dict:
        """
        MetaDrive config dict for the legacy "pgblock" path. Per-block radius control
        depends on the installed map API; the block string plus headless and
        traffic-free is the portable core.
        """
        block_str = spec.block_string() or "S"
        return {
            "use_render": False,               # headless: no window
            "image_observation": False,        # A1: state-based, NO rendering in the loop
            "traffic_density": 0.0,            # isolate the lane-keeping behaviour
            "num_scenarios": 1,
            "start_seed": int(seed),
            "map": block_str,                  # e.g. "CSCCS"
            "horizon": self.max_steps,
            "physics_world_step_size": self.physics_world_step_size,
            "decision_repeat": self.decision_repeat,
            "log_level": 50,                   # quiet
        }

    def _extract_state(self, env) -> dict:
        """
        The state dictionary the controller consumes.

        Carries lateral error, heading error, speed, position and the end-of-road
        flag, all in SI units and independent of how the simulator represents lanes;
        the Udacity arm builds the same dictionary from its telemetry.

        In "udacity" geometry the errors come from `road_frame` on the shared
        centreline. `beyond_end` is true when the projection has clamped to the last
        vertex, and the caller ends the run. The else branch serves the legacy
        "pgblock" mode, where the errors come from MetaDrive's lane API.
        """
        vehicle = getattr(env, "agent", None) or getattr(env, "vehicle", None)

        if self.geometry == "udacity" and self._local_centerline is not None:
            pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=float)
            heading = float(getattr(vehicle, "heading_theta", 0.0))
            spd_kmh = getattr(vehicle, "speed_km_h", None)
            if spd_kmh is None:
                spd_kmh = getattr(vehicle, "speed", 0.0)
            rf = road_frame(self._local_centerline,
                            float(pos[0]), float(pos[1]), heading)
            return {
                "lateral_error": float(rf.lateral_error),
                "heading_error": float(rf.heading_error),
                "speed": float(spd_kmh) / 3.6,
                "x": float(pos[0]), "y": float(pos[1]),
                "beyond_end": bool(rf.beyond_end),
            }

        # Current lane: prefer vehicle.lane, else the navigation module's current lane.
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            nav = getattr(vehicle, "navigation", None)
            lane = getattr(nav, "current_lane", None) if nav is not None else None
        pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=float)
        # Speed to m/s unambiguously: MetaDrive exposes speed_km_h (km/h). Fall back
        # to .speed (older API: also km/h) if the former is missing.
        spd_kmh = getattr(vehicle, "speed_km_h", None)
        if spd_kmh is None:
            spd_kmh = getattr(vehicle, "speed", 0.0)
        speed = float(spd_kmh) / 3.6                           # m/s
        heading = float(getattr(vehicle, "heading_theta", 0.0))

        lateral, lane_heading = 0.0, heading
        if lane is not None:
            try:
                long_, lateral = lane.local_coordinates(pos)   # lateral = XTE
                lane_heading = float(lane.heading_theta_at(long_))
            except Exception:
                pass
        heading_error = math.atan2(math.sin(heading - lane_heading),
                                   math.cos(heading - lane_heading))
        return {
            "lateral_error": float(lateral),
            "heading_error": float(heading_error),
            "speed": speed,
            "x": float(pos[0]), "y": float(pos[1]),
        }

    def _drive_episode(self, env, driver, tgt: float, budget: int) -> tuple:
        """
        Drive one episode: read, decide, execute, until something ends it.

        One iteration is one control decision, i.e. `decision_repeat` physics
        steps. Returns ``(traj, last_info, road_finished, infer_t, wait_t)``,
        where `traj` is the list of [x, y, lateral_error, steering] rows and the
        two times split the loop into deciding and simulating.
        """
        traj = []
        infer_t, wait_t = 0.0, 0.0
        last_info: dict = {}
        road_finished = False
        try:
            env.reset()
            done = False
            steps = 0
            # One iteration = one control decision = 0.1 s of simulated time.
            while not done and steps < budget:
                # Read: raw pose and speed from MetaDrive, errors from road_frame.
                state = self._extract_state(env)
                state["target_speed"] = tgt
                state["dt"] = self.decision_repeat * self.physics_world_step_size

                # Decide: proportional feedback law, scenarios/common/driver.py.
                # It corrects in proportion to how wrong it is right now -- no
                # memory, no lookahead, just the current error times a fixed gain:
                #   steering = -0.35 * lateral_error   (metres off the centreline)
                #              -0.8  * heading_error   (radians off the tangent)
                #   throttle =  0.3  * (target_speed - speed)
                t0 = time.perf_counter()
                steering, throttle = driver.act(state)
                t1 = time.perf_counter()
                if self._record is not None:      # BC data collection (teacher)
                    self._record.append((dict(state), float(steering)))
                # Execute: `decision_repeat` physics steps holding this command.
                step_out = env.step([float(steering), float(throttle)])
                t2 = time.perf_counter()
                infer_t += (t1 - t0)      # deciding
                wait_t += (t2 - t1)       # simulating

                if len(step_out) == 5:
                    _, _, terminated, truncated, info = step_out
                    done = bool(terminated) or bool(truncated)
                else:
                    _, _, done, info = step_out
                    done = bool(done)
                last_info = info if isinstance(info, dict) else {}

                traj.append([state["x"], state["y"],
                             state["lateral_error"], float(steering)])
                # Off-road: left the lane by more than MAX_XTE.
                if abs(state["lateral_error"]) > MAX_XTE:
                    done = True

                # Normal exit: the road is over.
                if state.get("beyond_end"):
                    road_finished = True
                    done = True

                steps += 1
        finally:
            try:
                env.close()
            except Exception:
                pass
        return traj, last_info, road_finished, infer_t, wait_t

    def _simulate_one(self, row: np.ndarray, ncols: int, seed: int = 0,
                      verbose: bool = False):
        """
        One episode -> (traj (L, 4), fidelity).

        The horizon is the per-scenario budget computed by `_make_env`, and
        `state["dt"]` is `decision_repeat * physics_world_step_size`, which the shared
        controller uses to convert its per-second limits.

        An episode ends on any of four conditions: MetaDrive's own termination,
        |xte| above MAX_XTE, the road being over (`beyond_end`, the normal exit), or
        the step budget running out. `fidelity` carries control_hz, meters_per_step,
        the per-step timings and the outcome label.
        """
        spec = build_scenario_spec(row, ncols)
        tgt = target_speed(spec) * self.speed_scale
        driver: Driver = self._driver_factory()
        driver.reset()

        env = self._make_env(spec, seed, row=row)

        budget = self._budget_steps if self.geometry == "udacity" else self.max_steps

        traj, last_info, road_finished, infer_t, wait_t = self._drive_episode(
            env, driver, tgt, budget)

        traj = np.asarray(traj, dtype=np.float32).reshape(-1, 4)
        L = traj.shape[0]
        outcome = _classify_outcome(last_info, road_finished, traj)
        hz = self.control_hz_nominal
        mps = _metres_per_step(traj)
        fidelity = {
            "control_hz": hz,                 # exact, constant by construction
            "meters_per_step": mps,           # exact: mean metres between decisions
            "infer_ms": (infer_t / L * 1000.0) if L else np.nan,
            "wait_ms": (wait_t / L * 1000.0) if L else np.nan,
            "outcome": outcome,
        }
        if verbose:
            print(f"  [md] seed={seed} steps={L} block={spec.block_string()} "
                  f"hz={hz:.1f} m/step={mps:.3f} outcome={outcome}", flush=True)
        return traj, fidelity
