"""
Lane-keeping scenario on the MetaDrive backend (Step A / variant A1: state-based).

Drop-in alternative to the Unity/Docker LaneKeepingScenario. It implements the
SAME BaseScenario contract (identical param_bounds and param_distributions, same
(N, T, 4) trajectory layout [x, y, xte, steering], same composite QoI), so the
orchestrator, POD, rare_event and the runner scripts work unchanged — you only
add a registry entry and select `--scenario lane_keeping_md`.

Why this exists (see the design doc): the Unity+Docker+Xvfb stack pins the
control loop at ~9 Hz because software rendering dominates each step, and makes
`meters_per_step` depend on machine load (hence the fidelity gate). MetaDrive
advances the physics by `decision_repeat * physics_world_step_size` per step, so
the CONTROL RATE IS AN EXACT CONFIG PARAMETER:

    control_hz = 1 / (decision_repeat * physics_world_step_size)   # e.g. 5*0.02 -> 10 Hz

That makes `meters_per_step = mean_speed / control_hz` exact and reproducible,
and lets you SWEEP the control rate as an experimental variable — the whole
point of the envelope result. The fidelity gate is therefore unnecessary here.

A1 uses a state-based PurePursuitDriver (no rendering -> headless, no GPU, no
Docker). A camera-vision driver (A2) can later implement the same Driver.act()
interface for end-to-end vision testing; the rest of this class is unchanged.

METADRIVE GLUE: `_build_md_config` / `_extract_state` / `_simulate_one` are the
only MetaDrive-coupled parts and import MetaDrive lazily, so this module can be
imported and its pure logic unit-tested WITHOUT MetaDrive installed. Verify the
attribute names in those methods against your installed MetaDrive version.
"""
from __future__ import annotations

import math
import numpy as np

from scenarios.base_scenario import BaseScenario
from scenarios.lane_keeping.qoi_lane import composite_lane_qoi
from scenarios.lane_keeping_md.map_builder import (
    build_scenario_spec, target_speed, ScenarioSpec,
)
from scenarios.lane_keeping_md.driver import PurePursuitDriver, Driver

MAX_XTE = 2.5  # metres — mirror of the Unity constant, used for QoI + termination


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
    ):
        # Control rate = 1 / (decision_repeat * physics_world_step_size).
        self.decision_repeat = int(decision_repeat)
        self.physics_world_step_size = float(physics_world_step_size)
        self.max_steps = int(max_steps)
        # Scales the target cruising speed. Lower -> more control margin, so the
        # failure outcome depends on scenario difficulty (curvature x speed)
        # instead of sitting on a knife-edge -> a learnable boundary for Part B.
        self.speed_scale = float(speed_scale)
        self.n_jobs = int(n_jobs)
        # Pluggable driver (A1 default). Callable returning a fresh Driver per run.
        self._driver_factory = driver_factory or (lambda: PurePursuitDriver())

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

    @property
    def control_hz_nominal(self) -> float:
        return 1.0 / (self.decision_repeat * self.physics_world_step_size)

    # ── Parameter space (identical to LaneKeepingScenario) ────────────────────

    def param_bounds(self) -> dict:
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
        """REALISTIC operational distribution per parameter (mirror of the Unity
        scenario): gentle curves and intermediate speeds are more likely."""
        from scipy import stats
        b = self.param_bounds()
        lo = np.asarray(lower if lower is not None else b["lower"], dtype=float)
        hi = np.asarray(upper if upper is not None else b["upper"], dtype=float)

        def _uniform(a, c):
            return stats.uniform(loc=a, scale=max(c - a, 1e-9))

        def _truncnorm(a, c, mu, sigma):
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
        return 0.0

    # ── Simulation ────────────────────────────────────────────────────────────

    def run_simulation(self, params: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Run N MetaDrive episodes and return zero-padded trajectories (N, T, 4).
        Channels: 0=x, 1=y, 2=xte (lateral offset from lane centre), 3=steering.

        Stores real run lengths in self._run_lengths and exact fidelity arrays so
        compute_qoi() can mask padding and report control_hz / meters_per_step.
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

    def _run_parallel(self, params, ncols, verbose):
        import concurrent.futures
        results = [None] * params.shape[0]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            futs = {ex.submit(self._simulate_one, params[i], ncols, i, False): i
                    for i in range(params.shape[0])}
            for fut in concurrent.futures.as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

    # ── QoI (shared composite metric) ─────────────────────────────────────────

    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        qoi, meta = composite_lane_qoi(
            trajectories, params,
            run_lengths=self._run_lengths,
            control_hz=self._control_hz,
            meters_per_step=self._meters_per_step,
            # Fidelity gate intentionally OFF: control rate is exact on MetaDrive.
            min_control_hz=0.0, max_meters_per_step=0.0,
            max_xte=MAX_XTE,
        )
        N = trajectories.shape[0]

        def _tail(arr):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            return arr[-N:] if arr.shape[0] >= N else np.full(N, np.nan)

        # MetaDrive's own termination is authoritative for failures: force a
        # negative (severity-ordered) margin for out_of_road / crash runs, which
        # the lane-relative XTE can under-detect.
        outcomes = self._md_outcome or []
        if len(outcomes) != N:
            outcomes = list(outcomes)[-N:]
        vm = np.asarray(meta["valid_mask"], dtype=bool)
        survival = np.asarray(meta["survival"], dtype=float)
        for i in range(N):
            oc = outcomes[i] if i < len(outcomes) else "unknown"
            if oc in ("out_of_road", "crash") and vm[i]:
                frac = float(survival[i]) / max(1, self.max_steps)
                forced = -0.001 - (1.0 - frac)      # earlier failure -> more negative
                cur = qoi[i]
                qoi[i] = forced if (np.isnan(cur) or forced < cur) else cur

        self._last_survival = meta["survival"]
        self._valid_mask = meta["valid_mask"]
        self._n_degenerate = meta["n_degenerate"]
        self._n_low_fidelity = meta["n_low_fidelity"]
        self._n_invalid = meta["n_invalid"]
        self._last_control_hz = meta["control_hz"]
        self._last_meters_per_step = meta["meters_per_step"]
        self._last_infer_ms = _tail(self._infer_ms)
        self._last_wait_ms = _tail(self._wait_ms)
        return qoi

    # ── MetaDrive glue (lazy import; verify attr names vs installed version) ───

    def _make_env(self, spec: ScenarioSpec, seed: int):
        """Create a headless MetaDrive env for one scenario. Isolated so tests can
        monkeypatch it. Imports MetaDrive lazily."""
        from metadrive.envs import MetaDriveEnv
        self._apply_curve_geometry(spec)      # make our angles actually shape the road
        return MetaDriveEnv(self._build_md_config(spec, seed))

    def _apply_curve_geometry(self, spec: ScenarioSpec) -> None:
        """Make the scenario's angles CONTROL the road curvature.

        By default MetaDrive samples each Curve block's radius at random per seed,
        so our angle parameters were inert (confirmed by diagnostic). Here we
        override the Curve block's PARAMETER_SPACE so radius/angle/length come from
        the scenario. This is a global, per-episode override (safe for sequential
        and process-pool execution: each episode sets it before building its env).
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
            # Start from the block's default parameter dict (keyed by the Parameter
            # enum, values are MetaDrive Space objects) and fix curvature; keep
            # `dir` as its original space so the road still winds both ways.
            base = dict(BlockParameterSpace.CURVE)
            base[Parameter.radius] = ConstantSpace(radius)
            base[Parameter.angle] = ConstantSpace(angle_deg)
            base[Parameter.length] = ConstantSpace(seg)
            Curve.PARAMETER_SPACE = ParameterSpace(base)
        except Exception as e:                            # API drift: fail loud, don't crash
            print(f"[geometry] curve override failed ({e}); angles remain inert", flush=True)

    def _build_md_config(self, spec: ScenarioSpec, seed: int) -> dict:
        """Assemble the MetaDrive config dict. The block sequence is derived from
        the scenario spec (curves vs straights). Per-block radius control depends
        on the installed MetaDrive map API; the block string + traffic-free,
        headless setup below is the portable core — tune the map_config to your
        version if you need exact per-block radii."""
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
        """Backend-neutral state for the driver, from the ego vehicle. Uses getattr
        fallbacks so a minor API drift degrades gracefully rather than crashing."""
        vehicle = getattr(env, "agent", None) or getattr(env, "vehicle", None)
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

    def _simulate_one(self, row: np.ndarray, ncols: int, seed: int = 0,
                      verbose: bool = False):
        """Run ONE MetaDrive episode with the driver; return (traj (L,4), fidelity).
        traj channels: [x, y, xte, steering]. Isolated + lazy import so the rest of
        the module is testable without MetaDrive (tests override this method)."""
        import time as _time
        spec = build_scenario_spec(row, ncols)
        tgt = target_speed(spec) * self.speed_scale
        driver: Driver = self._driver_factory()
        driver.reset()

        env = self._make_env(spec, seed)
        traj = []
        infer_t, wait_t = 0.0, 0.0
        last_info: dict = {}
        try:
            reset_out = env.reset()
            done = False
            steps = 0
            while not done and steps < self.max_steps:
                state = self._extract_state(env)
                state["target_speed"] = tgt

                t0 = _time.perf_counter()
                steering, throttle = driver.act(state)
                t1 = _time.perf_counter()
                if self._record is not None:      # BC data collection (teacher)
                    self._record.append((dict(state), float(steering)))
                step_out = env.step([float(steering), float(throttle)])
                t2 = _time.perf_counter()
                infer_t += (t1 - t0)
                wait_t += (t2 - t1)

                # Gymnasium 5-tuple (obs, reward, terminated, truncated, info);
                # tolerate the legacy 4-tuple too.
                if len(step_out) == 5:
                    _, _, terminated, truncated, info = step_out
                    done = bool(terminated) or bool(truncated)
                else:
                    _, _, done, info = step_out
                    done = bool(done)
                last_info = info if isinstance(info, dict) else {}

                traj.append([state["x"], state["y"],
                             state["lateral_error"], float(steering)])
                # Off-road termination: left the lane by more than MAX_XTE.
                if abs(state["lateral_error"]) > MAX_XTE:
                    done = True
                steps += 1
        finally:
            try:
                env.close()
            except Exception:
                pass

        traj = np.asarray(traj, dtype=np.float32).reshape(-1, 4)
        L = traj.shape[0]
        # Failure signal from MetaDrive's OWN termination info (authoritative): a
        # lane-relative XTE saturates once the car leaves the lane, so we trust
        # out_of_road / crash from the simulator over our XTE threshold alone.
        outcome = "max_step"
        if last_info.get("arrive_dest") or last_info.get("arrive_destination"):
            outcome = "arrive_dest"
        if last_info.get("out_of_road"):
            outcome = "out_of_road"
        if last_info.get("crash") or last_info.get("crash_vehicle") or last_info.get("crash_object"):
            outcome = "crash"
        if L and abs(float(traj[-1, 2])) > MAX_XTE:
            outcome = "out_of_road"
        hz = self.control_hz_nominal
        # meters_per_step from the exact per-step travelled distance.
        if L >= 2:
            d = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)
            mps = float(np.mean(d)) if d.size else 0.0
        else:
            mps = 0.0
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
