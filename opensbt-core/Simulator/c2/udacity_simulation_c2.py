"""
Udacity simulator for the C2 arm -- a subclass, not a modification.

It reuses `UdacitySimulator` for everything Unity-related (process startup, gym
env, road generation) and overrides only `simulate()`, because the original loop
does not pass the agent the information the state-based controller needs:

  * `pos`  -- the position, to project onto the centreline;
  * the run's centreline, set at reset;
  * `dt`   -- the real time step. On Udacity the control rate is NOT
             configurable: it depends on load and on the number of workers
             (measured: 20.8 Hz with 1 container, 8.5 Hz with 4). The shared
             controller's steering rate limiter is in units per second, so
             without the real `dt` its responsiveness would change with how the
             campaign is launched -- and the difference would be charged to the
             simulator.

The rest of the contract is identical: same `UdacitySimulationOutput`, same
termination conditions, same statistics. The client cannot tell the two arms
apart.
"""
from __future__ import annotations

import time

import gym
import numpy as np
from UdacitySimulatorIO import UdacitySimulationOutput, UdacitySimulatorConfig

from ..lanekeeping.config import CAP_XTE, MAX_XTE, UDACITY_SIM_NAME
from ..lanekeeping.road_generator.custom_road_generator import CustomRoadGenerator
from ..lanekeeping.self_driving.road import Road
from ..lanekeeping.udacity.udacity_simulation import UdacitySimulator
from .config import (
    DRIVER_SEED,
    MAX_SPEED,
    MIN_SPEED,
    OBS_LAG_TAU,
    OBS_LATENCY,
    SPEED_SCALE,
    STEER_NOISE,
    STEERING_SIGN,
)
from .state_based_agent import StateBasedAgent


class UdacitySimulatorC2(UdacitySimulator):
    """Udacity simulator running the shared state-based controller."""

    def __init__(self) -> None:
        # `super().__init__()` is deliberately not called: it would build the
        # SupervisedAgent, i.e. load TensorFlow and the DNN weights -- 600 MB and
        # several seconds, for an agent that would then be thrown away. Only the
        # part that is needed is replicated: the Unity environment.
        from ..lanekeeping.config import UDACITY_EXE_PATH
        from ..lanekeeping.udacity.env.udacity_gym_env import UdacityGymEnv_RoadGen
        from ..shared.driver import LateralFeedbackDriver

        driver = LateralFeedbackDriver(
            obs_latency=OBS_LATENCY,
            obs_lag_tau=OBS_LAG_TAU,
            steer_noise=STEER_NOISE,
            seed=DRIVER_SEED,
        )
        self.agent = StateBasedAgent(
            env_name=UDACITY_SIM_NAME,
            min_speed=MIN_SPEED,
            max_speed=MAX_SPEED,
            speed_scale=SPEED_SCALE,
            driver=driver,
            steering_sign=STEERING_SIGN,
        )
        print(f"[C2] braccio state-based | speed_scale={SPEED_SCALE} "
              f"| steering_sign={STEERING_SIGN:+.0f} "
              f"| max_steer_rate={driver.max_steer_rate} u/s "
              f"| obs_latency={OBS_LATENCY} obs_lag_tau={OBS_LAG_TAU} "
              f"steer_noise={STEER_NOISE}", flush=True)

        self.env = UdacityGymEnv_RoadGen(seed=1, exe_path=UDACITY_EXE_PATH)

    def simulate(self, simulator_config: UdacitySimulatorConfig) -> UdacitySimulationOutput:
        self.agent.setSpeedLimits(minSpeed=simulator_config.minSpeed,
                                  maxSpeed=simulator_config.maxSpeed)

        test_generator = CustomRoadGenerator(
            map_size=simulator_config.map_size,
            num_control_nodes=len(simulator_config.angles),
            seg_length=simulator_config.segLength)

        simulationOutput = UdacitySimulationOutput()

        road: Road = test_generator.generate(
            starting_pos=simulator_config.initial_position,
            angles=simulator_config.angles,
            simulator_name=UDACITY_SIM_NAME)

        simulationOutput.road = road.get_concrete_representation(to_plot=True)
        waypoints: str = road.get_string_repr()

        obs = self.env.reset(skip_generation=False, track_string=waypoints)

        # The centreline is the SAME polyline handed to Unity: the two backends
        # drive the same road by construction, not by resemblance.
        self.agent.set_road([(p.x, p.y) for p in road.road_points])

        speed: float = 0.0
        pos = (road.road_points[0].x, road.road_points[0].y, 0.0)
        xte_raw: float = 0.0

        # -- Episode horizon ---------------------------------------------------
        # `simulator_config.maxTime` is a global constant in WALL CLOCK seconds.
        # Two problems, both measured:
        #
        #  1. it is not enough. At speed_scale=0.42 the target speed is ~6.25 m/s
        #     and the roads are ~203 m: that needs ~32 s, while maxTime grants
        #     30. The car therefore covered ~92% of the track (less in practice,
        #     because the target speed is not held for the whole trip), and
        #     `angle_5` was almost never reached.
        #
        #  2. being wall clock, the portion of road covered depends on MACHINE
        #     LOAD: at 20.8 Hz (1 worker) 624 decisions fit in 30 s, at 8.5 Hz
        #     (4 workers) only 255. The same scenario covers two different
        #     portions depending on how many containers are running.
        #
        # The budget here is derived from the scenario with the same rule
        # MetaDrive uses (`scenarios/common/episode_budget.py`, vendored into
        # `Simulator/shared/`), so the backends share the horizon as well as the
        # geometry. It is still wall clock -- on Unity simulated time runs in
        # real time -- but it is no longer a blind constant.
        #
        # The NORMAL exit is still `agent.reached_end`: this is only the cap.
        try:
            from ..shared.driver import target_speed
            from ..shared.episode_budget import (
                budget_seconds, polyline_length,
            )

            # The same target speed the controller chases: the budget must match
            # the actual speed, not a nominal one.
            _v = target_speed(simulator_config.minSpeed, simulator_config.maxSpeed,
                              SPEED_SCALE)
            _pts = [(p.x, p.y) for p in road.road_points]
            max_time = budget_seconds(polyline_length(_pts), _v)
        except Exception as e:                       # degenerate geometry, broken import
            max_time = float(simulator_config.maxTime)
            print(f"[C2] per-scenario budget not computable ({e}); "
                  f"falling back to maxTime={max_time}s", flush=True)

        done_flag = False
        # dt measured between two consecutive decisions: the real time elapsed
        # between commands applied to the vehicle, which on Udacity varies with
        # load (20.8 Hz with 1 worker, 8.5 Hz with 4).
        last_decision_t = None
        loop_start = time.time()
        iterations = 0
        predictSeconds = 0.0
        stepSeconds = 0.0

        while not done_flag:
            t0 = time.perf_counter()
            dt = (t0 - last_decision_t) if last_decision_t is not None else None
            last_decision_t = t0
            actions = self.agent.predict(
                obs=obs,
                state=dict(speed=speed, simulator_name=UDACITY_SIM_NAME,
                           pos=pos, cte=xte_raw, dt=dt))
            predictSeconds += time.perf_counter() - t0

            if isinstance(self.env.action_space, gym.spaces.Box):   # type: ignore
                actions = np.clip(actions, self.env.action_space.low,
                                  self.env.action_space.high)       # type: ignore

            t0 = time.perf_counter()
            obs, done, info = self.env.step(actions)
            stepSeconds += time.perf_counter() - t0

            speed = info.get("speed", 0.0)
            pos = info.get("pos", pos)
            xte_raw = info["cte"]

            xte = xte_raw
            if CAP_XTE and abs(xte) > MAX_XTE:
                xte = MAX_XTE if xte > 0 else -MAX_XTE

            simulationOutput.addStats(
                position=info["pos"], speed=speed, xte=xte,
                steering=actions[0][0], throttle=actions[0][1])

            elapsed = time.time() - loop_start
            if (elapsed > max_time
                    or abs(xte_raw) > simulator_config.maxXTE
                    or done):
                done_flag = True
            # End of the road: past the last point the projection clamps to the
            # endpoint and the lateral error stops being a lateral error.
            # Carrying on would produce "failures" caused by the end of the
            # track. The roads are ~100 m long: at 15 m/s they are over in under
            # 7 s against the 30 s of maxTime, so this is the normal exit.
            if self.agent.reached_end:
                done_flag = True

            iterations += 1

        elapsedTime = time.time() - loop_start
        self.env.reset(skip_generation=False, track_string=waypoints)

        simulationOutput.elapsedTime = elapsedTime
        simulationOutput.iterations = iterations
        simulationOutput.predictSeconds = predictSeconds
        simulationOutput.stepSeconds = stepSeconds

        # Diagnostics: tells apart two faults with the same symptom.
        #   signs agree + small mismatch -> geometry OK, steering inverted
        #   segni discordi o mismatch grande  -> geometria sbagliata
        tot = self.agent.n_sign_agree + self.agent.n_sign_disagree
        agreeing = (self.agent.n_sign_agree / tot * 100.0) if tot else float("nan")
        print(f"[C2] xte_mismatch_max={self.agent.last_cte_mismatch:.3f} m | "
              f"sign agrees with Unity cte={agreeing:.0f}% ({tot} samples) | "
              f"reached_end={self.agent.reached_end} | step={iterations} | "
              f"{iterations / max(elapsedTime, 1e-9):.1f} Hz", flush=True)

        return simulationOutput
