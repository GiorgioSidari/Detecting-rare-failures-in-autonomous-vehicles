import os
from typing import Dict, Tuple

import numpy as np

from .autopilot_model import AutopilotModel
from ..global_log import GlobalLog
from ..self_driving.agent import Agent
from ..self_driving.utils.dataset_utils import preprocess
from ..config import UDACITY_SIM_NAME, STEERING_CORRECTION


# ── Steering damping + speed gain (stability at low control rate) ──────────────
# At a low control rate the car covers several metres between two steers: the model reacts to a
# stale error and overcorrects, saturating the steering and oscillating to the opposite side. Two
# mitigations, both tunable via environment variable:
#
#   1) RATE LIMITER  - caps |delta steering| per step: trims the jerks that lead to saturation.
#      Active by default, gentle (0.20). 0 = disabled.
#
#   2) SPEED GAIN    - reduces steering authority as speed grows (like a real car: less steering at
#      speed), to counter the high-speed failure mode (~27 m/s) found by the rare-event analysis.
#      Active by default, gentle. Speed is now handled correctly in m/s (see UNIT FIX in predict),
#      so STEER_SPEED_REF is in m/s. gain = 1 / (1 + K*(v - ref)) for v > ref, else 1.
#
# Reversible: setting both to 0 restores the raw DNN steering.
STEER_MAX_RATE     = float(os.getenv("LK_STEER_MAX_RATE", "0.20"))     # |delta| max/step; 0=off
                                                                       # 0.15 over-damps (late
                                                                       # correction -> more mild fails)
STEER_SPEED_GAIN_K = float(os.getenv("LK_STEER_SPEED_GAIN_K", "0.03")) # 0=off; attenuate steering with v
STEER_SPEED_REF    = float(os.getenv("LK_STEER_SPEED_REF", "12.0"))    # m/s: threshold above which to attenuate


class SupervisedAgent(Agent):
    def __init__(
        self,
        env_name: str,
        model_path: str,
        max_speed: int,
        min_speed: int,
        input_shape: Tuple[int],
        predict_throttle: bool = False,
        fake_images: bool = False,
    ):
        super().__init__(env_name=env_name)

        self.logger = GlobalLog("supevised_agent")

        self.agent = AutopilotModel(
            env_name=env_name, input_shape=input_shape, predict_throttle=predict_throttle)
        self.agent.load(model_path=model_path)
        self.agent.model.compile(loss="sgd", metrics=["mse"])

        self.predict_throttle = predict_throttle
        self.model_path = model_path
        self.fake_images = fake_images

        self.max_speed = max_speed
        self.min_speed = min_speed

        # Rate-limiter state: last applied steering. Reset at the start of each run (when speed is
        # 0 at spawn) so the filter does not inherit the previous run's steering.
        self.prev_steering = 0.0

    def setSpeedLimits(self, minSpeed: int, maxSpeed: int):
        """Sets the speed limits for the agent.

        Args:
            minSpeed (int): minimum speed
            maxSpeed (int): maximum speed
        """
        self.min_speed = minSpeed
        self.max_speed = maxSpeed

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        obs = preprocess(image=obs, env_name=self.env_name,
                         fake_images=self.fake_images)

        # the model expects a 4D array
        obs = np.array([obs])

        speed = 0.0 if state.get("speed", None) is None else state["speed"]
        # UNIT FIX (km/h -> m/s). Udacity telemetry provides speed in km/h
        # (udacity_sim.py: `speed = float(data["speed"]) * 3.6`), while the ODD min/max_speed are
        # in m/s. The original code compared km/h with m/s: the regulator was thus almost always
        # in "slow down" (speed_limit=min_speed) and the (speed/speed_limit)^2 term was wrong ->
        # throttle often zeroed and effective speed decoupled from max_speed. We convert to m/s so
        # the speed regulator and the speed gain work in consistent units.
        speed_mps = speed / 3.6
        # Run start: speed is 0 at spawn -> reset the filter state so damping does not start from
        # the previous step's steering.
        if speed == 0.0:
            self.prev_steering = 0.0

        if self.predict_throttle:
            # TF fast path: model(obs) instead of model.predict().
            action = self.agent.model(obs, training=False)
            steering = float(np.asarray(action[0]).reshape(-1)[0])
            throttle = float(np.asarray(action[1]).reshape(-1)[0])
        else:
            # Inference on the TF fast path: model(obs, training=False) instead of model.predict(),
            # which rebuilds its predict function on every call. Fewer ms/step -> faster loop ->
            # higher control rate (numerically equivalent to .predict for a single obs).
            steering_raw = float(np.asarray(
                self.agent.model(obs, training=False)).reshape(-1)[0])
            if state["simulator_name"] == UDACITY_SIM_NAME:
                steering_raw = STEERING_CORRECTION * steering_raw

            steering = steering_raw

            # (2) Speed gain: less steering authority at high speed (speed_mps and STEER_SPEED_REF
            #     both in m/s).
            if STEER_SPEED_GAIN_K > 0.0:
                over = max(speed_mps - STEER_SPEED_REF, 0.0)
                steering *= 1.0 / (1.0 + STEER_SPEED_GAIN_K * over)

            # (1) Rate limiter: cap the per-step steering jump (anti-jerk). Trims the escalation
            #     that leads to saturation and overshoot to the opposite side.
            if STEER_MAX_RATE > 0.0:
                delta = float(np.clip(steering - self.prev_steering,
                                      -STEER_MAX_RATE, STEER_MAX_RATE))
                steering = self.prev_steering + delta
            self.prev_steering = steering

            if speed_mps > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            throttle = np.clip(a=1.0 - steering**2 - (speed_mps / speed_limit) ** 2,
                               a_min=0.0, a_max=1.0)

            # if the track begins with a curve the model steers at the maximum and the throttle
            # would be 0 since the speed is 0. Give a non-zero throttle so the car can start moving.
            # (checked on the RAW steering: the rate limiter must not mask a full-lock curve start)
            if abs(steering_raw) >= 1.0 and throttle == 0.0 and speed == 0.0 and len(state) > 0:
                self.logger.warn(
                    "Road starts with a curve! Giving the car an extra throttle")
                throttle = 0.5

        return np.asarray([[steering, throttle]], dtype=np.float32)
