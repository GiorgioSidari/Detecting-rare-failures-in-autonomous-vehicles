import os
from typing import Dict, Tuple

import numpy as np

from .autopilot_model import AutopilotModel
from ..global_log import GlobalLog
from ..self_driving.agent import Agent
from ..self_driving.utils.dataset_utils import preprocess
from ..config import UDACITY_SIM_NAME, STEERING_CORRECTION


# Steering stabilisation at low control rate. At a low rate the car covers several metres between
# two steers, so it reacts to a stale error, overcorrects and oscillates. Two mitigations, both
# tunable via env var, 0 = off (setting both to 0 restores the raw DNN steering):
#   RATE LIMITER - caps |Δsteering|/step to trim the jerks that saturate the steering.
#   SPEED GAIN   - reduces steering authority as speed grows, to counter the high-speed failure mode.
STEER_MAX_RATE     = float(os.getenv("LK_STEER_MAX_RATE", "0.20"))      # |Δ| max/step; 0 = off
STEER_SPEED_GAIN_K = float(os.getenv("LK_STEER_SPEED_GAIN_K", "0.03"))  # attenuate steering with speed
STEER_SPEED_REF    = float(os.getenv("LK_STEER_SPEED_REF", "12.0"))     # m/s: gain = 1/(1+K*(v-ref)) above this


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
        # UNIT FIX: Udacity telemetry is in km/h but the ODD min/max_speed are in m/s. The original
        # code compared the two directly, keeping the regulator stuck in "slow down" and zeroing the
        # throttle. Convert to m/s so the speed regulator and the speed gain use consistent units.
        speed_mps = speed / 3.6
        # Run start (speed 0 at spawn): reset the filter so damping doesn't carry over the last run.
        if speed == 0.0:
            self.prev_steering = 0.0

        if self.predict_throttle:
            # TF fast path: model(obs) instead of model.predict().
            action = self.agent.model(obs, training=False)
            steering = float(np.asarray(action[0]).reshape(-1)[0])
            throttle = float(np.asarray(action[1]).reshape(-1)[0])
        else:
            # TF fast path: model(obs) instead of model.predict() (which rebuilds its predict
            # function every call) -> faster loop, higher control rate, same result for one obs.
            steering_raw = float(np.asarray(
                self.agent.model(obs, training=False)).reshape(-1)[0])
            if state["simulator_name"] == UDACITY_SIM_NAME:
                steering_raw = STEERING_CORRECTION * steering_raw

            steering = steering_raw

            # Speed gain: less steering authority at high speed.
            if STEER_SPEED_GAIN_K > 0.0:
                over = max(speed_mps - STEER_SPEED_REF, 0.0)
                steering *= 1.0 / (1.0 + STEER_SPEED_GAIN_K * over)

            # Rate limiter: cap the per-step steering jump (anti-jerk) to avoid saturation/overshoot.
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

            # Track starting with a curve: the model steers full-lock at speed 0, so throttle is 0
            # and the car never moves -> give it a nudge. Checked on the RAW steering so the rate
            # limiter can't mask a full-lock start.
            if abs(steering_raw) >= 1.0 and throttle == 0.0 and speed == 0.0 and len(state) > 0:
                self.logger.warn(
                    "Road starts with a curve! Giving the car an extra throttle")
                throttle = 0.5

        return np.asarray([[steering, throttle]], dtype=np.float32)
