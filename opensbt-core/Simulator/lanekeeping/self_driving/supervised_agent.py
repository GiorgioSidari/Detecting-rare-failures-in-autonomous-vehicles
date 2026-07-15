import os
from typing import Dict, Tuple

import numpy as np

from ..road_generator.custom_road_generator import CustomRoadGenerator
from .autopilot_model import AutopilotModel
from ..custom_types import GymEnv
from ..global_log import GlobalLog
from ..self_driving.agent import Agent
from ..self_driving.utils.dataset_utils import preprocess
from ..config import UDACITY_SIM_NAME, STEERING_CORRECTION


# ── Smorzamento sterzo + gain-velocita' (stabilita' a bassa cadenza) ──────────
# A ~1.5 Hz di controllo l'auto percorre ~2-3 m tra due sterzate: il modello
# reagisce a un errore ormai "vecchio" e sovra-corregge, saturando lo sterzo e
# oscillando dal lato opposto (modo di fallimento #17/#20). Due mitigazioni,
# entrambe tarabili via variabile d'ambiente:
#
#   1) RATE LIMITER  — limita |Δsterzo| per step: taglia i salti/jerk che portano
#      alla saturazione. ATTIVO di default, gentile (0.20). 0 = disattivato.
#
#   2) GAIN-VELOCITA' — riduce l'autorita' di sterzo al crescere della velocita'
#      (come un'auto vera: meno sterzo in velocita'). DISATTIVO di default (K=0)
#      perche' state["speed"] arriva in km/h mentre max_speed e' in m/s: il
#      coefficiente va calibrato sui dati prima di attivarlo (LK_STEER_SPEED_*).
#
# Sono unit-indipendenti dove possibile e completamente reversibili: mettendo
# tutto a 0 si torna al comportamento originale (solo sterzo grezzo del DNN).
STEER_MAX_RATE     = float(os.getenv("LK_STEER_MAX_RATE", "0.20"))    # |Δ| max/step; 0=off
                                                                      # 0.15 sovra-smorza (correzione
                                                                      # tardiva -> piu' fallimenti miti)
STEER_SPEED_GAIN_K = float(os.getenv("LK_STEER_SPEED_GAIN_K", "0.0")) # 0=off
STEER_SPEED_REF    = float(os.getenv("LK_STEER_SPEED_REF", "20.0"))   # unita' di state["speed"]


class SupervisedAgent(Agent):
    def __init__(
        self,
        # env: GymEnv,
        env_name: str,
        model_path: str,
        max_speed: int,
        min_speed: int,
        input_shape: Tuple[int],
        predict_throttle: bool = False,
        fake_images: bool = False,
    ):
        super().__init__(
            # env=env,
            env_name=env_name)

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

        # Stato del rate-limiter: ultimo sterzo applicato. Viene azzerato
        # all'inizio di ogni run (quando la velocita' e' 0 allo spawn) cosi' il
        # filtro non "eredita" lo sterzo del run precedente.
        self.prev_steering = 0.0

    def setSpeedLimits(self, minSpeed:int, maxSpeed:int):
        """Sets the speed limits for the agent

        Args:
            minSpeed (int): minimum Speed
            maxSpeed (int): maximum Speed
        """
        self.min_speed = minSpeed
        self.max_speed = maxSpeed

    def predict(self, obs: np.ndarray, state: Dict) -> np.ndarray:
        obs = preprocess(image=obs, env_name=self.env_name,
                         fake_images=self.fake_images)

        # the model expects 4D array
        obs = np.array([obs])

        speed = 0.0 if state.get("speed", None) is None else state["speed"]
        # Inizio run: allo spawn la velocita' e' 0 -> azzera lo stato del filtro
        # cosi' lo smorzamento non parte dallo sterzo dell'ultimo step precedente.
        if speed == 0.0:
            self.prev_steering = 0.0

        if self.predict_throttle:
            # Percorso veloce di TF: model(obs) invece di model.predict()
            action = self.agent.model(obs, training=False)
            steering = float(np.asarray(action[0]).reshape(-1)[0])
            throttle = float(np.asarray(action[1]).reshape(-1)[0])
        else:
            multiplier = 1
            # Inferenza sul percorso veloce di TF: model(obs, training=False)
            # invece di model.predict(), che ricostruisce la funzione di predizione
            # a ogni chiamata. Meno ms/step -> loop piu' veloce -> cadenza di
            # controllo piu' alta (equivalente numerico di .predict per singolo obs).
            steering_raw = float(np.asarray(
                self.agent.model(obs, training=False)).reshape(-1)[0])
            if state["simulator_name"] == UDACITY_SIM_NAME:
                steering_raw = STEERING_CORRECTION * steering_raw

            steering = steering_raw

            # ── (2) Gain-velocita': meno autorita' di sterzo alle alte velocita' ──
            if STEER_SPEED_GAIN_K > 0.0:
                over = max(float(speed) - STEER_SPEED_REF, 0.0)
                steering *= 1.0 / (1.0 + STEER_SPEED_GAIN_K * over)

            # ── (1) Rate limiter: limita il salto di sterzo per step (anti-jerk) ──
            #     Taglia proprio l'escalation che porta alla saturazione e alla
            #     sovra-correzione dal lato opposto.
            if STEER_MAX_RATE > 0.0:
                delta = float(np.clip(steering - self.prev_steering,
                                      -STEER_MAX_RATE, STEER_MAX_RATE))
                steering = self.prev_steering + delta
            self.prev_steering = steering

            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            throttle = multiplier * \
                np.clip(a=1.0 - steering**2 - (speed / speed_limit)
                        ** 2, a_min=0.0, a_max=1.0)

            # if the track begins with a curve the model steers at the maximum and the throttle will be 0 since the
            # speed is 0. To counteract this give a non-zero throttle such that the car can start going
            # (controllo sullo sterzo GREZZO: il rate-limiter non deve mascherare
            #  una partenza in curva a tutto sterzo)
            if abs(steering_raw) >= 1.0 and throttle == 0.0 and speed == 0.0 and len(state) > 0:
                self.logger.warn(
                    "Road starts with a curve! Giving the car an extra throttle")
                throttle = 0.5

        return np.asarray([[steering, throttle]], dtype=np.float32)
