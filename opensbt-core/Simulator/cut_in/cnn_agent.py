from __future__ import annotations

"""
Wraps the trained cut_in CNN as a controller_fn compatible with
CutInCarlaEpisode.run_episode() — mirrors
opensbt-core/Simulator/emergency_braking/cnn_agent.py.

Preprocessing constants MUST match scenarios/cut_in/train_cnn.py exactly.
"""

import numpy as np

from simulators.common.camera_dataset import preprocess

CARLA_CROP_TOP = 40
TARGET_WIDTH = 200
TARGET_HEIGHT = 66


def _crop(image: np.ndarray) -> np.ndarray:
    return image[CARLA_CROP_TOP:, :, :]


class CnnAgent:
    def __init__(self, model_path: str):
        import tensorflow as tf
        # safe_mode=False: trusted, locally-trained model — see
        # emergency_braking/cnn_agent.py for why this is required (Lambda layer).
        self.model = tf.keras.models.load_model(model_path, safe_mode=False)

    def __call__(self, frame, ego_speed: float, t: float, cutter_in_lane: bool) -> float:
        if frame is None:
            return 0.0
        arr = np.frombuffer(frame.raw_data, dtype=np.uint8)
        arr = arr.reshape(frame.height, frame.width, 4)[:, :, :3]
        img = preprocess(arr, crop_fn=_crop, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        img = np.expand_dims(img, axis=0)
        force = self.model(img, training=False).numpy().flatten()[0]
        return float(force)
