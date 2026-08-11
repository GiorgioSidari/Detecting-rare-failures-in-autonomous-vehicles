from __future__ import annotations

"""
Wraps the trained emergency_braking CNN as a controller_fn compatible with
EmergencyBrakingCarlaEpisode.run_episode() — the CARLA-side counterpart of
opensbt-core/Simulator/lanekeeping/self_driving/supervised_agent.py.

Preprocessing constants MUST match scenarios/emergency_braking/train_cnn.py
exactly (crop/resize) — they are duplicated here rather than imported to
keep this module (meant to run inside the simulator-side service/container)
independent from the main repo's scenarios/ package.
"""

import numpy as np

from simulators.common.camera_dataset import preprocess

CARLA_CROP_TOP = 40
TARGET_WIDTH = 200
TARGET_HEIGHT = 66


def _crop(image: np.ndarray) -> np.ndarray:
    return image[CARLA_CROP_TOP:, :, :]


class CnnAgent:
    """Loads the trained Keras model once; __call__ matches controller_fn's signature."""

    def __init__(self, model_path: str):
        import tensorflow as tf
        # safe_mode=False: the model's Lambda normalisation layer (see
        # simulators/common/pilotnet.py::build_pilotnet) is a plain Python
        # lambda, which Keras 3 refuses to deserialize by default (arbitrary
        # code execution risk for untrusted artifacts). We trust it — it's
        # our own model, trained locally by train_cnn.py.
        self.model = tf.keras.models.load_model(model_path, safe_mode=False)

    def __call__(self, frame, ego_speed: float, t: float, lead_braking: bool) -> float:
        if frame is None:
            return 0.0
        arr = np.frombuffer(frame.raw_data, dtype=np.uint8)
        arr = arr.reshape(frame.height, frame.width, 4)[:, :, :3]   # BGRA -> BGR
        img = preprocess(arr, crop_fn=_crop, width=TARGET_WIDTH, height=TARGET_HEIGHT)
        img = np.expand_dims(img, axis=0)
        force = self.model(img, training=False).numpy().flatten()[0]
        return float(force)
