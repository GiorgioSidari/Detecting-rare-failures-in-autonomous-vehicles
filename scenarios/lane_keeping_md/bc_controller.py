"""
Learned behavioral-cloning (BC) state controller for the MetaDrive lane-keeping
backend — variant A2 (state-based), Part C item 1.

Rationale (see the design doc, Part C): the pure-pursuit driver is robust, so its
failures had to be injected, which is a validity threat. A small MLP trained by
behavioral cloning of the pure-pursuit teacher, but ONLY on an easy training
distribution (gentle curves, low speed), extrapolates poorly outside its support:
on sharp curves / high speed it steers wrong and leaves the lane. The failure
boundary then coincides with the edge of the training distribution — a GENUINE
generalization failure, the state-based analogue of the Udacity DNN's fragility,
with no rendering and no hand-injected degradation.

Implementation note: we use scikit-learn's MLPRegressor rather than Keras/TF on
purpose. TensorFlow's native runtime conflicts with panda3d (MetaDrive) when both
are loaded in the same process on Windows (DLL init failure). sklearn coexists
with MetaDrive cleanly (Part B already runs GP + MetaDrive together), so the whole
A2 pipeline — collect on MetaDrive, then train — runs in one process.
"""
from __future__ import annotations

import numpy as np

from scenarios.lane_keeping_md.driver import Driver, _clip

# Feature order fed to the MLP, and fixed scales to normalise them to ~O(1) so
# training and inference agree (lane half-width, pi, ODD mid speed).
FEATURE_SCALE = np.array([2.5, np.pi, 15.0, 15.0], dtype=np.float64)


def state_to_features(state: dict) -> np.ndarray:
    """(lateral_error, heading_error, speed, target_speed) -> scaled (4,) array."""
    raw = np.array([
        float(state.get("lateral_error", 0.0)),
        float(state.get("heading_error", 0.0)),
        float(state.get("speed", 0.0)),
        float(state.get("target_speed", 0.0)),
    ], dtype=np.float64)
    return raw / FEATURE_SCALE


def build_bc_mlp(hidden=(32, 32), max_iter: int = 300, seed: int = 0):
    """Small sklearn MLP regressor: scaled state features -> steering. Deliberately
    low capacity so it fits the teacher in-distribution but extrapolates poorly
    outside it (the source of genuine generalization failures)."""
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=tuple(hidden), activation="relu",
                        solver="adam", alpha=1e-4, max_iter=int(max_iter),
                        random_state=seed)


def save_model(model, path: str) -> None:
    import joblib
    joblib.dump(model, path)


def load_model(path: str):
    import joblib
    return joblib.load(path)


class LearnedDriver(Driver):
    """
    Driver whose STEERING comes from a trained model; throttle stays a simple
    proportional speed regulator. Accepts a fitted model object OR a path to a
    joblib-saved sklearn model. Also accepts a plain callable (for tests).

    Parameters
    ----------
    model      : fitted estimator with .predict, or a callable feat->steer.
    model_path : path to a joblib-saved model (loaded lazily) if `model` is None.
    k_throttle : proportional gain of the speed regulator.
    max_rate   : steering rate limiter per step (0 = off).
    """

    def __init__(self, model=None, model_path: str | None = None,
                 k_throttle: float = 0.3, max_rate: float = 0.20):
        self._model = model
        self._model_path = model_path
        self.k_throttle = float(k_throttle)
        self.max_rate = float(max_rate)
        self._prev_steer = 0.0

    def _ensure_model(self):
        if self._model is None:
            if self._model_path is None:
                raise ValueError("LearnedDriver needs `model` or `model_path`.")
            self._model = load_model(self._model_path)
        return self._model

    def reset(self) -> None:
        self._prev_steer = 0.0

    def _predict_steer(self, feat: np.ndarray) -> float:
        """feat: (4,) scaled features -> scalar steering."""
        model = self._ensure_model()
        x = feat.reshape(1, -1)
        if hasattr(model, "predict"):          # sklearn estimator
            out = model.predict(x)
        else:                                  # plain callable (tests)
            out = model(x)
        return float(np.asarray(out).ravel()[0])

    def act(self, state: dict) -> tuple[float, float]:
        feat = state_to_features(state)
        steer = _clip(self._predict_steer(feat), -1.0, 1.0)

        if self.max_rate and self.max_rate > 0.0:
            delta = _clip(steer - self._prev_steer, -self.max_rate, self.max_rate)
            steer = self._prev_steer + delta
        self._prev_steer = steer

        speed = float(state.get("speed", 0.0))
        target = float(state.get("target_speed", 0.0))
        throttle = _clip(self.k_throttle * (target - speed), -1.0, 1.0)
        return steer, throttle
