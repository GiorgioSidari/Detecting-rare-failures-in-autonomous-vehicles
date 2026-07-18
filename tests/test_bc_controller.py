"""
Tests for the learned BC controller (Part C item 1 / A2).

The pure parts (feature scaling, LearnedDriver with a dummy model) need no
TensorFlow; the MLP builder test is skipped if TensorFlow is absent.

Run:  pytest tests/test_bc_controller.py -q
"""
import numpy as np
import pytest

from scenarios.lane_keeping_md.bc_controller import (
    state_to_features, LearnedDriver, FEATURE_SCALE,
)


def test_state_to_features_scaling():
    s = {"lateral_error": 2.5, "heading_error": np.pi, "speed": 15.0, "target_speed": 15.0}
    f = state_to_features(s)
    assert np.allclose(f, [1.0, 1.0, 1.0, 1.0], atol=1e-6)
    # zero state -> zero features
    assert np.allclose(state_to_features({}), [0, 0, 0, 0])


def test_learned_driver_uses_model_and_regulates_throttle():
    # Dummy model: steer = -lateral_feature (steer back to centre). Returns (B,1).
    def model(x):
        return -x[:, [0]]
    d = LearnedDriver(model=model, max_rate=0.0)
    steer, throttle = d.act({"lateral_error": 2.5, "heading_error": 0,
                             "speed": 5.0, "target_speed": 10.0})
    assert steer < 0                 # car left of centre -> steer right
    assert throttle > 0              # below target speed -> accelerate


def test_learned_driver_rate_limiter():
    def model(x):
        return np.array([[1.0]])     # always commands full-left steering
    d = LearnedDriver(model=model, max_rate=0.1)
    st = {"lateral_error": 0, "heading_error": 0, "speed": 0, "target_speed": 0}
    s1, _ = d.act(st)
    s2, _ = d.act(st)
    assert abs(s1) <= 0.1 + 1e-9
    assert abs(s2 - s1) <= 0.1 + 1e-9


def test_output_clipped_to_unit():
    def model(x):
        return np.array([[5.0]])     # out of range
    d = LearnedDriver(model=model, max_rate=0.0)
    s, _ = d.act({"lateral_error": 0, "heading_error": 0, "speed": 0, "target_speed": 0})
    assert -1.0 <= s <= 1.0


def test_build_and_fit_sklearn_mlp():
    from scenarios.lane_keeping_md.bc_controller import build_bc_mlp
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(FEATURE_SCALE)))
    y = -X[:, 0]                                   # steer ~ -lateral
    m = build_bc_mlp(hidden=(8, 8), max_iter=50)
    m.fit(X, y)
    out = m.predict(X[:3])
    assert out.shape == (3,)


def test_learned_driver_from_fitted_model():
    from scenarios.lane_keeping_md.bc_controller import build_bc_mlp, LearnedDriver
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_SCALE)))
    y = -X[:, 0]                                   # steer opposite to lateral feature
    m = build_bc_mlp(hidden=(16, 16), max_iter=200).fit(X, y)
    d = LearnedDriver(model=m, max_rate=0.0)
    steer, _ = d.act({"lateral_error": 2.5, "heading_error": 0, "speed": 5, "target_speed": 10})
    assert steer < 0                               # learned to steer back to centre
