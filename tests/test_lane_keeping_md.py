"""
Tests for the MetaDrive lane-keeping backend (Step A / A1).

These tests DO NOT require MetaDrive: the only MetaDrive-coupled method
(_simulate_one) is overridden with a fake, so we validate the pure logic
(param mapping, driver, shared QoI) and the BaseScenario interface conformance
in isolation. The real MetaDrive smoke test (one headless episode) is a separate
manual check on a machine with `metadrive-simulator` installed.

Run:  pytest tests/test_lane_keeping_md.py -q
"""
import numpy as np
import pytest

from scenarios.lane_keeping.qoi_lane import composite_lane_qoi, MAX_XTE
from scenarios.lane_keeping_md.map_builder import (
    angle_to_radius, build_scenario_spec, target_speed,
    MIN_RADIUS, STRAIGHT_RADIUS, STRAIGHT_ANGLE_EPS,
)
from scenarios.lane_keeping_md.driver import PurePursuitDriver
from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario


# ── map_builder ───────────────────────────────────────────────────────────────

def test_angle_to_radius_monotonic():
    # Sharper angle -> smaller radius (over a fixed arc length).
    r_gentle = angle_to_radius(10, seg_length=25)
    r_sharp  = angle_to_radius(80, seg_length=25)
    assert r_sharp < r_gentle
    # Near-zero angle -> treated as straight.
    assert angle_to_radius(0.0, 25) >= STRAIGHT_RADIUS
    assert angle_to_radius(STRAIGHT_ANGLE_EPS / 2, 25) >= STRAIGHT_RADIUS
    # Radius stays within clamps.
    assert angle_to_radius(85, 5) >= MIN_RADIUS


def test_build_scenario_spec_layout():
    row = np.array([0, 30, 60, 0, 85, 6.0, 20.0, 25.0, 300.0])
    spec = build_scenario_spec(row, ncols=len(row))
    assert len(spec.blocks) == 5
    assert spec.min_speed == 6.0 and spec.max_speed == 20.0
    assert spec.map_size == 300.0
    # angle 0 -> straight, angle 85 -> curve
    assert spec.blocks[0].kind == "S"
    assert spec.blocks[4].kind == "C"
    # sharpest curve is the one with the largest angle (smallest radius)
    assert spec.sharpest_curve_radius <= spec.blocks[2].radius
    assert target_speed(spec) == pytest.approx(13.0)


# ── driver ────────────────────────────────────────────────────────────────────

def test_driver_steers_back_to_centre():
    d = PurePursuitDriver(max_rate=0.0)  # disable rate limit for a clean sign test
    # Car to the LEFT of centre (lateral_error > 0) must steer RIGHT (negative).
    steer, _ = d.act({"lateral_error": 1.0, "heading_error": 0.0,
                      "speed": 5.0, "target_speed": 10.0})
    assert steer < 0
    d.reset()
    steer2, _ = d.act({"lateral_error": -1.0, "heading_error": 0.0,
                       "speed": 5.0, "target_speed": 10.0})
    assert steer2 > 0


def test_driver_rate_limiter_caps_delta():
    d = PurePursuitDriver(max_rate=0.1)
    s1, _ = d.act({"lateral_error": 5.0, "heading_error": 0.0, "speed": 0, "target_speed": 0})
    assert abs(s1 - 0.0) <= 0.1 + 1e-9      # first step limited from prev=0
    s2, _ = d.act({"lateral_error": 5.0, "heading_error": 0.0, "speed": 0, "target_speed": 0})
    assert abs(s2 - s1) <= 0.1 + 1e-9


def test_driver_throttle_regulator_sign():
    d = PurePursuitDriver()
    _, th_up = d.act({"lateral_error": 0, "heading_error": 0, "speed": 2.0, "target_speed": 10.0})
    _, th_dn = d.act({"lateral_error": 0, "heading_error": 0, "speed": 10.0, "target_speed": 2.0})
    assert th_up > 0 and th_dn < 0


# ── shared QoI ─────────────────────────────────────────────────────────────────

def _traj(xte, steer):
    n = len(xte)
    t = np.zeros((1, n, 4), dtype=np.float32)
    t[0, :, 2] = xte
    t[0, :, 3] = steer
    return t


def test_qoi_safe_case_value():
    t = _traj([0.5] * 5, [0.1] * 5)
    qoi, meta = composite_lane_qoi(t, np.zeros((1, 9)), run_lengths=[5])
    # M1 = 2.5-0.5=2.0 ; M2=0 ; M3=0 -> qoi = 1.2
    assert qoi[0] == pytest.approx(1.2, abs=1e-6)
    assert meta["valid_mask"][0]
    assert meta["n_invalid"] == 0


def test_qoi_failure_case_negative():
    t = _traj([0, 1, 2, 3, 3], [0, 0.2, 0.5, 0.8, 0.8])
    qoi, _ = composite_lane_qoi(t, np.zeros((1, 9)), run_lengths=[5])
    assert qoi[0] < 0            # left the lane -> failure


def test_qoi_degenerate_run_is_invalid():
    t = _traj([0.1, 0.1, 0.1, 0.1, 0.1], [0.0] * 5)
    qoi, meta = composite_lane_qoi(t, np.zeros((1, 9)), run_lengths=[2])  # < MIN_VALID_STEPS
    assert np.isnan(qoi[0])
    assert not meta["valid_mask"][0]
    assert meta["n_degenerate"] == 1


def test_qoi_bad_params_incoherent_speed():
    t = _traj([0.2] * 5, [0.0] * 5)
    params = np.zeros((1, 9)); params[0, 5] = 20.0; params[0, 6] = 10.0  # min>max
    qoi, meta = composite_lane_qoi(t, params, run_lengths=[5])
    assert np.isnan(qoi[0]) and not meta["valid_mask"][0]


# ── scenario interface (MetaDrive mocked out) ──────────────────────────────────

class _FakeMDScenario(LaneKeepingMetaDriveScenario):
    """Overrides the only MetaDrive-coupled method with a deterministic fake."""
    def _simulate_one(self, row, ncols, seed=0, verbose=False):
        # Length grows with the max angle, so runs have different lengths (tests padding).
        L = 4 + int(row[:5].max() // 20)
        traj = np.zeros((L, 4), dtype=np.float32)
        traj[:, 0] = np.arange(L)                    # x
        traj[:, 2] = 0.3 * np.sin(np.arange(L))      # xte within lane
        traj[:, 3] = 0.05                            # steering
        fid = {"control_hz": self.control_hz_nominal,
               "meters_per_step": 1.0, "infer_ms": 0.1, "wait_ms": 0.2}
        return traj, fid


def test_param_bounds_and_distributions():
    sc = LaneKeepingMetaDriveScenario()
    b = sc.param_bounds()
    assert len(b["lower"]) == 9 and len(b["upper"]) == 9 and len(b["names"]) == 9
    dists = sc.param_distributions(b["lower"], b["upper"])
    assert len(dists) == 9
    # each is a frozen scipy dist with a ppf
    u = np.linspace(0.1, 0.9, 5)
    for d in dists:
        vals = d.ppf(u)
        assert np.all(np.isfinite(vals))
    assert sc.failure_threshold() == 0.0


def test_control_rate_is_exact_config():
    sc = LaneKeepingMetaDriveScenario(decision_repeat=5, physics_world_step_size=0.02)
    assert sc.control_hz_nominal == pytest.approx(10.0)
    sc2 = LaneKeepingMetaDriveScenario(decision_repeat=2, physics_world_step_size=0.02)
    assert sc2.control_hz_nominal == pytest.approx(25.0)


def test_run_simulation_shape_and_padding():
    sc = _FakeMDScenario()
    params = np.array([
        [0,  0,  0,  0,  0,  5.0, 10.0, 25.0, 250.0],   # short run
        [80, 40, 0,  0,  0,  6.0, 20.0, 25.0, 250.0],   # longer run
        [85, 85, 85, 85, 85, 8.0, 25.0, 25.0, 250.0],   # longest run
    ], dtype=float)
    traj = sc.run_simulation(params)
    assert traj.ndim == 3 and traj.shape[0] == 3 and traj.shape[2] == 4
    assert len(sc._run_lengths) == 3
    assert traj.shape[1] == max(sc._run_lengths)         # padded to the longest
    # QoI runs end-to-end and populates the orchestrator-facing attributes.
    qoi = sc.compute_qoi(traj, params)
    assert qoi.shape == (3,)
    assert sc._valid_mask is not None and len(sc._valid_mask) == 3
    assert sc._last_control_hz[0] == pytest.approx(sc.control_hz_nominal)
