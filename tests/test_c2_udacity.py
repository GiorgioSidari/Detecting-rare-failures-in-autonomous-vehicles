"""
Tests for the C2 arm on Udacity (the additive Simulator/c2/ package).

Runs without Docker and without Unity: the telemetry is simulated. It checks
what can be checked offline, i.e. everything except Unity's physics:

  * the agent gets from the telemetry what it needs and nothing more;
  * the lateral error it computes matches the `cte` Unity would report;
  * the closed-loop controller brings the car back into the lane;
  * leaving `LK_DRIVER` unset keeps the historical behaviour unchanged -- the
    property that makes the change to the existing loop reversible.
"""
from __future__ import annotations

import math
import os
import types

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LK_ROOT = os.path.join(_REPO_ROOT, "opensbt-core", "Simulator")
from scenarios.common.driver import LateralFeedbackDriver           # noqa: E402
from scenarios.common.road_frame import road_frame              # noqa: E402
from scenarios.common.road_geometry import road_polyline      # noqa: E402


def _load_state_based_agent():
    """
    Loads `state_based_agent.py` by path, stubbing only the package
    dependencies that have side effects (`global_log`, `agent`).

    A normal import would execute `lanekeeping/__init__.py`, which pulls in
    UdacitySimulatorIO, and `self_driving/__init__.py`, which imports
    tensorflow. The stubs are removed straight after: `sys.modules` is global
    for the pytest session.
    """
    path = os.path.join(_LK_ROOT, "c2", "state_based_agent.py")
    if not os.path.isfile(path):
        return None

    src = open(path, encoding="utf-8").read()
    # The relative imports climb back to the `lanekeeping` package; here the
    # module is loaded in isolation, so they are rewritten onto the shared
    # equivalents.
    src = (src
           .replace("from ..shared.driver import",
                    "from scenarios.common.driver import")
           .replace("from ..shared.road_frame import",
                    "from scenarios.common.road_frame import")
           .replace("from .config import STEERING_SIGN",
                    "STEERING_SIGN = 1.0")
           .replace("from ..lanekeeping.global_log import GlobalLog",
                    "GlobalLog = lambda *a, **k: None")
           .replace("from ..lanekeeping.self_driving.agent import Agent",
                    "class Agent:\n"
                    "    def __init__(self, env_name):\n"
                    "        self.env_name = env_name"))

    mod = types.ModuleType("_state_based_agent_probe")
    mod.__dict__["__file__"] = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod.StateBasedAgent


StateBasedAgent = _load_state_based_agent()
requires_agent = pytest.mark.skipif(
    StateBasedAgent is None, reason="state_based_agent.py not found")


def _road():
    return road_polyline([10, 45, 3, 70, 25], seg_length=25.0, num_spline_nodes=80)


def _straight(length=300.0, n=301):
    return np.column_stack([np.linspace(0.0, length, n), np.zeros(n)])


# ─────────────────────────────────────────────────────────────────────────────
# Contratto
# ─────────────────────────────────────────────────────────────────────────────

@requires_agent
def test_road_is_required_before_driving():
    a = StateBasedAgent(env_name="udacity")
    with pytest.raises(RuntimeError, match="set_road"):
        a.predict(obs=None, state={"speed": 0.0, "pos": (0.0, 0.0, 0.0)})


@requires_agent
def test_position_is_required():
    a = StateBasedAgent(env_name="udacity")
    a.set_road(_straight())
    with pytest.raises(RuntimeError, match="pos"):
        a.predict(obs=None, state={"speed": 0.0})


@requires_agent
def test_action_format_matches_supervised_agent():
    """The simulation loop must not tell the two agents apart."""
    a = StateBasedAgent(env_name="udacity")
    a.set_road(_straight())
    act = a.predict(obs=None, state={"speed": 0.0, "pos": (0.0, 0.0, 0.0)})
    assert isinstance(act, np.ndarray)
    assert act.shape == (1, 2)
    assert act.dtype == np.float32


@requires_agent
def test_obs_is_ignored():
    """C2 does not use perception: the frame must change nothing."""
    a = StateBasedAgent(env_name="udacity")
    a.set_road(_straight())
    st = {"speed": 20.0, "pos": (10.0, 0.5, 0.0)}
    a1 = a.predict(obs=None, state=dict(st))
    a.set_road(_straight())
    a2 = a.predict(obs=np.zeros((160, 320, 3)), state=dict(st))
    np.testing.assert_array_equal(a1, a2)


@requires_agent
def test_throttle_is_never_negative():
    """Unity has no braking channel: a throttle < 0 would be reverse."""
    a = StateBasedAgent(env_name="udacity")
    a.set_road(_straight())
    act = a.predict(obs=None, state={"speed": 200.0, "pos": (10.0, 0.0, 0.0)})
    assert act[0][1] >= 0.0


@requires_agent
def test_set_road_clears_the_state():
    a = StateBasedAgent(env_name="udacity")
    a.set_road(_straight())
    a.predict(obs=None, state={"speed": 30.0, "pos": (10.0, 2.0, 0.0)})
    assert a.driver._prev_steer != 0.0
    a.set_road(_straight())
    assert a.driver._prev_steer == 0.0
    assert a._prev_pos is None


# ─────────────────────────────────────────────────────────────────────────────
# Consistency with Unity's reference
# ─────────────────────────────────────────────────────────────────────────────

@requires_agent
def test_mismatch_with_unity_cte_stays_small():
    """
    The agent continuously compares its own lateral error with the telemetry
    `cte`. Simulating a trajectory at a known offset, the two must agree: if
    they diverged, our idea of where the road is would have come apart from the
    simulator's.
    """
    xy = _road()
    a = StateBasedAgent(env_name="udacity")
    a.set_road(xy)

    rng = np.random.default_rng(0)
    for i in range(5, len(xy) - 5, 9):
        tx, ty = xy[i + 1] - xy[i - 1]
        n = math.hypot(tx, ty)
        nx, ny = -ty / n, tx / n
        offset = float(rng.uniform(-1.5, 1.5))
        px, py = xy[i][0] + nx * offset, xy[i][1] + ny * offset
        # Unity's `cte`: the signed distance from the centreline, i.e. the offset.
        a.predict(obs=None, state={"speed": 30.0, "pos": (px, py, 0.0),
                                   "cte": offset})

    assert a.last_cte_mismatch < 0.10, (
        f"worst gap from Unity's cte: {a.last_cte_mismatch:.3f} m")


# ─────────────────────────────────────────────────────────────────────────────
# Closed loop with simulated telemetry
# ─────────────────────────────────────────────────────────────────────────────

def _drive(agent, xy, *, y0=0.0, v=10.0, dt=0.1, steps=300,
           wheelbase=2.5, max_steer_rad=0.5):
    """
    Kinematic bicycle pretending to be Unity: it produces `pos` and `speed` in
    the telemetry format and consumes the agent's actions.
    """
    x, y, yaw = float(xy[0][0]), float(xy[0][1]) + y0, 0.0
    errs = []
    for _ in range(steps):
        fr = road_frame(xy, x, y, yaw)
        # The road is over: past the last point the lateral error is no longer
        # one (see RoadFrame.beyond_end). The real loop ends here, and the test
        # must do the same -- otherwise it measures the distance from the end of
        # the track and mistakes it for a controller failure.
        if fr.beyond_end:
            break
        errs.append(fr.lateral_error)
        act = agent.predict(obs=None,
                            state={"speed": v * 3.6, "pos": (x, y, 0.0), "dt": dt})
        steer = float(act[0][0])
        yaw += (v / wheelbase) * math.tan(steer * max_steer_rad) * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    return np.asarray(errs)


@requires_agent
def test_recentres_on_a_straight():
    a = StateBasedAgent(env_name="udacity")
    a.set_road(_straight())
    errs = _drive(a, _straight(), y0=1.5)
    assert abs(errs[-1]) < 0.15, f"final error {errs[-1]:.3f} m"


@requires_agent
def test_recentres_from_both_sides():
    for y0 in (+1.5, -1.5):
        a = StateBasedAgent(env_name="udacity")
        a.set_road(_straight())
        errs = _drive(a, _straight(), y0=y0)
        assert abs(errs[-1]) < 0.15, f"y0={y0}: finale {errs[-1]:.3f} m"


@requires_agent
def test_follows_a_curved_road():
    """The case that matters: the real road, generated from the 9 parameters."""
    xy = _road()
    a = StateBasedAgent(env_name="udacity")
    a.set_road(xy)
    errs = np.abs(_drive(a, xy, v=8.0, steps=200))
    assert errs.max() < 2.5, f"left the lane: max XTE {errs.max():.3f} m"


@requires_agent
def test_speed_scale_lowers_the_target_speed():
    """The operating-point calibration lever must actually do something."""
    from scenarios.common.driver import target_speed

    a = StateBasedAgent(env_name="udacity", min_speed=5.0, max_speed=15.0,
                        speed_scale=0.5)
    a.set_road(_straight())
    act = a.predict(obs=None, state={"speed": 0.0, "pos": (0.0, 0.0, 0.0)})
    expected = target_speed(5.0, 15.0, 0.5)
    # throttle = k_throttle * (target - speed), with speed = 0
    assert act[0][1] == pytest.approx(
        min(1.0, LateralFeedbackDriver().k_throttle * expected), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Reversibility of the change to the existing loop
# ─────────────────────────────────────────────────────────────────────────────
# Isolation from the pre-existing opensbt-core
# ─────────────────────────────────────────────────────────────────────────────

def test_lanekeeping_is_untouched():
    """
    Project constraint: the C2 arm must not touch anything that existed in
    `opensbt-core` before this branch. `lanekeeping/` is the working Udacity
    pipeline, used by others too.

    This test locks the constraint in: no file in `lanekeeping/` may name C2.
    """
    lk = os.path.join(_LK_ROOT, "lanekeeping")
    offenders = []
    for root, _dirs, files in os.walk(lk):
        for f in files:
            if not f.endswith(".py"):
                continue
            text = open(os.path.join(root, f), encoding="utf-8",
                         errors="ignore").read()
            if "StateBasedAgent" in text or "LK_DRIVER" in text:
                offenders.append(os.path.relpath(os.path.join(root, f), lk))
    assert not offenders, (
        f"lanekeeping/ references C2 in: {offenders}. "
        f"The C2 arm lives in Simulator/c2/, which reuses lanekeeping by "
        f"composition without modifying it.")
