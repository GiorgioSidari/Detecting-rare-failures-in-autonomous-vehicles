"""
Geometric parity between backends: the same theta must produce the same road.

This is the assumption the whole cross-simulator comparison rests on, and it was
false. MetaDrive built the road with its own PGBlocks starting from the MEAN of
the 5 angles, with turn directions taken from the seed: at the same theta the two
roads were uncorrelated (Spearman between the lengths -0.026), and the comparison
that came out -- rho=-0.020, Fisher p=1.000 -- looked like it said "the
simulators disagree" while it only said "the roads are different".

The tests below fall into two groups:

  * the PURE ones, which do not require MetaDrive and always run:
    metrica di distanza, costruzione dello ScenarioDescription, e la prova
    that the old geometry really did diverge;
  * the INTEGRATION ones, skipped when MetaDrive is missing, which check the
    corsia effettivamente costruita dal simulatore.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from scenarios.common.road_geometry import road_polyline            # noqa: E402
from scenarios.lane_keeping_md.scenario_map import (                # noqa: E402
    build_scenario_description,
)

from diag_road_parity import (                                      # noqa: E402
    EDGE_MARGIN_M, THRESHOLD_M, point_to_polyline_distance,
    _centerline_metadrive, _reference_centerline,
)

_has_metadrive = True
try:
    import metadrive  # noqa: F401
except Exception:                                                   # pragma: no cover
    _has_metadrive = False

requires_metadrive = pytest.mark.skipif(
    not _has_metadrive, reason="MetaDrive is not installed in this environment")


def _theta(n: int = 5, seed: int = 42) -> np.ndarray:
    from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

    from scenarios.lane_keeping.config import LaneKeepingScenario
    b = LaneKeepingScenario().param_bounds()
    s = LatinHypercube(d=len(b["lower"]), seed=seed)
    return qmc_scale(s.random(n=n), b["lower"], b["upper"])


# ── la metrica ───────────────────────────────────────────────────────────────

def test_distance_is_point_to_segment_not_point_to_vertex():
    """
    The midpoint of a segment is ZERO from the polyline, not half a step away.

    This is the measurement mistake that makes a correct alignment look broken:
    with point-to-vertex distance, two perfectly overlapping curves whose
    vertices are 1.34 m apart measure ~34 cm of deviation instead of ~0.
    """
    C = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    P = np.array([[5.0, 0.0], [15.0, 0.0]])          # esattamente sui segmenti
    assert point_to_polyline_distance(P, C).max() < 1e-12

    # and a point genuinely off is measured for what it is
    assert abs(point_to_polyline_distance(np.array([[5.0, 3.0]]), C)[0] - 3.0) < 1e-12


def test_distance_on_an_offset_curve():
    """A known offset is measured as such."""
    t = np.linspace(0, 20, 200)
    C = np.column_stack([t, np.zeros_like(t)])
    P = np.column_stack([t, np.full_like(t, 0.25)])
    d = point_to_polyline_distance(P, C)
    assert np.allclose(d, 0.25, atol=1e-9)


# ── lo ScenarioDescription ───────────────────────────────────────────────────

def test_scenario_description_uses_the_shared_centreline():
    """La corsia dichiarata a MetaDrive è ESATTAMENTE la spline di Udacity."""
    row = _theta(1)[0]
    sd = build_scenario_description(row)
    poly = sd["map_features"]["lane_0"]["polyline"]
    expected = np.asarray(road_polyline(row), dtype=float)
    assert np.allclose(poly, expected, atol=0.0), "the polyline is not the canonical one"
    assert sd["map_features"]["lane_0"]["type"] == "LANE_SURFACE_STREET"


def test_ego_track_spans_the_whole_road():
    """
    MetaDrive's route comes from the first and last point of the ego track: a
    degenerate track (start = finish) breaks the route computation, and the
    vehicle has no destination.
    """
    row = _theta(1)[0]
    sd = build_scenario_description(row)
    pos = sd["tracks"]["ego"]["state"]["position"]
    poly = sd["map_features"]["lane_0"]["polyline"]
    assert np.allclose(pos[:, :2], poly)
    assert np.linalg.norm(pos[-1, :2] - pos[0, :2]) > 10.0


def test_ego_heading_matches_the_tangent():
    """A wrong initial heading makes the vehicle start sideways."""
    row = _theta(1)[0]
    sd = build_scenario_description(row)
    poly = sd["map_features"]["lane_0"]["polyline"]
    hdg = sd["tracks"]["ego"]["state"]["heading"]
    t = poly[1] - poly[0]
    expected = math.atan2(t[1], t[0])
    diff = abs(math.atan2(math.sin(hdg[0] - expected), math.cos(hdg[0] - expected)))
    assert diff < math.radians(5.0), f"initial heading off by {math.degrees(diff):.1f} deg"


@pytest.mark.skipif(not _has_metadrive, reason="needs the MetaDrive schema")
def test_scenario_description_passes_the_sanity_check():
    from metadrive.scenario.scenario_description import ScenarioDescription as SD
    SD.sanity_check(build_scenario_description(_theta(1)[0]))


# -- the proof that the old geometry was broken ------------------------------

def test_legacy_geometry_diverges():
    """
    Il gate DEVE bocciare i PGBlock.

    If one day this test went green, it would mean the gate no longer
    discriminates -- i.e. that it protects against nothing.
    """
    scarti = []
    for row in _theta(5):
        ref = _reference_centerline(row)
        got = _centerline_metadrive(row, legacy=True)
        scarti.append(point_to_polyline_distance(got, ref).max())
    assert min(scarti) > 1.0, (
        f"la geometria legacy sembra allineata (max scostamento {min(scarti):.3f} m): "
        f"either it was fixed, or the gate stopped discriminating")


def test_legacy_lengths_are_uncorrelated():
    """
    The quantitative signature of the bug: road length, the crudest possible
    summary, did not correlate between the two backends.
    """
    from scipy.stats import spearmanr

    def L(C):
        d = np.diff(C, axis=0)
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    th = _theta(20)
    lu = [L(_reference_centerline(r)) for r in th]
    lm = [L(_centerline_metadrive(r, legacy=True)) for r in th]
    rho = spearmanr(lu, lm).statistic
    assert abs(rho) < 0.5, f"expected uncorrelated, rho={rho:.3f}"
    # and Udacity's length is nearly constant: `segment_length` is almost inert there
    assert (max(lu) - min(lu)) / np.mean(lu) < 0.05


# -- the lane MetaDrive actually builds ---------------------------------------

@requires_metadrive
@pytest.mark.parametrize("i", [0, 1, 2])
def test_metadrive_lane_matches_udacity(i):
    """The load-bearing test: parity within threshold on the actual lane."""
    row = _theta(3)[i]
    ref = _reference_centerline(row)
    got = _centerline_metadrive(row)

    err = point_to_polyline_distance(got, ref)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(got, axis=0).T))])
    inside = (s > EDGE_MARGIN_M) & (s < s[-1] - EDGE_MARGIN_M)

    assert err[inside].max() <= THRESHOLD_M, (
        f"worst deviation {err[inside].max()*100:.1f} cm above the threshold "
        f"di {THRESHOLD_M*100:.0f} cm")
    assert err[inside].mean() < 0.01


@requires_metadrive
def test_lane_length_within_one_percent():
    for row in _theta(3):
        ref = _reference_centerline(row)
        got = _centerline_metadrive(row)
        lr = np.hypot(*np.diff(ref, axis=0).T).sum()
        lg = np.hypot(*np.diff(got, axis=0).T).sum()
        assert abs(lg / lr - 1.0) < 0.01, f"lunghezze {lg:.1f} vs {lr:.1f}"


# -- the sign of the lateral error --------------------------------------------

@requires_metadrive
def test_xte_does_not_use_the_metadrive_convention():
    """
    `ScenarioLane.local_coordinates` returns the lateral value with the OPPOSITE
    sign to `road_frame`. Given to the controller, that sign makes it steer the
    wrong way: the vehicle diverged monotonically and left the road within
    ~30 steps su OGNI scenario.

    This test pins the choice: the XTE comes from `road_frame` on the shared
    centreline -- the same definition the C2 arm uses on Udacity -- so the
    measurement is identical on every backend and does not depend on internal
    conventions
    di ciascun simulatore.
    """
    from scenarios.common.road_frame import road_frame
    from scenarios.lane_keeping_md.scenario_map import lane_of, make_online_env

    row = _theta(1)[0]
    ref = _reference_centerline(row)
    env = make_online_env(row, decision_repeat=5, physics_world_step_size=0.02,
                          max_steps=10)
    try:
        env.reset()
        ln = lane_of(env)
        t = ref[1] - ref[0]
        t = t / np.linalg.norm(t)
        p = ref[0] + 10.0 * t + 2.0 * np.array([-t[1], t[0]])

        _, lat_md = ln.local_coordinates(p)
        rf = road_frame(ref, float(p[0]), float(p[1]), math.atan2(t[1], t[0]))

        assert abs(abs(lat_md) - abs(rf.lateral_error)) < 1e-3, "magnitudini diverse"
        assert np.sign(lat_md) != np.sign(rf.lateral_error), (
            "le convenzioni ora coincidono: se MetaDrive ha cambiato segno, "
            "revisit _extract_state, which assumes road_frame as the reference")
    finally:
        env.close()


@requires_metadrive
def test_vehicle_stays_in_lane():
    """
    End-to-end check: with the right sign the controller holds the lane.

    Before the fix this test failed with out_of_road within ~30 steps.
    """
    from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario

    sc = LaneKeepingMetaDriveScenario(speed_scale=0.1766, max_steps=120)
    traj = sc.run_simulation(_theta(2), verbose=False)
    for t in traj:
        assert len(t) >= 100, f"episodio troncato a {len(t)} steps"
        assert np.abs(t[:, 2]).max() < 2.5, "uscito di corsia"


# -- the legacy mode stays reachable ------------------------------------------

def test_gate_does_not_claim_ok_without_verifying(monkeypatch, capsys):
    """
    Saltare un backend NON è un successo.

    In its first version the gate printed "OK" and exited with 0 after skipping
    the only backend requested, because it was launched with the wrong
    interpreter (MetaDrive lives in a Python 3.10 venv). A gate like that
    protects only those who do not
    ne ha bisogno: basta invocarlo male per avere via libera.
    """
    import diag_road_parity as g

    def explode(row, **kw):
        raise ImportError("No module named 'metadrive'")

    monkeypatch.setitem(g.BACKEND, "metadrive", explode)
    monkeypatch.setattr(sys, "argv",
                        ["diag_road_parity.py", "--n", "2", "--backend", "metadrive"])

    assert g.main() == 1, "the gate claimed success without verifying anything"
    out = capsys.readouterr().out
    assert "NOT VERIFIED" in out
    assert "INCONCLUSIVE" in out
    # and it must say how to fix it, not only that it went wrong
    assert "venv310" in out or "Python < 3.12" in out


def test_gate_says_ok_only_with_a_verified_backend(monkeypatch, capsys):
    """The positive case: 'OK' appears only if something was really measured."""
    import diag_road_parity as g

    monkeypatch.setitem(g.BACKEND, "metadrive", g._reference_centerline)
    monkeypatch.setattr(sys, "argv",
                        ["diag_road_parity.py", "--n", "2", "--backend", "metadrive"])
    assert g.main() == 0
    out = capsys.readouterr().out
    assert "OK -- verified: metadrive" in out


def test_legacy_mode_is_selectable_and_validated():
    """
    The old geometry stays available to reproduce campaigns already run, but it
    is no longer the default and an invented value is rejected straight away
    instead of silently producing a different road.
    """
    from scenarios.lane_keeping_md.config import LaneKeepingMetaDriveScenario as S

    assert S().geometry == "udacity"
    assert S(geometry="pgblock").geometry == "pgblock"
    with pytest.raises(ValueError, match="geometry"):
        S(geometry="qualcosa")
