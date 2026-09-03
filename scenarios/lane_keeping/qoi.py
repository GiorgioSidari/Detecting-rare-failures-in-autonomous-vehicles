"""
Lane-keeping QoI (Quality of Interest): the composite safety margin.

Pure, testable computation of the per-scenario safety margin from trajectories, used by
LaneKeepingScenario.compute_qoi. Failure when the margin < 0. The margin combines three signals:

  M1 - XTE margin (weight 0.6):  MAX_XTE - max(|xte|).  Positive = in lane, negative = off-road.
  M2 - Steering peak deviation (weight 0.2): -(max|s| - mean|s|)/STEER_RANGE_NORM, clipped [-1, 0]
       (penalises sudden jerks / unstable control).
  M3 - Early boundary approach (weight 0.2): -(valid - first_near)/valid, in [-1, 0]
       (penalises approaching the lane edge early).

Zero-padded timesteps are excluded from M2/M3 via run_lengths. Invalid runs get margin = NaN so the
orchestrator excludes them from the rates: degenerate (too few steps), incoherent params
(min_speed > max_speed), or low control fidelity (under-sampled: too many metres per steer).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
MAX_XTE          = 2.5   # metres - lane half-width used by the QoI margin. The Udacity
                         # pipeline caps the telemetry XTE at its own MAX_XTE = 3 m
                         # (opensbt-core/Simulator/lanekeeping/config.py), so this
                         # threshold is the stricter of the two.
STEER_RANGE_NORM = 0.4   # normalisation for the steering peak deviation (max|s| - mean|s|);
                         # 0.4 keeps M2 unsaturated for most safe runs
EARLY_FRAC       = 0.7   # fraction of MAX_XTE that flags "approaching the boundary"
MIN_VALID_STEPS  = 3     # runs shorter than this are degenerate/aborted (not safe driving)


@dataclass
class LaneQoIResult:
    """Output of composite_lane_qoi: the margin plus everything the scenario stores/reports."""
    qoi: np.ndarray                   # (N,) composite margin; NaN = invalid (excluded from rates)
    survival: np.ndarray              # (N,) run length in steps (severity tiebreak)
    valid_mask: np.ndarray            # (N,) bool, True = counted in the rates
    control_hz: np.ndarray | None     # (N,) tail-aligned, for reporting
    meters_per_step: np.ndarray | None
    infer_ms: np.ndarray | None
    wait_ms: np.ndarray | None
    n_degenerate: int
    n_low_fidelity: int
    n_invalid: int


def _validity_mask(run_lengths: np.ndarray, N: int, T: int) -> tuple:
    """
    Which timesteps are real driving and which are zero padding.

    `run_lengths` may carry an extra leading row (a "nominal" sample dropped by
    the orchestrator), so it is aligned to the tail. `valid_count` never goes to
    zero: a run with no valid step would otherwise divide by it.
    """
    run_lengths = list(run_lengths) if run_lengths is not None else [T] * N
    if len(run_lengths) != N:
        run_lengths = run_lengths[-N:]
    valid = np.zeros((N, T), dtype=bool)
    for i, L in enumerate(run_lengths):
        valid[i, :min(L, T)] = True
    valid_count = valid.sum(axis=1).astype(np.float64)
    return run_lengths, valid, np.where(valid_count > 0, valid_count, 1.0)


def _margin_terms(xte_m: np.ndarray, steering_m: np.ndarray, valid_count: int) -> tuple:
    """
    The three components of the margin: lane departure, steering peak, earliness.

    A run of length 0 (a job lost to a stuck simulator, see run_simulation) has
    no valid timestep at all, so its row is entirely NaN and the nan-aware
    reductions warn about an empty slice. NaN is the answer we want -- the run
    is marked invalid by the caller -- so the expected warning is silenced
    rather than the bug.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="All-NaN|Mean of empty slice")
        m1 = MAX_XTE - np.nanmax(np.abs(xte_m), axis=1)                      # (N,)

        steer_abs = np.abs(steering_m)
        steer_peak = np.nanmax(steer_abs, axis=1) - np.nanmean(steer_abs, axis=1)
    m2 = -np.clip(steer_peak / STEER_RANGE_NORM, 0.0, 1.0)                   # [-1, 0]

    near = np.abs(xte_m) > EARLY_FRAC * MAX_XTE
    near = np.where(np.isnan(xte_m), False, near)
    any_near = near.any(axis=1)
    first_idx = np.where(any_near, near.argmax(axis=1).astype(np.float64), valid_count)
    m3 = -(valid_count - first_idx) / valid_count                            # [-1, 0]
    return m1, m2, m3


def _tail(arr: np.ndarray, N: int):
    """The last N entries of a per-run array, NaN-filled when it is too short."""
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    return arr[-N:] if arr.shape[0] >= N else np.full(N, np.nan)


def _invalidity_flags(survival: np.ndarray, params: np.ndarray,
                      control_hz: float, meters_per_step: float,
                      min_control_hz: float, max_meters_per_step: float) -> tuple:
    """
    The three reasons a run carries no information, as separate masks.

    Degenerate: too few steps to be real driving, so the QoI would wrongly
    reward it. Incoherent parameters: min_speed above max_speed. Low fidelity:
    the control loop ran too slowly, which is excluded so the rates do not
    depend on how many workers were up (gate off when the threshold is 0).
    """
    N = len(survival)
    degenerate = survival < MIN_VALID_STEPS

    if params.shape[1] > 6:
        bad_params = np.asarray(params)[:, 5] > np.asarray(params)[:, 6]
    else:
        bad_params = np.zeros(N, dtype=bool)

    low_fidelity = np.zeros(N, dtype=bool)
    hz, mps = _tail(control_hz, N), _tail(meters_per_step, N)
    if min_control_hz > 0 and hz is not None:
        low_fidelity |= np.nan_to_num(hz, nan=np.inf) < min_control_hz
    if max_meters_per_step > 0 and mps is not None:
        low_fidelity |= np.nan_to_num(mps, nan=0.0) > max_meters_per_step
    return degenerate, bad_params, low_fidelity


def composite_lane_qoi(
    trajectories: np.ndarray,
    run_lengths: np.ndarray,
    params: np.ndarray,
    *,
    control_hz=None,
    meters_per_step=None,
    infer_ms=None,
    wait_ms=None,
    min_control_hz: float = 0.0,
    max_meters_per_step: float = 0.0,
) -> LaneQoIResult:
    """
    Compute the composite lane-keeping safety margin (see module docstring).

    trajectories : (N, T, 4) with channels [x, y, xte, steering].
    run_lengths  : actual steps per run (None -> the full length T is used, e.g. in tests).
    params       : (N, D) sampled parameters (cols 5, 6 = min/max speed, checked for coherence).
    control_hz / meters_per_step / infer_ms / wait_ms : per-run fidelity arrays (optional).
    min_control_hz / max_meters_per_step : fidelity gate thresholds (0 = gate off).
    """
    N, T, _ = trajectories.shape
    xte      = trajectories[:, :, 2].astype(np.float64)   # (N, T)
    steering = trajectories[:, :, 3].astype(np.float64)   # (N, T)

    run_lengths, valid, valid_count = _validity_mask(run_lengths, N, T)

    # Padded timesteps -> NaN so nan-aware functions ignore them.
    xte_m      = np.where(valid, xte,      np.nan)
    steering_m = np.where(valid, steering, np.nan)

    m1, m2, m3 = _margin_terms(xte_m, steering_m, valid_count)
    qoi = 0.6 * m1 + 0.2 * m2 + 0.2 * m3

    survival = np.asarray(run_lengths, dtype=float)
    degenerate, bad_params, low_fidelity = _invalidity_flags(
        survival, params, control_hz, meters_per_step,
        min_control_hz, max_meters_per_step)
    control_hz, meters_per_step, infer_ms, wait_ms = (
        _tail(control_hz, N), _tail(meters_per_step, N),
        _tail(infer_ms, N), _tail(wait_ms, N))

    invalid = degenerate | bad_params | low_fidelity
    qoi = np.where(invalid, np.nan, qoi)

    return LaneQoIResult(
        qoi=qoi,
        survival=survival,
        valid_mask=~invalid,
        control_hz=control_hz,
        meters_per_step=meters_per_step,
        infer_ms=infer_ms,
        wait_ms=wait_ms,
        n_degenerate=int(degenerate.sum()),
        n_low_fidelity=int(low_fidelity.sum()),
        n_invalid=int(invalid.sum()),
    )
