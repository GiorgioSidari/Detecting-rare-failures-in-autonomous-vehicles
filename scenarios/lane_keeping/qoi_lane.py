"""
Shared lane-keeping QoI (Quality of Interest).

This module extracts the composite safety metric used by the Unity/Docker
lane-keeping scenario into a *pure* function, so that a second simulator
backend (MetaDrive) can compute the IDENTICAL metric and stay directly
comparable. The maths mirrors LaneKeepingScenario.compute_qoi exactly.

Composite safety metric (failure when < 0.0):

    M1 - XTE margin              (weight 0.6):  MAX_XTE - max(|xte|)
    M2 - Steering peak deviation (weight 0.2): -(max|s| - mean|s|)/STEER_RANGE_NORM, clip[-1,0]
    M3 - Early boundary approach (weight 0.2): -(valid - first_near)/valid, in [-1,0]

Zero-padded timesteps are excluded from M2/M3 via `run_lengths`.

Degenerate/invalid handling (excluded from rates, margin -> NaN):
  - degenerate  : runs shorter than MIN_VALID_STEPS (aborted sims)
  - bad_params  : min_speed > max_speed (columns 5 > 6), an incoherent sample
  - low_fidelity: control rate too low / meters-per-step too high (OFF by default;
                  irrelevant for MetaDrive, where the control rate is exact)

NOTE: constants mirror scenarios/lane_keeping/config.py. Kept local so this
module is standalone (no import of the Docker-coupled config).
"""
from __future__ import annotations

import numpy as np

# ── Physical constants (mirror scenarios/lane_keeping/config.py) ──────────────
MAX_XTE          = 2.5   # metres — lane half-width used as the XTE margin reference
STEER_RANGE_NORM = 0.4   # normalisation for the steering peak deviation (M2)
EARLY_FRAC       = 0.7   # fraction of MAX_XTE that flags "approaching the boundary"
MIN_VALID_STEPS  = 3     # runs shorter than this are degenerate/aborted


def composite_lane_qoi(
    trajectories: np.ndarray,
    params: np.ndarray,
    run_lengths=None,
    *,
    control_hz=None,
    meters_per_step=None,
    min_control_hz: float = 0.0,
    max_meters_per_step: float = 0.0,
    max_xte: float = MAX_XTE,
    steer_range_norm: float = STEER_RANGE_NORM,
    early_frac: float = EARLY_FRAC,
    min_valid_steps: int = MIN_VALID_STEPS,
):
    """
    Compute the composite lane-keeping QoI for a batch of trajectories.

    Parameters
    ----------
    trajectories : (N, T, D>=4)  channels [x, y, xte, steering] (0,1,2,3)
    params       : (N, d)        used only to detect min_speed > max_speed (cols 5,6)
    run_lengths  : optional list[int] of real step counts (else T for all).
                   May be longer than N (a leading 'nominal' row); the tail is used.
    control_hz, meters_per_step : optional (N,) fidelity arrays for the (usually off) gate
    min_control_hz, max_meters_per_step : fidelity gate thresholds (0 = disabled)

    Returns
    -------
    (qoi, meta) where
      qoi  : (N,) float — composite safety margin, NaN for invalid runs
      meta : dict with keys
             survival (N,), valid_mask (N,), control_hz (N,|None),
             meters_per_step (N,|None), n_degenerate, n_low_fidelity, n_invalid
    """
    trajectories = np.asarray(trajectories)
    N, T, _ = trajectories.shape
    xte      = trajectories[:, :, 2].astype(np.float64)   # (N, T)
    steering = trajectories[:, :, 3].astype(np.float64)   # (N, T)

    # Validity mask: True for real steps, False for zero padding.
    if run_lengths is None:
        run_lengths = [T] * N
    run_lengths = list(run_lengths)
    if len(run_lengths) != N:
        run_lengths = run_lengths[-N:]                    # align to the tail (drop nominal head)
    valid = np.zeros((N, T), dtype=bool)
    for i, L in enumerate(run_lengths):
        valid[i, :min(int(L), T)] = True
    valid_count = valid.sum(axis=1).astype(np.float64)
    valid_count = np.where(valid_count > 0, valid_count, 1.0)

    xte_m      = np.where(valid, xte,      np.nan)
    steering_m = np.where(valid, steering, np.nan)

    # M1 — XTE margin
    with np.errstate(invalid="ignore"):
        m1 = max_xte - np.nanmax(np.abs(xte_m), axis=1)

    # M2 — steering peak deviation
    steer_abs  = np.abs(steering_m)
    with np.errstate(invalid="ignore"):
        steer_peak = np.nanmax(steer_abs, axis=1) - np.nanmean(steer_abs, axis=1)
    m2 = -np.clip(steer_peak / steer_range_norm, 0.0, 1.0)

    # M3 — time of first near-boundary approach
    near_boundary = np.abs(xte_m) > early_frac * max_xte
    near_boundary = np.where(np.isnan(xte_m), False, near_boundary)
    any_near      = near_boundary.any(axis=1)
    first_idx     = np.where(
        any_near,
        near_boundary.argmax(axis=1).astype(np.float64),
        valid_count,
    )
    m3 = -(valid_count - first_idx) / valid_count

    qoi = 0.6 * m1 + 0.2 * m2 + 0.2 * m3

    # ── Validity: degenerate / incoherent params / low fidelity ───────────────
    survival = np.asarray(run_lengths, dtype=float)
    degenerate = survival < min_valid_steps

    params = np.asarray(params)
    if params.shape[1] > 6:
        bad_params = params[:, 5] > params[:, 6]
    else:
        bad_params = np.zeros(N, dtype=bool)

    def _tail(arr):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        return arr[-N:] if arr.shape[0] >= N else np.full(N, np.nan)

    control_hz_t      = _tail(control_hz)
    meters_per_step_t = _tail(meters_per_step)

    low_fidelity = np.zeros(N, dtype=bool)
    if min_control_hz > 0 and control_hz_t is not None:
        low_fidelity |= np.nan_to_num(control_hz_t, nan=np.inf) < min_control_hz
    if max_meters_per_step > 0 and meters_per_step_t is not None:
        low_fidelity |= np.nan_to_num(meters_per_step_t, nan=0.0) > max_meters_per_step

    invalid = degenerate | bad_params | low_fidelity
    qoi = np.where(invalid, np.nan, qoi)

    meta = {
        "survival":        survival,
        "valid_mask":      ~invalid,
        "control_hz":      control_hz_t,
        "meters_per_step": meters_per_step_t,
        "n_degenerate":    int(degenerate.sum()),
        "n_low_fidelity":  int(low_fidelity.sum()),
        "n_invalid":       int(invalid.sum()),
    }
    return qoi, meta
