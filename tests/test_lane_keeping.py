"""
Diagnostic test script for LaneKeepingScenario.

Runs two simulations (straight and zigzag) and prints in detail:
  - the parameters sent to Docker
  - trajectory statistics (XTE, steering, speed)
  - the three components of the composite QoI (M1, M2, M3)
  - a final verdict

Usage:
    cd /path/to/Detecting-rare-failures-in-autonomous-vehicles
    python3 tests/test_lane_keeping.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scenarios.lane_keeping.config import (
    LaneKeepingScenario, MAX_XTE, STEER_RANGE_NORM, EARLY_FRAC
)

# ── Helpers ───────────────────────────────────────────────────────────────────

SEP  = "─" * 60
SEP2 = "═" * 60

def print_section(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

def compute_qoi_components(sc, traj, params):
    """Restituisce M1, M2, M3 e la QoI finale separatamente."""
    N, T, _ = traj.shape
    xte      = traj[:, :, 2].astype(np.float64)
    steering = traj[:, :, 3].astype(np.float64)

    run_lengths = sc._run_lengths if sc._run_lengths else [T] * N
    valid = np.zeros((N, T), dtype=bool)
    for i, L in enumerate(run_lengths):
        valid[i, :L] = True
    valid_count = valid.sum(axis=1).astype(np.float64)
    valid_count = np.where(valid_count > 0, valid_count, 1.0)

    xte_m      = np.where(valid, xte,      np.nan)
    steering_m = np.where(valid, steering, np.nan)

    m1 = MAX_XTE - np.nanmax(np.abs(xte_m), axis=1)
    steer_abs  = np.abs(steering_m)
    steer_peak = np.nanmax(steer_abs, axis=1) - np.nanmean(steer_abs, axis=1)
    m2 = -np.clip(steer_peak / STEER_RANGE_NORM, 0.0, 1.0)

    near = np.where(np.isnan(xte_m), False, np.abs(xte_m) > EARLY_FRAC * MAX_XTE)
    any_near  = near.any(axis=1)
    first_idx = np.where(any_near, near.argmax(axis=1).astype(float), valid_count)
    m3 = -(valid_count - first_idx) / valid_count

    qoi = 0.6 * m1 + 0.2 * m2 + 0.2 * m3
    return m1, m2, m3, qoi

def run_and_report(sc, label, params, param_names):
    print_section(label)

    # Stampa parametri
    print("\nParametri inviati al simulatore:")
    for name, val in zip(param_names, params[0]):
        print(f"  {name:<22} = {val:.1f}")

    # Simulation
    print(f"\n⏳ Sending the simulation to {sc.simulator_url}...")
    traj = sc.run_simulation(params)

    run_len = sc._run_lengths[0]
    T_max   = traj.shape[1]
    xte     = traj[0, :run_len, 2]
    steer   = traj[0, :run_len, 3]
    x       = traj[0, :run_len, 0]
    y       = traj[0, :run_len, 1]

    # Traiettoria
    print("\nTraiettoria:")
    print(f"  Step totali (run reale):  {run_len}")
    print(f"  Steps ended early: {'yes — XTE > limit' if run_len < T_max else 'No — run completo'}")
    print(f"  Distanza percorsa:  X {x[0]:.1f}m → {x[-1]:.1f}m  (Δ {x[-1]-x[0]:.1f}m)")
    print(f"  Deriva laterale:    Y {y[0]:.4f}m → {y[-1]:.4f}m")

    # XTE
    print("\nCross-Track Error (XTE) — distanza dal centro corsia:")
    print(f"  Max |XTE|:   {np.abs(xte).max():.4f}m   (limit = {MAX_XTE}m)")
    print(f"  Media |XTE|: {np.abs(xte).mean():.4f}m")
    print(f"  Soglia M3:   {EARLY_FRAC * MAX_XTE:.4f}m  ({EARLY_FRAC*100:.0f}% del limit)")
    near_steps = (np.abs(xte) > EARLY_FRAC * MAX_XTE).sum()
    print(f"  Step vicino al bordo: {near_steps} / {run_len}  ({near_steps/run_len*100:.1f}%)")

    # Sterzo
    print("\nSterzo (steering):")
    print(f"  Media:       {steer.mean():.4f}")
    steer_peak_val = np.abs(steer).max() - np.abs(steer).mean()
    print(f"  Std dev:     {steer.std():.4f}")
    print(f"  Peak dev:    {steer_peak_val:.4f}   (max|s|-mean|s|, STEER_RANGE_NORM={STEER_RANGE_NORM})")
    print(f"  Min / Max:   {steer.min():.4f} / {steer.max():.4f}")

    # QoI
    m1, m2, m3, qoi = compute_qoi_components(sc, traj, params)
    print("\nQoI composita:")
    print(f"  M1 (weight 0.6) -- XTE margin:      {m1[0]:+.4f}   (MAX_XTE - max|XTE|)")
    print(f"  M2 (weight 0.2) -- steering peak:   {m2[0]:+.4f}   (-(max|s|-mean|s|)/STEER_RANGE_NORM)")
    print(f"  M3 (weight 0.2) -- edge approach:   {m3[0]:+.4f}   (0=never; -1=at once)")
    print(f"  {SEP}")
    print(f"  QoI finale:  {qoi[0]:+.4f}")

    verdict = "SAFE  (QoI > 0)" if qoi[0] > 0 else "FAILURE  (QoI < 0)"
    print(f"\n  Verdict: {verdict}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sc = LaneKeepingScenario()
    bounds = sc.param_bounds()
    names  = bounds["names"]

    # Test 1 -- slow straight: every angle = 0 deg
    run_and_report(
        sc,
        label  = "TEST 1 -- Straight (all angles = 0 deg)",
        params = np.array([[0, 0, 0, 0, 0, 10.0, 15.0, 25.0, 250.0]]),
        param_names = names,
    )

    # Test 2 — zigzag estremo: angoli alternati a 85° e 0°
    run_and_report(
        sc,
        label  = "TEST 2 — Zigzag estremo (angoli [85,0,85,0,85])",
        params = np.array([[85, 0, 85, 0, 85, 12.0, 28.0, 35.0, 300.0]]),
        param_names = names,
    )

    print(f"\n{SEP2}")
    print("  Fine test diagnostico")
    print(SEP2)
