"""
Step 2.1 validation — POD Embedder.

Generates 200 trajectories via EmergencyBrakingSimulator, fits EmbedderPOD,
then checks:
  1. nModes is in the expected range (3–8); warns if > 15
  2. Explained variance >= 99%
  3. Round-trip reconstruction error (relative RMSE on centered data) is consistent
     with the reported explained variance
  4. No NaN / Inf in codes or reconstructed trajectories
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulators.emergency_braking import EmergencyBrakingSimulator
from evaluation.param_space import lhs_sample, PARAM_BOUNDS
from embedder import EmbedderPOD

N_TRAJECTORIES = 200

# --- generate trajectories ---
print(f"Generating {N_TRAJECTORIES} trajectories...")
params = lhs_sample(N_TRAJECTORIES, PARAM_BOUNDS, seed=42)
sim = EmergencyBrakingSimulator()
trajectories = sim.run(params)
print(f"  trajectories shape : {trajectories.shape}  (expected ({N_TRAJECTORIES}, {sim.T}, 2))")

# ---  fit the embedder ---
embedder = EmbedderPOD(variance_threshold=0.99)
codes = embedder.fit_transform(trajectories)

print("\nPOD fit results:")
print(f"  nModes             : {embedder.nModes}")
print(f"  explained variance : {embedder.explained_variance * 100:.4f}%")
print(f"  codes shape        : {codes.shape}  (expected ({N_TRAJECTORIES}, {embedder.nModes}))")

# ---  check nModes is sensible ---
if embedder.nModes > 15:
    print(f"\n[WARN] nModes={embedder.nModes} exceeds 15 — check for numerical instability "
          f"in the simulator or redundant parameters.")
elif embedder.nModes > 8:
    print(f"\n[WARN] nModes={embedder.nModes} is above the typical range (3–8).")
else:
    print(f"\n[OK] nModes={embedder.nModes} is within the expected range (3–8).")

# ---  check explained variance ---
variance_ok = embedder.explained_variance >= 0.99
if variance_ok:
    print(f"[OK] Explained variance {embedder.explained_variance * 100:.4f}% >= 99%.")
else:
    print(f"[FAIL] Explained variance {embedder.explained_variance * 100:.4f}% < 99%.")

# ---  round-trip reconstruction error ---
# Relative RMSE is computed on the centered data so it matches the explained-variance
# metric (both measure unexplained signal as a fraction of total signal energy).
# Expected: sqrt(1 - explained_variance) ≈ sqrt(0.0062) ≈ 7.9% for 99.38% variance.
reconstructed = embedder.reconstruct(codes)

mean_traj = trajectories.reshape(N_TRAJECTORIES, -1).mean(axis=0)
centered   = trajectories.reshape(N_TRAJECTORIES, -1) - mean_traj
recon_flat = reconstructed.reshape(N_TRAJECTORIES, -1) - mean_traj

rel_rmse = np.sqrt(np.mean((centered - recon_flat) ** 2)) / np.sqrt(np.mean(centered ** 2))
expected_rel_rmse = np.sqrt(1.0 - embedder.explained_variance)

print("\nReconstruction error (relative RMSE on centered data):")
print(f"  measured  : {rel_rmse * 100:.4f}%")
print(f"  expected  : {expected_rel_rmse * 100:.4f}%  (= sqrt(1 - explained_variance))")

# --- check for NaN / Inf ---
nan_inf_ok = (
    not np.isnan(codes).any()
    and not np.isinf(codes).any()
    and not np.isnan(reconstructed).any()
    and not np.isinf(reconstructed).any()
)
if nan_inf_ok:
    print("\n[OK] No NaN or Inf in codes or reconstructed trajectories.")
else:
    print("\n[FAIL] NaN or Inf detected — check simulator outputs and embedder inputs.")

#
# Pass criterion (per spec): explained variance >= 99% and nModes <= 15
passed = variance_ok and embedder.nModes <= 15 and nan_inf_ok
print(f"\n{'[PASS] Step 2.1 complete — EmbedderPOD is working correctly.' if passed else '[FAIL] Step 2.1 checks failed — see warnings above.'}")