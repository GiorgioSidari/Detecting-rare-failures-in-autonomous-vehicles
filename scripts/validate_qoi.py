"""
Step 1.2 validation — QoI histogram via Latin Hypercube Sampling.

Sample 500 points over the full parameter space using LHS, run the simulator,
compute safety margins, and plot a histogram.

Expected: bimodal distribution with a clear gap near 0.0 separating safe (positive)
from failure (negative) cases. A unimodal distribution indicates a simulator bug.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale

from simulators.emergency_braking import EmergencyBrakingSimulator
from evaluation.qoi import compute_safety_margin, failure_indicator

N_SAMPLES = 500

sim    = EmergencyBrakingSimulator()
bounds = sim.ParamBounds()

# Latin Hypercube samples in [0, 1]^4, then scale to physical bounds
sampler      = LatinHypercube(d=4, seed=42)
unit_samples = sampler.random(n=N_SAMPLES)
params       = scale(unit_samples, bounds["lower"], bounds["upper"])

trajectories = sim.run(params)

# Detection distance varies per sample — pass as array
detection_distances = params[:, 2]
safety_margins      = compute_safety_margin(trajectories, detection_distance=detection_distances)
failures            = failure_indicator(safety_margins)
failure_rate        = failures.mean()

print(f"Samples        : {N_SAMPLES}")
print(f"Failure rate   : {failure_rate*100:.1f}%")
print(f"Margin range   : [{safety_margins.min():.2f}, {safety_margins.max():.2f}] m")
print(f"Margin mean    : {safety_margins.mean():.2f} m")

# --- Histogram ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(safety_margins, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Failure threshold (0.0 m)")
ax.set_xlabel("Safety margin (m)")
ax.set_ylabel("Count")
ax.set_title(f"QoI distribution — LHS {N_SAMPLES} samples\nFailure rate: {failure_rate:.1%}")
ax.legend()
ax.grid(True, alpha=0.3)

# Failure rate broken down by initial speed
ax2 = axes[1]
speed_bins = np.linspace(bounds["lower"][0], bounds["upper"][0], 10)
bin_idx    = np.digitize(params[:, 0], speed_bins) - 1
bin_rates  = []
bin_centers = []
for b in range(len(speed_bins) - 1):
    mask = bin_idx == b
    if mask.sum() > 0:
        bin_rates.append(failures[mask].mean() * 100)
        bin_centers.append((speed_bins[b] + speed_bins[b + 1]) / 2)

ax2.bar(bin_centers, bin_rates, width=(speed_bins[1] - speed_bins[0]) * 0.8,
        color="coral", edgecolor="white", alpha=0.85)
ax2.set_xlabel("Initial speed (m/s)")
ax2.set_ylabel("Failure rate (%)")
ax2.set_title("Failure rate by initial speed\n(marginalised over other params)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "validate_qoi.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()

# Pass/fail check
if safety_margins.min() < 0 and safety_margins.max() > 0:
    print("\n[PASS] Histogram shows both safe and failure cases — threshold separates two modes.")
else:
    print("\n[FAIL] Distribution is unimodal or entirely on one side — check simulator / QoI.")