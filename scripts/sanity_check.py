"""
Step 1.1 sanity check — expected S-curve failure rate vs initial speed.

Fix: friction_coefficient=0.8, detection_distance=50 m, nominal_delay=0.2 s
Vary initial_speed from 5 to 45 m/s in steps of 2 m/s.
Run 200 Monte Carlo samples per speed.

Expected: ~0% failure below ~22 m/s, ~100% above ~34 m/s, smooth S-curve in between.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from simulators.emergency_braking import EmergencyBrakingSimulator
from evaluation.qoi import compute_safety_margin, failure_indicator

N_MC          = 200          # Monte Carlo samples per speed
SPEEDS        = np.arange(5, 46, 2, dtype=float)
FRICTION      = 0.8
DETECTION_D   = 50.0         # metres
NOMINAL_DELAY = 0.2          # seconds

sim = EmergencyBrakingSimulator()
failure_rates = []

for v0 in SPEEDS:
    params = np.column_stack([
        np.full(N_MC, v0),
        np.full(N_MC, FRICTION),
        np.full(N_MC, DETECTION_D),
        np.full(N_MC, NOMINAL_DELAY),
    ])
    trajectories    = sim.run(params)
    margins         = compute_safety_margin(trajectories, detection_distance=DETECTION_D)
    failures        = failure_indicator(margins)
    failure_rates.append(failures.mean())

failure_rates = np.array(failure_rates)

plt.figure(figsize=(8, 5))
plt.plot(SPEEDS, failure_rates * 100, marker="o", linewidth=2, color="steelblue")
plt.axhline(0,   color="green",  linestyle="--", linewidth=0.8, label="0% (all safe)")
plt.axhline(100, color="red",    linestyle="--", linewidth=0.8, label="100% (all fail)")
plt.axvline(22,  color="orange", linestyle=":",  linewidth=0.8, label="Expected ~22 m/s boundary")
plt.axvline(34,  color="purple", linestyle=":",  linewidth=0.8, label="Expected ~34 m/s boundary")
plt.xlabel("Initial speed (m/s)")
plt.ylabel("Failure rate (%)")
plt.title("Sanity check — Emergency braking failure rate vs initial speed\n"
          f"friction={FRICTION}, d={DETECTION_D} m, delay={NOMINAL_DELAY} s, N={N_MC} MC")
plt.legend(fontsize=8)
plt.ylim(-5, 105)
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "sanity_check.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()

# Quick pass/fail assessment
low_speed_fail  = failure_rates[SPEEDS <= 20].mean()
high_speed_fail = failure_rates[SPEEDS >= 36].mean()
print(f"\nFailure rate below 20 m/s : {low_speed_fail*100:.1f}%  (expected ~0%)")
print(f"Failure rate above 36 m/s : {high_speed_fail*100:.1f}%  (expected ~100%)")

if low_speed_fail < 0.05 and high_speed_fail > 0.95:
    print("\n[PASS] S-curve looks correct — simulator is working as expected.")
else:
    print("\n[FAIL] Unexpected failure rates — check physics implementation.")