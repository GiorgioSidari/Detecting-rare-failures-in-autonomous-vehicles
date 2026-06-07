# Detecting Rare Failures in Autonomous Vehicles

This project studies how rare safety-critical failures occur in autonomous vehicle emergency braking systems. The idea is to simulate the braking scenario many times under varying and stochastic conditions, and figure out which combinations of parameters push the system into failure — without having to run millions of simulations to find them.

## What the simulator does

The framework is built around a generic `BaseSimulator` that any scenario can extend. The only scenario implemented so far is `EmergencyBrakingSimulator`, which models an AV approaching an obstacle and attempting an emergency stop. Each simulation run is controlled by four parameters — the vehicle's initial speed, the road friction, how far away the obstacle was when it was detected, and the nominal reaction delay of the system.

On top of those, the simulator adds realistic noise: braking efficiency varies slightly on every run (modelled after ISO 26262 brake-by-wire tolerance bands), and the actual reaction delay has log-normal jitter to capture sensor/actuator stack variability. This is why results differ even with identical input parameters, and why Monte Carlo sampling is needed.

The output is a trajectory for each run — position and velocity at every timestep — which feeds into the QoI.

## How failure is defined

The Quantity of Interest (QoI) lives in `evaluation/qoi.py`. It computes a **safety margin** for each run: how much stopping distance was left before the obstacle (minus a 2 m buffer). A positive margin means the vehicle stopped safely; a negative margin means it hit or overshot the obstacle. The failure indicator just turns that into a binary 0/1.

This separation is intentional — the simulator produces trajectories, the QoI judges them. Neither knows about the other.

## Validation steps

Before doing anything more sophisticated, the simulator is validated in two steps:

**Step 1.1 — Sanity check** (`scripts/sanity_check.py`): fixes friction, detection distance and delay, then sweeps initial speed from 5 to 45 m/s with 200 Monte Carlo samples per speed. The expected result is a smooth S-curve — near-zero failures at low speeds and near-100% failures at high speeds, with a transition around 22–34 m/s. If the curve looks wrong, the physics implementation is broken.

**Step 1.2 — QoI validation** (`scripts/validate_qoi.py`): draws 500 Latin Hypercube samples across the full 4D parameter space and plots the distribution of safety margins. The histogram should be bimodal, with a clear gap near zero separating safe from failing runs. A unimodal distribution would mean the parameter space is miscalibrated or the simulator has a bug.

## Project layout

```
simulators/
  base_simulator.py       # abstract base class
  emergency_braking.py    # the braking scenario
evaluation/
  qoi.py                  # safety margin + failure indicator
scripts/
  sanity_check.py         # step 1.1
  validate_qoi.py         # step 1.2
```

## Running it

```bash
pip install -e .
python scripts/sanity_check.py
python scripts/validate_qoi.py
```

Requires Python ≥ 3.9, numpy, scipy, and matplotlib. Generated plots are saved next to each script and are not tracked by git.
