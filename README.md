# Detecting Rare Failures in Autonomous Vehicles

This project studies how rare safety-critical failures occur in autonomous vehicle emergency braking systems. The idea is to simulate the braking scenario many times under varying and stochastic conditions, figure out which combinations of parameters push the system into failure — and do it without having to run millions of simulations to find them.

## What the simulator does

The framework is built around a generic `BaseSimulator` that any scenario can extend. The only scenario implemented so far is `EmergencyBrakingSimulator`, which models an AV approaching an obstacle and attempting an emergency stop. Each run is controlled by four parameters: the vehicle's initial speed, the road friction, how far away the obstacle was when detected, and the system's nominal reaction delay.

On top of those, the simulator adds realistic noise: braking efficiency varies slightly on every run (modelled after ISO 26262 brake-by-wire tolerance bands), and the actual reaction delay has log-normal jitter to capture sensor/actuator stack variability. This is why results differ even with identical inputs, and why Monte Carlo sampling is needed.

The output of each run is a trajectory — position and velocity at every timestep — which feeds into the evaluation pipeline.

## How failure is defined

The Quantity of Interest lives in `evaluation/qoi.py`. It computes a **safety margin** for each run: how much stopping distance was left before the obstacle (minus a 2 m buffer). A positive margin means the vehicle stopped safely; negative means it hit or overshot. The failure indicator turns that into a binary 0/1.

The simulator produces trajectories; the QoI judges them. Neither knows about the other.

## Sampling the parameter space

Rather than sampling parameters naively, the project uses **Latin Hypercube Sampling** (LHS) to cover the 4D parameter space efficiently. The bounds and the sampler live in `evaluation/param_space.py` and are shared across all scripts, so every experiment draws from the same well-calibrated region.

## Validation

Before doing anything more sophisticated, the simulator is validated in two steps.

**Step 1.1 — Sanity check** (`scripts/sanity_check.py`): fixes friction, detection distance, and delay, then sweeps initial speed from 5 to 45 m/s with 200 Monte Carlo samples per speed. The expected result is a smooth S-curve — near-zero failures at low speeds, near-100% at high speeds, with a transition around 22–34 m/s. If the curve looks wrong, the physics implementation is broken.

**Step 1.2 — QoI validation** (`scripts/validate_qoi.py`): draws 500 LHS samples across the full 4D space and plots the distribution of safety margins. The histogram should be bimodal, with a clear gap near zero separating safe from failing runs. A unimodal distribution would mean the parameter space is miscalibrated or the simulator has a bug.

## Compressing trajectories with POD

Raw trajectories are high-dimensional: each run produces a time series of position and velocity. That's a lot of data to feed into any downstream model.

The `embedder/pod.py` module addresses this with **Proper Orthogonal Decomposition** (POD). It learns the dominant directions of variation across a set of trajectories using SVD, then projects each trajectory down to a small number of coordinates — the POD codes — that capture at least 99% of the total variance. The mean trajectory is subtracted first so the decomposition focuses on how trajectories differ from one another, not their shared shape.

In practice, 250 LHS-sampled trajectories compress to just a handful of modes with negligible reconstruction error. That means downstream steps can work with compact numeric codes rather than full time series, which matters a lot when those steps are expensive.

**Step 2.1 — POD test** (`simulators/test/test_pod.py`): generates 250 trajectories, fits the embedder, reports how many modes were selected and the reconstruction error, and saves the trajectories, codes, and parameters to `simulators/test/data/` for inspection.

## Project layout

```
simulators/
  base_simulator.py        # abstract base class
  emergency_braking.py     # the braking scenario
  test/
    test_pod.py            # step 2.1 — POD embedder validation
    data/                  # saved trajectories, codes, and parameters
evaluation/
  qoi.py                   # safety margin + failure indicator
  param_space.py           # parameter bounds + LHS sampler
embedder/
  pod.py                   # POD trajectory compressor
scripts/
  sanity_check.py          # step 1.1
  validate_qoi.py          # step 1.2
```

## Running it

```bash
pip install -e .

# Validation
python scripts/sanity_check.py
python scripts/validate_qoi.py

# POD embedder test
python simulators/test/test_pod.py
```

Requires Python ≥ 3.9, numpy, scipy, and matplotlib. Generated plots and test datasets are not tracked by git.