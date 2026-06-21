# Detecting Rare Failures in Autonomous Vehicles

A framework for finding statistically rare, safety-critical failures in AV control systems using Latin Hypercube Sampling, neural network behavioral cloning, POD trajectory compression, and a FastAPI + browser-based frontend.

## What it does

The system runs N simulations of an AV scenario (Emergency Braking, Cut-In, Lane Keeping), compresses the resulting trajectories with Proper Orthogonal Decomposition, computes a safety margin for each run, and surfaces the bottom-k% worst failures as "rare failures" — the cases that not only crash, but crash hardest.

Each scenario has two modes:
- **Physics mode** — ideal controller, deterministic physics with realistic noise
- **NN mode** — a neural network trained via behavioral cloning drives the vehicle; its generalisation errors are the source of rare failures

---

## Quick start

### Prerequisites

- Python ≥ 3.9
- A virtual environment (`.venv/` or similar)
- Docker only if you want the **Lane Keeping** scenario (opensbt-core)

### 1 — Install dependencies

```bash
pip install -e .
```

### 2 — Train the Emergency Braking MLP (one-time, ~15 min on CPU)

```bash
# Step 2a: generate behavioral cloning dataset (~30 seconds)
python -m scenarios.emergency_braking.train --dataset-only

# Step 2b: train the MLP (~10-20 min on CPU, less with GPU)
python -m scenarios.emergency_braking.train
```

The trained model is saved to `scenarios/emergency_braking/models/emergency_braking_mlp.keras`.
A `training_curves.png` plot is also saved there.

> **Note:** model files and datasets are excluded from git (`.gitignore`).
> Anyone cloning the repo must run this step before using NN mode.

### 3 — Validate the pipeline (optional but recommended)

```bash
python scripts/validate_pipeline.py --n 200
```

Should print `ALL PASS` for both Physics and NN modes.

### 4 — Start the API server

```bash
# Windows: double-click start.bat, or:
uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
```

### 5 — Open the frontend

Open `frontend/index.html` directly in your browser. The header shows `⬤ API online` when the server is reachable.

---

## Project structure

```
scenarios/
  emergency_braking/
    config.py           # EmergencyBrakingScenario (use_nn flag)
    train.py            # dataset generation + MLP training
    nn_controller.py    # BrakingMLP (Keras Sequential)
    nn_simulator.py     # time integration driven by MLP
    dataset/            # ← generated, not in git
    models/             # ← generated, not in git
  cut_in/               # Blocco 2 (in development)
  lane_keeping/         # Blocco 3 (requires Docker)
  base_scenario.py      # abstract BaseScenario interface
  __init__.py           # SCENARIOS registry

simulators/
  base_simulator.py
  emergency_braking.py  # physics simulator (+ run_with_actuals)

embedder/
  pod.py                # Proper Orthogonal Decomposition via SVD

pipeline/
  orchestrator.py       # LHS → sim → QoI → POD → rare failures
  rare_failures.py      # find_rare_failures(), summarise_rare_params()

api/
  server.py             # FastAPI: /scenarios /run /status /explain /health
  schemas.py            # Pydantic models

evaluation/
  qoi.py                # safety margin + failure indicator

frontend/
  index.html            # 3-screen SPA (scenario → config → results)

scripts/
  validate_pipeline.py  # end-to-end smoke test
  sanity_check.py       # physics sanity (speed sweep)
  validate_qoi.py       # QoI distribution check
  validate_pod.py       # POD embedder check
```

---

## Scenario parameters

### Emergency Braking

| Parameter | Range | Description |
|---|---|---|
| `initial_speed` | 5–50 m/s | Vehicle speed at obstacle detection |
| `friction_coefficient` | 0.3–1.0 | Road grip (0.3 = wet/icy, 1.0 = dry asphalt) |
| `detection_distance` | 10–100 m | Distance to obstacle when detected |
| `nominal_delay` | 0.05–0.5 s | Braking system reaction delay |

The **safety margin** = `detection_distance - final_position - 2 m`. Negative = crash.

---

## Pipeline parameters (frontend)

| Parameter | Description |
|---|---|
| **Samples (N)** | Number of LHS points to simulate. More = better rare-failure coverage, but slower. |
| **Seed** | Random seed — same seed reproducibly gives the same LHS points. |
| **Rare fraction (%)** | Bottom-k% of failures to flag as "rare". 5% = only the worst crashes. |
| **Param bounds** | Editable lower/upper per parameter — narrow them to focus sampling on a specific region. |

---

## Scenarios status

| Scenario | Physics | NN | Requires |
|---|---|---|---|
| Emergency Braking | ✅ | ✅ (train first) | nothing |
| Cut-In | 🚧 Blocco 2 | 🚧 Blocco 2 | nothing |
| Lane Keeping | ✅ (existing) | ✅ (existing Udacity DNN) | Docker (opensbt-core) |

---

## Docker (Lane Keeping only)

The Lane Keeping scenario calls the opensbt-core Unity simulator via HTTP. To use it:

```bash
cd opensbt-core
docker compose up --build
```

The Emergency Braking and Cut-In scenarios do **not** require Docker.
