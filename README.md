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

## Lane Keeping — how it works and how to run it

The Lane Keeping scenario stress-tests a neural-network autopilot (the Udacity "chauffeur" DNN) by driving it on procedurally generated roads and measuring how well it stays inside the lane. Unlike Emergency Braking and Cut-In, which run in-process, here the actual driving happens inside the opensbt-core Unity simulator, which the Python code drives over HTTP: for each sampled scenario the client sends the road parameters to the simulator, the simulator runs the drive, and returns the trajectory (position, cross-track error and steering, step by step).

### Architecture: a pool of parallel simulators

Each simulator container holds a single Unity instance and runs **one** simulation at a time — it is inherently sequential. To avoid waiting for hours when sampling dozens of scenarios, the system starts **several containers in parallel** (4 by default), each on its own port (8000, 8001, ...), and distributes the jobs across the pool through a shared queue. With W workers the throughput is roughly W times that of a single container. The client health-checks every worker before starting and skips the unreachable ones; on top of that each job has an individual timeout, so a stuck container only fails its own scenario instead of freezing the whole batch. Full details are in `opensbt-core/PARALLELIZZAZIONE.md`.

### One-time setup

The model and the Unity executable are not in the repo (excluded via `.gitignore`) and must be downloaded from the Google Drive linked in `opensbt-core/README.md`, then placed as follows:

- the model `mixed-chauffeur.h5` in `opensbt-core/Simulator/SelfDrivingModels/`
- the Ubuntu build of the simulator in `opensbt-core/Simulator/SimulatorExec/ubuntu_binaries/` (the file `ubuntu.x86_64` must exist, with execute permission)

### Starting the simulators

From `opensbt-core/`, generate the compose file for the desired number of workers and start the containers:

```bash
cd opensbt-core
python gen_parallel_compose.py 4          # 4 containers on ports 8000-8003
docker compose -f docker-compose.parallel.yml up --build
```

Alternatively, `./run_parallel.sh 4` does both and keeps the containers and `NUM_WORKERS` in sync. Leave this window open: it is the server. You can check that the workers respond by opening `http://localhost:8000/health` ... `8003/health`.

### Running experiments

In a second terminal, using the same number of workers, run the script that executes the full pipeline (LHS sampling → simulations on the pool → QoI → POD compression → rare-failure detection) and prints a readable report:

```bash
python scripts/run_lanekeeping.py --n 50 --workers 4
```

Main options:

- `--n` number of sampled scenarios (higher = more accurate estimates but slower)
- `--workers` number of parallel containers (must match the ones you started)
- `--preset full|realistic` — `full` is the complete extended space (angles up to 85°, speed up to 30 m/s); `realistic` narrows to a plausible operational design domain (gentler angles, plausible speeds with non-overlapping min/max) so the measured failure rate reflects realistic conditions rather than an artificially wide space
- `--max-speed X` force the speed cap (m/s) on any preset, handy for studying the effect of speed
- `--trace-worst` print the step-by-step trace (x, XTE, steering) of the worst scenario, to see *how* it fails
- `--seed`, `--quiet` for reproducibility and to silence the per-job progress

A typical run for an informative analysis and its diagnosis:

```bash
python scripts/run_lanekeeping.py --n 30 --preset realistic --max-speed 6 --trace-worst
```

### Reading the report

The report shows the **failure rate** (share of scenarios with safety margin < 0), the **rare failure rate** (the worst bottom-k%, 5% by default), the distribution of margins sorted from worst to best, and the table of the most critical scenarios with their parameters and the number of steps survived. It also reports **invalid scenarios** that are *excluded* from the analysis rather than counted: degenerate runs (simulations aborted in very few steps), physically-inconsistent parameter combinations (e.g. `min_speed > max_speed`) and, when the fidelity gate is enabled, **under-sampled** runs (see *Control fidelity* below). Rates and rare failures are computed only over the valid scenarios, so the failure rate is not polluted by non-real cases. For each rare failure the runner also prints a **data-driven explanation** derived from its trajectory: the failure mode (unstable oscillation / uncorrected drift / wrong-way steering), which side it left the lane, after how many steps, and the scenario context (sharpest curve, speed band).

The safety margin combines three signals: how well the car stays inside the lane (weight 0.6), the abruptness of the steering (0.2) and how early it approaches the edge (0.2). Because the cross-track error saturates once the car leaves the lane, many "full" failures end up with the same margin; to tell them apart, among failures with equal margin the ones that leave the lane **earlier** are considered rarer (survival time as a tie-breaker).

### Control fidelity (parallelism without losing accuracy)

The autopilot runs a closed loop — `predict → step → next Unity frame` — at whatever frequency the machine can sustain. When several containers compete for CPU, each Unity instance slows down and the car travels *more metres between two steering decisions*: the DNN, trained at near-real-time cadence, then steers on stale, widely-spaced observations, and failures start to reflect the machine load rather than the model. To keep parallelism *and* accuracy, the runner now measures two fidelity indicators for every run, computed from data the simulator already returns (`elapsedTime`, `iterations`, `speeds`): the **control frequency** (`iterations / elapsedTime`, Hz) and the **spatial resolution** (`meters_per_step` = mean speed × seconds-per-step). It prints them in a `FEDELTÀ DEL CONTROLLO` block (min / median / max), so you can watch how the cadence changes as you vary `--workers`.

An optional **fidelity gate** marks under-sampled runs as invalid (excluded from rates and rare failures, exactly like degenerate runs), so results become independent of how many workers were running. It is controlled by two environment variables, **off by default** (measure only, no exclusion):

```bash
LK_MIN_CONTROL_HZ=5        # exclude runs slower than 5 Hz
LK_MAX_METERS_PER_STEP=2   # exclude runs coarser than 2 m per steering decision
```

Empirically, on this setup the cadence sits around **~1.5 Hz / ~2.7 m per steering decision and is essentially the same at 1 and 4 workers** — the bottleneck is Unity's frame delivery, not CPU contention — so parallelism here is safe to use (the 4× speedup comes at no measurable accuracy cost). The gate is provided mainly as a guardrail for heavier worker pools or slower machines.

### Autopilot stability tuning

The autopilot (`opensbt-core/.../self_driving/supervised_agent.py`) has two knobs aimed at the oscillatory instability seen on curves (drift → steering saturates → overshoots to the opposite side). First, inference uses TF's fast path (`model(obs, training=False)`) instead of `model.predict()`, which rebuilds its predict function on every call — a few milliseconds saved per step (though, per the note above, Unity, not inference, is the cadence bottleneck here). Second, a **steering rate-limiter** caps `|Δsteering|` per step to damp the jerks that lead to saturation; it is active by default at `0.20` and tunable via `LK_STEER_MAX_RATE` (`0` disables it, restoring the raw DNN steering). It mainly reshapes the failure *distribution* — it removes the catastrophic opposite-side overshoots and makes the worst case milder — but it does not reliably lower the overall failure rate; tightening it too much (e.g. `0.15`) over-damps, delaying legitimate curve corrections and pushing borderline runs into mild failures. A **speed-dependent steering gain** (reduce authority at higher speed, as a real car does) is wired but **off by default** (`LK_STEER_SPEED_GAIN_K=0`), because `state["speed"]` arrives in km/h while `max_speed` is in m/s, so its coefficient must be calibrated before enabling it (`LK_STEER_SPEED_REF` sets the reference speed).

Because this code runs **inside the Docker image**, changes take effect only after rebuilding (`docker compose ... up --build`), and the `LK_STEER_*` variables must be set in the container's `environment:` rather than in the host shell (the baked-in defaults apply otherwise).

### A note on the results

On this model the failure rate stays high even when narrowing geometry and speed: speed is a modest lever (from ~94% down to ~85% going from 30 to 6 m/s) and the traces show an oscillatory instability of the controller on curves. Part of this is a genuine fidelity limit rather than a bug: at ~1.5 Hz the car is effectively "blind" for almost three metres between steering updates, and that cadence is imposed by the simulator, not by the Python code. The steering rate-limiter shaves the worst overshoots (the catastrophic opposite-side overshoot becomes milder) but does not reliably lower the overall failure rate, and tightening it too far over-damps the correction; controller tuning alone plateaus here. In practice the lane-keeper is reliable only within gentle curves and low speeds; the value of the analysis lies in characterising *which* combinations of curvature and speed push it into failure — which is exactly what the pipeline (QoI + POD + rare failures) surfaces. Lowering the baseline further is less a matter of controller tuning than of the model+ODD pair: narrowing the operational design domain, revisiting the QoI threshold, or retraining the DNN.

Emergency Braking and Cut-In do **not** require Docker.
