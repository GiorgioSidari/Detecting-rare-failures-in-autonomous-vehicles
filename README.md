# Detecting Rare Failures in Autonomous Vehicles

A framework for finding statistically rare, safety-critical failures in AV control systems using Latin Hypercube Sampling, neural network behavioral cloning, POD trajectory compression, and a FastAPI + browser-based frontend.

## What it does

The system runs N simulations of an AV scenario (Emergency Braking, Cut-In, Lane Keeping), compresses the resulting trajectories with Proper Orthogonal Decomposition, computes a safety margin for each run, and surfaces the worst failures.

It distinguishes two axes explicitly:

- **Severity** — the bottom-k% worst failures by safety margin (the cases that crash hardest), tie-broken by how early they leave the road.
- **Rarity** — the *probability* of failure under a realistic operational distribution. With distribution-aware sampling (`--sampling realistic`) the failure fraction becomes an estimate of P(failure), reported with a Wilson confidence interval. For genuinely low probabilities the Cross-Entropy rare-event estimator (`scripts/run_rare_event.py`) samples adaptively toward the failure region and reweights, reaching the same estimate with far fewer runs than plain Monte Carlo.

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
  lane_keeping/         # Blocco 3 (Udacity DNN, requires Docker)
  base_scenario.py      # abstract BaseScenario interface
  __init__.py           # SCENARIOS registry

simulators/
  base_simulator.py
  emergency_braking.py  # physics simulator (+ run_with_actuals)

embedder/
  pod.py                # Proper Orthogonal Decomposition via SVD

pipeline/
  orchestrator.py       # run(): LHS → sim → QoI → POD → rare failures + P(failure)+CI
                        # run_rare_event(): Cross-Entropy rare-event estimate
  severity.py           # find_severe_failures() (severity axis, bottom-k%)
  rare_event.py         # Cross-Entropy + importance sampling P(failure) estimator
  active_boundary.py    # active-learning of the fail/safe boundary (GP + ARD importance)

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
  run_lanekeeping.py    # lane-keeping pipeline runner (readable report)
  run_rare_event.py     # Cross-Entropy rare-event runner (Docker)
  sweep_lanekeeping.py  # ODD grid sweep (failure rate over speed x angle)
  envelope_lanekeeping.py # P(failure) vs meters-per-steer curve (operational envelope)
  validate_rare_probability.py # in-process validation of the P estimate + Wilson CI
  validate_rare_event.py       # in-process validation of the Cross-Entropy estimator
  # --- active-learning boundary (feature branch) ---
  run_active_boundary.py   # active-learning boundary on the Unity DNN (any BaseScenario)
  validate_active_boundary.py # active-boundary P vs brute-force MC (correctness + efficiency)
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
- `--sampling realistic|uniform` — `realistic` (default) draws from the operational distribution (via ppf of `param_distributions`), so the failure fraction estimates **P(failure)** under the ODD; `uniform` is a plain LHS baseline
- `--preset full|realistic` — `full` is the complete extended space (angles up to 85°, speed up to 30 m/s); `realistic` narrows the bounds to a plausible operational design domain
- `--max-speed X` / `--max-angle A` / `--min-speed Y` cap those bounds on any preset (narrow the ODD; use `--min-speed` with a low `--max-speed` to avoid overlapping min/max bands)
- `--trace-worst` print the step-by-step trace (x, XTE, steering) of the worst scenario, to see *how* it fails
- `--seed`, `--quiet` for reproducibility and to silence the per-job progress

**Other tools** (all use the same pool; run in a second terminal):

- `scripts/sweep_lanekeeping.py` — sweeps a grid of ODDs (max speed × max angle) and tabulates the failure rate per cell, to locate an informative band.
- `scripts/envelope_lanekeeping.py` — pools runs across conditions and plots **P(failure) vs meters-per-steer** (the operational-envelope curve), deriving the threshold and the required control rate.
- `scripts/run_rare_event.py` — Cross-Entropy rare-event estimator on the real simulator, for a genuinely low P (with the `--max-speed/--max-angle/--min-speed` caps to target a rare ODD).
- `scripts/validate_rare_probability.py` and `scripts/validate_rare_event.py` — in-process validations (no Docker) of the P estimate/Wilson CI and of the Cross-Entropy estimator against brute force.

### Reading the report

The report shows the two axes separately: **P(failure)** under the ODD (the rarity axis, with a Wilson 95% CI) and the **worst-case** severity (the bottom-k%, 5% by default, plus the margin distribution). It then lists the most critical scenarios with their parameters and steps survived. It also reports **invalid scenarios** that are *excluded* rather than counted: degenerate runs (aborted in very few steps), physically-inconsistent parameter combinations (e.g. `min_speed > max_speed`) and, when the fidelity gate is enabled, **under-sampled** runs (see *Control fidelity* below). Everything is computed over the valid scenarios only. A `FEDELTÀ DEL CONTROLLO` block reports the control rate, the meters-per-steer, and the per-step time split (inference vs waiting for Unity). For each rare failure the runner prints a **data-driven explanation** from its trajectory: the failure mode (unstable oscillation / uncorrected drift / wrong-way steering), which side it left the lane, after how many steps, and the scenario context (sharpest curve, speed band).

The safety margin combines three signals: how well the car stays inside the lane (weight 0.6), the abruptness of the steering (0.2) and how early it approaches the edge (0.2). Because the cross-track error saturates once the car leaves the lane, many "full" failures end up with the same margin; to tell them apart, among failures with equal margin the ones that leave the lane **earlier** are considered rarer (survival time as a tie-breaker).

### Control fidelity and rendering

The autopilot runs a closed loop — `predict → step → next Unity frame` — at whatever rate the machine sustains. Its **control rate** matters: if the car travels too many metres between two steering decisions, the DNN steers on stale observations, oscillates and leaves the lane. The runner measures two fidelity indicators per run, from data the simulator already returns (`elapsedTime`, `iterations`, `speeds`): the **control rate** (`iterations / elapsedTime`, Hz) and the **spatial resolution** (`meters_per_step` = mean speed × seconds-per-step). It also splits the per-step time into **inference** (`agent.predict`) vs **waiting for Unity** (`env.step`), so the bottleneck is visible. All of this prints in the `FEDELTÀ DEL CONTROLLO` block.

The time split showed the loop was dominated (~85%) by Unity's **software rendering** (llvmpipe under Xvfb), not by CPU/inference. Lowering the render resolution therefore raised the control rate from ~1.5 Hz to **~9 Hz** (and cut meters-per-step accordingly). The resolution is tunable via environment variables baked into the image:

```bash
XVFB_RESOLUTION=320x240x24   # Xvfb virtual display resolution
UNITY_SCREEN_WIDTH=320        # Unity -screen-width
UNITY_SCREEN_HEIGHT=240       # Unity -screen-height
UNITY_SCREEN_QUALITY=Fastest  # Unity -screen-quality
```

An optional **fidelity gate** marks under-sampled runs as invalid (excluded from rates and rare failures), so results are independent of machine load. It is controlled by two host-side environment variables, **off by default** (measure only):

```bash
LK_MIN_CONTROL_HZ=5        # exclude runs slower than 5 Hz
LK_MAX_METERS_PER_STEP=2   # exclude runs coarser than 2 m per steering decision
```

### Autopilot control (`opensbt-core/.../self_driving/supervised_agent.py`)

Three changes to the controller:

- **Unit fix (km/h → m/s).** The telemetry reports speed in km/h while the ODD `min/max_speed` are in m/s; the original throttle regulator compared the two directly and kept the car crawling far below the commanded speed. Converting to m/s makes the regulator actually reach `max_speed`. This is a correctness fix, and it *exposes* the model's real behaviour at the commanded speeds (see the note below).
- **Steering rate-limiter** — caps `|Δsteering|` per step to damp the jerks that lead to saturation. Active by default at `0.20` (`LK_STEER_MAX_RATE`, `0` disables). It softens the catastrophic opposite-side overshoots; tightening it too far (e.g. `0.15`) over-damps.
- **Speed-dependent steering gain** — reduces steering authority as speed grows (like a real car), targeting the high-speed failure mode. Now that speed is handled in m/s it is active by default (`LK_STEER_SPEED_GAIN_K=0.03`, `LK_STEER_SPEED_REF=12` m/s); `gain = 1/(1 + K·(v − ref))`.
- Inference uses TF's fast path (`model(obs, training=False)`) instead of `model.predict()`.

Because this code runs **inside the Docker image**, changes take effect only after rebuilding (`docker compose ... up --build`), and the `LK_STEER_*` variables must be set in the container's `environment:` (the baked-in defaults apply otherwise).

### Key result: the operational envelope

Pooling runs across speeds and binning by their *measured* meters-per-steer (`scripts/envelope_lanekeeping.py`) gives a clean step function: failures are governed not by nominal speed but by the **spatial control resolution** — the metres the car covers between two steering decisions, `meters_per_step ≈ speed / control_rate`. The lane keeper is deterministically safe below **~0.6 m/step**, fails above ~0.8, with a narrow stochastic band in between. Speed alone gave a non-monotonic, confounded picture; binning by meters-per-steer removes the confound.

This yields a concrete requirement: `control_rate ≥ target_speed / 0.6`. At Unity's ~9 Hz the model is safe only up to ~5–6 m/s; driving the ODD's higher speeds (up to 14 m/s) would need **~22 Hz** — about 2.5× the current rate, i.e. faster rendering (GPU) or a retrained controller. The rate-limiter and speed gain soften the failure mode but do not move this boundary, which is set by the control resolution.

### Environment variables (lane keeping)

| Variable | Where | Default | Effect |
|---|---|---|---|
| `NUM_WORKERS` / `SIMULATOR_URLS` | host | 4 | worker pool size / explicit endpoints |
| `LK_MIN_CONTROL_HZ` | host | 0 (off) | exclude runs below this control rate |
| `LK_MAX_METERS_PER_STEP` | host | 0 (off) | exclude runs above this spatial resolution |
| `LK_STEER_MAX_RATE` | container | 0.20 | steering rate-limiter (0 = off) |
| `LK_STEER_SPEED_GAIN_K` | container | 0.03 | speed-dependent steering attenuation (0 = off) |
| `LK_STEER_SPEED_REF` | container | 12 | reference speed (m/s) for the gain |
| `XVFB_RESOLUTION` | container | 320x240x24 | virtual display resolution |
| `UNITY_SCREEN_WIDTH/HEIGHT/QUALITY` | container | 320 / 240 / Fastest | Unity render resolution/quality |

Emergency Braking and Cut-In do **not** require Docker.

---

## Active-learning of the failure boundary (feature branch)

On top of the existing pipeline this branch adds a method that *learns* where the controller fails and applies it to the real Udacity DNN. It is **additive** — the existing scenarios and pipeline are untouched — and plugs into the same `BaseScenario` interface. See `RESULTS.md` for the empirical findings.

### Method (`pipeline/active_boundary.py`)

Where `run()` gives severity (bottom-k%) and `run_rare_event()` gives rarity (P), this learns **where and why** a scenario fails: a Gaussian-Process model of the safety margin over the parameter space, refined by sampling adaptively near the fail/safe boundary. It returns P(failure) with a credible interval, the ARD **feature importance** (which parameters drive the failure), and the concrete failing scenarios. It is **backend-agnostic**: it runs on the real Unity DNN (`--scenario lane_keeping`) exactly as on any `BaseScenario`.

```bash
python scripts/run_active_boundary.py --scenario lane_keeping                  # real DNN (needs Docker)
python scripts/run_active_boundary.py --scenario lane_keeping --max-angle 8 --max-speed 10 --max-seg 14
```

Both `run_active_boundary.py` and `run_rare_event.py` accept ODD-narrowing flags to target the rare regime: `--max-angle`, `--max-speed`, `--min-speed`, `--max-seg`, `--min-seg`. Narrowing the ODD until failures become rare is how a genuine rare failure is surfaced (see `RESULTS.md`).

### What it produced on the real model

Applied to the real Udacity DNN in Unity, the active-boundary method was **cross-validated against the existing Cross-Entropy estimator** (they agree on P at the same ODD), and used to **quantify the model's safety envelope in speed** and to extract concrete, reproducible **rare-failure scenarios** (P ≈ 2% at 9–10 m/s). Full numbers in `RESULTS.md`.

### Validation

- `python scripts/validate_active_boundary.py` — active-boundary P vs brute-force Monte Carlo (correctness confirmed; the GP surrogate is **not** more sample-efficient than plain MC for estimating P — an honest limitation, so the method's value is the boundary + importance, not the P estimate).
- Tests: `pytest tests/test_active_boundary.py -q`.
