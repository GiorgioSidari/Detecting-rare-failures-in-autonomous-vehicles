# Detecting Rare Failures in Autonomous Vehicles

A framework for finding statistically rare, safety-critical failures in AV control systems using Latin Hypercube Sampling, neural network behavioral cloning, POD trajectory compression, and a FastAPI + browser-based frontend.

## What it does

The system runs N simulations of an AV scenario (Emergency Braking, Cut-In, Lane Keeping), compresses the resulting trajectories with Proper Orthogonal Decomposition, computes a safety margin for each run, and surfaces the bottom-k% worst failures as "rare failures" — the cases that not only crash, but crash hardest.

### Modes, in plain terms

Each scenario is a fixed situation — same obstacle, same road, same sampled parameters. A **mode** only changes *who decides the vehicle's action*, i.e. which controller is actually driving. Three kinds of controller appear across the three scenarios (not every scenario has all three — see the table below):

- **Physics** — an idealised controller with perfect knowledge of the true state (exact distance, exact speed) and a hand-written formula. It never fails; it's the "expert" / ground truth that the other two modes are trained to imitate.
- **Scalar-MLP** (Emergency Braking only) — a small neural network fed 3 hand-picked numbers (speed, distance, elapsed time). It learns to imitate the physics controller from that summary, and occasionally gets it wrong because a 3-number summary is a lossy approximation of the true state.
- **Video-CNN** (all three scenarios) — a convolutional network fed a raw camera frame, exactly what a real self-driving car would see. It has to infer the hazard from pixels alone, with no privileged access to exact distances. This is the closest analogue to a real perception-based autopilot, and its mistakes are the ones we're most interested in finding.

| Mode | Pros | Cons |
|---|---|---|
| **Physics** | Instant (thousands of runs/second), nothing to train, no external software, never wrong — a perfect baseline and label source for the other two | Not realistic: no real AV has exact ground-truth distance/speed sensing, so it can't itself be the *subject* of a "does it fail" study |
| **Scalar-MLP** | Still fast (milliseconds/run), quick to train (minutes), no external simulator needed | Still not realistic — a real AV doesn't get 3 clean numbers handed to it; mainly a convenient stand-in for testing the pipeline without installing CARLA |
| **Video-CNN** | The realistic mode — tests an actual perception-based controller reacting to raw pixels like a real AV; its failures generalise to what could genuinely happen on the road | Needs CARLA (or Docker+Udacity) installed and, for CARLA, a GPU; training requires collecting a frame dataset first; each run takes seconds instead of milliseconds, so realistic sample sizes are in the tens/hundreds, not thousands |

---

## Quick start

### Prerequisites

- Python ≥ 3.9, in a virtual environment (`.venv/` or similar)
- **Docker**, only for the **Lane Keeping** scenario (opensbt-core / Udacity simulator)
- **CARLA 0.9.16** + an **NVIDIA GPU**, only for the **video-CNN mode** of Emergency Braking / Cut-In — not required for physics mode or the scalar-MLP mode. Budget ~8 GB disk for the CARLA package.

### 1 — Install dependencies

```bash
pip install -e .
```

### 2 — Train the Emergency Braking scalar-MLP (one-time, ~15 min on CPU)

```bash
# Step 2a: generate behavioral cloning dataset (~30 seconds)
python -m scenarios.emergency_braking.train --dataset-only

# Step 2b: train the MLP (~10-20 min on CPU, less with GPU)
python -m scenarios.emergency_braking.train
```

The trained model is saved to `scenarios/emergency_braking/models/emergency_braking_mlp.keras`.

> **Note:** model files and datasets are excluded from git (`.gitignore`).
> Anyone cloning the repo must run this step (and the CARLA training steps below, if wanted) before using NN mode.

### 3 — Validate the pipeline (optional but recommended)

```bash
python scripts/validate_pipeline.py --n 200
```

### 4 — Start the API server

```bash
# Windows: double-click start.bat, or:
uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
```

### 5 — Open the frontend

Open `frontend/index.html` directly in your browser. The header shows `⬤ API online` when the server is reachable.

---

## The three scenarios at a glance

| Scenario | Code location | Modes available | Registry key(s) |
|---|---|---|---|
| **Lane Keeping** | `scenarios/lane_keeping/` + `opensbt-core/Simulator/lanekeeping/` | video-CNN only (Udacity DNN) | `lane_keeping`, `lane_keeping_chauffeur`, `lane_keeping_ch2` |
| **Emergency Braking** | `scenarios/emergency_braking/` + `simulators/emergency_braking.py` + `opensbt-core/Simulator/emergency_braking/` | physics · scalar-MLP · video-CNN (CARLA) | `emergency_braking`, `emergency_braking_carla` |
| **Cut-In** | `scenarios/cut_in/` + `opensbt-core/Simulator/cut_in/` | physics · video-CNN (CARLA) | `cut_in`, `cut_in_carla` |

The `_carla` registry keys only appear once their model has been trained (auto-detected in `scenarios/__init__.py` from the presence of the `.h5` file) — see each scenario's section for how to get there.

### How the modes differ

|  | **Physics** | **Scalar-MLP** (Emergency Braking only) | **Video-CNN** (all three) |
|---|---|---|---|
| Controller input | Ground-truth state (exact distances/speeds) | 3 hand-picked numbers (velocity, distance, time) | Raw camera frame — the network has to *see* the hazard, like a real perception stack |
| Simulator | In-process numpy, vectorised (thousands of runs/second) | In-process numpy | A real 3D simulator (CARLA or Udacity/Unity) driven over HTTP, one run at a time per worker |
| Training needed | No | Yes (behavioral cloning on the physics expert) | Yes (behavioral cloning on frames + the physics expert's decisions) |
| External requirement | None | None | CARLA (Emergency Braking/Cut-In) or Docker+Udacity (Lane Keeping) |
| Speed | Milliseconds for N=1000 | Milliseconds for N=1000 | ~5-10s *per run* — realistically N in the tens/low hundreds per sitting |
| Why it fails | It doesn't (used as the "expert"/ground truth) | Learned approximation of the expert from scalar features | Learned approximation of the expert from *vision* — reaction is emergent, not hand-coded, so its failures look like real perception mistakes (reacts late, misjudges distance, etc.) |

---

## Project structure

```
scenarios/
  emergency_braking/
    config.py                 # EmergencyBrakingScenario (use_nn: False | True | "carla")
    train.py                  # scalar dataset generation + BrakingMLP training
    nn_controller.py          # BrakingMLP (Keras Sequential, 3 scalar features)
    nn_simulator.py           # time integration driven by BrakingMLP
    collect_carla_dataset.py  # Fase 2: runs N CARLA episodes with the physics expert,
                               #         records (frame, braking_force) pairs
    train_cnn.py               # Fase 2: trains the PilotNet CNN on the CARLA dataset
    nn_simulator_carla.py      # Fase 2: HTTP client -> opensbt-core CARLA service
    dataset/, dataset_carla/   # ← generated, not in git
    models/                    # ← generated, not in git (.keras = scalar-MLP, .h5 = CNN)
  cut_in/
    simulator.py               # CutInSimulator (physics: sigmoid lateral merge + braking)
    qoi.py                      # compute_min_gap()
    config.py                   # CutInScenario (use_nn: False | "carla")
    collect_carla_dataset.py    # Fase 3, mirrors emergency_braking's
    train_cnn.py                 # Fase 3, mirrors emergency_braking's
    nn_simulator_carla.py        # Fase 3, mirrors emergency_braking's
    dataset_carla/, models/      # ← generated, not in git
  lane_keeping/
    config.py                   # LaneKeepingScenario, talks to opensbt-core over HTTP
  base_scenario.py              # abstract BaseScenario interface (incl. pod_channels())
  __init__.py                   # SCENARIOS registry

simulators/
  base_simulator.py
  emergency_braking.py          # physics simulator (+ run_with_actuals, the "expert")
  common/                       # shared by the video-CNN scenarios, not scenario-specific
    pilotnet.py                 # the CNN architecture (build_pilotnet/train_pilotnet)
    camera_dataset.py           # crop/resize/augment + CameraDataGenerator + archive I/O
    http_worker_pool.py         # job-queue/health-check/pool pattern (all HTTP scenarios)

opensbt-core/                   # vendored Udacity sim (lane_keeping) + our own CARLA services
  Simulator/
    SimulatorServer.py          # Udacity/lanekeeping FastAPI service (port 8000+)
    lanekeeping/                # vendored from the "maxitwo" project — do not edit lightly
    emergency_braking/          # Fase 2: carla_episode.py, cnn_agent.py, SimulatorServer.py (port 8100+)
    cut_in/                     # Fase 3: carla_episode.py, cnn_agent.py, SimulatorServer.py (port 8200+)
  CarlaSimulator_0916/          # ← CARLA itself, downloaded separately, not in git
  scenario_runner/               # ← optional, design reference only, not a runtime dependency

embedder/
  pod.py                        # Proper Orthogonal Decomposition via SVD

pipeline/
  orchestrator.py               # LHS → sim → QoI → POD → rare failures
  rare_failures.py              # find_rare_failures(), summarise_rare_params()
  rare_event.py                 # Cross-Entropy rare-event probability estimation

api/
  server.py                     # FastAPI: /scenarios /run /status /explain /health
  schemas.py                    # Pydantic models

evaluation/
  qoi.py                        # emergency_braking safety margin + failure indicator

frontend/
  index.html                    # 3-screen SPA (scenario → config → results)

scripts/
  validate_pipeline.py          # end-to-end smoke test
  run_lanekeeping.py            # lane_keeping runner with a readable report
  run_emergency_braking.py      # same report style, emergency_braking (physics/scalar-MLP/CARLA)
  run_cut_in.py                 # same report style, cut_in (physics/CARLA)
```

> **API/frontend note:** `GET /scenarios` + `POST /run/{name}` work generically for *any* registered scenario, including the `_carla` ones, so the browser frontend already shows failure rate / rare failures / trajectory chart for them. It does **not** yet show the richer terminal-only content (P(fallimento) confidence interval, control-fidelity block, the data-driven "why it failed" text, the worst-run trace) — those live only in the `scripts/run_*.py` runners below, `api/schemas.py::StatusResponse` doesn't carry those fields yet.

---

## Scenario parameters

### Emergency Braking

| Parameter | Range | Physics / scalar-MLP meaning | Video-CNN (CARLA) meaning |
|---|---|---|---|
| `initial_speed` | 5–50 m/s | Vehicle speed at obstacle detection | Cruise speed both vehicles reach before the episode clock starts |
| `friction_coefficient` | 0.3–1.0 | Road grip (0.3 = wet/icy, 1.0 = dry asphalt) | Same — applied to the ego's tires via CARLA's vehicle physics |
| `detection_distance` | 10–100 m | Distance to obstacle when detected | Initial gap to the lead vehicle |
| `nominal_delay` | 0.05–0.5 s | **Ego's** braking reaction delay | **Lead vehicle's** braking-onset delay — the ego's own reaction is emergent (whatever the CNN decides from the frames), so this became a hazard-timing knob instead of a direct input to the ego |

The **safety margin** = `detection_distance - final_position - 2 m`. Negative = crash.

### Cut-In

| Parameter | Range | Physics meaning | Video-CNN (CARLA) meaning |
|---|---|---|---|
| `ego_speed` | 10–40 m/s | Ego's initial/cruise speed | Same |
| `cutter_speed` | 10–40 m/s | Speed of the vehicle cutting in | Same |
| `lateral_gap` | 1–4 m | Lateral distance between vehicles at cut-in start | Same |
| `reaction_delay` | 0.05–0.8 s | **Ego's** braking reaction delay after the cutter enters the lane | Shifts **when the cutter starts** its lateral merge — same reinterpretation as Emergency Braking's `nominal_delay`, for the same reason |

The **safety margin** = min bumper-to-bumper gap while the cutter is in the ego's lane, minus a 1.5 m safety buffer. Negative = crash.

---

## CARLA setup (Emergency Braking & Cut-In video-CNN mode)

Both scenarios' video-CNN mode share the same CARLA installation and the same operational quirks — covered once here instead of twice.

### Install

1. **CARLA 0.9.16** (not in git, ~7.8 GB): download the Windows package from `https://tiny.carla.org/carla-0-9-16-windows` and extract it — it unpacks flat, so you should end up with `opensbt-core/CarlaSimulator_0916/CarlaUE4.exe` directly (no extra nested folder).
2. **Matching Python client**: `pip install carla==0.9.16`.
   > Client and server versions must match **exactly**. An adjacent version (e.g. client 0.9.16 talking to a 0.9.15 server) connects fine for basic calls (`get_server_version()` etc.) but silently corrupts camera sensor data — this is not a warning you can ignore.
3. `carla-simulator/scenario_runner` (optional, `opensbt-core/scenario_runner/`) is **not** needed at runtime — our own `carla_episode.py` talks to the raw CARLA API directly. It's kept around purely as the design reference for the scenario parametrisation (see its `srunner/scenarios/{follow_leading_vehicle,cut_in}.py`).

### Running CARLA itself

```bash
opensbt-core/CarlaSimulator_0916/CarlaUE4.exe -RenderOffScreen -nosound
```

Headless, no window. Give it **15-30 seconds** to finish booting before pointing a client at it — connecting too early doesn't crash CARLA, but see the troubleshooting note below.

### Important: one CARLA instance = one client at a time

CARLA's synchronous mode (`world.tick()`) is driven by a **single** client. This means:
- You can **not** run the Emergency Braking service and the Cut-In service against the same CARLA instance at the same time — pick one, or start a second CARLA instance on a different port (`-carla-rpc-port=3000`) for the other.
- Only **one worker per CARLA instance** is currently supported (`EB_NUM_WORKERS`/`CI_NUM_WORKERS` default to 1). A real multi-worker pool (like Lane Keeping's 4 parallel Docker containers) would need one CARLA instance *per* worker, each on its own port — not set up yet (see Fase 4 in the project plan).

### Troubleshooting

- **A job times out after 90s and the service looked healthy the whole time**: almost always means the service's background simulation thread died right after startup — most commonly because it tried to connect to CARLA before CARLA had finished booting. `/health` keeps answering OK regardless (it's a separate, always-up route), which is what makes this confusing. The services retry the connection a few times on startup (~30s total) to absorb this, but if you started the service *immediately* after launching `CarlaUE4.exe`, wait for CARLA to be reachable (`python -c "import carla; carla.Client('localhost', 2000).get_server_version()"`) before starting the service, or just restart the service once CARLA is confirmed up.
- **`RuntimeError: Spawn failed because of collision at spawn position`**: happens occasionally when a sampled parameter combination places a vehicle on top of static map geometry. Handled automatically (small position retry) as of this version — if you still see it propagate all the way up, restart CARLA (a failed spawn can occasionally leave the world in a state where the same spot keeps failing) and retry.
- **Nothing responds / everything hangs**: kill `CarlaUE4.exe` and any `uvicorn ...SimulatorServer` processes and start over — CARLA has no persisted state that needs preserving between runs.

---

## Emergency Braking — how to run each mode

**Physics** (default, no setup): `SCENARIOS["emergency_braking"]` with `use_nn=False` — just run the pipeline, nothing to install.

**Scalar-MLP**: train once (`python -m scenarios.emergency_braking.train`, see Quick Start step 2), then `EmergencyBrakingScenario(use_nn=True)` is used automatically once the model file exists.

**Video-CNN (CARLA)** — the vehicle is driven by a CNN reading camera frames from inside CARLA. Do the [CARLA setup](#carla-setup-emergency-braking--cut-in-video-cnn-mode) above once, then:

1. **Collect the training dataset** (runs N episodes with the physics "expert" controller and records camera frames + its braking decisions — not video files, frame sequences, see `simulators/common/camera_dataset.py`):
   ```bash
   opensbt-core/CarlaSimulator_0916/CarlaUE4.exe -RenderOffScreen -nosound   # separate terminal, wait ~20s
   python -m scenarios.emergency_braking.collect_carla_dataset --n 200
   ```
   Sanity check on completion: with `--n 200` expect on the order of a few thousand frame-label pairs (~7-8k is typical), a handful of collisions (the expert is near-perfect, not perfect), and an occasional "SALTATO" line (skipped episode, spawn collision — normal, a few out of 200 is fine).
2. **Train the CNN**:
   ```bash
   python -m scenarios.emergency_braking.train_cnn --epochs 25
   ```
   Saves to `scenarios/emergency_braking/models/emergency_braking_cnn.h5`. Once this file exists, `emergency_braking_carla` appears automatically in the scenario registry. Expect `val_loss` (binary cross-entropy) to land somewhere around 0.1-0.3 — much higher suggests something went wrong upstream (check the dataset's braking fraction isn't close to 0% or 100%).
3. **Start the simulation service** (CARLA must already be running and reachable, see troubleshooting above):
   ```bash
   cd opensbt-core
   uvicorn Simulator.emergency_braking.SimulatorServer:app --host 0.0.0.0 --port 8100
   ```
4. **Run experiments** with a readable report (mirrors `run_lanekeeping.py`, see that section below for how to read it):
   ```bash
   python scripts/run_emergency_braking.py --n 20
   python scripts/run_emergency_braking.py --n 30 --seed 7 --trace-worst
   python scripts/run_emergency_braking.py --scenario emergency_braking            # physics, no CARLA needed
   python scripts/run_emergency_braking.py --scenario emergency_braking --use-nn   # scalar-MLP, no CARLA needed
   ```
   Or use the API/frontend (Quick Start steps 4-5) — pick `emergency_braking_carla` from the scenario grid.

For more than one worker, point `EB_SIMULATOR_URLS` (comma-separated) or `EB_NUM_WORKERS`/`EB_SIMULATOR_BASE_PORT` at several CARLA+service pairs — see the one-client-per-CARLA-instance note above.

---

## Cut-In — how to run each mode

**Physics** (default, no setup): `SCENARIOS["cut_in"]` with `use_nn=False`.

**Video-CNN (CARLA)** — same recipe as Emergency Braking, on port 8200 instead of 8100 (do the [CARLA setup](#carla-setup-emergency-braking--cut-in-video-cnn-mode) once if you haven't already):

```bash
opensbt-core/CarlaSimulator_0916/CarlaUE4.exe -RenderOffScreen -nosound   # separate terminal, wait ~20s
python -m scenarios.cut_in.collect_carla_dataset --n 200
python -m scenarios.cut_in.train_cnn --epochs 30
cd opensbt-core
uvicorn Simulator.cut_in.SimulatorServer:app --host 0.0.0.0 --port 8200
```

Then, in another terminal:

```bash
python scripts/run_cut_in.py --n 20
python scripts/run_cut_in.py --n 30 --seed 7 --trace-worst
python scripts/run_cut_in.py --scenario cut_in   # physics, no CARLA needed
```

Once `scenarios/cut_in/models/cut_in_cnn.h5` exists, `cut_in_carla` appears automatically in the registry. Worker pool env vars: `CI_SIMULATOR_URLS` / `CI_NUM_WORKERS` / `CI_SIMULATOR_BASE_PORT`.

Unlike Emergency Braking, the cutting vehicle is **kinematically scripted** (its transform is set directly from the same sigmoid-merge formula as the physics simulator) rather than physics-driven — this keeps the longitudinal/lateral gap bookkeeping exactly consistent with `scenarios/cut_in/qoi.py` while still rendering a real vehicle for the ego's camera to see merging in. Only the ego is a real CARLA physics vehicle, driven by the controller (expert or CNN).

---

## Lane Keeping — how it works and how to run it

The Lane Keeping scenario stress-tests a neural-network autopilot (the Udacity "chauffeur" DNN) by driving it on procedurally generated roads and measuring how well it stays inside the lane. Unlike Emergency Braking and Cut-In's physics/scalar-MLP modes, which run in-process, here the actual driving always happens inside the opensbt-core Unity simulator, which the Python code drives over HTTP: for each sampled scenario the client sends the road parameters to the simulator, the simulator runs the drive, and returns the trajectory (position, cross-track error and steering, step by step).

Its DNN (`mixed-chauffeur.h5`) is **not trained in this repo** — it's downloaded pre-trained from the upstream "maxitwo" project (see `opensbt-core/Simulator/lanekeeping/udacity/README.md`). This is the one thing Lane Keeping does *not* have in common with Emergency Braking/Cut-In's video-CNN mode: there, we collect the data and train the model ourselves (see their sections above); here, we only run inference on someone else's model.

### Architecture: a pool of parallel simulators

Each simulator container holds a single Unity instance and runs **one** simulation at a time — it is inherently sequential. To avoid waiting for hours when sampling dozens of scenarios, the system starts **several containers in parallel** (4 by default), each on its own port (8000, 8001, ...), and distributes the jobs across the pool through a shared queue. With W workers the throughput is roughly W times that of a single container. The client health-checks every worker before starting and skips the unreachable ones; on top of that each job has an individual timeout, so a stuck container only fails its own scenario instead of freezing the whole batch. Full details are in `opensbt-core/PARALLELIZZAZIONE.md`. The same pool/health-check/timeout pattern (`simulators/common/http_worker_pool.py`) is reused by Emergency Braking and Cut-In's CARLA services — currently at 1 worker each, see the CARLA setup section above for why.

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

Main options (shared, with small variations, by `run_emergency_braking.py` and `run_cut_in.py` too):

- `--n` number of sampled scenarios (higher = more accurate estimates but slower)
- `--workers` number of parallel containers/CARLA instances (must match what you started)
- `--seed`, `--quiet` for reproducibility and to silence the per-job progress
- `--trace-worst` print the step-by-step trace of the worst scenario, to see *how* it fails (columns differ per scenario: x/XTE/steering for Lane Keeping, position/velocity for Emergency Braking, longitudinal/lateral gap for Cut-In)
- `--rare-fraction` bottom-k% treated as "rare" (default 5%)
- `--sampling uniform|realistic` — `realistic` samples from an operational-design-domain distribution instead of uniformly, so the failure fraction estimates P(failure) *under realistic driving conditions* rather than under an arbitrary uniform space. Only implemented for Lane Keeping so far (`param_distributions()`); Emergency Braking/Cut-In accept the flag but silently fall back to `uniform`.
- Lane Keeping only: `--preset full|realistic` (bounds preset) and `--max-speed`/`--max-angle` (bound overrides) — these override the parameter *ranges themselves*; not meaningful for the other two scenarios' parameter sets, so they don't have an equivalent.
- Emergency Braking only: `--scenario emergency_braking [--use-nn]` to run physics/scalar-MLP instead of the CARLA default, without needing CARLA at all.

A typical run for an informative analysis and its diagnosis:

```bash
python scripts/run_lanekeeping.py --n 30 --preset realistic --max-speed 6 --trace-worst
```

### Reading the report

The report shows the **failure rate** (share of scenarios with safety margin < 0) reframed as **P(fallimento)** with a 95% confidence interval (Wilson score, robust for small N or extreme rates), the **rare failure rate** (the worst bottom-k%, 5% by default — a *severity* measure, distinct from the probability above), the distribution of margins sorted from worst to best, and the table of the most critical scenarios with their parameters and the number of steps survived. It also reports **invalid scenarios** that are *excluded* from the analysis rather than counted: degenerate runs (simulations aborted in very few steps), physically-inconsistent parameter combinations (e.g. `min_speed > max_speed`, Lane Keeping only) and, when the fidelity gate is enabled, **under-sampled** runs (see *Control fidelity* below, Lane Keeping only). Rates and rare failures are computed only over the valid scenarios, so the failure rate is not polluted by non-real cases. For each rare/worst failure the runner also prints a **data-driven explanation** derived from its trajectory — this is scenario-specific: unstable oscillation / uncorrected drift / wrong-way steering for Lane Keeping, reaction timing + residual speed at impact for Emergency Braking, minimum gap and when it occurred for Cut-In.

The safety margin for Lane Keeping combines three signals: how well the car stays inside the lane (weight 0.6), the abruptness of the steering (0.2) and how early it approaches the edge (0.2). Because the cross-track error saturates once the car leaves the lane, many "full" failures end up with the same margin; to tell them apart, among failures with equal margin the ones that leave the lane **earlier** are considered rarer (survival time as a tie-breaker). Emergency Braking and Cut-In's margins are simpler (remaining stopping distance / remaining gap, see *Scenario parameters* above) and don't need this tie-breaking.

### Control fidelity (parallelism without losing accuracy) — Lane Keeping only

The autopilot runs a closed loop — `predict → step → next Unity frame` — at whatever frequency the machine can sustain. When several containers compete for CPU, each Unity instance slows down and the car travels *more metres between two steering decisions*: the DNN, trained at near-real-time cadence, then steers on stale, widely-spaced observations, and failures start to reflect the machine load rather than the model. To keep parallelism *and* accuracy, the runner now measures two fidelity indicators for every run, computed from data the simulator already returns (`elapsedTime`, `iterations`, `speeds`): the **control frequency** (`iterations / elapsedTime`, Hz) and the **spatial resolution** (`meters_per_step` = mean speed × seconds-per-step). It prints them in a `FEDELTÀ DEL CONTROLLO` block (min / median / max), so you can watch how the cadence changes as you vary `--workers`. Emergency Braking/Cut-In don't have this instrumentation yet (they currently only ever run 1 CARLA worker, so there's no parallelism to lose accuracy to).

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

Emergency Braking and Cut-In's video-CNN models are much younger by comparison (first-pass CNNs trained on ~200 episodes each, see their sections above) — expect their failure-mode characterisation to be less mature than Lane Keeping's until they've been run and analysed at larger N.

---

## Scenarios status

| Scenario | Physics | Scalar-MLP | Video-CNN | Requires (video-CNN) | Rich terminal report |
|---|---|---|---|---|---|
| Emergency Braking | ✅ | ✅ | ✅ (Fase 2) | CARLA 0.9.16 | `scripts/run_emergency_braking.py` |
| Cut-In | ✅ | — | ✅ (Fase 3) | CARLA 0.9.16 | `scripts/run_cut_in.py` |
| Lane Keeping | — | — | ✅ (pre-trained DNN) | Docker (opensbt-core) | `scripts/run_lanekeeping.py` |
