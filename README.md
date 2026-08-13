# Detecting Rare Failures in Autonomous Vehicles

Search algorithms for **rare, safety-critical failures** of a lane-keeping
controller, compared head to head on two simulators at a matched budget.

**The full experimental report is [`docs/risultati_sperimentazione.md`](docs/risultati_sperimentazione.md)**
(Italian). It is the document to read: everything below is how to run the code
that produced it.

---

## The research question

Given a lane-keeping scenario parameterised by 9 variables and a fixed budget of
120 simulations, which search strategy surfaces the most *rare* failures?

Two factors, crossed:

| factor | levels |
|---|---|
| search family | `active_boundary` (GP surrogate + boundary acquisition), `cross_entropy` (CE + importance sampling), `plain_sampling` (the floor) |
| sampling design | `lhs` (Latin Hypercube) vs `random` (i.i.d.) |

Six arms, run on two backends (**Udacity + Docker**, a DNN driving from camera
images; **MetaDrive**, an in-process lateral controller on exact state), with a
pre-registered hypothesis sequence and 12 seeds per campaign.

### The answers, in one paragraph

Active boundary finds **13 to 30 times more rare failures per simulation** than
any other arm, twice demonstrated on MetaDrive (p <= 0.00073, paired by seed).
**LHS does not beat random sampling** — a clean, replicated negative result
across two simulators, two ODDs and three search families. **Cross-entropy is
the worst method for finding rare events** here, and its P(failure) estimate is
unusable at this budget. And the two simulators **do not agree** on which
scenarios are hard (Spearman rho = -0.020 over the same 60 scenarios).

Numbers, statistical tests, caveats and limits: `docs/risultati_sperimentazione.md`.

---

## Repository map

```
docs/
  risultati_sperimentazione.md      # THE REPORT — read this first
  preregistrazione_classifica.json          # hypothesis sequence, committed before the data
  preregistrazione_replica_degradata.json   # same, for the replication campaign

pipeline/                           # the algorithms under comparison
  samplers.py                       # LHSSampler / RandomSampler behind one interface
  active_boundary.py                # GP surrogate on the continuous margin
  active_boundary_random.py         # ... as an arm, with either design
  rare_event.py                     # cross-entropy + importance sampling
  rare_event_random.py              # ... as an arm, with either design
  model_comparison.py               # campaign harness: every arm at a matched budget
  arm_ranking.py                    # rarity metric, paired tests, pre-registered sequence
  failure_regions.py                # DBSCAN regions, axis_spread, structure_score
  operating_point.py                # bisection calibration of speed_scale / obs_lag
  odd_presets.py                    # ODD presets and narrowing rules
  cross_simulator.py                # Udacity <-> MetaDrive comparison on a shared design
  qoi_optimizer.py                  # worst-case search (BayesOpt / CMA-ES)

scenarios/
  common/                           # shared across backends: driver, road frame,
                                    # road geometry, episode budget
  lane_keeping/                     # backend A — Udacity DNN over HTTP (Docker)
  lane_keeping_md/                  # backend B — MetaDrive, in-process, headless
  base_scenario.py                  # the interface every backend implements

scripts/                            # campaign entry points (see "Reproducing")
tests/                              # 177 tests, no Docker and no MetaDrive needed
results/                            # campaign outputs (git-ignored, see report §11)
opensbt-core/                       # Udacity simulator + containers
```

Modules that pre-date this experiment (`api/`, `embedder/`, `frontend/`,
`evaluation/`, `simulators/`, the `emergency_braking` and `cut_in` scenarios)
are the earlier framework the project was built on. They are untouched and
documented in the appendix at the bottom.

---

## Setup

Two environments are needed, because MetaDrive requires Python < 3.12:

```bash
python -m venv .venv        && .venv/Scripts/pip install -e .          # Udacity + pipeline
python3.10 -m venv .venv310 && .venv310/Scripts/pip install -e . metadrive-simulator
.venv310/Scripts/python -m metadrive.pull_asset
```

The **Udacity** backend additionally needs Docker and two files that are not in
the repo (download links in `opensbt-core/README.md`):

- `mixed-chauffeur.h5` in `opensbt-core/Simulator/SelfDrivingModels/`
- the Ubuntu simulator build in `opensbt-core/Simulator/SimulatorExec/ubuntu_binaries/`
  (`ubuntu.x86_64`, executable)

MetaDrive needs neither Docker nor a GPU.

---

## Reproducing the campaigns

The exact commands behind every number in the report are in
`docs/risultati_sperimentazione.md` §11. In short:

```powershell
# --- MetaDrive campaigns (~1.4-2.0 s per simulation) ---
.venv310\Scripts\python.exe scripts\run_model_comparison_md.py --budget 120 ^
       --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --out results\cmp_md12

# --- Udacity campaigns (~30 s per simulation; start the container pool first) ---
python scripts\run_model_comparison.py --budget 120 --seeds 0 1 2 3 4 5 ^
       --max-angle 8 --max-speed 9.6 --max-seg 12 --out results\cmp_rare

# --- operating-point calibration ---
.venv310\Scripts\python.exe scripts\calibrate_operating_point.py lane_keeping_md ^
       --lever obs_lag --sampling odd --max-angle 8 --max-speed 9.6 --max-seg 12 ^
       --n 120 --low 0.02 --high 0.10 --out results\taratura_md_narrow_lag_odd.json

# --- rankings and the pre-registered sequence (no simulation, seconds) ---
python scripts\rank_arms.py results\cmp_md12_raw.npz --regions ^
       --plan docs\preregistrazione_classifica.json --out results\rank_md12

# --- cross-simulator comparison on a shared design ---
.venv310\Scripts\python.exe scripts\run_cross_simulator.py collect lane_keeping_md --n 60 --speed-scale 0.1766
python scripts\run_cross_simulator.py collect lane_keeping --n 60 --speed-scale 0.42
python scripts\run_cross_simulator.py compare results\cross_lane_keeping_md.json results\cross_lane_keeping.json
```

Analysis scripts (`rank_arms.py`, `reanalyze_regions.py`, `run_cross_simulator.py compare`)
re-read the saved `.npz`/`.json` and run in seconds — they never re-simulate, so
every table in the report can be regenerated from the files in `results/`
without a simulator.

Supporting scripts, all optional:

| script | what it answers |
|---|---|
| `scripts/reanalyze_regions.py` | structure vs shuffled z-scores (report §7) |
| `scripts/check_repeatability.py` | how deterministic is the simulator? (report §1.4) |
| `scripts/validate_model_comparison.py` | LHS-vs-random on analytic regions, no Docker |
| `scripts/diag_road_parity.py` | same theta, same road? the cross-simulator gate |
| `scripts/diag_odd_feasibility.py` | is the ODD physically drivable at all? |
| `scripts/plot_boundary_md.py` | the learned fail/safe boundary in (curvature, speed) |

---

## Tests

```bash
python -m pytest tests -q
```

177 tests pass without Docker and without MetaDrive; 38 more are skipped unless
those are installed. `tests/test_arm_ranking.py` alone carries 33 tests,
including a regression for every pipeline defect listed in report §8.

---

## What was set aside

The final cleanup moved out of the repo everything the reported experiment does
not use. Nothing was deleted: it all sits under `_archive/` (git-ignored) and
in the git history.

| archived | why |
|---|---|
| `_archive/carla/` | the CARLA backend never became available (report §10.10): the geometric parity gate fails at 19.3 cm and the GPU is too small. The canonical road generator that lived inside it was moved to `scenarios/common/road_geometry.py` first, where it belongs. |
| `_archive/behavior_cloning/` | a learned MetaDrive driver, never used: every reported campaign runs the shared lateral controller. |
| `_archive/diagnostici/` | one-off diagnostics for geometry bugs that are now fixed and covered by `tests/test_road_parity.py`. |
| `results/_archive/` | campaigns no table in the report cites: a trial run, a superseded `obs_lag` calibration, a probe. |
| `_archive/RESULTS.md` | an earlier results document, superseded by `docs/risultati_sperimentazione.md`. |
| `results/_archive/pre_rinomina_campi/` | the seven JSON files as they were before their field names were translated to English. Backups, not campaigns. |

---

## Appendix — Lane Keeping on Udacity: operational details

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

---

## Appendix — the pre-existing framework

The repository started as a general rare-failure framework: an LHS sampler, a
POD trajectory embedder, a QoI module, a FastAPI server and a small browser
frontend, with two in-process scenarios (Emergency Braking, Cut-In). None of it
is on the path of the experiment reported here, and none of it was touched by
the cleanup.

```bash
python -m scenarios.emergency_braking.train --dataset-only   # generate the dataset
python -m scenarios.emergency_braking.train                  # train the MLP (~15 min CPU)
python scripts/validate_pipeline.py --n 200                  # end-to-end smoke test
uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload   # API; frontend/index.html is the UI
```

| Scenario | Physics | NN | Requires |
|---|---|---|---|
| Emergency Braking | yes | yes (train first) | nothing |
| Cut-In | in development | in development | nothing |
| Lane Keeping (Udacity) | — | yes | Docker |
| Lane Keeping (MetaDrive) | yes | — | metadrive-simulator |
