# Running the Lane-Keeping simulator on x86_64 (native)

The lane-keeping simulator combines two components with **conflicting emulation
needs on Apple Silicon**:

- **TensorFlow 2.9.2** needs the CPU **AVX** instructions → works under QEMU, but
  Rosetta does not reliably expose AVX (TF aborts with *"compiled to use AVX…"*).
- **Unity (Mono runtime)** does dynamic amd64 code generation → works under Rosetta,
  but **crashes under QEMU** (`Assertion … tramp-amd64.c`, `qemu: signal 6`).

There is no emulation mode on ARM that runs both. **Run it on a native x86_64
machine** and every one of these problems disappears (native AVX, native Mono JIT,
software GL). This works on any Intel/AMD box: a lab/university machine, a Linux
PC, an Intel Mac, or a cloud VM. **No GPU is required** — rendering uses Mesa's
software rasteriser.

---

## Prerequisites

- An **x86_64 / amd64** machine running Linux (Ubuntu 22.04+ recommended).
- **Docker** + **docker compose** installed and running.
  ```bash
  # Ubuntu quick install:
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER      # then log out/in so `docker` works without sudo
  docker compose version             # confirm compose v2 is available
  ```
- Confirm you're actually on x86_64:
  ```bash
  uname -m        # must print: x86_64
  ```

---

## Steps

### 1. Get the code

```bash
git clone <your-repo-url> Detecting-rare-failures-in-autonomous-vehicles
cd Detecting-rare-failures-in-autonomous-vehicles
git checkout lanekeeping_step_improvement
```

### 2. Add the two binaries that are NOT in git

These are gitignored (large files). Download them from the MaxiTwo SWITCHdrive
(see `Simulator/lanekeeping/udacity/README.md`) and place them exactly here:

```
opensbt-core/Simulator/SelfDrivingModels/mixed-chauffeur.h5          # the DNN model
opensbt-core/Simulator/SimulatorExec/ubuntu_binaries/ubuntu.x86_64   # the Unity build
opensbt-core/Simulator/SimulatorExec/ubuntu_binaries/ubuntu_Data/    # (comes with it)
opensbt-core/Simulator/SimulatorExec/ubuntu_binaries/UnityPlayer.so  # (comes with it)
```

Make the Unity executable runnable:

```bash
chmod +x opensbt-core/Simulator/SimulatorExec/ubuntu_binaries/ubuntu.x86_64
```

### 3. (Optional) drop the amd64 pin

The `platform: linux/amd64` lines in `docker-compose.yml` / `docker-compose-ch2.yml`
were added for Apple Silicon. On a native x86_64 host they're harmless (the host IS
amd64), so you can leave them or remove them — either works.

### 4. Build and start the simulator

```bash
cd opensbt-core
docker compose up --build
```

First build takes a few minutes (TensorFlow + deps). On native x86 there is **no**
AVX abort and **no** Mono crash.

### 5. Verify it's healthy

In another terminal:

```bash
curl http://localhost:8000/health          # -> {"status": "Up and running"}
```

The container log should get **past** `sleep...and repeat to connect` (Unity
connects to the socket) instead of looping forever.

### 6. Run a simulation

From the repo root (in a venv with `pip install -e .`):

```bash
# quick diagnostic (2 runs, prints QoI breakdown)
python tests/test_lane_keeping.py

# full pipeline
python -c "from pipeline.orchestrator import run; print(run('lane_keeping', n_samples=50).failure_rate)"
```

---

## Step D — multi-model comparison

Once the single model works, add the second model and run both containers:

```bash
# place the second model, e.g. opensbt-core/Simulator/SelfDrivingModels/dave2-ch2.h5
cd opensbt-core
docker compose up --build -d                           # chauffeur → :8000
docker compose -f docker-compose-ch2.yml up --build -d  # second model → :8001
cd ..
python compare_models.py --n-samples 200 --save results/
```

See `Simulator/SelfDrivingModels/README.md` for how to obtain the second model.

---

## Cloud VM notes (if you don't have a physical x86 machine)

Any Intel/AMD Linux instance works — pick the **x86_64 / amd64** image (NOT ARM/Graviton).
- CPU-only is fine; no GPU needed.
- 4 vCPU / 8 GB RAM is comfortable; 2 vCPU works but simulations are slower.
- Open/allow ports 8000 (and 8001 for Step D) if you want to reach the API from
  your laptop; otherwise run the pipeline on the VM itself.

After provisioning, follow the same Steps 1–6 above.
