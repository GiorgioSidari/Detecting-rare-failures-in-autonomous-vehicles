# Results — Rare-failure detection on the real Udacity DNN

This document reports the empirical findings of the `lanekeeping_step_improvement` branch.
The headline: the active-learning boundary method (`pipeline/active_boundary.py`) was applied
to the **real Udacity "chauffeur" DNN** in the Unity simulator, **cross-validated against the
existing Cross-Entropy estimator**, and used to quantify the model's safety envelope and to
extract concrete, reproducible **rare-failure scenarios**.

All numbers below are on the real vision model in Unity (`--scenario lane_keeping`), not on a
stand-in controller.

---

## 1. The model is broadly fragile

On the full ODD (angles up to 85°, speed up to 30 m/s) the DNN fails almost everywhere:
`P(failure) ≈ 100%` (active-boundary, 27 sims). Failures are the norm, not the exception — so
"rare failures" only exist once the ODD is narrowed toward the model's safe envelope.

The learned feature importance shifts with the regime: on the wide ODD **curvature (road
angles)** dominates; in the narrow/rare regime the dominant driver becomes **segment_length**
(the length of road segments), with speed second.

---

## 2. Safety envelope in speed (gentle curves ≤ 8°, short segments ≤ 14 m)

Narrowing the ODD one axis at a time maps a clean, monotonic safety staircase:

| Speed band (m/s) | P(failure) | Method | Notes |
|---|---|---|---|
| ≤ 9 | **0%** (0 / 440) | Cross-Entropy | genuinely safe |
| 9 – 10 | **≈ 2%**  CI95 [0.9, 3.7] | active-boundary | **rare failures** (see §4) |
| 10 – 11 | **≈ 30–32%** | active-boundary **31.7%** ≈ Cross-Entropy **30.5%** | **cross-validated** (see §3) |
| road segments ≥ 32 m | **100%** | active-boundary | catastrophic regardless of speed/curvature |

The safety boundary in speed sits at **~9–10 m/s**. Long road segments break the model
systematically — a non-obvious finding surfaced automatically by the ARD importance.

---

## 3. Cross-validation of the two methods

On the identical ODD (angles ≤ 8°, speed 10–11 m/s, segments ≤ 14 m), two independent
estimators agree on P(failure):

- **active-boundary (GP surrogate):** P = **31.70%**, CI95 [30.9, 32.4]
- **Cross-Entropy (importance sampling):** P = **30.46%**, CI95 [19.8, 42.4]

Agreement within confidence intervals validates the active-boundary estimator against the
project's pre-existing rare-event method, on the real model.

---

## 4. Concrete rare failures (P ≈ 2%, speed 9–10 m/s)

In a regime where the DNN succeeds ~98% of the time, the active-boundary method isolated
specific, reproducible failing scenarios. Sorted worst-first (safety margin, negative = crash):

| margin | angles (°) | min_speed | max_speed | seg (m) | map (m) |
|---|---|---|---|---|---|
| -0.277 | 6, 5, 5, 0, 4 | 5.6 | 9.9 | 12.8 | 233 |
| -0.267 | 5, 8, 6, 8, 7 | 6.9 | 9.9 | 12.2 | 248 |
| -0.164 | 2, 7, 0, 1, 4 | 7.6 | 9.8 | 11.9 | 153 |
| -0.153 | 6, 6, 0, 1, 1 | 8.0 | 9.6 | 12.4 | 225 |
| -0.097 | 5, 7, 6, 3, 3 | 8.3 | 9.8 | 12.8 | 287 |

(10 / 48 sampled points failed; these are the 5 worst.) Each row is a fully specified,
re-runnable scenario — the kind of concrete rare failure the project set out to detect.

---

## 5. How to read the output

`scripts/run_active_boundary.py` prints a progress log followed by a result block. This is what
each field means.

**Progress log.**

- `[scenario]` / `[ODD] narrowed: ...` — the scenario under test and the operational design domain
  after applying the `--max-*` / `--min-*` caps. A `None` cap means that bound was left at its
  native value.
- `[seed] K valid / N   failure rate=...` — the `N` initial LHS points (`--n-seed`) that are
  actually simulated to bootstrap the GP. `K` are valid; the rest are discarded as degenerate
  (aborted runs) or incoherent (e.g. `min_speed > max_speed`). The seed is roughly uniform, so its
  failure rate is close to the natural one.
- `[iter i/n]  +m valid  total=T  cumulative failure rate=...` — each active-learning iteration
  simulates a `--batch` of new points chosen **on the fail/safe boundary** (where the GP predicts
  `P(fail) ≈ 0.5`); `+m` are valid, `total` is the cumulative valid count, and the cumulative
  failure rate is the failed fraction over all valid points so far. This rate climbs toward ~50%
  **by construction** — boundary points fail about half the time — so it reflects the sampling
  strategy, not the model's true failure probability.

**Result block.**

- `total evaluations` — the real simulations actually spent (the analysis budget ≈
  `n_seed + batch × n_iter`, minus invalid points).
- `empirical fail rate` — the failed fraction among the evaluated points. High **by construction**
  (boundary sampling); do **not** read it as a probability.
- `P(failure) [ODD (param_distributions)]` with `CI95` — **the rarity number.** The learned GP
  gives `P(fail | θ)` at every point; this value is its Monte-Carlo integral against the realistic
  operational distribution (`param_distributions`) over the ODD, i.e. the probability that a
  *realistically drawn* scenario fails. The `CI95` is a 95% credible interval from the GP posterior
  (the model's own uncertainty). The large gap between `empirical fail rate` and `P(failure)` is the
  visible signature of boundary sampling: the first weights points by *where we sampled* (the
  boundary, ~50/50), the second by *how realistic* each scenario is (most are safe → low P).
- `most influential parameters (ARD importance)` — which parameters drive failure, from the ARD
  length-scales of the GP kernel (a short length-scale on a dimension = the margin is sensitive to
  it). The **group** signal (e.g. speed + segment_length above the angles) is robust across runs;
  the fine ranking is somewhat unstable.
- `failing scenarios found: F/total (worst first)` — the concrete failing scenarios, sorted by
  `margin` (negative = failure, more negative = worse crash). Each row lists all nine parameters:
  `margin`, the five road `angles` (°, cumulative per-segment turns), `minV`/`maxV` = min/max speed
  (m/s), `seg` = segment length (m), `map` = map size (m). Every row is a fully specified,
  re-runnable failing scenario.

**Rule of thumb.** Read `P(failure)` (with its CI) as the *rarity*, the failing-scenario list as
the *severity / concrete cases*, and the ARD block as the *cause*. The `empirical fail rate` is
diagnostic of the method (boundary concentration), not a result. The other runners share the same
two-axis reading: `scripts/run_lanekeeping.py` reports `P(failure)` with a Wilson CI plus the
bottom-k% worst cases, and `scripts/run_rare_event.py` reports the Cross-Entropy `P(failure)` with a
bootstrap CI.

---

## 6. Reproduce

Requires the Unity simulator running (see the Lane Keeping section of `README.md`).

```bash
# real DNN, full ODD (fails ~everywhere)
python scripts/run_active_boundary.py --scenario lane_keeping --n-seed 12 --batch 6 --n-iter 3

# rare regime: concrete rare failures at P ≈ 2%
python scripts/run_active_boundary.py --scenario lane_keeping \
    --max-angle 8 --max-speed 10 --max-seg 14 --n-seed 24 --batch 8 --n-iter 4

# cross-validation at the 10–11 m/s band
python scripts/run_active_boundary.py --scenario lane_keeping --max-angle 8 --max-speed 11 --max-seg 14
python scripts/run_rare_event.py     --scenario lane_keeping --max-angle 8 --max-speed 11 --max-seg 14 \
    --spi 40 --final 200 --max-iter 6
```
