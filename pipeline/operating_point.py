"""
Calibration of a backend's operating point.

The problem
-----------
A backend that fails in 0% of cases, or in 100%, **carries no information**:
there is no variance to explain, the boundary is not learnable, and the
comparison between search methods becomes empty because either everyone finds
everything or nobody finds anything.

Observed in practice at `speed_scale = 1.0`:

    Udacity  (C2, 1 worker)   5/5 falliti
    MetaDrive (C2, smoke)     2/2 falliti

With those numbers any comparison between the two would measure noise.

The opposite problem is just as real: lower the speed too far and nothing fails
any more, so there is nothing left to search for.

The lever
---------
**One only**, so that the calibration can be declared rather than being a set of
opaque adjustments: `speed_scale`, which scales the cruising speed.

Why speed in particular: it acts at once on the control margin (more time to
correct) and on the metres covered between two decisions, which is the central
failure mechanism of this work. It is also the only parameter that means the
same thing on every backend.

The value found **must be declared in the report**: it is a parameter of the
experiment, not an implementation detail. Two backends tuned to different
`speed_scale` remain comparable on the SHAPE of the failure set, not on
absolute failure rates.

Metodo
------
Bisection on the failure rate, which is non-increasing in `speed_scale`: slower
= fewer failures. Monotonicity is not exact (sampling is stochastic), so the
SAME seed and the SAME parameter design are used at every evaluation: without
that, the noise between two evaluations would be mistaken for the effect of
speed and the bisection would not converge.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class Evaluation:
    """One point of the bisection."""
    speed_scale: float
    failure_rate: float
    n_valid: int
    n_total: int
    median_margin: float


@dataclass
class CalibrationResult:
    """Outcome of the calibration, serialisable for the report."""
    backend: str
    speed_scale: float               # the chosen value
    failure_rate: float              # the rate it reached
    target_low: float
    target_high: float
    centered: bool                   # False = the band could not be centred
    n_samples: int
    seed: int
    history: List[Evaluation] = field(default_factory=list)
    note: str = ""
    # The ODD the calibration was performed on. speed_scale is NOT a property of
    # the backend: the failure rate it produces depends on the box the scenarios
    # are drawn from, so a value tuned on one ODD says nothing about another.
    # Recorded here so a campaign on different bounds cannot silently inherit it.
    odd_lower: List[float] = field(default_factory=list)
    odd_upper: List[float] = field(default_factory=list)
    #: Which parameter was calibrated: "speed_scale" (default) or "obs_lag",
    #: which degrades the controller's observation. The second one CHANGES the
    #: system under test, so it must always be read together with the value.
    lever: str = "speed_scale"
    fixed_speed_scale: float = None

    def report(self) -> str:
        lines = [
            "=" * 64,
            f" OPERATING-POINT CALIBRATION -- {self.backend}",
            "=" * 64,
            f" Target band  : {self.target_low:.0%} - {self.target_high:.0%}",
            f" Samples/eval : {self.n_samples}   (seed {self.seed}, fixed design)",
            f" ODD          : {self._odd_str()}",
            f" Lever        : {self.lever}" + (
                f"   (speed_scale held at {self.fixed_speed_scale})"
                if self.lever == "obs_lag" else ""),
            "-" * 64,
            f" {self.lever:>12} | {'failure':>8} | {'valid':>7} | median margin",
        ]
        for v in self.history:
            lines.append(f" {v.speed_scale:>12.4f} | {v.failure_rate:>7.1%} | "
                         f"{v.n_valid:>3}/{v.n_total:<3} | {v.median_margin:>8.3f}")
        lines += [
            "-" * 64,
            f" CHOSEN: {self.lever} = {self.speed_scale:.4f}  "
            f"-> failure rate {self.failure_rate:.1%}",
        ]
        if not self.centered:
            lines.append(f" !  {self.note}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def _odd_str(self) -> str:
        if not self.odd_lower:
            return "not recorded (calibration predates this field)"
        return (f"angles [{self.odd_lower[0]:.0f}, {self.odd_upper[0]:.0f}]  "
                f"max_speed [{self.odd_lower[6]:.1f}, {self.odd_upper[6]:.1f}]  "
                f"segment [{self.odd_lower[7]:.0f}, {self.odd_upper[7]:.0f}]"
                if len(self.odd_lower) >= 8 else str(self.odd_lower))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        d = asdict(self)
        # `speed_scale` is the historical field name, but with --lever obs_lag
        # the value is NOT a speed: it is a time constant in seconds. A reader
        # who takes that number and passes it to --speed-scale runs a different
        # experiment from the one calibrated, and nothing tells them. So the
        # lever name and the flag to use are exposed explicitly.
        d["lever_value"] = float(self.speed_scale)
        d["cli_flag"] = ("--obs-lag" if self.lever == "obs_lag"
                         else "--speed-scale")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


def calibrate(
    evaluate: Callable[[float], Evaluation],
    *,
    backend: str,
    target_low: float = 0.10,
    target_high: float = 0.20,
    scale_min: float = 0.15,
    scale_max: float = 1.0,
    max_iter: int = 8,
    n_samples: int = 40,
    seed: int = 42,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Bisection on `speed_scale` to centre the failure rate inside the band.

    `evaluate(scale) -> Evaluation` is injected: the calibration knows nothing
    about backends and is tested with a synthetic function of known truth.

    Cases that are NOT an error and must be reported, not hidden:

      * even at the lowest scale the backend fails too often -> the controller
        cannot hold that scenario even at walking pace, and it is the
        controller that needs revisiting before the campaign, not the speed;
      * even at the highest scale it never fails -> the parameter space is too
        easy, and the ODD must be widened or the controller degraded
        (`obs_latency`, `obs_lag_tau`).

    In both cases the endpoint closest to the band is returned with
    `centered=False` and an explicit note.
    """
    history: List[Evaluation] = []

    def _log(v: Evaluation) -> None:
        history.append(v)
        if verbose:
            print(f"  speed_scale={v.speed_scale:.4f} -> "
                  f"failure {v.failure_rate:.1%} "
                  f"({v.n_valid}/{v.n_total} valid, "
                  f"median margin {v.median_margin:+.3f})", flush=True)

    if verbose:
        print(f"[calibration {backend}] target band "
              f"{target_low:.0%}-{target_high:.0%}, {n_samples} samples/evaluation",
              flush=True)

    def _within_band(v: Evaluation) -> CalibrationResult:
        return CalibrationResult(
            backend=backend, speed_scale=v.speed_scale,
            failure_rate=v.failure_rate, target_low=target_low,
            target_high=target_high, centered=True, n_samples=n_samples,
            seed=seed, history=history)

    # Endpoints: they say straight away whether the band is reachable.
    #
    # The band is checked RIGHT AFTER each endpoint, before evaluating the
    # other: one evaluation costs `n_samples` simulations, which on Udacity
    # means minutes.
    high_end = evaluate(scale_max); _log(high_end)      # faster = more failures
    if target_low <= high_end.failure_rate <= target_high:
        return _within_band(high_end)
    if high_end.failure_rate < target_low:
        return CalibrationResult(
            backend=backend, speed_scale=scale_max,
            failure_rate=high_end.failure_rate, target_low=target_low,
            target_high=target_high, centered=False, n_samples=n_samples,
            seed=seed, history=history,
            note=(f"Even at speed_scale={scale_max} the failure rate does not "
                  f"reach {target_low:.0%}. The parameter space is too easy: "
                  f"widen the ODD, or degrade the controller "
                  f"(obs_latency / obs_lag_tau) to make a boundary emerge."))

    low_end = evaluate(scale_min); _log(low_end)
    if target_low <= low_end.failure_rate <= target_high:
        return _within_band(low_end)
    if low_end.failure_rate > target_high:
        return CalibrationResult(
            backend=backend, speed_scale=scale_min,
            failure_rate=low_end.failure_rate, target_low=target_low,
            target_high=target_high, centered=False, n_samples=n_samples,
            seed=seed, history=history,
            note=(f"Even at speed_scale={scale_min} it fails in "
                  f"{low_end.failure_rate:.0%} of cases. This is not a speed "
                  f"problem: the controller does not hold the lane even at "
                  f"walking pace. Check the steering sign and the gains BEFORE "
                  f"the campaign -- calibrating here would hide the defect."))

    lo, hi = scale_min, scale_max
    best = min((high_end, low_end),
                   key=lambda v: _distance_from_band(v.failure_rate,
                                                       target_low, target_high))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        v = evaluate(mid); _log(v)

        if _distance_from_band(v.failure_rate, target_low, target_high) < \
           _distance_from_band(best.failure_rate, target_low, target_high):
            best = v

        if target_low <= v.failure_rate <= target_high:
            return _within_band(v)

        # Too many failures -> slow down; too few -> speed up.
        if v.failure_rate > target_high:
            hi = mid
        else:
            lo = mid

    return CalibrationResult(
        backend=backend, speed_scale=best.speed_scale,
        failure_rate=best.failure_rate, target_low=target_low,
        target_high=target_high, centered=False, n_samples=n_samples,
        seed=seed, history=history,
        note=(f"Bisection exhausted after {max_iter} iterations without "
              f"centring the band. The value returned is the closest one. With "
              f"{n_samples} samples the uncertainty on the rate is about "
              f"+/-{1.96 * (0.25 / n_samples) ** 0.5:.0%}: if that is comparable "
              f"to the band width, what is needed is more samples, not more "
              f"iterations."))


def _distance_from_band(rate: float, low: float, high: float) -> float:
    if rate < low:
        return low - rate
    if rate > high:
        return rate - high
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on a real scenario
# ─────────────────────────────────────────────────────────────────────────────

def scenario_evaluator(
    build_scenario: Callable[[float], object],
    *,
    n_samples: int = 40,
    seed: int = 42,
    verbose: bool = False,
    lower=None,
    upper=None,
    sampling: str = "uniform",
) -> Callable[[float], Evaluation]:
    """
    Builds the evaluation function for a scenario from the registry.

    `build_scenario(speed_scale)` must return a `BaseScenario` tuned to that
    speed.

    **The parameter design is FIXED** across evaluations: same seed, same
    thetas. Without that, the difference between two evaluations would mix the
    effect of speed with sampling noise, and the bisection would chase the
    noise instead of the signal.

    **`sampling` decides WHAT is being calibrated.** With `"uniform"` the design
    is uniform over the box; with `"odd"` it goes through the operational
    marginals (`param_distributions`), which is what the campaign's
    `plain_sampling` arms do. The two rates do NOT coincide -- on the wide ODD
    the uniform calibration gave 17.2% while `plain_sampling` measured 3.2%, a
    factor of 5 -- so calibrating a band in uniform and then reading it under
    the ODD puts the campaign out of band. Use `"odd"` when the target is the
    rate of the blind-sampling arms.
    """
    from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

    def _evaluate(speed_scale: float) -> Evaluation:
        scenario = build_scenario(speed_scale)
        bounds = scenario.param_bounds()
        lo = np.asarray(bounds["lower"] if lower is None else lower, float)
        hi = np.asarray(bounds["upper"] if upper is None else upper, float)

        # Deterministic LHS design, identical at every call.
        sampler = LatinHypercube(d=len(lo), seed=seed)
        unit = sampler.random(n=n_samples)
        if sampling == "odd" and hasattr(scenario, "param_distributions"):
            import numpy as _np
            dists = scenario.param_distributions(lo, hi)
            u = _np.clip(unit, 1e-12, 1 - 1e-12)
            params = _np.column_stack([d.ppf(u[:, j])
                                       for j, d in enumerate(dists)])
        else:
            params = qmc_scale(unit, lo, hi)

        traj = scenario.run_simulation(params, verbose=verbose)
        qoi = np.asarray(scenario.compute_qoi(traj, params), dtype=float)

        valid = np.asarray(getattr(scenario, "_valid_mask",
                                   ~np.isnan(qoi)), dtype=bool)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return Evaluation(speed_scale, float("nan"), 0, n_samples, float("nan"))

        rate = float((qoi[valid] < scenario.failure_threshold()).mean())
        return Evaluation(
            speed_scale=speed_scale,
            failure_rate=rate,
            n_valid=n_valid,
            n_total=n_samples,
            median_margin=float(np.nanmedian(qoi[valid])),
        )

    return _evaluate
