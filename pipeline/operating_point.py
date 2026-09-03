"""
Calibration of a backend's operating point through `speed_scale`.

`calibrate` searches for the value of `speed_scale` -- the multiplier applied to
the cruising speed by `scenarios.common.driver.target_speed` -- that puts the
backend's failure rate inside a target band (by default 10-20%).

How the search works:

* `scenario_evaluator` builds the callable that, given a `speed_scale`, runs a
  batch of simulations and returns an `Evaluation` (failure rate, number of
  valid runs, number of runs). The same seed and the same parameter design are
  reused at every evaluation, so two evaluations differ only in `speed_scale`.
* `calibrate` first evaluates the bracket endpoints, then calls `_bisect`, which
  halves the interval assuming the failure rate is non-increasing in
  `speed_scale`: if the rate at the midpoint is above the band the search moves
  down, otherwise up.
* `_distance_from_band` scores each evaluation by its distance from the target
  band; the best-scoring evaluation seen is what gets returned.
* `TOO_EASY_NOTE`, `CONTROLLER_BROKEN_NOTE` and `EXHAUSTED_NOTE` are the
  diagnostic strings attached to `CalibrationResult` when the whole bracket
  fails below the band, above the band, or when the evaluation budget runs out
  before the band is reached.

`CalibrationResult` holds the chosen `speed_scale`, every `Evaluation` in
search order and those notes, and serialises to JSON via `asdict`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, List

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


def _bisect(evaluate, log, lo: float, hi: float, best: Evaluation,
            target_low: float, target_high: float, max_iter: int) -> tuple:
    """
    Halve the interval until the failure rate lands inside the band.

    Returns ``(hit, best)``: `hit` is the evaluation inside the band, or None if
    the iterations ran out; `best` is the closest evaluation seen either way, so
    an exhausted search still returns something usable.
    """
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        v = evaluate(mid); log(v)

        if _distance_from_band(v.failure_rate, target_low, target_high) < \
           _distance_from_band(best.failure_rate, target_low, target_high):
            best = v

        if target_low <= v.failure_rate <= target_high:
            return v, best

        # Too many failures -> slow down; too few -> speed up.
        if v.failure_rate > target_high:
            hi = mid
        else:
            lo = mid
    return None, best


# The three ways calibration ends without centring the band. Kept as templates
# because each one tells the reader to fix a different thing, and the wording is
# the actionable part of the result.
TOO_EASY_NOTE = (
    "Even at speed_scale={scale} the failure rate does not reach {low:.0%}. "
    "The parameter space is too easy: widen the ODD, or degrade the controller "
    "(obs_latency / obs_lag_tau) to make a boundary emerge.")

CONTROLLER_BROKEN_NOTE = (
    "Even at speed_scale={scale} it fails in {rate:.0%} of cases. This is not a "
    "speed problem: the controller does not hold the lane even at walking pace. "
    "Check the steering sign and the gains BEFORE the campaign -- calibrating "
    "here would hide the defect.")

EXHAUSTED_NOTE = (
    "Bisection exhausted after {max_iter} iterations without centring the band. "
    "The value returned is the closest one. With {n_samples} samples the "
    "uncertainty on the rate is about +/-{half_width:.0%}: if that is comparable "
    "to the band width, what is needed is more samples, not more iterations.")


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

    `evaluate(scale) -> Evaluation` is injected, so the calibration depends on no
    backend.

    Three outcomes end the search without centring the band, each returning the
    evaluation closest to it with ``centered=False`` and a note: the rate stays
    above the band even at `scale_min`, it stays below it even at `scale_max`, or
    the iterations run out.
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

    def _endpoint(v: Evaluation, note: str) -> CalibrationResult:
        return CalibrationResult(
            backend=backend, speed_scale=v.speed_scale,
            failure_rate=v.failure_rate, target_low=target_low,
            target_high=target_high, centered=False, n_samples=n_samples,
            seed=seed, history=history, note=note)

    # Endpoints: they say straight away whether the band is reachable, and the
    # band is checked RIGHT AFTER each one, before evaluating the other -- a
    # single evaluation costs `n_samples` simulations, which on Udacity means
    # minutes.
    high_end = evaluate(scale_max); _log(high_end)      # faster = more failures
    if target_low <= high_end.failure_rate <= target_high:
        return _within_band(high_end)
    if high_end.failure_rate < target_low:
        return _endpoint(high_end, TOO_EASY_NOTE.format(
            scale=scale_max, low=target_low))

    low_end = evaluate(scale_min); _log(low_end)
    if target_low <= low_end.failure_rate <= target_high:
        return _within_band(low_end)
    if low_end.failure_rate > target_high:
        return _endpoint(low_end, CONTROLLER_BROKEN_NOTE.format(
            scale=scale_min, rate=low_end.failure_rate))

    hit, best = _bisect(evaluate, _log, scale_min, scale_max,
                        min((high_end, low_end),
                            key=lambda v: _distance_from_band(
                                v.failure_rate, target_low, target_high)),
                        target_low, target_high, max_iter)
    if hit is not None:
        return _within_band(hit)

    return _endpoint(best, EXHAUSTED_NOTE.format(
        max_iter=max_iter, n_samples=n_samples,
        half_width=1.96 * (0.25 / n_samples) ** 0.5))


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
    Build the evaluation function for a scenario from the registry.

    `build_scenario(speed_scale)` must return a `BaseScenario` tuned to that
    speed. The parameter design is fixed across evaluations -- same seed, same
    thetas -- so consecutive evaluations differ only by the speed.

    `sampling` selects the distribution the design is drawn from: `"uniform"`
    over the box, or `"odd"` through the operational marginals
    (`param_distributions`), which is what the campaign's `plain_sampling` arms
    use. The two give different failure rates on the same scenario.
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
