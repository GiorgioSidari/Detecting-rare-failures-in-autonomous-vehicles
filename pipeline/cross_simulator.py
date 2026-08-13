"""
Confronto cross-simulatore: Udacity ↔ MetaDrive ↔ CARLA.

The problem this class solves
-----------------------------
Comparing two backends is NOT comparing their failure rates. Changing the
simulator changes three things at once:

    (a) how hard the simulator is  -- geometry, vehicle dynamics, control rate
    (b) how good the controller is -- the same controller is not equally good
    (c) the operating point        -- the backends are tuned to different `speed_scale`

A higher rate on one backend cannot be attributed to any of the three. The
metrics here are chosen because they **survive** an offset in difficulty:

  1. **Spearman on the QoI** -- do the two backends agree on WHICH scenarios
     are hard? Independent of the QoI scale and of the absolute rate. It is
     the most informative number about the relation between two backends.
  2. **Failure-region overlap** -- delegated to
     `pipeline.failure_regions.compare_failure_regions`, which is already
     label-agnostic: it takes `{label: (theta, margins)}` and does not know
     whether the labels are search methods or simulators.
  3. **Failure rate** -- reported for completeness, with its `speed_scale`
     next to it, and flagged as NOT comparable.

Structural constraint: the work happens on FILES
------------------------------------------------
The backends do not share a Python environment:

    Udacity    Python 3.8 inside Docker (gym, tensorflow 2.9)
    MetaDrive  Python <3.12 in locale (gymnasium)
    CARLA      Python 3.8 in container + GPU

So the comparison cannot call the backends in-process: each one saves its own
result and this class reads them back. It is also more robust -- campaigns run
for hours and you do not want to lose them to a bug in the comparison.

The shared design
-----------------
Spearman requires that both backends evaluated **the same thetas**. If the
designs differ the correlation is meaningless: it correlates scenario i of A
with scenario i of B, which are different scenarios. The class checks this and
REFUSES to compute it, rather than returning a number that says nothing.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# One backend's run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BackendRun:
    """
    One campaign on one backend, with everything needed to interpret it.


    `speed_scale` is not a detail: it is the operating point the backend was
    tuned to, and without it the failure rates cannot even be read.
    """

    backend: str
    theta: np.ndarray               # (N, d) evaluated parameters
    qoi: np.ndarray                 # (N,)   safety margin, <0 = failure
    valid: Optional[np.ndarray] = None      # (N,) bool; None = all valid except NaN
    speed_scale: float = 1.0
    control_hz: Optional[np.ndarray] = None
    meters_per_step: Optional[np.ndarray] = None
    threshold: float = 0.0
    note: str = ""

    def __post_init__(self):
        self.theta = np.asarray(self.theta, dtype=float)
        self.qoi = np.asarray(self.qoi, dtype=float)
        if self.theta.ndim != 2:
            raise ValueError(f"theta must be (N, d), got {self.theta.shape}")
        if len(self.qoi) != len(self.theta):
            raise ValueError(f"theta ({len(self.theta)}) and qoi ({len(self.qoi)}) "
                             f"have different lengths")
        if self.valid is None:
            self.valid = ~np.isnan(self.qoi)
        self.valid = np.asarray(self.valid, dtype=bool)

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    @property
    def failure_rate(self) -> float:
        """NaN when no run is valid: not 0, which would read as 'never failed'."""
        if self.n_valid == 0:
            return float("nan")
        return float((self.qoi[self.valid] < self.threshold).mean())

    @property
    def theta_failed(self) -> np.ndarray:
        m = self.valid & (self.qoi < self.threshold)
        return self.theta[m]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        d = {
            "backend": self.backend,
            "speed_scale": self.speed_scale,
            "threshold": self.threshold,
            "note": self.note,
            "theta": self.theta.tolist(),
            "qoi": [None if np.isnan(v) else float(v) for v in self.qoi],
            "valid": self.valid.tolist(),
            "control_hz": (None if self.control_hz is None
                           else np.asarray(self.control_hz, dtype=float).tolist()),
            "meters_per_step": (None if self.meters_per_step is None
                                else np.asarray(self.meters_per_step, dtype=float).tolist()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BackendRun":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        qoi = np.array([np.nan if v is None else v for v in d["qoi"]], dtype=float)
        return cls(
            backend=d["backend"],
            theta=np.asarray(d["theta"], dtype=float),
            qoi=qoi,
            valid=np.asarray(d["valid"], dtype=bool),
            speed_scale=d.get("speed_scale", 1.0),
            control_hz=(None if d.get("control_hz") is None
                        else np.asarray(d["control_hz"], dtype=float)),
            meters_per_step=(None if d.get("meters_per_step") is None
                             else np.asarray(d["meters_per_step"], dtype=float)),
            # Files written before the fields were renamed to English still
            # carry the Italian key. Reading them must keep working: the .npz
            # and .json in results/ are hours of simulator time and cannot be
            # regenerated cheaply.
            threshold=d.get("threshold", d.get("threshold", 0.0)),
            note=d.get("note", d.get("nota", "")),
        )

    @classmethod
    def from_scenario(cls, scenario, theta: np.ndarray, qoi: np.ndarray,
                    speed_scale: float = 1.0, note: str = "") -> "BackendRun":
        """Also pulls the fidelity arrays off the scenario, when it exposes them."""
        return cls(
            backend=getattr(scenario, "name", "sconosciuto"),
            theta=theta, qoi=qoi,
            valid=getattr(scenario, "_valid_mask", None),
            speed_scale=speed_scale,
            control_hz=getattr(scenario, "_last_control_hz", None),
            meters_per_step=getattr(scenario, "_last_meters_per_step", None),
            threshold=scenario.failure_threshold(),
            note=note,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Confronto
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonOutcome:
    """Risultato del confronto, serializzabile."""
    backend: List[str] = field(default_factory=list)
    failure_rates: Dict[str, float] = field(default_factory=dict)
    speed_scale: Dict[str, float] = field(default_factory=dict)
    n_valid: Dict[str, int] = field(default_factory=dict)
    median_control_hz: Dict[str, float] = field(default_factory=dict)
    spearman: Dict[str, float] = field(default_factory=dict)
    spearman_p: Dict[str, float] = field(default_factory=dict)
    shared_design: Dict[str, bool] = field(default_factory=dict)
    jaccard: Dict[str, float] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    n_regions: int = 0
    #: How many regions ONLY that backend found. The most direct form of
    #: "does testing on a single simulator leave something uncovered?".
    exclusive_regions: Dict[str, int] = field(default_factory=dict)
    #: Does the failure cloud have joint structure, or is it indistinguishable
    #: from shuffled axes? Without structure, Jaccard and coverage describe noise.
    structure: Dict[str, object] = field(default_factory=dict)
    eps_dbscan: float = float("nan")
    caveats: List[str] = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


class CrossSimulatorComparison:
    """
    Compares two or more backends on the STRUCTURE of their failures.

        cmp = CrossSimulatorComparison()
        cmp.add(BackendRun.load("results/md.json"))
        cmp.add(BackendRun.load("results/udacity.json"))
        outcome = cmp.compare(lower=b["lower"], upper=b["upper"])
        print(cmp.report())
    """

    #: Tolerance below which two parameter designs count as the same one.
    #: The backends round angles to integers before sending them, so a
    #: difference of a few decimals is expected and harmless.
    DESIGN_TOLERANCE = 0.51

    def __init__(self) -> None:
        self.results: List[BackendRun] = []
        self.outcome: Optional[ComparisonOutcome] = None
        self._regions = None

    def add(self, r: BackendRun) -> "CrossSimulatorComparison":
        if any(x.backend == r.backend for x in self.results):
            raise ValueError(f"backend '{r.backend}' already present: labels "
                             f"must be distinct, otherwise the pairwise "
                             f"comparison becomes ambiguous")
        self.results.append(r)
        return self

    # ── design shared ─────────────────────────────────────────────────────

    def _same_design(self, a: BackendRun, b: BackendRun) -> bool:
        """
        Did the two backends evaluate the same thetas?

        A necessary condition for Spearman: without it, scenario i of A would be
        correlated with scenario i of B, which are DIFFERENT scenarios. The
        resulting number looks like a correlation but is not one.
        """
        if a.theta.shape != b.theta.shape:
            return False
        return bool(np.max(np.abs(a.theta - b.theta)) <= self.DESIGN_TOLERANCE)

    # ── metriche ─────────────────────────────────────────────────────────────

    def compare(self, lower: Sequence[float], upper: Sequence[float], *,
                  param_names: Optional[Sequence[str]] = None,
                  dists=None, tau: Optional[float] = None) -> ComparisonOutcome:
        if len(self.results) < 2:
            raise ValueError("at least two backends are needed to compare")

        outcome = ComparisonOutcome(backend=[r.backend for r in self.results])
        for r in self.results:
            outcome.failure_rates[r.backend] = r.failure_rate
            outcome.speed_scale[r.backend] = r.speed_scale
            outcome.n_valid[r.backend] = r.n_valid
            if r.control_hz is not None and len(r.control_hz):
                hz = np.asarray(r.control_hz, dtype=float)
                hz = hz[~np.isnan(hz)]
                outcome.median_control_hz[r.backend] = (
                    float(np.median(hz)) if len(hz) else float("nan"))

        self._spearman(outcome)
        self._failure_regions(outcome, lower, upper, param_names, dists, tau)
        self._collect_caveats(outcome)

        self.outcome = outcome
        return outcome

    def _spearman(self, outcome: ComparisonOutcome) -> None:
        """
        Rank correlation between the QoIs, over the samples valid in BOTH.

        The most informative metric, because it answers "do the two simulators
        agree on which scenarios are hard?" without depending on the QoI scale
        or on the failure rate.
        """
        from scipy.stats import spearmanr

        for i, a in enumerate(self.results):
            for b in self.results[i + 1:]:
                key = f"{a.backend} ~ {b.backend}"
                shared = self._same_design(a, b)
                outcome.shared_design[key] = shared
                if not shared:
                    # Better NaN than a number that means nothing.
                    outcome.spearman[key] = float("nan")
                    outcome.spearman_p[key] = float("nan")
                    continue

                m = a.valid & b.valid
                if int(m.sum()) < 3:
                    outcome.spearman[key] = float("nan")
                    outcome.spearman_p[key] = float("nan")
                    continue

                # QoI constant on one side: the correlation is undefined (not
                # "zero"). Happens on backends degenerate at 0% or 100%, which
                # are already flagged among the caveats.
                if np.ptp(a.qoi[m]) == 0.0 or np.ptp(b.qoi[m]) == 0.0:
                    outcome.spearman[key] = float("nan")
                    outcome.spearman_p[key] = float("nan")
                    continue

                rho, p = spearmanr(a.qoi[m], b.qoi[m])
                outcome.spearman[key] = float(rho)
                outcome.spearman_p[key] = float(p)

    def _failure_regions(self, outcome, lower, upper, param_names, dists, tau) -> None:
        """
        Delegates to `compare_failure_regions`, which takes `{label: (theta, margins)}`
        and does not know whether the labels are methods or simulators: already generic.

        `eps=None` is not optional: the module documents that a DBSCAN radius
        tuned in 2-D declares every point isolated in 9-D, and the comparison
        would return Jaccard 0 everywhere as an artefact rather than a result.
        """
        from pipeline.failure_regions import compare_failure_regions

        runs = {r.backend: (r.theta, r.qoi) for r in self.results}
        if all(len(r.theta_failed) == 0 for r in self.results):
            outcome.caveats.append(
                "No backend produced any failure: there are no regions to "
                "compare. Raise speed_scale or widen the ODD.")
            return

        try:
            self._regions = compare_failure_regions(
                runs=runs, lower=lower, upper=upper,
                param_names=param_names, dists=dists,
                eps=None, tau=tau, weighting="odd",
            )
        except Exception as exc:                       # noqa: BLE001
            outcome.caveats.append(f"compare_failure_regions ha fallito: {exc}")
            return

        outcome.n_regions = len(getattr(self._regions, "regions", []) or [])

        # `pairwise` is keyed by (a, b) with values {"jaccard", "coverage_a_in_b", ...}.
        for pair, m in (getattr(self._regions, "pairwise", {}) or {}).items():
            key = f"{pair[0]} ~ {pair[1]}" if isinstance(pair, tuple) else str(pair)
            if "jaccard" in m:
                outcome.jaccard[key] = float(m["jaccard"])
            for k in ("coverage_a_in_b", "coverage_b_in_a"):
                if k in m:
                    outcome.coverage[f"{key} [{k}]"] = float(m[k])

        # Regions ONE backend alone found: the most direct form of the question
        # "does testing on a single simulator leave something uncovered?".
        for r in self.results:
            try:
                outcome.exclusive_regions[r.backend] = len(
                    self._regions.exclusive_regions(r.backend))
            except Exception:                          # noqa: BLE001
                pass

        # `structure` says whether the failure cloud has joint structure or is
        # indistinguishable from shuffled axes. Without structure, Jaccard and
        # coverage describe noise.
        st = getattr(self._regions, "structure", None)
        if isinstance(st, dict):
            outcome.structure = {str(k): (float(v) if isinstance(v, (int, float)) else str(v))
                               for k, v in st.items()}
        outcome.eps_dbscan = float(getattr(self._regions, "eps", float("nan")))

    def _collect_caveats(self, outcome: ComparisonOutcome) -> None:
        """
        The traps that make a comparison meaningless without anything flagging
        it. Every one of them was observed in practice.
        """
        # Degenerate rates: no variance left to explain.
        for name, rate in outcome.failure_rates.items():
            if np.isnan(rate):
                outcome.caveats.append(f"{name}: no valid run.")
            elif rate <= 0.0:
                outcome.caveats.append(
                    f"{name}: 0% failures. There is nothing to compare -- "
                    f"raise speed_scale or widen the ODD.")
            elif rate >= 1.0:
                outcome.caveats.append(
                    f"{name}: 100% failures. Check that the ODD is physically "
                    f"drivable at that speed "
                    f"(scripts/diag_odd_feasibility.py) before blaming "
                    f"the controller.")

        # Different operating points: the failure rates are not comparable.
        scale = set(round(s, 4) for s in outcome.speed_scale.values())
        if len(scale) > 1:
            outcome.caveats.append(
                f"different speed_scale across backends {outcome.speed_scale}: "
                f"the failure rates are NOT comparable with each other. What "
                f"stays valid is the comparison on SHAPE (Spearman, regions) "
                f"and any comparison internal to a single backend.")

        # Different designs: Spearman cannot be computed.
        for key, ok in outcome.shared_design.items():
            if not ok:
                outcome.caveats.append(
                    f"{key}: DIFFERENT parameter designs, Spearman cannot be "
                    f"computed. To get it, evaluate the same thetas on both "
                    f"backends (shared design).")

        # Very different control rates: a known confounder.
        hz = outcome.median_control_hz
        if len(hz) >= 2:
            v = [x for x in hz.values() if not np.isnan(x)]
            if v and max(v) / max(min(v), 1e-9) > 1.5:
                outcome.caveats.append(
                    f"Very different control rates {hz}: this is a confounder. "
                    f"The steering rate limiter is in units/s so the controller "
                    f"compensates, but the spatial resolution of the "
                    f"observations still differs. Declare it, or align the rates.")

    # ── report ───────────────────────────────────────────────────────────────

    def report(self) -> str:
        if self.outcome is None:
            raise RuntimeError("call compare() before report()")
        e = self.outcome
        lines = [
            "=" * 72,
            " CROSS-SIMULATOR COMPARISON",
            "=" * 72,
            f" {'backend':<22}{'speed_scale':>12}{'valid':>9}{'failure':>10}{'Hz':>8}",
            "-" * 72,
        ]
        for name in e.backend:
            hz = e.median_control_hz.get(name, float("nan"))
            lines.append(
                f" {name:<22}{e.speed_scale.get(name, float('nan')):>12.4f}"
                f"{e.n_valid.get(name, 0):>9}"
                f"{e.failure_rates.get(name, float('nan')):>9.1%}"
                f"{hz:>8.1f}")

        lines += [
            "-" * 72,
            " !  Failure rates are NOT comparable across backends tuned to",
            "    different speed_scale. They only serve to check that none is",
            "    degenerate (0% or 100%).",
            "",
            " AGREEMENT ON DIFFICULTY (Spearman on the QoI)",
            "-" * 72,
        ]
        if e.spearman:
            for key, rho in e.spearman.items():
                if np.isnan(rho):
                    lines.append(f"   {key:<40} n/a (design not shared)")
                else:
                    p = e.spearman_p.get(key, float("nan"))
                    lines.append(f"   {key:<40} rho={rho:+.3f}  p={p:.4f}")
            lines += [
                "",
                "   high rho -> the simulators agree on WHICH scenarios are",
                "               hard: results from one generalise to the other.",
                "   low rho  -> they order difficulty differently: testing on",
                "               a single simulator leaves something uncovered.",
            ]
        else:
            lines.append("   (no comparable pair)")

        lines += ["", " FAILURE REGIONS", "-" * 72,
                  f"   regions found (union of the backends): {e.n_regions}"]
        for key, j in e.jaccard.items():
            lines.append(f"   Jaccard  {key:<34} {j:.3f}")
        for key, c in e.coverage.items():
            lines.append(f"   coverage {key:<34} {c:.3f}")
        for name, n in e.exclusive_regions.items():
            lines.append(f"   regions found ONLY by {name:<24} {n}")
        if not np.isnan(e.eps_dbscan):
            lines.append(f"   DBSCAN eps estimated from the data: {e.eps_dbscan:.3f}")
        if e.n_regions:
            lines += [
                "",
                "   A low Jaccard WITH many failures on both sides is the most",
                "   interesting result: the two simulators expose DISJOINT",
                "   regions, i.e. testing only one leaves part of the domain",
                "   uncovered.",
            ]

        if e.caveats:
            lines += ["", " CAVEATS", "-" * 72]
            lines += [f"   • {a}" for a in e.caveats]

        lines.append("=" * 72)
        return "\n".join(lines)

    def report_regioni(self) -> str:
        """Detailed `compare_failure_regions` report, when available."""
        if self._regions is None:
            return "(no region analysis available)"
        return self._regions.report()
