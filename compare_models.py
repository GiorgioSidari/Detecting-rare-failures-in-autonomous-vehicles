#!/usr/bin/env python3
from __future__ import annotations

"""
Multi-model comparison for the lane-keeping scenario (Step D).

Runs two autopilot models through the identical pipeline and compares *where*
and *how often* each one fails. Because both runs use the same LHS seed and the
same parameter bounds, sample i is the exact same road/speed configuration for
both models — so failures can be compared sample-by-sample, not just in
aggregate.

What it reports
---------------
1. Failure rates            — overall + rare-failure rate per model.
2. Per-sample agreement     — both fail / only-A / only-B / both-safe,
                              Jaccard overlap of the two failure sets,
                              and correlation of the safety margins.
3. Parameter-space zones    — centroids of shared vs. model-specific failures,
                              so you can see which geometry/speed regions are
                              model-specific rather than universally hard.
4. POD failure-cluster overlap — embeds each model's failing trajectories in a
                              shared POD basis and measures how separated /
                              mixed the two failure clusters are.

Usage
-----
Live (needs both Docker containers up — see opensbt-core/docker-compose*.yml):
    python compare_models.py --n-samples 200

Save the raw pipeline output so the analysis can be re-run without re-simulating:
    python compare_models.py --n-samples 200 --save results/

Re-run the analysis offline from saved runs:
    python compare_models.py --load-a results/lane_keeping_chauffeur.npz \\
                             --load-b results/lane_keeping_ch2.npz

Emit machine-readable metrics too:
    python compare_models.py --n-samples 200 --json results/comparison.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# Make the repo root importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder.pod import EmbedderPOD  # noqa: E402

DEFAULT_A = "lane_keeping_chauffeur"
DEFAULT_B = "lane_keeping_ch2"

# Trajectory channels fed to POD — same choice as the orchestrator:
# x position (temporal structure) + XTE + steering; y is dropped (redundant with XTE).
POD_CHANNELS = [0, 2, 3]


# ──────────────────────────────────────────────────────────────────────────────
# Container for the subset of a PipelineResult we need (also what we save/load)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RunData:
    name: str
    params: np.ndarray            # (N, d)
    trajectories: np.ndarray      # (N, T, 4)
    safety_margins: np.ndarray    # (N,)
    failures: np.ndarray          # (N,) binary (0/1)
    rare_failure_idx: np.ndarray  # (r,) indices into N
    failure_rate: float
    rare_failure_rate: float
    param_names: list[str]

    @classmethod
    def from_result(cls, res) -> "RunData":
        return cls(
            name=res.scenario_name,
            params=np.asarray(res.params, dtype=float),
            trajectories=np.asarray(res.trajectories, dtype=np.float32),
            safety_margins=np.asarray(res.safety_margins, dtype=float),
            failures=np.asarray(res.failures, dtype=float),
            rare_failure_idx=np.asarray(res.rare_failure_idx, dtype=int),
            failure_rate=float(res.failure_rate),
            rare_failure_rate=float(res.rare_failure_rate),
            param_names=list(res.param_names),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            name=self.name,
            params=self.params,
            trajectories=self.trajectories,
            safety_margins=self.safety_margins,
            failures=self.failures,
            rare_failure_idx=self.rare_failure_idx,
            failure_rate=self.failure_rate,
            rare_failure_rate=self.rare_failure_rate,
            param_names=np.array(self.param_names, dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "RunData":
        z = np.load(path, allow_pickle=True)
        return cls(
            name=str(z["name"]),
            params=z["params"],
            trajectories=z["trajectories"],
            safety_margins=z["safety_margins"],
            failures=z["failures"],
            rare_failure_idx=z["rare_failure_idx"],
            failure_rate=float(z["failure_rate"]),
            rare_failure_rate=float(z["rare_failure_rate"]),
            param_names=list(z["param_names"]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pad_time(traj: np.ndarray, T: int) -> np.ndarray:
    """Zero-pad a (N, t, D) array along the time axis up to length T."""
    N, t, D = traj.shape
    if t == T:
        return traj
    out = np.zeros((N, T, D), dtype=traj.dtype)
    out[:, :t, :] = traj
    return out


def pod_failure_overlap(a: RunData, b: RunData, variance_threshold: float = 0.99) -> dict:
    """
    Fit a shared POD basis on all trajectories from both models, embed each
    model's *failing* trajectories, and quantify how separated the two failure
    clusters are in that shared latent space.

    Returns metric dict (or a 'note' key when there aren't enough failures).
    """
    fa = a.failures.astype(bool)
    fb = b.failures.astype(bool)
    n_fa, n_fb = int(fa.sum()), int(fb.sum())
    if n_fa < 2 or n_fb < 2:
        return {"note": f"not enough failures for POD overlap (A={n_fa}, B={n_fb}); need >=2 each"}

    # Common time length, then keep the POD channels and stack both models.
    T = max(a.trajectories.shape[1], b.trajectories.shape[1])
    tA = _pad_time(a.trajectories, T)[:, :, POD_CHANNELS]
    tB = _pad_time(b.trajectories, T)[:, :, POD_CHANNELS]

    pod = EmbedderPOD(variance_threshold=variance_threshold)
    pod.fit(np.concatenate([tA, tB], axis=0))

    codes_a = pod.embed(tA[fa])   # (n_fa, k)
    codes_b = pod.embed(tB[fb])   # (n_fb, k)

    # Standardise per-dimension using the pooled spread so no single mode dominates.
    pooled = np.concatenate([codes_a, codes_b], axis=0)
    std = pooled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    za, zb = codes_a / std, codes_b / std

    # (1) Normalised centroid distance: 0 = identical centres; grows with separation.
    centroid_dist = float(np.linalg.norm(za.mean(0) - zb.mean(0)))

    # (2) Mixing index via cross nearest-neighbour labelling.
    #     For every failure point, is its nearest *other* failure point from the
    #     opposite model? ~0.5 => clusters fully interleaved (models fail on the
    #     same kinds of trajectory); ~0.0 => disjoint, model-specific failure shapes.
    pts = np.concatenate([za, zb], axis=0)
    labels = np.concatenate([np.zeros(len(za)), np.ones(len(zb))])
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(axis=1)
    mixing_index = float((labels[nn] != labels).mean())

    return {
        "pod_n_modes": int(pod.nModes),
        "pod_explained_variance": float(pod.explained_variance),
        "n_failures_a": n_fa,
        "n_failures_b": n_fb,
        "centroid_distance_std": centroid_dist,
        "mixing_index": mixing_index,
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compare(a: RunData, b: RunData, variance_threshold: float = 0.99) -> dict:
    """Compute every comparison metric and return them as a dict."""
    aligned = (
        a.params.shape == b.params.shape
        and np.allclose(a.params, b.params, rtol=1e-4, atol=1e-6)
    )

    fa = a.failures.astype(bool)
    fb = b.failures.astype(bool)

    metrics: dict = {
        "scenario_a": a.name,
        "scenario_b": b.name,
        "n_samples": int(len(a.failures)),
        "params_aligned": bool(aligned),
        "failure_rate_a": a.failure_rate,
        "failure_rate_b": b.failure_rate,
        "rare_failure_rate_a": a.rare_failure_rate,
        "rare_failure_rate_b": b.rare_failure_rate,
    }

    if aligned:
        both = fa & fb
        only_a = fa & ~fb
        only_b = ~fa & fb
        neither = ~fa & ~fb
        union = fa | fb
        jaccard = float(both.sum() / union.sum()) if union.sum() else 1.0
        metrics.update(
            both_fail=int(both.sum()),
            only_a_fails=int(only_a.sum()),
            only_b_fails=int(only_b.sum()),
            both_safe=int(neither.sum()),
            jaccard_failure_overlap=jaccard,
            agreement_rate=float((fa == fb).mean()),
            safety_margin_corr=_corr(a.safety_margins, b.safety_margins),
        )
        # Parameter-space centroids of each failure zone.
        def centroid(mask):
            return a.params[mask].mean(axis=0).tolist() if mask.sum() else None
        metrics["param_centroids"] = {
            "shared_failures": centroid(both),
            "only_a": centroid(only_a),
            "only_b": centroid(only_b),
        }
    else:
        metrics["note_alignment"] = (
            "params not aligned (different seed/bounds) — per-sample and "
            "parameter-space comparisons skipped; rates + POD overlap still valid."
        )

    metrics["pod_overlap"] = pod_failure_overlap(a, b, variance_threshold)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def print_report(m: dict, param_names: list[str]) -> None:
    line = "─" * 68
    print("\n" + "═" * 68)
    print(f"  MULTI-MODEL COMPARISON   {m['scenario_a']}   vs   {m['scenario_b']}")
    print("═" * 68)
    print(f"  samples per model : {m['n_samples']}")

    print("\n" + line)
    print("  FAILURE RATES")
    print(line)
    print(f"  {m['scenario_a']:<28}  failure={m['failure_rate_a']:.3f}   rare={m['rare_failure_rate_a']:.3f}")
    print(f"  {m['scenario_b']:<28}  failure={m['failure_rate_b']:.3f}   rare={m['rare_failure_rate_b']:.3f}")

    if m.get("params_aligned"):
        print("\n" + line)
        print("  PER-SAMPLE AGREEMENT  (same road/speed config per index)")
        print(line)
        print(f"  both fail        : {m['both_fail']}")
        print(f"  only {m['scenario_a']} : {m['only_a_fails']}")
        print(f"  only {m['scenario_b']} : {m['only_b_fails']}")
        print(f"  both safe        : {m['both_safe']}")
        print(f"  Jaccard overlap of failure sets : {m['jaccard_failure_overlap']:.3f}")
        print(f"  agreement rate (labels match)   : {m['agreement_rate']:.3f}")
        print(f"  safety-margin correlation       : {m['safety_margin_corr']:.3f}")

        print("\n" + line)
        print("  FAILURE ZONES IN PARAMETER SPACE  (centroid per parameter)")
        print(line)
        pc = m["param_centroids"]
        header = "  zone            " + "".join(f"{n[:11]:>13}" for n in param_names)
        print(header)
        for key, label in [("shared_failures", "shared"),
                           ("only_a", f"only {m['scenario_a']}"),
                           ("only_b", f"only {m['scenario_b']}")]:
            vals = pc.get(key)
            if vals is None:
                print(f"  {label:<15} (none)")
            else:
                print(f"  {label:<15}" + "".join(f"{v:>13.2f}" for v in vals))
    else:
        print("\n  " + m.get("note_alignment", ""))

    print("\n" + line)
    print("  POD FAILURE-CLUSTER OVERLAP")
    print(line)
    po = m["pod_overlap"]
    if "note" in po:
        print(f"  {po['note']}")
    else:
        print(f"  POD modes / explained variance : {po['pod_n_modes']}  /  {po['pod_explained_variance']:.3f}")
        print(f"  failures embedded              : A={po['n_failures_a']}  B={po['n_failures_b']}")
        print(f"  centroid distance (std units)  : {po['centroid_distance_std']:.3f}   (0 = same centre)")
        print(f"  mixing index                   : {po['mixing_index']:.3f}   "
              f"(~0.5 = same failure shapes, ~0 = model-specific)")
    print("═" * 68 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def _run_scenario(name: str, n_samples: int, seed: int) -> RunData:
    from pipeline.orchestrator import run  # imported lazily so --load works without Docker
    print(f"  running pipeline for '{name}' (n={n_samples}, seed={seed}) …", flush=True)
    return RunData.from_result(run(name, n_samples=n_samples, seed=seed))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two lane-keeping autopilots (Step D).")
    p.add_argument("--scenario-a", default=DEFAULT_A)
    p.add_argument("--scenario-b", default=DEFAULT_B)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42,
                   help="Same seed for both runs => identical params => per-sample comparison.")
    p.add_argument("--variance-threshold", type=float, default=0.99)
    p.add_argument("--load-a", help="Load run A from an .npz instead of simulating.")
    p.add_argument("--load-b", help="Load run B from an .npz instead of simulating.")
    p.add_argument("--save", help="Directory to save raw runs (<scenario>.npz) for offline re-analysis.")
    p.add_argument("--json", dest="json_out", help="Write comparison metrics to this JSON path.")
    args = p.parse_args(argv)

    a = RunData.load(args.load_a) if args.load_a else _run_scenario(args.scenario_a, args.n_samples, args.seed)
    b = RunData.load(args.load_b) if args.load_b else _run_scenario(args.scenario_b, args.n_samples, args.seed)

    if args.save:
        a.save(os.path.join(args.save, f"{a.name}.npz"))
        b.save(os.path.join(args.save, f"{b.name}.npz"))
        print(f"  saved raw runs to {args.save}/")

    metrics = compare(a, b, variance_threshold=args.variance_threshold)
    print_report(metrics, a.param_names)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  wrote metrics to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
