"""
Blocco 1d — End-to-end pipeline validation for Emergency Braking.

Runs the full pipeline twice:
  1. Physics simulator  (use_nn=False)  — always works, no model needed
  2. NN simulator       (use_nn=True)   — only if model file exists on disk

For each run, checks:
  - Shapes are correct at every stage
  - failure_rate is in a sensible range
  - rare_failure_idx are a subset of failures
  - POD codes have the right dimensions
  - No NaN / Inf anywhere

Usage (from project root, with venv activated):
    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --n 200   # faster smoke test
"""

import argparse
import os
import sys
import traceback

import numpy as np

# ── Project root must be on sys.path ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.orchestrator import run, PipelineResult
from scenarios.emergency_braking.config import _DEFAULT_MODEL_PATH

# ── Colour codes (skip on Windows without ANSI support) ──────────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
    _ANSI = True
except Exception:
    _ANSI = sys.platform != "win32"

GREEN  = "\033[92m" if _ANSI else ""
RED    = "\033[91m" if _ANSI else ""
YELLOW = "\033[93m" if _ANSI else ""
BOLD   = "\033[1m"  if _ANSI else ""
RESET  = "\033[0m"  if _ANSI else ""


# ──────────────────────────────────────────────────────────────────────────────
# Validation logic
# ──────────────────────────────────────────────────────────────────────────────

def check(condition: bool, msg: str) -> bool:
    """Print ✓ / ✗ and return True if passed."""
    if condition:
        print(f"  {GREEN}✓{RESET}  {msg}")
    else:
        print(f"  {RED}✗  {msg}{RESET}")
    return condition


def validate_result(result: PipelineResult, n_samples: int) -> bool:
    """Run all assertion checks on a PipelineResult. Returns True if all pass."""
    N  = n_samples
    T  = result.trajectories.shape[1]
    D  = result.trajectories.shape[2]
    d  = result.params.shape[1]
    k  = result.pod_codes.shape[1]

    all_ok = True

    # ── Shapes ────────────────────────────────────────────────────────────────
    all_ok &= check(result.params.shape       == (N, d),     f"params shape          : {result.params.shape} == ({N}, {d})")
    all_ok &= check(result.trajectories.shape == (N, T, D),  f"trajectories shape     : {result.trajectories.shape} == ({N}, {T}, {D})")
    all_ok &= check(result.safety_margins.shape == (N,),     f"safety_margins shape   : {result.safety_margins.shape} == ({N},)")
    all_ok &= check(result.failures.shape      == (N,),      f"failures shape         : {result.failures.shape} == ({N},)")
    all_ok &= check(result.pod_codes.shape     == (N, k),    f"pod_codes shape        : {result.pod_codes.shape} == ({N}, {k})")
    all_ok &= check(k >= 1,                                  f"POD modes              : {k} modes (≥ 1)")

    # ── Numerical sanity ──────────────────────────────────────────────────────
    no_nan_traj = not np.any(np.isnan(result.trajectories))
    no_inf_traj = not np.any(np.isinf(result.trajectories))
    no_nan_qoi  = not np.any(np.isnan(result.safety_margins))
    all_ok &= check(no_nan_traj and no_inf_traj, "trajectories          : no NaN / Inf")
    all_ok &= check(no_nan_qoi,                  "safety_margins        : no NaN")

    # ── Semantic sanity ───────────────────────────────────────────────────────
    fr = result.failure_rate
    all_ok &= check(0.0 <= fr <= 1.0,
                    f"failure_rate           : {fr:.1%}  (0–100%)")
    all_ok &= check(0.0 < fr < 1.0,
                    f"failure_rate non-trivial: {fr:.1%}  (neither 0% nor 100%)")

    if len(result.rare_failure_idx) > 0:
        all_rare_are_failures = np.all(result.failures[result.rare_failure_idx] == 1)
        all_ok &= check(all_rare_are_failures,
                        f"rare failures ⊆ failures: {len(result.rare_failure_idx)} rare / {int(result.failures.sum())} total failures")
        rare_margins = result.safety_margins[result.rare_failure_idx]
        sorted_ok = np.all(np.diff(rare_margins) >= 0)
        all_ok &= check(sorted_ok,
                        f"rare failures sorted worst-first: margins {rare_margins[:3].round(2)} …")
    else:
        all_ok &= check(int(result.failures.sum()) == 0,
                        "rare_failure_idx empty  : expected 0 failures")

    # ── Velocities non-negative ───────────────────────────────────────────────
    velocities = result.trajectories[:, :, 1]
    all_ok &= check(np.all(velocities >= -1e-6),
                    "velocities             : all ≥ 0 (no reverse motion)")

    # ── Nominal trajectory shape ──────────────────────────────────────────────
    all_ok &= check(result.nominal_trajectory.shape == (1, T, D),
                    f"nominal_trajectory shape: {result.nominal_trajectory.shape} == (1, {T}, {D})")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Per-run driver
# ──────────────────────────────────────────────────────────────────────────────

def run_validation(label: str, use_nn: bool, n_samples: int, seed: int) -> bool:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}[{label}]{RESET}  n_samples={n_samples}  use_nn={use_nn}")
    print(f"{'─'*60}")

    # Build a fresh scenario with the requested use_nn flag
    from scenarios.emergency_braking.config import EmergencyBrakingScenario
    from scenarios import SCENARIOS

    # Temporarily override the registry entry for this test
    original = SCENARIOS.get("emergency_braking")
    try:
        SCENARIOS["emergency_braking"] = EmergencyBrakingScenario(use_nn=use_nn)

        result = run(
            scenario_name="emergency_braking",
            n_samples=n_samples,
            seed=seed,
            rare_fraction=0.05,
        )

        print()
        print(f"  {BOLD}Summary:{RESET}")
        print(f"    T = {result.trajectories.shape[1]} timesteps, "
              f"D = {result.trajectories.shape[2]} state dims")
        print(f"    failure_rate      : {result.failure_rate:.1%}")
        print(f"    rare_failure_rate : {result.rare_failure_rate:.2%}")
        print(f"    POD modes         : {result.pod_n_modes}")
        print(f"    safety_margins    : min={result.safety_margins.min():.2f}  "
              f"mean={result.safety_margins.mean():.2f}  "
              f"max={result.safety_margins.max():.2f}")
        print()

        ok = validate_result(result, n_samples)

        if ok:
            print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED{RESET}")
        else:
            print(f"\n  {RED}{BOLD}SOME CHECKS FAILED — see ✗ above{RESET}")

        return ok

    except Exception as e:
        print(f"\n  {RED}{BOLD}EXCEPTION:{RESET} {e}")
        traceback.print_exc()
        return False
    finally:
        if original is not None:
            SCENARIOS["emergency_braking"] = original


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end pipeline validation")
    parser.add_argument("--n",    type=int, default=500, help="Number of LHS samples (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-only", action="store_true",
                        help="Skip NN validation even if model exists")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    results = []

    # ── 1. Physics simulator (always) ─────────────────────────────────────────
    ok = run_validation(
        label="PHYSICS SIMULATOR",
        use_nn=False,
        n_samples=args.n,
        seed=args.seed,
    )
    results.append(("Physics", ok))

    # ── 2. NN simulator (only if model trained) ───────────────────────────────
    if not args.physics_only:
        model_exists = os.path.exists(_DEFAULT_MODEL_PATH)
        if model_exists:
            ok_nn = run_validation(
                label="NN SIMULATOR",
                use_nn=True,
                n_samples=args.n,
                seed=args.seed,
            )
            results.append(("NN", ok_nn))
        else:
            print(f"\n{YELLOW}[NN SIMULATOR]{RESET}  Skipped — model not found at:")
            print(f"  {_DEFAULT_MODEL_PATH}")
            print("  Run:  python -m scenarios.emergency_braking.train")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}VALIDATION SUMMARY{RESET}")
    print(f"{'═'*60}")
    all_passed = True
    for name, ok in results:
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {name:20s}  {status}")
        all_passed = all_passed and ok

    print(f"{'═'*60}")
    if all_passed:
        print(f"{GREEN}{BOLD}Overall: ALL PASS{RESET}\n")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}Overall: FAILURES DETECTED{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
