"""
Batch run + control-rate sweep for the MetaDrive lane-keeping scenario.

Two things in one:

1) BATCH — runs the FULL pipeline through the orchestrator
   (LHS -> MetaDrive -> QoI -> POD -> rare-failure detection) and reports the
   failure rate, P(failure) with a Wilson CI, and the worst scenarios. This
   validates the end-to-end integration of `lane_keeping_md` with the existing
   pipeline.

2) SWEEP — varies the control rate (via decision_repeat) and tabulates the mean
   meters-per-step against the failure rate. This reproduces the meters-per-steer
   ENVELOPE, now with the control rate as an EXACT swept variable — impossible on
   the Unity/Docker backend, where the rate was an emergent property of rendering.

Run:
    python scripts/run_lanekeeping_md.py                 # defaults
    python scripts/run_lanekeeping_md.py --n 40 --sweep-n 25
"""
from __future__ import annotations

import argparse
import numpy as np

from scenarios import SCENARIOS
from scenarios.lane_keeping_md.driver import PurePursuitDriver
from pipeline.orchestrator import run


def _apply_driver(obs_latency: int, steer_noise: float, seed: int,
                  obs_lag: float = 0.0, quiet: bool = False) -> None:
    """Swap the registry scenario's driver factory to a (possibly degraded) one."""
    sc = SCENARIOS["lane_keeping_md"]
    sc._driver_factory = lambda: PurePursuitDriver(
        obs_latency=obs_latency, obs_lag=obs_lag, steer_noise=steer_noise, seed=seed
    )
    if (obs_latency or steer_noise or obs_lag) and not quiet:
        print(f"[driver] degraded: obs_latency={obs_latency} steps, "
              f"obs_lag={obs_lag}, steer_noise={steer_noise}")


def lag_scan(n: int, seed: int, steer_noise: float, lags) -> None:
    """Scan the continuous observation sluggishness (obs_lag) at a fixed control
    rate to find a value that yields a MID failure band (neither 0% nor 100%) —
    the regime where the failure boundary depends on the scenario, i.e. what the
    Part-B active classifier needs."""
    from collections import Counter
    print("=" * 74)
    print(f" OBS-LAG SCAN  (fixed rate, n={n})  -> find a mid failure band for Part B")
    print("=" * 74)
    print(f" {'obs_lag':>8} | {'failure %':>9} | outcomes")
    print(" " + "-" * 60)
    for lag in lags:
        _apply_driver(0, steer_noise, seed, obs_lag=lag, quiet=True)
        r = run("lane_keeping_md", n_samples=n, sampling="realistic", seed=seed)
        sc = SCENARIOS["lane_keeping_md"]
        oc = dict(Counter(sc._md_outcome[-r.n_samples:])) if sc._md_outcome else {}
        print(f" {lag:>8.2f} | {r.failure_rate*100:>8.1f}% | {oc}")
    print("\n Pick the obs_lag with an intermediate failure % (~30-70%): that is where")
    print(" the boundary depends on the parameters and Part B has something to learn.")


def _fmt_ci(ci):
    if not ci:
        return "n/a"
    lo, hi = ci
    return f"[{lo*100:.1f}, {hi*100:.1f}]%"


def batch(n: int, sampling: str, seed: int) -> None:
    print("=" * 74)
    print(f" BATCH  scenario=lane_keeping_md  n={n}  sampling={sampling}  seed={seed}")
    print("=" * 74)
    r = run("lane_keeping_md", n_samples=n, sampling=sampling, seed=seed, verbose=True)

    valid = r.valid_mask if r.valid_mask is not None else np.ones(r.n_samples, bool)
    print(f"\n valid runs        : {int(valid.sum())}/{r.n_samples}"
          f"   (invalid={r.n_invalid}, degenerate={r.n_degenerate})")
    print(f" failure rate      : {r.failure_rate*100:.1f}%")
    print(f" P(failure)        : {r.failure_probability*100:.2f}%   CI95 {_fmt_ci(r.failure_probability_ci)}")
    sc = SCENARIOS["lane_keeping_md"]
    if getattr(sc, "_md_outcome", None):
        from collections import Counter
        oc = Counter(sc._md_outcome[-r.n_samples:])
        print(f" outcomes          : {dict(oc)}")
    print(f" POD modes         : {r.pod_n_modes}")
    if r.meters_per_step is not None:
        mps = np.asarray(r.meters_per_step, float)
        print(f" meters_per_step   : mean={np.nanmean(mps):.3f}  "
              f"min={np.nanmin(mps):.3f}  max={np.nanmax(mps):.3f}")

    # Worst scenarios by safety margin (most severe failures first).
    order = np.argsort(np.where(np.isnan(r.safety_margins), np.inf, r.safety_margins))
    print("\n worst scenarios (safety margin, lower = worse):")
    names = r.param_names
    for idx in order[:5]:
        m = r.safety_margins[idx]
        row = r.params[idx]
        ang = ",".join(f"{int(round(a)):2d}" for a in row[:5])
        print(f"   margin={m:+.3f}  angles=[{ang}]  "
              f"minV={row[5]:.1f} maxV={row[6]:.1f}  steps={int(r.trajectories[idx].any(axis=1).sum())}")


def sweep(n: int, seed: int, decision_repeats) -> None:
    print("\n" + "=" * 74)
    print(" CONTROL-RATE SWEEP  (envelope: meters_per_step vs failure rate)")
    print("=" * 74)
    sc = SCENARIOS["lane_keeping_md"]
    dt = sc.physics_world_step_size
    original = sc.decision_repeat
    print(f" {'decision_repeat':>16} | {'control_hz':>10} | {'mean m/step':>11} | {'failure %':>9}")
    print(" " + "-" * 60)
    try:
        for dr in decision_repeats:
            sc.decision_repeat = int(dr)
            r = run("lane_keeping_md", n_samples=n, sampling="realistic", seed=seed)
            hz = 1.0 / (dr * dt)
            mps = np.nanmean(np.asarray(r.meters_per_step, float)) if r.meters_per_step is not None else float("nan")
            print(f" {dr:>16} | {hz:>9.1f}  | {mps:>11.3f} | {r.failure_rate*100:>8.1f}%")
    finally:
        sc.decision_repeat = original  # restore

    print("\n Note: lower control_hz -> more metres per decision -> more failures.")
    print("       If the 'failure %' column stays ~0 everywhere, the driver is too")
    print("       robust: a harder ODD or a weaker controller is needed before Part B")
    print("       (the classifier needs signal).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="batch sample count")
    ap.add_argument("--sampling", choices=["uniform", "realistic"], default="realistic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-n", type=int, default=20, help="samples per sweep point")
    ap.add_argument("--no-sweep", action="store_true", help="skip the control-rate sweep")
    ap.add_argument("--obs-latency", type=int, default=0,
                    help="driver: act on the state from N steps ago (surfaces failures)")
    ap.add_argument("--obs-lag", type=float, default=0.0,
                    help="driver: continuous observation sluggishness EMA in [0,1)")
    ap.add_argument("--steer-noise", type=float, default=0.0,
                    help="driver: std of gaussian steering noise")
    ap.add_argument("--lag-scan", action="store_true",
                    help="scan obs_lag at fixed rate to find a mid failure band, then stop")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="scale the target cruising speed (lower = more control margin)")
    args = ap.parse_args()

    SCENARIOS["lane_keeping_md"].speed_scale = args.speed_scale
    if args.speed_scale != 1.0:
        print(f"[scenario] speed_scale={args.speed_scale}")

    if args.lag_scan:
        lag_scan(args.n, args.seed, args.steer_noise,
                 lags=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8])
        return

    _apply_driver(args.obs_latency, args.steer_noise, seed=args.seed, obs_lag=args.obs_lag)
    batch(args.n, args.sampling, args.seed)
    if not args.no_sweep:
        # 3->16.7Hz, 5->10Hz, 8->6.25Hz, 12->4.17Hz, 20->2.5Hz
        sweep(args.sweep_n, seed=args.seed + 1, decision_repeats=[3, 5, 8, 12, 20])


if __name__ == "__main__":
    main()
