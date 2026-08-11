#!/usr/bin/env python3
"""
Runner con output leggibile per lo scenario cut_in — mirror di
scripts/run_lanekeeping.py e scripts/run_emergency_braking.py. Di default usa
la modalita' video-CNN su CARLA (Fase 3 del piano); con --scenario cut_in
passi alla fisica ideale.

Uso:
    python scripts/run_cut_in.py                       # 20 campioni, video-CNN (CARLA)
    python scripts/run_cut_in.py --n 50
    python scripts/run_cut_in.py --scenario cut_in      # fisica ideale
    python scripts/run_cut_in.py --n 30 --seed 7 --quiet --trace-worst

Prerequisiti (modalita' video-CNN, default):
    opensbt-core/CarlaSimulator_0916/CarlaUE4.exe -RenderOffScreen -nosound
    cd opensbt-core && uvicorn Simulator.cut_in.SimulatorServer:app --port 8200
Vedi README.md, sezione "Cut-In — how to run each mode".
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LANE_WIDTH = 3.5   # metres — matches scenarios/cut_in/simulator.py's LANE_WIDTH


def main() -> None:
    ap = argparse.ArgumentParser(description="Esegue la pipeline cut_in con output chiaro.")
    ap.add_argument("--n", type=int, default=20, help="numero di campioni LHS (default 20)")
    ap.add_argument("--workers", type=int, default=None,
                    help="numero di istanze CARLA parallele (default: CI_NUM_WORKERS env o 1)")
    ap.add_argument("--seed", type=int, default=42, help="seed (default 42)")
    ap.add_argument("--scenario", default="cut_in_carla",
                    help="'cut_in_carla' (default, video-CNN) | 'cut_in' (fisica ideale)")
    ap.add_argument("--rare-fraction", type=float, default=0.05,
                    help="frazione bottom-k trattata come 'rara' (default 0.05)")
    ap.add_argument("--quiet", action="store_true", help="non mostrare il progresso per-simulazione")
    ap.add_argument("--trace-worst", action="store_true",
                    help="stampa la traccia passo-passo (gap longitudinale/laterale) del run peggiore")
    ap.add_argument("--sampling", choices=["uniform", "realistic"], default="uniform",
                    help="'realistic' richiede che lo scenario esponga param_distributions() "
                         "(non ancora implementato per cut_in: ricade su 'uniform')")
    args = ap.parse_args()

    # IMPORTANTE: va impostato PRIMA di importare scenarios/pipeline, perche' il
    # pool viene costruito all'import di scenarios/.
    if args.workers is not None:
        os.environ["CI_NUM_WORKERS"] = str(args.workers)
    n_workers = int(os.environ.get("CI_NUM_WORKERS", "1"))

    import numpy as np
    from pipeline.orchestrator import run

    is_carla = args.scenario.endswith("_carla")

    line = "=" * 64
    sub = "-" * 64
    print(line)
    print(f" {args.scenario.upper()} — esecuzione pipeline")
    print(line)
    print(f" Campioni (N)   : {args.n}")
    print(f" Seed           : {args.seed}")
    print(f" Campionamento  : {args.sampling}")
    if is_carla:
        print(f" Worker pool    : {n_workers}  (CI_NUM_WORKERS={n_workers})")
    print(sub)
    if not args.quiet:
        print(" Progresso simulazioni (una riga per run completato):")

    t0 = time.time()
    r = run(args.scenario, n_samples=args.n, seed=args.seed,
            rare_fraction=args.rare_fraction, sampling=args.sampling,
            verbose=not args.quiet)
    dt = time.time() - t0

    m = np.asarray(r.safety_margins, dtype=float)
    N = args.n
    valid = (np.asarray(r.valid_mask, dtype=bool)
             if getattr(r, "valid_mask", None) is not None else ~np.isnan(m))
    n_valid = int(valid.sum())
    n_invalid = int(getattr(r, "n_invalid", N - n_valid))
    order = np.argsort(np.where(valid, m, np.inf))[:n_valid]   # solo validi, worst->best
    mv = m[valid]
    n_rare = len(r.rare_failure_idx)

    print(sub)
    print(" RISULTATI")
    print(sub)
    print(f" Tempo totale        : {dt:6.1f}s   (~{dt / max(N, 1):.1f}s per campione)")
    inv_note = f"   ({n_invalid} non validi esclusi)" if n_invalid else ""
    print(f" Scenari validi      : {n_valid}/{N}{inv_note}")
    if n_valid == 0:
        print(" Nessuno scenario valido: impossibile calcolare i tassi.")
        print(line)
        return

    n_fail = int((mv < 0).sum())
    p_fail = float(getattr(r, "failure_probability", r.failure_rate))
    ci = getattr(r, "failure_probability_ci", None)
    ci_txt = f"  CI95% [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]%" if ci is not None else ""
    print(f" P(fallimento)       : {p_fail * 100:5.1f}%   ({n_fail}/{n_valid} falliti){ci_txt}")
    print(f" Worst-case (severita'): bottom-{args.rare_fraction * 100:.0f}% = {n_rare}/{n_valid} scenari"
          f"   | margine: min {mv.min():+.3f} | mediana {np.median(mv):+.3f} | max {mv.max():+.3f}")
    print(f" Modi POD             : {r.pod_n_modes}")
    print(sub)

    print(" Margini ordinati (peggiore -> migliore, solo scenari validi):")
    vals = ", ".join(f"{m[i]:+.2f}" for i in order)
    print(f"   [{vals}]")
    print("   (valori < 0 = collisione; piu' negativo = peggiore)")
    print(sub)

    survival = None
    try:
        traj = np.asarray(r.trajectories)
        survival = (np.abs(traj).sum(axis=2) > 0).sum(axis=1)
    except Exception:
        survival = None

    def fmt_params(row):
        row = np.asarray(row, dtype=float)
        return (f"ego={row[0]:.1f}m/s  cutter={row[1]:.1f}m/s  "
                f"gap_laterale0={row[2]:.1f}m  ritardo={row[3]:.2f}s")

    rare = list(r.rare_failure_idx)
    if rare:
        print(" SCENARIO RARO (il piu' critico):" if len(rare) == 1
              else f" SCENARI RARI (i {len(rare)} piu' critici):")
        rows = list(rare)
    else:
        print(" Nessun rare failure isolato. I 5 scenari peggiori:")
        rows = list(order[:5])
    for i in rows:
        extra = f"  step={int(survival[i])}" if survival is not None else ""
        print(f"   #{int(i):<3d} margin={m[i]:+.3f}{extra}   {fmt_params(r.params[i])}")
    print(line)

    # Spiegazione data-driven: quando il cutter e' entrato in corsia e quanto
    # si e' avvicinato il gap longitudinale in quel momento.
    def _explain(i):
        traj = np.asarray(r.trajectories[i], dtype=float)
        L = int(survival[i]) if survival is not None else traj.shape[0]
        L = max(1, min(L, traj.shape[0]))
        long_gap, lat_gap = traj[:L, 0], traj[:L, 1]
        row = np.asarray(r.params[i], dtype=float)
        in_lane = lat_gap < LANE_WIDTH / 2.0
        if in_lane.any():
            masked = np.where(in_lane, long_gap, np.inf)
            idx_min = int(masked.argmin())
            min_gap = float(masked[idx_min])
            merge_frac = idx_min / max(L - 1, 1)
            detail = f"gap minimo {min_gap:.1f}m al {merge_frac * 100:.0f}% della corsa (step {idx_min})"
        else:
            detail = "il cutter non e' mai entrato del tutto in corsia in questo run"
        mode = f"collisione: {detail}" if m[i] < 0 else f"gap mantenuto: {detail}"
        ctx = (f"; ego={row[0]:.0f}m/s cutter={row[1]:.0f}m/s "
               f"gap_laterale0={row[2]:.1f}m ritardo={row[3]:.2f}s")
        return f"{mode}{ctx}"

    print(" PERCHE' SONO FALLITI (lettura dai dati):")
    for i in rows:
        print(f"   #{int(i):<3d} {_explain(i)}")
    print(line)

    if args.trace_worst:
        wi = int(order[0])
        wtraj = np.asarray(r.trajectories[wi])
        L = int(survival[wi]) if survival is not None else wtraj.shape[0]
        L = min(L, wtraj.shape[0])
        print(f" TRACCIA run peggiore  #{wi}  (margin {m[wi]:+.3f}, {L} step)")
        print("   step | gap longitudinale | gap laterale")
        for t in range(L):
            print(f"   {t:4d} | {wtraj[t, 0]:16.3f}m | {wtraj[t, 1]:11.3f}m")
        print("   (gap laterale -> 0 = il cutter e' entrato in corsia; "
              "gap longitudinale piccolo in quel momento = collisione)")
        print(sub)

    fr = r.failure_rate
    if fr >= 0.8:
        hint = ("Tasso di fallimento molto alto: i fallimenti NON sono rari — "
                "il modello e' fragile su questo spazio di parametri.")
    elif fr <= 0.2:
        hint = "Tasso basso: buon regime per isolare rare failure significativi."
    else:
        hint = "Tasso intermedio: ragionevole per confrontare scenari sicuri e falliti."
    print(" Lettura:", hint)
    print(line)


if __name__ == "__main__":
    main()
