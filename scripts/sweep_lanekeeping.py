#!/usr/bin/env python3
"""
Sweep dell'ODD per lo scenario lane_keeping.

Dopo aver risolto la cadenza di controllo (render Unity leggero -> ~7 Hz), nell'ODD
realistico il modello non fallisce quasi mai (failure rate ~0%). Per un framework che
cerca *rare failures* serve invece un regime in cui i fallimenti tornino, ma stavolta
VERI (debolezze del modello a cadenza corretta), non artefatti del simulatore.

Questo script gira la pipeline su una GRIGLIA di ODD, variando i due assi piu'
significativi -- velocita' massima e angolo massimo di curva -- e tabella per ogni
cella il failure rate (e rare rate, margine mediano, cadenza). Cosi' localizzi la
banda informativa (tipicamente ~20-40%) da usare come ODD di riferimento.

Uso:
    python scripts/sweep_lanekeeping.py
    python scripts/sweep_lanekeeping.py --n 20 --workers 4 \
        --speeds 14,20,25,30 --angles 45,60,75,85
    python scripts/sweep_lanekeeping.py --csv sweep.csv

Prerequisiti: i container del simulatore in esecuzione (come per run_lanekeeping.py).

Tempo: ~ (n. celle) x N x tempo/campione. Con render leggero ~10 s/campione, quindi
una griglia 3x3 a N=15 ~ 22 min. Parti coarse, poi infittisci la banda interessante.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Project root su sys.path (eseguibile da qualsiasi cartella)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ODD di base = preset "realistic" (vedi run_lanekeeping.py). Lo sweep sovrascrive
# solo l'angolo massimo (colonne 0-4) e la velocita' massima (colonna 6, upper).
#   angoli   : [0, angle_max] per segmento
#   min_speed: [5, 8]  m/s (banda fissa, non sovrapposta a max_speed)
#   max_speed: [9, speed_max] m/s
#   seg_len  : [20, 40] m ; map_size: [200, 350] m
BASE_LOWER = [0, 0, 0, 0, 0, 5.0, 9.0, 20.0, 200.0]
BASE_UPPER = [45, 45, 45, 45, 45, 8.0, 14.0, 40.0, 350.0]


def build_bounds(angle_max: float, speed_max: float):
    """Costruisce (lower, upper) per una cella della griglia."""
    lower = list(map(float, BASE_LOWER))
    upper = list(map(float, BASE_UPPER))
    for k in range(5):                 # angoli 1..5
        upper[k] = float(angle_max)
    upper[6] = float(speed_max)        # max_speed (upper)
    # Sicurezza: nessuna dimensione con lower >= upper (scipy.scale lo rifiuta).
    for i in range(len(lower)):
        if upper[i] <= lower[i]:
            upper[i] = lower[i] + 1e-6
    return lower, upper


def parse_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep ODD (velocita' max x angolo max) per lane_keeping.")
    ap.add_argument("--n", type=int, default=15, help="campioni LHS per cella (default 15)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker/container paralleli (default: NUM_WORKERS env o 4)")
    ap.add_argument("--seed", type=int, default=42, help="seed (default 42)")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--speeds", default="14,22,30",
                    help="lista velocita' max (m/s), separate da virgola (default 14,22,30)")
    ap.add_argument("--angles", default="45,65,85",
                    help="lista angoli max (gradi), separate da virgola (default 45,65,85)")
    ap.add_argument("--rare-fraction", type=float, default=0.05)
    ap.add_argument("--csv", default=None, help="salva i risultati anche in questo CSV")
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)
    n_workers = int(os.environ.get("NUM_WORKERS", "4"))

    import numpy as np
    from pipeline.orchestrator import run

    speeds = parse_list(args.speeds)
    angles = parse_list(args.angles)

    line = "=" * 72
    sub = "-" * 72
    print(line)
    print(" LANE_KEEPING — sweep ODD (failure rate su griglia velocita' x angolo)")
    print(line)
    print(f" Campioni/cella : {args.n}    Seed: {args.seed}    Worker: {n_workers}")
    print(f" Velocita' max  : {speeds}  m/s")
    print(f" Angoli max     : {angles}  gradi")
    print(f" Celle totali   : {len(speeds) * len(angles)}   "
          f"(~{args.n * len(speeds) * len(angles)} simulazioni)")
    print(sub)

    # cella -> metriche
    rows = []  # dict per cella
    t0 = time.time()
    for a in angles:
        for s in speeds:
            lower, upper = build_bounds(a, s)
            r = run(args.scenario, n_samples=args.n, seed=args.seed,
                    rare_fraction=args.rare_fraction,
                    param_lower=lower, param_upper=upper, verbose=False)
            m = np.asarray(r.safety_margins, dtype=float)
            valid = (np.asarray(r.valid_mask, dtype=bool)
                     if getattr(r, "valid_mask", None) is not None else ~np.isnan(m))
            mv = m[valid]
            chz = getattr(r, "control_hz", None)
            chz_med = (float(np.nanmedian(np.asarray(chz, float)))
                       if chz is not None and np.isfinite(np.asarray(chz, float)).any() else float("nan"))
            rows.append({
                "angle_max": a, "speed_max": s,
                "fail": r.failure_rate * 100.0,
                "rare": r.rare_failure_rate * 100.0,
                "n_valid": int(valid.sum()),
                "margin_med": float(np.median(mv)) if mv.size else float("nan"),
                "hz_med": chz_med,
            })
            print(f"  angolo_max={a:5.0f}  vel_max={s:5.0f}  ->  "
                  f"fail={r.failure_rate*100:5.1f}%  rare={r.rare_failure_rate*100:4.1f}%  "
                  f"margine_med={np.median(mv) if mv.size else float('nan'):+.2f}  "
                  f"cadenza_med={chz_med:4.1f}Hz  (validi {int(valid.sum())}/{args.n})",
                  flush=True)
    dt = time.time() - t0

    # ── Matrice failure rate: righe = angolo_max, colonne = velocita' max ──
    print(sub)
    print(" MATRICE FAILURE RATE (%)   righe = angolo max | colonne = velocita' max")
    print(sub)
    header = "  ang\\vel |" + "".join(f"{s:8.0f}" for s in speeds)
    print(header)
    print("  " + "-" * (len(header) - 2))
    grid = {(d["angle_max"], d["speed_max"]): d["fail"] for d in rows}
    for a in angles:
        cells = "".join(f"{grid[(a, s)]:8.1f}" for s in speeds)
        print(f"  {a:6.0f}  |{cells}")
    print(sub)

    # ── Lettura: celle piu' vicine alla banda informativa (20-40%) ──
    target_lo, target_hi = 20.0, 40.0
    in_band = [d for d in rows if target_lo <= d["fail"] <= target_hi]
    print(f" Banda informativa cercata: {target_lo:.0f}-{target_hi:.0f}% di fallimenti")
    if in_band:
        print(" Celle in banda (ODD candidati per la ricerca di rare failure):")
        for d in sorted(in_band, key=lambda x: abs(x["fail"] - 30.0)):
            print(f"   angolo_max={d['angle_max']:.0f}  vel_max={d['speed_max']:.0f}"
                  f"  ->  fail={d['fail']:.1f}%  rare={d['rare']:.1f}%")
    else:
        # niente in banda: indica la cella piu' vicina al centro (30%)
        nearest = min(rows, key=lambda x: abs(x["fail"] - 30.0))
        print(" Nessuna cella in banda. La piu' vicina al 30%:")
        print(f"   angolo_max={nearest['angle_max']:.0f}  vel_max={nearest['speed_max']:.0f}"
              f"  ->  fail={nearest['fail']:.1f}%")
        print(" Suggerimento: estendi/infittisci la griglia verso quel valore "
              "(--speeds/--angles) per centrare la banda.")
    print(sub)
    print(f" Tempo totale: {dt:.0f}s")

    # ── CSV opzionale ──
    if args.csv:
        import csv
        path = args.csv if os.path.isabs(args.csv) else os.path.join(_PROJECT_ROOT, args.csv)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["angle_max", "speed_max", "fail", "rare",
                                              "n_valid", "margin_med", "hz_med"])
            w.writeheader()
            for d in rows:
                w.writerow(d)
        print(f" Risultati salvati in: {path}")
    print(line)


if __name__ == "__main__":
    main()
