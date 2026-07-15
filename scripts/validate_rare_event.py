#!/usr/bin/env python3
"""
Validazione (M4) dell'estimatore Cross-Entropy di pipeline/rare_event.py.

Come per M3, si sostituisce il simulatore con un MARGINE sintetico e noto (cosi' la
probabilita' vera P_true e' calcolabile a forza bruta) e si verifica che la
Cross-Entropy:
  1. dia una stima non distorta di P_true (copertura del CI corretta),
  2. ci arrivi con MOLTI MENO run del Monte Carlo ingenuo a parita' di budget
     (riduzione di varianza) — il punto centrale del metodo.

Il margine sintetico e' UNIMODALE (fallisce se media(angoli) e max_speed sono entrambi
alti): e' la classe di regioni che la CE con proposta unimodale gestisce bene. Con
regioni multimodali la CE sottostima (limite documentato in rare_event.py).

Uso:
    python scripts/validate_rare_event.py
    python scripts/validate_rare_event.py --reps 12 --angle-thr 53 --speed-thr 28.5

Non richiede Docker.
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scenarios.lane_keeping.config import LaneKeepingScenario
from pipeline.rare_event import estimate_failure_probability, _sample_product


def main() -> None:
    ap = argparse.ArgumentParser(description="Validazione M4 (Cross-Entropy vs brute-force).")
    ap.add_argument("--reps", type=int, default=10, help="ripetizioni (seed) della CE (default 10)")
    ap.add_argument("--angle-thr", type=float, default=53.0,
                    help="soglia media-angoli della regola sintetica (default 53: P~1e-4)")
    ap.add_argument("--speed-thr", type=float, default=28.5,
                    help="soglia max_speed della regola sintetica (default 28.5)")
    ap.add_argument("--truth-n", type=int, default=3_000_000,
                    help="campioni del Monte Carlo di ground truth (default 3e6)")
    # budget CE (nella validazione i run sono gratis -> valori generosi)
    ap.add_argument("--spi", type=int, default=500, help="samples_per_iter della CE")
    ap.add_argument("--final", type=int, default=8000, help="campioni della stima finale")
    args = ap.parse_args()

    at, st = args.angle_thr, args.speed_thr

    def margin_fn(X):
        # UNIMODALE: margine < 0  <=>  media(angoli) > at  AND  max_speed > st
        return np.maximum(at - X[:, :5].mean(axis=1), st - X[:, 6])

    sc = LaneKeepingScenario()
    b = sc.param_bounds()
    lower, upper = b["lower"], b["upper"]
    f_dists = sc.param_distributions(lower, upper)
    d = len(lower)

    line = "=" * 70
    print(line)
    print(" VALIDAZIONE M4 — Cross-Entropy vs Monte Carlo (distribuzioni reali)")
    print(line)

    # ── Ground truth ──
    rng = np.random.default_rng(0)
    T = _sample_product(f_dists, args.truth_n, rng, d)
    p_true = float((margin_fn(T) < 0).mean())
    print(f" Regola sintetica (unimodale): media(angoli)>{at:g} AND max_speed>{st:g}")
    print(f" P_true (MC {args.truth_n:,}) = {p_true*100:.4f}%  ({p_true:.2e})")
    print("-" * 70)

    # ── Cross-Entropy su piu' seed ──
    ce_p, ce_eval, covered = [], [], 0
    for s in range(args.reps):
        res = estimate_failure_probability(
            margin_fn, f_dists, lower, upper,
            samples_per_iter=args.spi, final_samples=args.final, seed=s)
        ce_p.append(res.p_fail)
        ce_eval.append(res.n_evaluations)
        cov = res.ci[0] <= p_true <= res.ci[1]
        covered += cov
        print(f"  CE seed {s:2d}: p={res.p_fail*100:.4f}%  "
              f"CI[{res.ci[0]*100:.4f}, {res.ci[1]*100:.4f}]  "
              f"eval={res.n_evaluations}  iter={res.iterations}  cover={cov}")
    ce_p = np.array(ce_p)
    mean_eval = int(np.mean(ce_eval))
    # errore relativo empirico (std delle stime / P_true) = qualita' dell'estimatore
    ce_relerr = float(ce_p.std() / p_true) if p_true > 0 else float("inf")

    # ── Monte Carlo ingenuo con lo STESSO budget ──
    B = mean_eval
    mc_fails = (margin_fn(_sample_product(f_dists, B, np.random.default_rng(12345), d)) < 0)
    k_mc = int(mc_fails.sum())
    p_mc = k_mc / B
    mc_relerr = float(np.sqrt(p_mc * (1 - p_mc) / B) / p_mc) if p_mc > 0 else float("inf")

    print("-" * 70)
    print(f" Cross-Entropy : media stime = {ce_p.mean()*100:.4f}%   "
          f"copertura CI = {covered}/{args.reps}   err.rel. = {ce_relerr*100:.1f}%")
    print(f" MC ingenuo    : {k_mc} fallimenti su {B} run   stima = {p_mc*100:.4f}%   "
          f"err.rel. = {mc_relerr*100:.1f}%")
    print("-" * 70)
    speedup = (mc_relerr / ce_relerr) ** 2 if ce_relerr > 0 else float("inf")
    print(f" A parita' di budget ({B} run), la CE ha errore relativo ~{mc_relerr/max(ce_relerr,1e-9):.1f}x")
    print(f" piu' piccolo -> per pareggiarla il MC ingenuo servirebbe ~{speedup:.0f}x piu' run.")
    print("-" * 70)

    ok_cov = covered >= int(0.80 * args.reps)         # copertura ~>=80% (bootstrap)
    ok_bias = abs(ce_p.mean() - p_true) < 0.5 * p_true # bias medio < 50% (regione unimodale)
    ok_var = ce_relerr < mc_relerr                     # CE piu' precisa del MC
    all_ok = ok_cov and ok_bias and ok_var
    print(" ESITO:", "TUTTO OK" if all_ok else "ATTENZIONE — rivedere parametri/regime")
    print(line)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
