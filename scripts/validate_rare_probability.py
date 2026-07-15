#!/usr/bin/env python3
"""
Validazione (M3) della stima di probabilita' di fallimento del livello intermedio.

NON valida il simulatore (quello e' un altro discorso, gia' affrontato con la
cadenza di controllo): valida la *macchina statistica* introdotta nella pipeline —
campionamento distribution-aware via ppf + stima P(fallimento) + intervallo di Wilson.

Idea: si sostituisce il simulatore con una regola di fallimento SINTETICA e nota,
definita sugli stessi 9 parametri del lane_keeping. Sotto le distribuzioni operative
reali (`LaneKeepingScenario.param_distributions`) la probabilita' vera P_true di quella
regola e' calcolabile con un Monte Carlo indipendente ad altissimo N (ground truth).
Poi si verifica che l'estimatore usato dalla pipeline (LHS + ppf, come in
`pipeline.orchestrator.run`) a N moderato:
  1. CONVERGA a P_true (bias ~0),
  2. abbia un Wilson CI che COPRE P_true circa nel 95% dei casi (copertura calibrata;
     con LHS la varianza e' <= binomiale, quindi il CamI e' semmai conservativo).

Uso:
    python scripts/validate_rare_probability.py
    python scripts/validate_rare_probability.py --ns 50,200,1000 --reps 300

Non richiede Docker: e' tutto in-process.
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Riusa il CODICE REALE della pipeline: le distribuzioni dello scenario e il Wilson CI.
from scenarios.lane_keeping.config import LaneKeepingScenario
from pipeline.orchestrator import _wilson_ci


def synthetic_fail(params: np.ndarray, angle_thr: float, speed_thr: float) -> np.ndarray:
    """
    Regola di fallimento SINTETICA e nota sui 9 parametri (stessa forma del vero
    scenario: peggiora con curve strette E velocita' alta). Serve solo a fornire un
    ground truth: fallisce se max(5 angoli) > angle_thr E max_speed > speed_thr.
    Alzando le soglie si rende l'evento piu' RARO (default: regime raro, P ~ pochi %),
    che e' il caso interessante per validare il Wilson CI e motivare M4.
    """
    max_angle = params[:, :5].max(axis=1)
    max_speed = params[:, 6]
    return (max_angle > angle_thr) & (max_speed > speed_thr)


def ppf_sample(dists, unit):
    """Trasforma i campioni unitari con la ppf, ESATTAMENTE come fa l'orchestrator."""
    out = np.empty_like(unit)
    for j, d in enumerate(dists):
        out[:, j] = d.ppf(unit[:, j])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validazione M3 della stima di P(fallimento).")
    ap.add_argument("--ns", default="50,200,1000",
                    help="valori di N da testare, separati da virgola (default 50,200,1000)")
    ap.add_argument("--reps", type=int, default=300,
                    help="ripetizioni (seed) per stimare bias e copertura (default 300)")
    ap.add_argument("--truth-n", type=int, default=1_000_000,
                    help="campioni del Monte Carlo di ground truth (default 1e6)")
    ap.add_argument("--angle-thr", type=float, default=70.0,
                    help="soglia angolo della regola sintetica (default 70: regime raro)")
    ap.add_argument("--speed-thr", type=float, default=26.0,
                    help="soglia velocita' della regola sintetica (default 26: regime raro)")
    args = ap.parse_args()

    Ns = [int(x) for x in args.ns.split(",") if x.strip()]

    sc = LaneKeepingScenario()
    b = sc.param_bounds()
    lower, upper = b["lower"], b["upper"]
    dists = sc.param_distributions(lower, upper)   # distribuzioni REALI dello scenario
    d = len(lower)

    line = "=" * 68
    print(line)
    print(" VALIDAZIONE M3 — stima di P(fallimento) (distribuzioni reali dello scenario)")
    print(line)

    # ── Ground truth: MC indipendente ad altissimo N dalle distribuzioni ──
    rng = np.random.default_rng(0)
    truth = np.empty((args.truth_n, d))
    for j, dist in enumerate(dists):
        truth[:, j] = dist.rvs(size=args.truth_n, random_state=rng)
    fails_truth = synthetic_fail(truth, args.angle_thr, args.speed_thr)
    p_true = float(fails_truth.mean())
    se_truth = float(np.sqrt(p_true * (1 - p_true) / args.truth_n))
    print(f" Regola sintetica: max(angoli)>{args.angle_thr:g} AND max_speed>{args.speed_thr:g}")
    print(f" P_true (MC {args.truth_n:,} campioni) = {p_true*100:.3f}%  (±{se_truth*100:.3f}%)")
    print("-" * 68)
    print(f" {'N':>6} | {'stima media':>12} | {'bias':>8} | {'copertura CI95':>14} | {'ampiezza CI':>11}")
    print("-" * 68)

    all_ok = True
    for N in Ns:
        ests = np.empty(args.reps)
        covered = 0
        widths = np.empty(args.reps)
        for r in range(args.reps):
            unit = LatinHypercube(d=d, seed=1000 + r).random(n=N)   # come l'orchestrator
            params = ppf_sample(dists, unit)
            fails = synthetic_fail(params, args.angle_thr, args.speed_thr)
            k = int(fails.sum())
            ests[r] = k / N
            lo, hi = _wilson_ci(k, N)          # il Wilson CI REALE della pipeline
            widths[r] = hi - lo
            if lo <= p_true <= hi:
                covered += 1
        mean_est = float(ests.mean())
        bias = mean_est - p_true
        coverage = covered / args.reps
        # Criteri: bias piccolo (scende con N) e copertura >= ~0.93 (Wilson e' esatto/
        # conservativo; con LHS la varianza e' <= binomiale -> copertura non inferiore).
        ok_bias = abs(bias) < max(0.02, 2 * se_truth + 1.0 / N)
        ok_cov = coverage >= 0.90
        all_ok = all_ok and ok_bias and ok_cov
        flag = "OK" if (ok_bias and ok_cov) else "!!"
        print(f" {N:>6} | {mean_est*100:>11.2f}% | {bias*100:>+7.2f}% | "
              f"{coverage*100:>13.1f}% | {widths.mean()*100:>10.2f}%  {flag}")

    print("-" * 68)
    print(" Lettura: stima media ~ P_true (bias ~0, cala con N) e copertura ~95% =")
    print("          l'estimatore e il CI sono corretti. La convergenza mostra anche")
    print("          perche' a N piccolo lo 0/20 sperimentale NON e' 'P=0' ma solo un")
    print("          tetto largo: la stessa macchina, con piu' campioni, stringe il CI.")
    print(line)
    print(" ESITO:", "TUTTO OK" if all_ok else "ATTENZIONE — qualche criterio non soddisfatto")
    print(line)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
