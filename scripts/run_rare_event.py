#!/usr/bin/env python3
"""
Runner Cross-Entropy (M4) sul simulatore vero (Docker).

Stima P(fallimento) sotto la distribuzione operativa con la Cross-Entropy: campiona in
modo ADATTIVO verso la regione di fallimento invece di contare su campionamento cieco.
Utile quando i fallimenti sono rari (dove il Monte Carlo ingenuo sarebbe inefficiente).

Prerequisiti: come run_lanekeeping.py, i container del simulatore in esecuzione.

Uso:
    python scripts/run_rare_event.py                          # ODD full, budget contenuto
    python scripts/run_rare_event.py --preset full --spi 60 --final 300
    python scripts/run_rare_event.py --workers 4 --seed 1 --quiet

NOTA sul budget: ogni iterazione esegue `--spi` simulazioni + `--final` alla fine.
A ~10 s/run un budget di ~500-800 run richiede indicativamente 1-2 ore. Parti piccolo.

NOTA sull'ODD: sul preset 'realistic' a cadenza corretta i fallimenti sono ~assenti,
quindi la CE stimerebbe P~0 (risultato valido: "nessun fallimento in questo ODD").
Per vedere la CE al lavoro serve un ODD dove i fallimenti esistono ma sono rari:
il preset 'full' (angoli fino a 85, velocita' fino a 30) e' il default per questo.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Bounds del preset "realistic" (identici a run_lanekeeping.py).
REALISTIC_LOWER = [0,  0,  0,  0,  0,   5.0,  9.0, 20.0, 200.0]
REALISTIC_UPPER = [45, 45, 45, 45, 45,  8.0, 14.0, 40.0, 350.0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-Entropy (M4) sul simulatore lane_keeping.")
    ap.add_argument("--scenario", default="lane_keeping")
    ap.add_argument("--preset", choices=["full", "realistic"], default="full",
                    help="'full' = ODD ampio (default: i fallimenti esistono ma rari); "
                         "'realistic' = ODD ristretto (a cadenza corretta ~nessun fallimento)")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="forza il tetto di max_speed (m/s): restringe l'ODD verso il regime raro")
    ap.add_argument("--max-angle", type=float, default=None,
                    help="forza il tetto degli angoli (gradi): restringe l'ODD verso il regime raro")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker/container paralleli (default: NUM_WORKERS env o 4)")
    ap.add_argument("--spi", type=int, default=60, help="samples_per_iter della CE (default 60)")
    ap.add_argument("--final", type=int, default=300, help="campioni della stima finale (default 300)")
    ap.add_argument("--max-iter", type=int, default=10, help="max iterazioni CE (default 10)")
    ap.add_argument("--rho", type=float, default=0.2, help="frazione elite (default 0.2)")
    ap.add_argument("--alpha", type=float, default=0.2, help="quota mixture da f (default 0.2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="non mostrare la discesa gamma per iterazione")
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)
    n_workers = int(os.environ.get("NUM_WORKERS", "4"))

    import numpy as np
    from pipeline.orchestrator import run_rare_event
    from scenarios import SCENARIOS

    lower = REALISTIC_LOWER if args.preset == "realistic" else None
    upper = REALISTIC_UPPER if args.preset == "realistic" else None

    # Override espliciti per puntare direttamente un ODD piu' stretto (regime raro):
    # richiedono bound espliciti -> per il preset 'full' partiamo dai default scenario.
    if args.max_speed is not None or args.max_angle is not None:
        if lower is None or upper is None:
            _b = SCENARIOS[args.scenario].param_bounds()
            lower = list(np.asarray(_b["lower"], dtype=float))
            upper = list(np.asarray(_b["upper"], dtype=float))
        else:
            lower = list(map(float, lower)); upper = list(map(float, upper))
        if args.max_angle is not None:
            for k in range(5):                               # angoli 1..5
                upper[k] = float(args.max_angle)
                lower[k] = min(lower[k], upper[k])
        if args.max_speed is not None:
            X = float(args.max_speed)
            upper[6] = X                                     # max_speed (upper)
            lower[6] = max(1.0, min(lower[6], X - 1.0))
            upper[5] = min(upper[5], X)                      # min_speed non oltre X
            lower[5] = max(1.0, min(lower[5], upper[5] - 1.0))
        # Sicurezza: nessuna dimensione con lower >= upper.
        for i in range(len(lower)):
            if upper[i] <= lower[i]:
                upper[i] = lower[i] + 1e-6

    names = SCENARIOS[args.scenario].param_bounds()["names"]

    line = "=" * 66
    sub = "-" * 66
    print(line)
    print(f" {args.scenario.upper()} — stima rare-event (Cross-Entropy)")
    print(line)
    budget_max = args.spi * args.max_iter + args.final
    print(f" Preset ODD     : {args.preset}")
    if args.max_angle is not None:
        print(f" Max angle (cap): {args.max_angle} deg")
    if args.max_speed is not None:
        print(f" Max speed (cap): {args.max_speed} m/s")
    print(f" Campionamento  : Cross-Entropy adattiva + importance sampling")
    print(f" Budget CE      : {args.spi}/iter x max {args.max_iter} iter + {args.final} finali "
          f"(<= {budget_max} run)")
    print(f" Worker pool    : {n_workers}  (NUM_WORKERS={n_workers})")
    print(sub)
    if not args.quiet:
        print(" Discesa di gamma (la soglia di 'quasi-fallimento' verso 0):")

    t0 = time.time()
    res = run_rare_event(
        args.scenario, seed=args.seed,
        param_lower=lower, param_upper=upper,
        samples_per_iter=args.spi, final_samples=args.final,
        rho=args.rho, max_iter=args.max_iter, alpha=args.alpha,
        verbose=not args.quiet,
    )
    dt = time.time() - t0

    print(sub)
    print(" RISULTATI")
    print(sub)
    print(f" Tempo totale        : {dt:6.1f}s   ({res.n_evaluations} run eseguiti)")
    print(f" Iterazioni CE       : {res.iterations}")
    lo, hi = res.ci
    print(f" P(fallimento)       : {res.p_fail*100:.4f}%   CI95% [{lo*100:.4f}, {hi*100:.4f}]%")
    print(f" Fallimenti effettivi (stima finale): {res.n_fail_effective}")
    if res.gamma_history:
        gh = ", ".join(f"{g:+.3f}" for g in res.gamma_history)
        print(f" Gamma per iterazione: [{gh}]   (arriva a <=0 = soglia di fallimento)")
    print(sub)

    # ── Regione di fallimento: dove la CE ha spostato la proposta ──
    # Confronto tra la proposta finale (loc) e il centro dell'ODD: i parametri che si
    # sono spostati di piu' sono quelli che "guidano" il fallimento.
    print(" REGIONE DI FALLIMENTO (dove la proposta si e' concentrata):")
    bnd = SCENARIOS[args.scenario].param_bounds()
    mid = (np.asarray(bnd["lower"], float) + np.asarray(bnd["upper"], float)) / 2.0
    if lower is not None:
        mid = (np.asarray(lower, float) + np.asarray(upper, float)) / 2.0
    for j, nm in enumerate(names):
        shift = res.q_loc[j] - mid[j]
        arrow = "↑" if shift > 0 else ("↓" if shift < 0 else "·")
        print(f"   {nm:<20} loc={res.q_loc[j]:7.2f}  (centro ODD {mid[j]:7.2f})  {arrow}{abs(shift):6.2f}")
    print(sub)

    if res.p_fail <= 0.0 or res.n_fail_effective == 0:
        print(" Lettura: nessun fallimento trovato in questo ODD -> P stimata ~0. A cadenza")
        print("          corretta l'ODD ristretto e' molto sicuro; prova --preset full o un")
        print("          ODD piu' duro per far emergere (e stimare) i fallimenti rari.")
    else:
        print(" Lettura: la CE ha localizzato i fallimenti e stimato P con importance")
        print("          sampling; la 'regione di fallimento' sopra dice quali parametri li")
        print("          guidano. Confronta P col brute-force solo se puoi permettertelo.")
    print(line)


if __name__ == "__main__":
    main()
