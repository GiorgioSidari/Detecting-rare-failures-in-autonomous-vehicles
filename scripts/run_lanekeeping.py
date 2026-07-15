#!/usr/bin/env python3
"""
Runner con output leggibile per lo scenario lane_keeping (pool parallelo).

Uso:
    python scripts/run_lanekeeping.py                      # 20 campioni, 4 worker, bounds full
    python scripts/run_lanekeeping.py --n 50 --workers 4
    python scripts/run_lanekeeping.py --n 50 --preset realistic   # ODD realistico proposto
    python scripts/run_lanekeeping.py --n 30 --seed 7 --quiet

Prerequisiti: i container del simulatore devono essere in esecuzione, es.:
    cd opensbt-core && docker compose -f docker-compose.parallel.yml up --build
"""
from __future__ import annotations

import argparse
import os
import time
import sys

# Project root su sys.path (eseguibile da qualsiasi cartella)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Ordine parametri: angle1..5, min_speed, max_speed, seg_length, map_size.
#
# Preset "realistic": PROPOSTA di ODD realistico (da confermare/adeguare).
# - angoli 0-45 per segmento (evita tornanti irreali)
# - min_speed 5-8, max_speed 9-14 m/s: bande NON sovrapposte, cosi' non si
#   generano mai campioni incoerenti con min_speed > max_speed
# - segmenti piu' lunghi (curve piu' dolci)
REALISTIC_LOWER = [0,  0,  0,  0,  0,   5.0,  9.0, 20.0, 200.0]
REALISTIC_UPPER = [45, 45, 45, 45, 45,  8.0, 14.0, 40.0, 350.0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Esegue la pipeline lane_keeping con output chiaro.")
    ap.add_argument("--n", type=int, default=20, help="numero di campioni LHS (default 20)")
    ap.add_argument("--workers", type=int, default=None,
                    help="numero di container/worker paralleli (default: NUM_WORKERS env o 4)")
    ap.add_argument("--seed", type=int, default=42, help="seed (default 42)")
    ap.add_argument("--scenario", default="lane_keeping", help="nome scenario (default lane_keeping)")
    ap.add_argument("--preset", choices=["full", "realistic"], default="full",
                    help="'full' = spazio esteso completo (ODD ampio); "
                         "'realistic' = ODD realistico proposto (angoli/velocita' plausibili)")
    ap.add_argument("--rare-fraction", type=float, default=0.05,
                    help="frazione bottom-k trattata come 'rara' (default 0.05)")
    ap.add_argument("--quiet", action="store_true", help="non mostrare il progresso per-simulazione")
    ap.add_argument("--trace-worst", action="store_true",
                    help="stampa la traccia passo-passo (x, XTE, sterzo) del run peggiore")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="forza il limite superiore di max_speed (m/s) su qualsiasi preset")
    ap.add_argument("--max-angle", type=float, default=None,
                    help="forza il limite superiore degli angoli (gradi) su qualsiasi preset")
    ap.add_argument("--sampling", choices=["uniform", "realistic"], default="realistic",
                    help="'realistic' = campiona dalla distribuzione operativa (ppf): la "
                         "frazione di fallimenti stima P(fallimento) sull'ODD; "
                         "'uniform' = LHS uniforme sui bound (baseline)")
    args = ap.parse_args()

    # IMPORTANTE: NUM_WORKERS va impostato PRIMA di importare la pipeline,
    # perche' il pool viene costruito all'import di scenarios/.
    if args.workers is not None:
        os.environ["NUM_WORKERS"] = str(args.workers)
    n_workers = int(os.environ.get("NUM_WORKERS", "4"))

    import numpy as np
    from pipeline.orchestrator import run

    lower = REALISTIC_LOWER if args.preset == "realistic" else None
    upper = REALISTIC_UPPER if args.preset == "realistic" else None

    # Override espliciti (velocita'/angolo massimi). Richiedono bounds espliciti:
    # per il preset "full" partiamo dai default dello scenario.
    if args.max_speed is not None or args.max_angle is not None:
        if lower is None or upper is None:
            from scenarios import SCENARIOS
            _b = SCENARIOS[args.scenario].param_bounds()
            lower = list(np.asarray(_b["lower"], dtype=float))
            upper = list(np.asarray(_b["upper"], dtype=float))
        if args.max_angle is not None:
            for _k in range(5):                          # angoli 1..5 (il vero driver)
                upper[_k] = float(args.max_angle)
                lower[_k] = min(lower[_k], upper[_k])
        if args.max_speed is not None:
            X = float(args.max_speed)
            upper[6] = X                                 # max_speed (upper)
            lower[6] = max(1.0, min(lower[6], X - 1.0))
            upper[5] = min(upper[5], X)                  # min_speed non oltre X
            lower[5] = max(1.0, min(lower[5], upper[5] - 1.0))

    # Sicurezza: nessuna dimensione con lower >= upper (scipy.scale lo rifiuta).
    if lower is not None and upper is not None:
        lower = list(map(float, lower)); upper = list(map(float, upper))
        for _i in range(len(lower)):
            if upper[_i] <= lower[_i]:
                upper[_i] = lower[_i] + 1e-6

    line = "=" * 64
    sub  = "-" * 64
    print(line)
    print(f" {args.scenario.upper()} — esecuzione pipeline")
    print(line)
    print(f" Campioni (N)   : {args.n}")
    print(f" Seed           : {args.seed}")
    print(f" Preset bounds   : {args.preset}")
    print(f" Campionamento   : {args.sampling}"
          + ("  (distribuzione operativa via ppf)" if args.sampling == "realistic"
             else "  (LHS uniforme sui bound)"))
    if args.max_angle is not None:
        print(f" Max angle (cap) : {args.max_angle} deg")
    if args.max_speed is not None:
        print(f" Max speed (cap) : {args.max_speed} m/s")
    print(f" Worker pool     : {n_workers}  (NUM_WORKERS={n_workers})")
    print(sub)
    if not args.quiet:
        print(" Progresso simulazioni (una riga per run completato):")

    t0 = time.time()
    r = run(args.scenario, n_samples=args.n, seed=args.seed,
            rare_fraction=args.rare_fraction,
            param_lower=lower, param_upper=upper,
            sampling=args.sampling,
            verbose=not args.quiet)
    dt = time.time() - t0

    m = np.asarray(r.safety_margins, dtype=float)
    N = args.n
    valid = (np.asarray(r.valid_mask, dtype=bool)
             if getattr(r, "valid_mask", None) is not None else ~np.isnan(m))
    n_valid   = int(valid.sum())
    n_invalid = int(getattr(r, "n_invalid", N - n_valid))
    n_deg     = int(getattr(r, "n_degenerate", 0))
    order = np.argsort(np.where(valid, m, np.inf))[:n_valid]   # solo validi, worst->best
    mv = m[valid]
    n_rare = len(r.rare_failure_idx)

    print(sub)
    print(" RISULTATI")
    print(sub)
    print(f" Tempo totale        : {dt:6.1f}s   (~{dt/max(N,1):.1f}s per campione)")
    n_lowfid = int(getattr(r, "n_low_fidelity", 0))
    inv_note = ""
    if n_invalid:
        parts = []
        if n_deg:
            parts.append(f"{n_deg} abortiti")
        if n_lowfid:
            parts.append(f"{n_lowfid} sotto-campionati")
        detail = (": " + ", ".join(parts)) if parts else ""
        inv_note = f"   ({n_invalid} non validi esclusi{detail})"
    print(f" Scenari validi      : {n_valid}/{N}{inv_note}")
    if n_valid == 0:
        print(" Nessuno scenario valido: impossibile calcolare i tassi.")
        print(line)
        return
    n_fail = int((mv < 0).sum())
    # ── Asse RARITA' (probabilita') ──
    p_fail = float(getattr(r, "failure_probability", r.failure_rate))
    ci = getattr(r, "failure_probability_ci", None)
    odd_lbl = ("ODD realistico" if getattr(r, "sampling", "uniform") == "realistic"
               else "ODD uniforme")
    ci_txt = (f"  CI95% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]%"
              if ci is not None else "")
    print(f" P(fallimento)       : {p_fail*100:5.1f}%   [{odd_lbl}]"
          f"   ({n_fail}/{n_valid} falliti){ci_txt}")
    # ── Asse SEVERITA' (worst-case) ──
    print(f" Worst-case (severita'): bottom-{args.rare_fraction*100:.0f}% = {n_rare}/{n_valid} scenari"
          f"   | margine: min {mv.min():+.3f} | mediana {np.median(mv):+.3f} | max {mv.max():+.3f}")
    print(f" Modi POD             : {r.pod_n_modes}")
    print(sub)

    # ── Fedelta' del loop di controllo ────────────────────────────────────────
    # Quanto velocemente ha girato il loop DNN->Unity per ogni run. Se cala quando
    # aumenti --workers, la parallelizzazione ti sta togliendo accuratezza: l'auto
    # percorre piu' metri tra due sterzate e i fallimenti diventano artefatti di CPU,
    # non del modello. Tienilo alto (pochi metri/step) scegliendo bene i worker; i
    # run sotto-soglia si escludono con LK_MIN_CONTROL_HZ / LK_MAX_METERS_PER_STEP.
    chz = getattr(r, "control_hz", None)
    mps = getattr(r, "meters_per_step", None)
    if chz is not None and mps is not None:
        chz = np.asarray(chz, dtype=float); mps = np.asarray(mps, dtype=float)
        if np.isfinite(chz).any():
            print(" FEDELTA' DEL CONTROLLO (indipendenza dal carico/worker):")
            print(f"   Frequenza loop   : min {np.nanmin(chz):5.2f} | mediana "
                  f"{np.nanmedian(chz):5.2f} | max {np.nanmax(chz):5.2f}  Hz")
            print(f"   Metri per sterzata: min {np.nanmin(mps):5.2f} | mediana "
                  f"{np.nanmedian(mps):5.2f} | max {np.nanmax(mps):5.2f}  m/step")

            # ── Split del tempo per step: inferenza vs attesa Unity ──
            # Dice DOVE va il tempo del loop: se domina 'attesa Unity' il collo di
            # bottiglia e' il simulatore (I/O), se domina 'inferenza' e' la CPU.
            im = getattr(r, "infer_ms_per_step", None)
            wm = getattr(r, "wait_ms_per_step", None)
            if im is not None and wm is not None:
                im = np.asarray(im, dtype=float); wm = np.asarray(wm, dtype=float)
                if np.isfinite(im).any() and np.isfinite(wm).any():
                    im_med, wm_med = np.nanmedian(im), np.nanmedian(wm)
                    tot = im_med + wm_med
                    quota = (wm_med / tot * 100.0) if tot > 0 else float("nan")
                    print(f"   Tempo/step (mediana): inferenza {im_med:6.1f} ms | "
                          f"attesa Unity {wm_med:6.1f} ms  "
                          f"(attesa = {quota:4.1f}% del passo)")
                    dominante = "Unity/I-O" if wm_med >= im_med else "inferenza/CPU"
                    print(f"   -> collo di bottiglia: {dominante}")

            n_lowfid = int(getattr(r, "n_low_fidelity", 0))
            if n_lowfid:
                print(f"   Esclusi per bassa fedelta': {n_lowfid} "
                      f"(sotto-campionati: non contano come fallimenti)")
            else:
                print("   Gate fedelta' OFF (LK_MIN_CONTROL_HZ/LK_MAX_METERS_PER_STEP=0): "
                      "solo misura, nessuna esclusione")
            print(sub)

    print(" Margini ordinati (peggiore -> migliore, solo scenari validi):")
    vals = ", ".join(f"{m[i]:+.2f}" for i in order)
    print(f"   [{vals}]")
    print("   (valori < 0 = fallimento; piu' negativo = peggiore)")
    print(sub)

    # survival (n. step) ricavato dalle traiettorie (timestep non nulli), se disponibile.
    survival = None
    try:
        traj = np.asarray(r.trajectories)
        survival = (np.abs(traj).sum(axis=2) > 0).sum(axis=1)
    except Exception:
        survival = None

    def fmt_params(row):
        row = np.asarray(row, dtype=float)
        if len(row) >= 9:
            ang = ",".join(f"{int(round(a)):d}" for a in row[:5])
            return f"angoli=[{ang}]  speed=[{int(round(row[5]))},{int(round(row[6]))}]  seg={int(round(row[7]))}m  map={int(round(row[8]))}m"
        return "params=[" + ",".join(f"{v:.1f}" for v in row) + "]"

    rare = list(r.rare_failure_idx)
    if rare:
        if len(rare) == 1:
            print(" SCENARIO RARO (il piu' critico):")
        else:
            print(f" SCENARI RARI (i {len(rare)} piu' critici):")
        rows = list(rare)
    else:
        print(" Nessun rare failure isolato. I 5 scenari peggiori:")
        rows = list(order[:5])
    for i in rows:
        extra = f"  step={int(survival[i])}" if survival is not None else ""
        print(f"   #{int(i):<3d} margin={m[i]:+.3f}{extra}   {fmt_params(r.params[i])}")
    print(line)

    # Spiegazione data-driven del PERCHE' di ogni fallimento mostrato sopra:
    # classifica il modo di fallimento dai dati (XTE e sterzo passo-passo).
    def _explain(i):
        Li = int(survival[i]) if survival is not None else r.trajectories[i].shape[0]
        Li = max(1, min(Li, r.trajectories[i].shape[0]))
        t = np.asarray(r.trajectories[i], dtype=float)
        xte = t[:Li, 2]; steer = t[:Li, 3]
        xmin, xmax = float(xte.min()), float(xte.max())
        peak = xmax if abs(xmax) >= abs(xmin) else xmin
        fin = float(xte[-1]); steer_max = float(np.abs(steer).max())
        mask = np.abs(xte) > 0.2
        corr = float(np.mean(np.sign(steer[mask]) * np.sign(xte[mask]))) if mask.any() else 0.0
        both = (xmin < -1.0) and (xmax > 1.0)
        if both:
            mode = (f"oscillazione instabile: prima deriva (XTE {xmin:+.1f} m), poi lo sterzo "
                    f"{'satura e ' if steer_max >= 0.9 else ''}sovra-corregge fino a {xmax:+.1f} m "
                    f"dal lato opposto")
        elif corr > 0.15:
            mode = f"sterza nel verso sbagliato (asseconda la deriva) fino a XTE {peak:+.1f} m"
        else:
            quando = "tardi" if Li > 10 else "subito"
            mode = (f"deriva non corretta ({quando}): sterza contro l'errore ma non basta, "
                    f"XTE fino a {peak:+.1f} m")
        row = np.asarray(r.params[i], dtype=float)
        ctx = ""
        if len(row) >= 9:
            ctx = (f"; contesto: curva max {int(max(row[:5]))}°, "
                   f"velocita' {int(round(row[5]))}-{int(round(row[6]))} m/s")
        side = "XTE+ (positivo)" if peak > 0 else "XTE- (negativo)"
        return f"esce dal lato {side} dopo {Li} step -- {mode}{ctx}"

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
        print("   step |    x      |   XTE     | sterzo")
        for t in range(L):
            print(f"   {t:4d} | {wtraj[t,0]:9.3f} | {wtraj[t,2]:+9.3f} | {wtraj[t,3]:+7.3f}")
        print("   (XTE monotona in una direzione = deriva costante; oscilla crescendo = controllo instabile; parte gia alta = spawn/geometria)")
        print(sub)

    # Lettura sintetica
    fr = r.failure_rate
    if fr >= 0.8:
        hint = ("Tasso di fallimento molto alto: i fallimenti NON sono rari. "
                "Valuta un ODD piu' realistico (--preset realistic) o rivedi la "
                "soglia QoI; alto anche cosi' = modello intrinsecamente fragile.")
    elif fr <= 0.2:
        hint = "Tasso basso: buon regime per isolare rare failure significativi."
    else:
        hint = "Tasso intermedio: ragionevole per confrontare scenari sicuri e falliti."
    print(" Lettura:", hint)
    print(line)


if __name__ == "__main__":
    main()
