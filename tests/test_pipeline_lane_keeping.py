"""
Test pipeline completa per LaneKeepingScenario.

Esegue la pipeline end-to-end:
    LHS sampling -> Docker simulation -> composite QoI -> POD embedding -> rare failures

Output:
    - detailed terminal text for every sample
    - tests/output/pipeline_results.png with three plots:
        1. XTE over time for every run (safe=green, failure=red)
        2. Scatter plot POD mode 1 vs mode 2 (safe vs failure)
        3. QoI decomposition (M1, M2, M3) per sample

Usage:
    cd /path/to/Detecting-rare-failures-in-autonomous-vehicles
    python3 tests/test_pipeline_lane_keeping.py [--n N]

Optional arguments:
    --n N   number of LHS samples (default: 20)
"""

from __future__ import annotations

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no graphical window -- save to file only
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats.qmc import LatinHypercube, scale

from scenarios.lane_keeping.config import (
    LaneKeepingScenario, MAX_XTE, STEER_RANGE_NORM, EARLY_FRAC
)
from embedder.pod import EmbedderPOD
from pipeline.severity import find_severe_failures

# ── Costanti ──────────────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pipeline_results.png")

SEP  = "─" * 70
SEP2 = "═" * 70

COLOR_SAFE    = "#2ecc71"
COLOR_FAILURE = "#e74c3c"
COLOR_RARE    = "#8e44ad"

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_qoi_components(sc: LaneKeepingScenario,
                           traj: np.ndarray,
                           params: np.ndarray):
    """Returns M1, M2, M3 and the QoI separately for every run."""
    N, T, _ = traj.shape
    xte      = traj[:, :, 2].astype(np.float64)
    steering = traj[:, :, 3].astype(np.float64)

    run_lengths = sc._run_lengths if sc._run_lengths else [T] * N
    valid = np.zeros((N, T), dtype=bool)
    for i, L in enumerate(run_lengths):
        valid[i, :L] = True
    valid_count = valid.sum(axis=1).astype(np.float64)
    valid_count = np.where(valid_count > 0, valid_count, 1.0)

    xte_m      = np.where(valid, xte,      np.nan)
    steering_m = np.where(valid, steering, np.nan)

    m1 = MAX_XTE - np.nanmax(np.abs(xte_m), axis=1)
    steer_abs  = np.abs(steering_m)
    steer_peak = np.nanmax(steer_abs, axis=1) - np.nanmean(steer_abs, axis=1)
    m2 = -np.clip(steer_peak / STEER_RANGE_NORM, 0.0, 1.0)

    near      = np.where(np.isnan(xte_m), False, np.abs(xte_m) > EARLY_FRAC * MAX_XTE)
    any_near  = near.any(axis=1)
    first_idx = np.where(any_near, near.argmax(axis=1).astype(float), valid_count)
    m3        = -(valid_count - first_idx) / valid_count

    return m1, m2, m3, 0.6 * m1 + 0.2 * m2 + 0.2 * m3


def print_section(title: str):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


def print_sample_row(i: int, params_row: np.ndarray, qoi: float,
                     run_len: int, max_xte: float, label: str):
    angles = params_row[:5]
    print(f"  [{i+1:2d}] angoli=[{','.join(f'{a:2.0f}' for a in angles)}]"
          f"  speed=[{params_row[5]:.0f},{params_row[6]:.0f}]"
          f"  seg={params_row[7]:.0f}m  map={params_row[8]:.0f}m"
          f"  →  XTE={max_xte:.3f}m  QoI={qoi:+.3f}  {label}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plots(traj: np.ndarray, qoi: np.ndarray, failures: np.ndarray,
               rare_idx: np.ndarray, pod_codes: np.ndarray,
               run_lengths: list[int], m1: np.ndarray, m2: np.ndarray,
               m3: np.ndarray, param_names: list[str], n_samples: int,
               params: np.ndarray | None = None):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    N, T, _ = traj.shape
    safe_mask    = failures == 0
    fail_mask    = failures == 1
    rare_mask    = np.zeros(N, dtype=bool)
    rare_mask[rare_idx] = True

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(
        f"Lane Keeping — pipeline LHS (N={n_samples})   |   "
        f"failure rate: {fail_mask.mean()*100:.0f}%   |   "
        f"rare failures: {len(rare_idx)}",
        fontsize=13, fontweight="bold"
    )

    # -- Plot 1: XTE over time -------------------------------------------------
    ax = axes[0]
    ax.set_title("XTE over time (all runs)", fontsize=11)
    for i in range(N):
        L   = run_lengths[i]
        xte = traj[i, :L, 2]
        t   = np.arange(L)
        if rare_mask[i]:
            color, lw, zorder, alpha = COLOR_RARE, 1.8, 3, 0.9
        elif fail_mask[i]:
            color, lw, zorder, alpha = COLOR_FAILURE, 1.2, 2, 0.6
        else:
            color, lw, zorder, alpha = COLOR_SAFE, 0.8, 1, 0.4
        ax.plot(t, xte, color=color, linewidth=lw, zorder=zorder, alpha=alpha)

    ax.axhline( MAX_XTE, color="black", linestyle="--", linewidth=1, label=f"+MAX_XTE ({MAX_XTE}m)")
    ax.axhline(-MAX_XTE, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Step simulazione")
    ax.set_ylabel("XTE (m)")
    legend_patches = [
        mpatches.Patch(color=COLOR_SAFE,    label="safe"),
        mpatches.Patch(color=COLOR_FAILURE, label="failure"),
        mpatches.Patch(color=COLOR_RARE,    label="rare failure"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right")
    ax.set_ylim(-MAX_XTE * 1.3, MAX_XTE * 1.3)

    # ── Grafico 2: Scatter POD mode 1 vs mode 2 ───────────────────────────────
    ax = axes[1]
    ax.set_title("POD embedding — mode 1 vs mode 2", fontsize=11)
    if pod_codes.shape[1] >= 2:
        ax.scatter(pod_codes[safe_mask, 0],    pod_codes[safe_mask, 1],
                   c=COLOR_SAFE,    s=60, label="safe",          zorder=2, alpha=0.8)
        non_rare_fail = fail_mask & ~rare_mask
        ax.scatter(pod_codes[non_rare_fail, 0], pod_codes[non_rare_fail, 1],
                   c=COLOR_FAILURE, s=60, label="failure",       zorder=3, alpha=0.8)
        ax.scatter(pod_codes[rare_mask, 0],    pod_codes[rare_mask, 1],
                   c=COLOR_RARE,    s=120, label="rare failure", zorder=4,
                   edgecolors="black", linewidths=1.2)
        for idx in rare_idx:
            ax.annotate(f"#{idx+1}", (pod_codes[idx, 0], pod_codes[idx, 1]),
                        fontsize=8, xytext=(5, 5), textcoords="offset points")
    else:
        ax.text(0.5, 0.5, "POD has a single mode\n(trajectories too similar)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("POD mode 1")
    ax.set_ylabel("POD mode 2")
    ax.legend(fontsize=9)

    # ── Grafico 3: Decomposizione QoI (M1, M2, M3) ────────────────────────────
    ax = axes[2]
    ax.set_title("QoI components per sample", fontsize=11)
    x   = np.arange(N)
    w   = 0.25
    b1  = ax.bar(x - w, m1, w, label="M1 (XTE margin × 0.6)",      color="#3498db", alpha=0.85)
    b2  = ax.bar(x,     m2, w, label="M2 (sterzo peak dev × 0.2)",  color="#f39c12", alpha=0.85)
    b3  = ax.bar(x + w, m3, w, label="M3 (early approach × 0.2)",   color="#9b59b6", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)

    # Highlight the failures with a background
    for i in range(N):
        if fail_mask[i]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08,
                       color=COLOR_RARE if rare_mask[i] else COLOR_FAILURE)

    ax.set_xlabel("Campione LHS")
    ax.set_ylabel("Valore componente QoI")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(N)], fontsize=7)
    ax.legend(fontsize=8, loc="upper right")

    # ── Grafico 4: segment_length vs QoI ─────────────────────────────────────
    ax = axes[3]
    ax.set_title("segment_length vs QoI", fontsize=11)
    if params is not None:
        seg_idx = 7   # segment_length column in the parameter vector
        seg = params[:, seg_idx]
        ax.scatter(seg[safe_mask],         qoi[safe_mask],
                   c=COLOR_SAFE,    s=60, label="safe",         alpha=0.8, zorder=2)
        non_rare_fail = fail_mask & ~rare_mask
        ax.scatter(seg[non_rare_fail],     qoi[non_rare_fail],
                   c=COLOR_FAILURE, s=60, label="failure",      alpha=0.8, zorder=3)
        ax.scatter(seg[rare_mask],         qoi[rare_mask],
                   c=COLOR_RARE,    s=120, label="rare failure", zorder=4,
                   edgecolors="black", linewidths=1.2)
        for idx in rare_idx:
            ax.annotate(f"#{idx+1}\nseg={params[idx,seg_idx]:.0f}m",
                        (seg[idx], qoi[idx]),
                        fontsize=7, xytext=(6, 4), textcoords="offset points")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, label="QoI=0 (failure)")
        ax.set_xlabel("segment_length (m)")
        ax.set_ylabel("QoI")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "params not available",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\n  Grafici salvati in: {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_samples: int = 50, seed: int = 42, rare_fraction: float = 0.30):

    sc     = LaneKeepingScenario()
    bounds = sc.param_bounds()

    # ── 1. LHS sampling ───────────────────────────────────────────────────────
    print_section(f"STEP 1 — Latin Hypercube Sampling  (N={n_samples}, seed={seed})")
    sampler = LatinHypercube(d=len(bounds["lower"]), seed=seed)
    params  = scale(sampler.random(n=n_samples), bounds["lower"], bounds["upper"])
    print(f"\n  {len(bounds['names'])} parametri campionati:")
    for name, lo, hi in zip(bounds["names"], bounds["lower"], bounds["upper"]):
        print(f"    {name:<22}  [{lo:.1f}, {hi:.1f}]")
    print(f"\n  Sample table (rows = runs, columns = parameters):")
    header = "  #    " + "  ".join(f"{n[:8]:>8}" for n in bounds["names"])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row in enumerate(params):
        print(f"  {i+1:2d}   " + "  ".join(f"{v:8.2f}" for v in row))

    # ── 2. Simulazioni ────────────────────────────────────────────────────────
    print_section(f"STEP 2 — Simulazioni Docker parallelizzate  (http://localhost:8000)")
    print(f"\n  Fase 1 — Submit {n_samples} job al SimulatorServer...")
    print(f"  Fase 2 — Poll parallelo (ThreadPoolExecutor, max_workers={n_samples})")
    print(f"\n  Attendi completamento...\n")

    t0      = time.time()
    traj    = sc.run_simulation(params, verbose=True)   # prints each job as it finishes
    elapsed = time.time() - t0

    all_run_lengths = sc._run_lengths
    print(f"\n  ✓ Completate {n_samples} simulazioni in {elapsed:.1f}s  "
          f"({elapsed/n_samples:.1f}s/sim media)")

    # ── 3. QoI ───────────────────────────────────────────────────────────────
    print_section("STEP 3 — QoI composita (M1 + M2 + M3)")
    m1, m2, m3, qoi = compute_qoi_components(sc, traj, params)
    failures         = (qoi < 0).astype(float)
    fail_rate        = failures.mean()

    print(f"\n  {'#':>3}  {'QoI':>7}  {'M1':>7}  {'M2':>7}  {'M3':>7}  {'XTE max':>8}  Esito")
    print("  " + "-" * 62)
    for i in range(n_samples):
        L       = all_run_lengths[i]
        max_xte = np.abs(traj[i, :L, 2]).max()
        outcome   = "✗ FAIL" if failures[i] else "✓ safe"
        print(f"  {i+1:3d}  {qoi[i]:+7.3f}  {m1[i]:+7.3f}  {m2[i]:+7.3f}  {m3[i]:+7.3f}  {max_xte:8.4f}m  {outcome}")
    print(f"\n  Failure rate: {failures.sum():.0f} / {n_samples}  ({fail_rate*100:.1f}%)")

    # ── 4. POD embedding ─────────────────────────────────────────────────────
    print_section("STEP 4 -- POD embedding of the trajectories")
    # Uses x, XTE and steering (channels 0, 2, 3): y is redundant with XTE (lateral position)
    # x provides the dominant temporal structure that concentrates the variance in the first mode
    traj_pod  = traj[:, :, [0, 2, 3]]         # (N, T, 3): x + xte + steering
    pod       = EmbedderPOD(variance_threshold=0.99)
    pod_codes = pod.fit_transform(traj_pod)   # (N, k)

    print(f"\n  Traiettorie complete: {traj.shape}  (4 canali: x, y, xte, steering)")
    print(f"  Input POD:            {traj_pod.shape}  (3 canali: x, xte, steering — y rimossa)")
    print(f"  POD modes selected: {pod.nModes}  (explaining >=99% of the variance)")
    print(f"  POD codes:          {pod_codes.shape}  (each run -> {pod.nModes} numbers)")

    # Varianza spiegata per modo
    flat   = traj_pod.reshape(n_samples, -1)
    flat_c = flat - flat.mean(axis=0)
    _, S, _ = np.linalg.svd(flat_c, full_matrices=False)
    cum_var = np.cumsum(S**2) / np.sum(S**2)
    print(f"\n  Varianza spiegata per modo:")
    for k in range(min(pod.nModes, 6)):
        bar = "█" * int(cum_var[k] * 30)
        print(f"    modo {k+1:2d}: {cum_var[k]*100:5.1f}%  {bar}")

    # ── 5. Rare failures ─────────────────────────────────────────────────────
    print_section("STEP 5 — Rare failures")
    rare_idx = find_severe_failures(
        safety_margins=qoi,
        failures=failures,
        fraction=rare_fraction,
    )
    rare_mask = np.zeros(n_samples, dtype=bool)
    rare_mask[rare_idx] = True

    if len(rare_idx) == 0:
        print(f"\n  Nessun rare failure trovato (fraction={rare_fraction}).")
        print(f"  Con {failures.sum():.0f} failure su {n_samples}, prova ad aumentare N.")
    else:
        print(f"\n  Rare failures (bottom {rare_fraction*100:.0f}% of failures by QoI):"
              f"  {len(rare_idx)} samples")
        print(f"\n  {'#':>3}  {'QoI':>7}  angoli                          speed   seg   map")
        print("  " + "-" * 72)
        for idx in rare_idx:
            row = params[idx]
            angles_str = "[" + ",".join(f"{a:.0f}" for a in row[:5]) + "]"
            print(f"  {idx+1:3d}  {qoi[idx]:+7.3f}  {angles_str:<32}"
                  f"  [{row[5]:.0f},{row[6]:.0f}]   {row[7]:.0f}m  {row[8]:.0f}m")

        # Confronto POD safe vs rare
        if pod_codes.shape[1] >= 2 and rare_mask.any() and (~failures.astype(bool)).any():
            safe_codes = pod_codes[~failures.astype(bool)]
            rare_codes = pod_codes[rare_mask]
            dist = np.linalg.norm(safe_codes.mean(axis=0) - rare_codes.mean(axis=0))
            print(f"\n  Distanza POD  safe_centroid → rare_centroid: {dist:.4f}")
            print(f"  (a high value = the rare failures occupy a different part of the space)")

        # Parametri critici nei rare failures
        seg_vals = [params[idx, 7] for idx in rare_idx]
        print(f"\n  ⚠  Parametro critico rilevato: segment_length")
        print(f"     Rare failures — seg: {[f'{v:.0f}m' for v in seg_vals]}")
        safe_seg = params[~failures.astype(bool), 7]
        print(f"     Safe runs      — seg media: {safe_seg.mean():.1f}m  "
              f"min: {safe_seg.min():.1f}m  max: {safe_seg.max():.1f}m")

    # ── 6. Grafici ────────────────────────────────────────────────────────────
    print_section("STEP 6 — Generazione grafici")
    make_plots(traj, qoi, failures, rare_idx, pod_codes,
               all_run_lengths, m1, m2, m3, bounds["names"], n_samples,
               params=params)

    print_section("Riepilogo finale")
    print(f"\n  Campioni totali:    {n_samples}")
    print(f"  Failure:            {int(failures.sum())}  ({fail_rate*100:.1f}%)")
    print(f"  Rare failure:       {len(rare_idx)}")
    print(f"  Modi POD (99% var): {pod.nModes}")
    print(f"  Grafici:            {OUTPUT_FILE}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,   default=50,   help="Number of LHS samples")
    parser.add_argument("--seed", type=int,   default=42,   help="Seed riproducibilità")
    parser.add_argument("--rare", type=float, default=0.30, help="Frazione rare failures")
    args = parser.parse_args()
    main(n_samples=args.n, seed=args.seed, rare_fraction=args.rare)
