"""
Behavioral cloning trainer for the Emergency Braking MLP controller.

Pipeline
--------
1. Generate N trajectories with the optimal physics controller (LHS params).
2. At each timestep record (state, optimal_action) pairs.
3. Train BrakingMLP to imitate the optimal action.
4. Save weights to models/emergency_braking_mlp.keras

Optimal policy
--------------
The "expert" controller does exactly one thing:
  - braking_force = 0.0  while t < actual_delay   (reaction phase, no braking)
  - braking_force = 1.0  while t >= actual_delay AND velocity > 0  (full braking)
  - braking_force = 0.0  once velocity == 0   (already stopped, irrelevant)

The MLP learns a continuous approximation of this binary policy.
Its generalisation errors on unseen parameter combinations are the rare failures.
"""

import os
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

from simulators.emergency_braking import EmergencyBrakingSimulator

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "emergency_braking_mlp.keras")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

N_TRAIN = 5_000
SEED    = 42


def generate_dataset(n: int = N_TRAIN, seed: int = SEED, save: bool = True):
    """
    Generate behavioral-cloning dataset from the optimal physics controller.

    Parameters
    ----------
    n    : number of LHS parameter combinations to simulate
    seed : random seed for LHS (noise is always stochastic)
    save : if True, saves X.npy and y.npy to DATASET_DIR

    Returns
    -------
    X : (M, 3)  float32 — [velocity (m/s), distance_to_obstacle (m), elapsed_time (s)]
    y : (M, 1)  float32 — optimal braking_force in {0.0, 1.0}

    where M = n * T  (all timesteps of all trajectories, flattened)
    """
    np.random.seed(seed)

    sim    = EmergencyBrakingSimulator()
    bounds = sim.ParamBounds()
    T      = sim.T
    dt     = sim.dt

    # ── 1. Latin Hypercube sampling ──────────────────────────────────────────
    sampler      = LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n)
    params       = scale(unit_samples, bounds["lower"], bounds["upper"])  # (n, 4)

    detection_distances = params[:, 2]  # (n,) — needed for distance_to_obstacle feature

    print(f"  Simulating {n} trajectories (T={T} steps each) …")

    # ── 2. Simulate with the optimal physics controller ──────────────────────
    # run_with_actuals() exposes actual_delays so we can label every timestep.
    trajectories, actual_delays, _ = sim.run_with_actuals(params)
    # trajectories : (n, T, 2)  — [:, :, 0] = position, [:, :, 1] = velocity
    # actual_delays: (n,)        — real reaction delay (nominal + noise)

    print(f"  Building state-action pairs …")

    # ── 3. Build feature matrix X and label vector y ─────────────────────────
    # Time axis: shape (T,)
    time_axis = np.arange(T, dtype=np.float32) * dt  # [0.00, 0.01, ..., 9.99]

    # Broadcast time over trajectories: (n, T)
    times = np.broadcast_to(time_axis, (n, T))  # read-only view, no copy

    positions  = trajectories[:, :, 0]   # (n, T)
    velocities = trajectories[:, :, 1]   # (n, T)

    # distance_to_obstacle at each timestep = detection_distance - current_position
    # Shape: detection_distances is (n,), expand to (n, T)
    dist_to_obs = detection_distances[:, np.newaxis] - positions  # (n, T)
    dist_to_obs = np.clip(dist_to_obs, 0.0, None)   # never negative (crashed = 0)

    # Stack features → (n, T, 3)
    X_3d = np.stack([velocities, dist_to_obs, times], axis=2).astype(np.float32)

    # ── 4. Optimal action label ──────────────────────────────────────────────
    # Braking phase starts when t >= actual_delay AND vehicle is still moving.
    # actual_delays shape: (n,) → broadcast to (n, T)
    delay_broadcast = actual_delays[:, np.newaxis]        # (n, 1)
    braking_phase   = (times >= delay_broadcast)          # (n, T) bool
    still_moving    = (velocities > 0.0)                  # (n, T) bool

    y_2d = (braking_phase & still_moving).astype(np.float32)   # (n, T)  values: 0.0 or 1.0

    # ── 5. Flatten to (M, 3) and (M, 1) ─────────────────────────────────────
    M = n * T
    X = X_3d.reshape(M, 3)          # (M, 3)
    y = y_2d.reshape(M, 1)          # (M, 1)

    # ── 6. Sanity check ──────────────────────────────────────────────────────
    brake_fraction = y.mean()
    print(f"  Dataset shape  : X={X.shape}, y={y.shape}")
    print(f"  Braking fraction: {brake_fraction:.1%}  "
          f"(expected ~50-70% depending on delay distribution)")
    assert 0.2 < brake_fraction < 0.9, (
        f"Unexpected braking fraction {brake_fraction:.1%}. "
        "Check noise model or parameter bounds."
    )

    # ── 7. Optionally save to disk ────────────────────────────────────────────
    if save:
        os.makedirs(DATASET_DIR, exist_ok=True)
        np.save(os.path.join(DATASET_DIR, "X.npy"), X)
        np.save(os.path.join(DATASET_DIR, "y.npy"), y)
        np.save(os.path.join(DATASET_DIR, "params.npy"), params)
        print(f"  Saved to {DATASET_DIR}/")

    return X, y


def load_dataset():
    """Load a previously generated dataset from disk."""
    X = np.load(os.path.join(DATASET_DIR, "X.npy"))
    y = np.load(os.path.join(DATASET_DIR, "y.npy"))
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    return X, y


def train(
    n: int = N_TRAIN,
    seed: int = SEED,
    force_regenerate: bool = False,
    epochs: int = 50,
    batch_size: int = 2048,
    plot: bool = True,
) -> None:
    """
    Full training pipeline: generate dataset → build MLP → fit → save.

    Parameters
    ----------
    n               : number of LHS trajectories for the dataset
    seed            : random seed
    force_regenerate: if False and dataset already exists on disk, skip generation
    epochs          : max training epochs (EarlyStopping may stop sooner)
    batch_size      : 2048 works well for 5M rows; reduce if OOM
    plot            : save a loss-curve plot to MODELS_DIR/training_curves.png
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    x_path = os.path.join(DATASET_DIR, "X.npy")
    if not force_regenerate and os.path.exists(x_path):
        print("Dataset already on disk — loading …")
        X, y = load_dataset()
    else:
        print(f"Generating dataset ({n} trajectories) …")
        X, y = generate_dataset(n=n, seed=seed, save=True)

    print(f"\nDataset: {len(X):,} samples  "
          f"({y.mean():.1%} braking, {1-y.mean():.1%} coasting)")

    # ── 2. Build ──────────────────────────────────────────────────────────────
    from scenarios.emergency_braking.nn_controller import BrakingMLP

    print("\nBuilding MLP with input normalisation …")
    mlp = BrakingMLP()
    mlp.build(X_adapt=X)   # fits Normalization layer on training data
    mlp.summary()

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\nTraining …")
    history = mlp.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    # ── 4. Report ─────────────────────────────────────────────────────────────
    best_epoch    = int(np.argmin(history.history["val_loss"])) + 1
    final_val_loss = min(history.history["val_loss"])
    final_val_mae  = history.history["val_mae"][best_epoch - 1]

    print(f"\n{'─'*40}")
    print(f"Best epoch      : {best_epoch}")
    print(f"Val loss (BCE)  : {final_val_loss:.6f}")
    print(f"Val MAE         : {final_val_mae:.4f}  "
          f"({'good' if final_val_mae < 0.05 else 'acceptable' if final_val_mae < 0.15 else 'recheck training'})")
    print(f"{'─'*40}")

    # ── 5. Save model ─────────────────────────────────────────────────────────
    mlp.save(MODEL_PATH)

    # ── 6. Save loss curves ───────────────────────────────────────────────────
    if plot:
        _save_training_plot(history, best_epoch)

    print("\nDone. Run the pipeline with use_nn=True to test the trained controller.")


def _save_training_plot(history, best_epoch: int) -> None:
    """Save train/val loss and MAE curves to MODELS_DIR/training_curves.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0f1117")

        for ax in axes:
            ax.set_facecolor("#1a1d2e")
            ax.tick_params(colors="#9ca3af")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3148")

        epochs_range = range(1, len(history.history["loss"]) + 1)

        # Loss
        axes[0].plot(epochs_range, history.history["loss"],
                     color="#7c6ff7", label="train", lw=1.5)
        axes[0].plot(epochs_range, history.history["val_loss"],
                     color="#22c55e", label="val", lw=1.5)
        axes[0].axvline(best_epoch, color="#ef4444", ls="--", lw=1, label=f"best={best_epoch}")
        axes[0].set_title("Binary Cross-Entropy Loss", color="#e2e8f0")
        axes[0].set_xlabel("Epoch", color="#9ca3af")
        axes[0].legend(facecolor="#1a1d2e", labelcolor="#e2e8f0", edgecolor="#2d3148")

        # MAE
        axes[1].plot(epochs_range, history.history["mae"],
                     color="#7c6ff7", label="train", lw=1.5)
        axes[1].plot(epochs_range, history.history["val_mae"],
                     color="#22c55e", label="val", lw=1.5)
        axes[1].axvline(best_epoch, color="#ef4444", ls="--", lw=1)
        axes[1].set_title("Mean Absolute Error", color="#e2e8f0")
        axes[1].set_xlabel("Epoch", color="#9ca3af")
        axes[1].legend(facecolor="#1a1d2e", labelcolor="#e2e8f0", edgecolor="#2d3148")

        plt.tight_layout()
        out = os.path.join(MODELS_DIR, "training_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        plt.close()
        print(f"Loss curves saved → {out}")
    except Exception as e:
        print(f"(Could not save plot: {e})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BrakingMLP via behavioral cloning.")
    parser.add_argument("--n",           type=int,   default=N_TRAIN, help="Number of training trajectories")
    parser.add_argument("--seed",        type=int,   default=SEED)
    parser.add_argument("--epochs",      type=int,   default=50,      help="Max training epochs")
    parser.add_argument("--batch-size",  type=int,   default=2048,    help="Mini-batch size")
    parser.add_argument("--regen",       action="store_true",         help="Force dataset regeneration")
    parser.add_argument("--dataset-only",action="store_true",         help="Only generate dataset, skip training")
    parser.add_argument("--no-plot",     action="store_true",         help="Skip saving loss curve plot")
    args = parser.parse_args()

    if args.dataset_only:
        generate_dataset(n=args.n, seed=args.seed, save=True)
    else:
        train(
            n=args.n,
            seed=args.seed,
            force_regenerate=args.regen,
            epochs=args.epochs,
            batch_size=args.batch_size,
            plot=not args.no_plot,
        )
