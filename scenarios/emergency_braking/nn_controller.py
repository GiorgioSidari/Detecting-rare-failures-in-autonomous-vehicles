from __future__ import annotations

"""
MLP controller for the Emergency Braking scenario.

Replaces the hardcoded physics with a learned braking policy trained via
behavioral cloning from the optimal (physics-based) controller.

Architecture
------------
Input  : [velocity (m/s), distance_to_obstacle (m), elapsed_time (s)]  — shape (3,)
Hidden : Dense(64, ReLU) → Dense(32, ReLU)
Output : braking_force in [0, 1]  — shape (1,)  via Sigmoid

Why Sigmoid output?
    The optimal policy is binary (0 or 1), but Sigmoid lets the network output
    a continuous value. This means it will output ~0.7 where it "should" output 1.0
    in difficult conditions — exactly the imprecision that produces rare failures.

Input normalisation
    Raw features have very different scales (velocity ~5-50, distance ~0-100, time ~0-10).
    A normalisation layer (or manual z-score) speeds up training significantly.
    We use a Keras Normalization layer adapted to the training data mean/variance.
"""

import os
import numpy as np


class BrakingMLP:
    """
    Lightweight wrapper around a Keras Sequential model.

    Usage
    -----
    # Training (done once)
    mlp = BrakingMLP()
    mlp.build()
    mlp.fit(X_train, y_train)
    mlp.save("models/emergency_braking_mlp.keras")

    # Inference (at every simulation timestep)
    mlp = BrakingMLP(model_path="models/emergency_braking_mlp.keras")
    forces = mlp.predict(states)   # states: (N, 3)
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path is not None:
            self.load(model_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Architecture
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, X_adapt: np.ndarray | None = None) -> None:
        """
        Build the Keras Sequential model.

        Parameters
        ----------
        X_adapt : (M, 3) optional — training data used to fit the normalisation
                  layer. If None, normalisation is skipped (not recommended).

        Architecture
        ------------
        Normalization  (optional, fitted on training data)
        Dense(64, ReLU)
        Dense(32, ReLU)
        Dense(1, Sigmoid)
        """
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Normalization

        layers = []

        if X_adapt is not None:
            norm = Normalization(input_shape=(3,))
            norm.adapt(X_adapt.astype(np.float32))
            layers.append(norm)

        layers += [
            Dense(64, activation="relu",
                  input_shape=(3,) if X_adapt is None else None,
                  kernel_initializer="he_normal"),
            Dense(32, activation="relu",
                  kernel_initializer="he_normal"),
            Dense(1,  activation="sigmoid"),
        ]

        self.model = Sequential(layers, name="braking_mlp")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",   # better than MSE for 0/1 labels
            metrics=["mae"],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 2048,
        validation_split: float = 0.1,
        verbose: int = 1,
    ):
        """
        Train the model via behavioral cloning.

        Parameters
        ----------
        X              : (M, 3)  features  [velocity, dist_to_obs, time]
        y              : (M, 1)  labels    0.0 or 1.0
        epochs         : training epochs (50 is enough for this simple task)
        batch_size     : 2048 is fast with 5M rows; reduce if you hit OOM
        validation_split: fraction held out for val loss monitoring
        verbose        : 1 = progress bar, 0 = silent, 2 = one line per epoch

        Returns
        -------
        history : Keras History object (loss curves accessible via .history)
        """
        if self.model is None:
            raise RuntimeError("Call build() before fit().")

        import tensorflow as tf

        callbacks = [
            # Stop early if val_loss doesn't improve for 5 epochs
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            # Halve LR if val_loss plateaus for 3 epochs
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        return self.model.fit(
            X.astype(np.float32),
            y.astype(np.float32),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict braking force for a batch of states.

        Parameters
        ----------
        state : (N, 3)  — [velocity, distance_to_obstacle, elapsed_time]
                          or (3,) for a single state (auto-expanded)

        Returns
        -------
        braking_force : (N,)  float32 in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or fit() first.")

        single = state.ndim == 1
        if single:
            state = state[np.newaxis, :]   # (1, 3)

        force = self.model(
            state.astype(np.float32), training=False
        ).numpy().flatten()                # (N,)

        return force[0] if single else force

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, model_path: str) -> None:
        """Save full model (architecture + weights + normalisation) to disk."""
        if self.model is None:
            raise RuntimeError("Nothing to save — model not built.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved → {model_path}")

    def load(self, model_path: str) -> None:
        """Load a previously saved model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python -m scenarios.emergency_braking.train` first."
            )
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded ← {model_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print Keras model summary."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        self.model.summary()

    @property
    def is_ready(self) -> bool:
        """True if the model is built and ready for predict()."""
        return self.model is not None
