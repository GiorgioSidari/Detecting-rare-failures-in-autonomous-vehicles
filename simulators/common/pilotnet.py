from __future__ import annotations

"""
Shared PilotNet (NVIDIA end-to-end) CNN builder, generalized from
opensbt-core/Simulator/lanekeeping/self_driving/autopilot_model.py so that
emergency_braking and cut_in can train a camera-frame CNN without duplicating
the architecture per scenario.

Unlike the vendored lanekeeping version (which branches on env_name to select
a simulator-specific crop), this module is simulator-agnostic: callers pass
their own preprocessing via `simulators.common.camera_dataset` and only the
input shape / output layer differ between scenarios.
"""

import datetime
import os
from typing import Tuple

import numpy as np


def build_pilotnet(
    input_shape: Tuple[int, int, int],
    output_dim: int,
    output_activation: str | None = None,
    keep_probability: float = 0.5,
):
    """
    Build the "Modified NVIDIA model" (PilotNet, Bojarski et al. 2016) used by
    lanekeeping, generalized to an arbitrary output layer:

      - lane_keeping-style steering regression : output_dim=1, activation=None
      - emergency_braking braking_force        : output_dim=1, activation="sigmoid"
      - cut_in steering (+ optional brake)      : output_dim=1 or 2, activation=None

    Parameters
    ----------
    input_shape : (height, width, channels) of the *preprocessed* frame.
    output_dim  : number of output neurons.
    output_activation : Keras activation name for the output layer, or None
                        for a linear output (regression).
    keep_probability  : dropout keep-probability after the conv stack.
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))
    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Dropout(keep_probability))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(output_dim, activation=output_activation))

    return model


def train_pilotnet(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    save_path: str,
    model_name: str,
    input_shape: Tuple[int, int, int],
    output_dim: int,
    data_generator_cls,
    output_activation: str | None = None,
    loss: str = "mean_squared_error",
    metrics: list[str] | None = None,
    keep_probability: float = 0.5,
    learning_rate: float = 1e-4,
    nb_epoch: int = 200,
    batch_size: int = 128,
    early_stopping_patience: int = 5,
    save_best_only: bool = True,
):
    """
    Behavioral-cloning training loop shared by emergency_braking/cut_in CNNs.

    `data_generator_cls` must be a `simulators.common.camera_dataset.CameraDataGenerator`
    (or compatible Keras Sequence) already configured with the scenario's own
    preprocessing/augmentation — this function only wires model + callbacks + fit.

    Returns (model, history).
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam

    os.makedirs(save_path, exist_ok=True)
    model = build_pilotnet(
        input_shape=input_shape,
        output_dim=output_dim,
        output_activation=output_activation,
        keep_probability=keep_probability,
    )

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{model_name}-{timestamp}.h5"
    checkpoint = ModelCheckpoint(
        os.path.join(save_path, filename),
        monitor="val_loss", verbose=0, save_best_only=save_best_only, mode="auto",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_patience)

    model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate), metrics=metrics or [])

    train_gen = data_generator_cls(X=X_train, y=y_train, batch_size=batch_size, is_training=True)
    val_gen = data_generator_cls(X=X_val, y=y_val, batch_size=batch_size, is_training=False)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=nb_epoch,
        callbacks=[checkpoint, early_stopping],
        verbose=1,
    )

    return model, history
