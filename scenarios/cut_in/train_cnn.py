from __future__ import annotations

"""
Trains the cut_in CNN on the CARLA dataset collected by
collect_carla_dataset.py — mirrors scenarios/emergency_braking/train_cnn.py.

Usage:
    python -m scenarios.cut_in.train_cnn
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulators.common.camera_dataset import CameraDataGenerator, load_archive_into_dataset, preprocess  # noqa: E402
from simulators.common.pilotnet import train_pilotnet  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_carla")
ARCHIVE_PATH = os.path.join(DATASET_DIR, "cut_in_carla.npz")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_NAME = "cut_in_cnn"
STABLE_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.h5")

# Same camera setup as emergency_braking (carla_episode.CAMERA_WIDTH/HEIGHT = 320x160).
CARLA_CROP_TOP = 40
TARGET_WIDTH = 200
TARGET_HEIGHT = 66
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, 3)


def _crop(image):
    return image[CARLA_CROP_TOP:, :, :]


def _preprocess(image):
    return preprocess(image, crop_fn=_crop, width=TARGET_WIDTH, height=TARGET_HEIGHT)


class _CutInDataGenerator(CameraDataGenerator):
    def __init__(self, X, y, batch_size: int = 64, is_training: bool = True):
        super().__init__(
            X=X, y=y,
            preprocess_fn=_preprocess,
            input_shape=INPUT_SHAPE,
            batch_size=batch_size,
            is_training=is_training,
            # braking_force doesn't depend on left/right — see
            # scenarios/emergency_braking/train_cnn.py for why this must be False.
            steering_label=False,
        )


def train(epochs: int = 30, batch_size: int = 64, seed: int = 42, archive_path: str = ARCHIVE_PATH) -> None:
    if not os.path.exists(archive_path):
        raise FileNotFoundError(
            f"Dataset non trovato in {archive_path}. "
            "Esegui prima: python -m scenarios.cut_in.collect_carla_dataset"
        )

    X_train, X_val, y_train, y_val = load_archive_into_dataset([archive_path], seed=seed, test_split=0.15)
    print(f"Train: {len(X_train)} frame  |  Val: {len(X_val)} frame")
    print(f"Frazione frenata — train: {y_train.mean():.1%}  val: {y_val.mean():.1%}")

    model, history = train_pilotnet(
        X_train, X_val, y_train, y_val,
        save_path=MODELS_DIR,
        model_name=MODEL_NAME,
        input_shape=INPUT_SHAPE,
        output_dim=1,
        data_generator_cls=_CutInDataGenerator,
        output_activation="sigmoid",
        loss="binary_crossentropy",
        metrics=["mae"],
        learning_rate=1e-4,
        nb_epoch=epochs,
        batch_size=batch_size,
        early_stopping_patience=5,
    )

    best_epoch = int(min(range(len(history.history["val_loss"])), key=lambda i: history.history["val_loss"][i]))
    print(f"\nMiglior epoca: {best_epoch + 1}  "
          f"val_loss={history.history['val_loss'][best_epoch]:.4f}  "
          f"val_mae={history.history['val_mae'][best_epoch]:.4f}")

    model.save(STABLE_MODEL_PATH)
    print(f"Modello salvato (path stabile) → {STABLE_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allena la CNN cut_in sul dataset CARLA.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archive", default=ARCHIVE_PATH)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, archive_path=args.archive)
