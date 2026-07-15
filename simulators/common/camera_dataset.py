from __future__ import annotations

"""
Shared camera-frame dataset utilities, generalized from
opensbt-core/Simulator/lanekeeping/self_driving/utils/dataset_utils.py.

The vendored version branches preprocessing/crop on a fixed env_name
(donkey/udacity/beamng). This version takes the crop as a caller-supplied
callable instead, so emergency_braking/cut_in (both driven by a new
simulator, not one of the three above) don't need a new hardcoded branch —
each scenario just passes its own crop function.

Archive format (.npz) is unchanged from the vendored version so the same
mental model applies: 'observations' = array of frames, 'actions' = array of
expert labels. These are frame sequences, not video files.
"""

import os
from typing import Callable, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize the image to the input shape used by the network model."""
    return cv2.resize(image, (width, height), cv2.INTER_AREA)


def bgr2yuv(image: np.ndarray) -> np.ndarray:
    """Convert the image from BGR to YUV (what the NVIDIA model expects)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def preprocess(
    image: np.ndarray,
    crop_fn: Callable[[np.ndarray], np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """crop -> resize -> BGR2YUV, mirroring dataset_utils.preprocess()."""
    image = crop_fn(image)
    image = resize(image, width=width, height=height)
    image = bgr2yuv(image)
    return image


def random_flip(image: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly flip the image left<->right; y[0] (steering-like) is negated."""
    if np.random.rand() < 0.5:
        image = cv2.flip(src=image, flipCode=1)
        y[0] = -y[0]
    return image, y


def random_translate(
    image: np.ndarray, y: np.ndarray, range_x: int = 100, range_y: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly shift the image and adjust y[0] proportionally to the shift."""
    if np.random.rand() < 0.5:
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        y[0] += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
    return image, y


def random_brightness(image: np.ndarray) -> np.ndarray:
    """Randomly adjust brightness (HSV value channel)."""
    if np.random.rand() < 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def augment(image: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """flip -> translate -> brightness, adjusting the label in place."""
    image, y = random_flip(image, y)
    image, y = random_translate(image, y)
    image = random_brightness(image)
    return image, y


def save_archive(observations: List[np.ndarray], actions: List[np.ndarray], archive_path: str) -> None:
    """
    Save a (frames, labels) dataset as a compressed .npz — a sequence of
    numpy frames, not a video file. Mirrors opensbt-core's save_archive().
    """
    os.makedirs(os.path.dirname(os.path.abspath(archive_path)), exist_ok=True)
    np.savez_compressed(
        archive_path,
        observations=np.asarray(observations),
        actions=np.asarray(actions),
    )


def load_archive_into_dataset(
    archive_paths: List[str],
    seed: int,
    test_split: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate one or more archives, then train/val split."""
    obs, actions = [], []
    for path in archive_paths:
        z = np.load(path)
        obs.append(z["observations"])
        actions.append(z["actions"])
    X = np.concatenate(obs)
    y = np.concatenate(actions)
    return train_test_split(X, y, test_size=test_split, random_state=seed)


class CameraDataGenerator(keras.utils.Sequence):
    """
    Keras Sequence serving (frame, label) batches with on-the-fly
    preprocessing/augmentation, generalized from opensbt-core's DataGenerator
    (env_name branching removed — the caller supplies preprocess_fn).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        input_shape: Tuple[int, int, int],
        batch_size: int = 128,
        is_training: bool = True,
        augment_prob: float = 0.5,
    ):
        self.X = X
        self.y = y
        self.preprocess_fn = preprocess_fn
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.is_training = is_training
        self.augment_prob = augment_prob
        self.indexes = np.arange(len(X))
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index: int):
        batch_idx = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        out_dim = 1 if np.ndim(self.y[0]) == 0 else len(self.y[0])
        X_batch = np.empty((self.batch_size, *self.input_shape))
        y_batch = np.empty((self.batch_size, out_dim))

        for i, idx in enumerate(batch_idx):
            x_item = self.X[idx]
            y_item = np.atleast_1d(self.y[idx]).astype(np.float64).copy()

            if self.is_training and np.random.rand() < self.augment_prob:
                x_item, y_item = augment(x_item, y_item)

            X_batch[i] = self.preprocess_fn(x_item)
            y_batch[i] = y_item

        return X_batch, y_batch
