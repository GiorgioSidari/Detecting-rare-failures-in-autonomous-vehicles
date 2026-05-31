# Quantity of Interest (QoI) for the emergency braking scenario.
#
# Safety margin: how much stopping distance remains before the obstacle.
# Failure threshold: 0.0 m — any penetration into the 2 m safety buffer counts as a failure.

import numpy as np


def compute_safety_margin(
    trajectories: np.ndarray,
    detection_distance,
    safety_buffer: float = 2.0,
) -> np.ndarray:
    """
    Compute the remaining safety margin for each trajectory.

    Parameters
    ----------
    trajectories : (N, T, 2)  last dim = [position (m), velocity (m/s)]
    detection_distance : float or (N,) array — obstacle distance per scenario (m)
    safety_buffer : minimum acceptable stopping clearance (m), default 2.0

    Returns
    -------
    safety_margins : (N,)
        Positive  →  vehicle stopped with margin to spare (safe)
        Negative  →  vehicle hit or overshot the obstacle (failure)
    """
    final_positions = trajectories[:, -1, 0]            # shape (N,)
    d = np.asarray(detection_distance, dtype=float)     # broadcast-safe
    return d - final_positions - safety_buffer


def failure_indicator(safety_margins: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Binary failure indicator: 1 if failure, 0 if safe.

    Parameters
    ----------
    safety_margins : (N,) output of compute_safety_margin
    threshold      : failure boundary (default 0.0)

    Returns
    -------
    indicators : (N,) float array of 0s and 1s
    """
    return (safety_margins < threshold).astype(float)