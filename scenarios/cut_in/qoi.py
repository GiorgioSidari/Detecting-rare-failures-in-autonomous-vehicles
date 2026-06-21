"""
QoI for the Cut-In scenario.

Safety metric: minimum longitudinal gap reached during the entire manoeuvre
               while the cutter is in the ego's lane (lateral_gap < LANE_WIDTH/2).

Positive → ego always kept a safe distance
Negative → ego collided or came within IMPACT_THRESHOLD metres
"""

import numpy as np

IMPACT_THRESHOLD = 1.5   # metres
LANE_WIDTH       = 3.5   # metres


def compute_min_gap(
    trajectories: np.ndarray,
    safety_buffer: float = IMPACT_THRESHOLD,
) -> np.ndarray:
    """
    Parameters
    ----------
    trajectories : (N, T, 2)  last dim = [longitudinal_gap, lateral_gap]

    Returns
    -------
    safety_margins : (N,)
        min longitudinal_gap (while cutter is in lane) minus safety_buffer
    """
    long_gap = trajectories[:, :, 0]   # (N, T)
    lat_gap  = trajectories[:, :, 1]   # (N, T)

    in_lane_mask = lat_gap < (LANE_WIDTH / 2)   # cutter is in ego lane

    # Where cutter is not yet in lane, set gap to +inf so min ignores it
    masked_gap = np.where(in_lane_mask, long_gap, np.inf)
    min_gap = masked_gap.min(axis=1)   # (N,)

    # If cutter never entered the lane (all inf), scenario didn't trigger
    min_gap = np.where(np.isinf(min_gap), np.inf, min_gap)

    return min_gap - safety_buffer


def failure_indicator(safety_margins: np.ndarray) -> np.ndarray:
    """1 if failure (collision or near-miss), 0 if safe."""
    return (safety_margins < 0.0).astype(float)
