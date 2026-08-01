"""
Severity axis: rank the worst failures.

Definition: the bottom-k% of safety margins among the failures — the cases that
crashed the hardest (the extreme tail of the outcome distribution). This is
DISTINCT from the rarity axis (the probability P(failure) under the ODD),
estimated in orchestrator.run / rare_event / active_boundary.
"""
from __future__ import annotations

import numpy as np


def find_severe_failures(
    safety_margins: np.ndarray,
    failures: np.ndarray,
    fraction: float = 0.05,
    tiebreak: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return indices of the most severe (worst) failures.

    Parameters
    ----------
    safety_margins : (N,)  — lower = worse
    failures       : (N,)  — binary, 1 = failure
    fraction       : keep the worst `fraction` of all failures
                     e.g. 0.05 = bottom 5%

    Returns
    -------
    rare_idx : 1-D array of indices into the original N samples,
               sorted from worst to least-bad.
    """
    failure_idx = np.where(failures == 1)[0]

    if len(failure_idx) == 0:
        return np.array([], dtype=int)

    margins_sub = safety_margins[failure_idx]
    if tiebreak is not None:
        # Primary: margin ascending (more negative = worse). Secondary: 'tiebreak' ascending.
        # When the QoI saturates (car off-road, XTE capped) many failures share the same margin;
        # we use survival time (n. steps) as the tie-breaker, so among equivalent crashes the ones
        # that leave the lane earlier rank as more severe.
        tb_sub = np.asarray(tiebreak, dtype=float)[failure_idx]
        order = np.lexsort((tb_sub, margins_sub))
    else:
        order = np.argsort(margins_sub)

    sorted_by_margin = failure_idx[order]
    k = max(1, int(np.ceil(len(sorted_by_margin) * fraction)))
    return sorted_by_margin[:k]


def summarise_rare_params(
    rare_params: np.ndarray,
    param_names: list[str],
) -> list[dict]:
    """
    Build a list of dicts (one per rare failure) for the LLM prompt
    and the frontend table.

    Returns
    -------
    [{"index": i, "params": {"initial_speed": 38.2, ...}}, ...]
    """
    result = []
    for i, row in enumerate(rare_params):
        result.append({
            "index": i,
            "params": {name: round(float(val), 4)
                       for name, val in zip(param_names, row)},
        })
    return result
