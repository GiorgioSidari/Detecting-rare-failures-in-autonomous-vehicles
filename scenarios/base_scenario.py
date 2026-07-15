"""
Base interface for all scenarios.

Every scenario must implement this interface so that the orchestrator,
the API and the frontend can treat all scenarios uniformly.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseScenario(ABC):
    """
    A scenario bundles together:
      - a simulator  (physics or NN-controlled)
      - parameter bounds
      - a QoI function
      - metadata for the frontend (name, param labels, etc.)
    """

    # --- Metadata (override as class attributes) ---
    name: str = ""
    description: str = ""

    # --- Abstract interface ---

    @abstractmethod
    def param_bounds(self) -> dict:
        """
        Physical bounds for the controllable parameters sampled via LHS.

        Returns
        -------
        dict with keys:
            'names'  : list[str]       — human-readable labels for the frontend
            'lower'  : np.ndarray (d,) — lower bounds
            'upper'  : np.ndarray (d,) — upper bounds
        """
        pass

    @abstractmethod
    def run_simulation(self, params: np.ndarray) -> np.ndarray:
        """
        Run N simulations in batch.

        Parameters
        ----------
        params : (N, d)  — one row per scenario sample

        Returns
        -------
        trajectories : (N, T, D)
            T = number of timesteps
            D = state dimension (e.g. 2 for [position, velocity])
        """
        pass

    @abstractmethod
    def compute_qoi(self, trajectories: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute the scalar safety metric for each trajectory.

        Parameters
        ----------
        trajectories : (N, T, D)
        params       : (N, d)   — needed when QoI depends on input params
                                  (e.g. detection_distance in emergency braking)

        Returns
        -------
        safety_margins : (N,)
            Positive → safe, Negative → failure
        """
        pass

    @abstractmethod
    def failure_threshold(self) -> float:
        """
        Scalar threshold below which a safety_margin is a failure.
        Default 0.0 works for most scenarios.
        """
        pass

    # --- Concrete helpers (shared by all scenarios) ---

    def is_failure(self, safety_margins: np.ndarray) -> np.ndarray:
        """Binary failure indicator: 1 = failure, 0 = safe."""
        return (safety_margins < self.failure_threshold()).astype(float)

    def nominal_params(self) -> np.ndarray:
        """
        Mid-point of the parameter space — used as the 'ideal' reference
        trajectory shown in the frontend visualisation.
        """
        bounds = self.param_bounds()
        return ((bounds["lower"] + bounds["upper"]) / 2.0).reshape(1, -1)

    def pod_channels(self) -> list[int] | None:
        """
        Trajectory channels (last axis of run_simulation's output) fed to the
        POD embedder. None (default) means "use every channel" — correct for
        scenarios whose D channels are all part of the state (e.g. emergency
        braking's [position, velocity]). Override when only a subset is
        meaningful for shape-based embedding (e.g. lane_keeping drops the
        redundant y channel — see scenarios/lane_keeping/config.py).
        """
        return None
