"""
Emergency Braking simulator driven by the trained MLP controller.

How it differs from EmergencyBrakingSimulator
---------------------------------------------
The physics simulator uses:
    actual_decel = friction * 9.81 * braking_efficiency
    actual_delay = nominal_delay + LogNormal noise

This NNSimulator uses instead:
    braking_force = BrakingMLP.predict([velocity, dist_to_obstacle, t])  ∈ [0, 1]
    applied_decel = braking_force * friction * 9.81

The stochastic delay and braking_efficiency are intentionally removed.
The MLP's imperfect approximation of the optimal policy *is* the source of
variance. On unseen parameter combinations (especially edge cases like
high speed + low friction + long delay) the network may output:
  • a force too low  → insufficient braking → crash
  • braking too late → vehicle overshoots the obstacle

These systematic generalisation errors are the rare failures we want to detect.

Batched inference
-----------------
At each timestep t, we build a (N_active, 3) state matrix and call
BrakingMLP.predict() once — one forward pass per timestep for all active
trajectories. This is GPU-friendly and fast.
"""

import os
import numpy as np
from simulators.base_simulator import BaseSimulator
from scenarios.emergency_braking.nn_controller import BrakingMLP

# Default model path (relative to this file)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "emergency_braking_mlp.keras"
)


class EmergencyBrakingNNSimulator(BaseSimulator):
    """
    Drop-in replacement for EmergencyBrakingSimulator that uses a trained MLP
    to decide the braking force at each timestep.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras model (.keras).  Defaults to the standard
        location written by train.py.
    force_threshold : float  [0, 1]
        Any MLP output above this is treated as "braking" (used only for
        diagnostics; the physics always uses the raw continuous value).
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        force_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.controller      = BrakingMLP(model_path=model_path)
        self.force_threshold = force_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Core simulation
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, controllable_parameters: np.ndarray) -> np.ndarray:
        """
        Simulate N emergency braking scenarios using the trained MLP.

        Parameters
        ----------
        controllable_parameters : (N, 4)
            Columns: [initial_speed (m/s), friction_coefficient (-),
                      detection_distance (m), nominal_delay (s)]
            NOTE: nominal_delay is NOT passed to the MLP — it was only used
            during dataset generation to label timesteps. It is kept here so
            the signature stays identical to EmergencyBrakingSimulator.run().

        Returns
        -------
        trajectories : (N, T, 2)
            Last dim: [position (m), velocity (m/s)]
        """
        N = controllable_parameters.shape[0]

        initial_speeds      = controllable_parameters[:, 0]   # (N,)
        friction_coeffs     = controllable_parameters[:, 1]   # (N,)
        detection_distances = controllable_parameters[:, 2]   # (N,)
        # nominal_delay (col 3) is intentionally NOT used in the NN path

        # Maximum deceleration each vehicle is physically capable of.
        # braking_efficiency is not sampled separately; the NN's imprecision
        # plays that role implicitly.
        max_decels = friction_coeffs * 9.81   # (N,)  [m/s²]

        # ── State initialisation ──────────────────────────────────────────────
        trajectories = np.zeros((N, self.T, 2), dtype=np.float32)
        positions    = np.zeros(N,              dtype=np.float32)
        velocities   = initial_speeds.copy().astype(np.float32)
        active       = np.ones(N, dtype=bool)   # still need to integrate

        for t_idx in range(self.T):
            t = float(t_idx * self.dt)

            # Record current state
            trajectories[:, t_idx, 0] = positions
            trajectories[:, t_idx, 1] = velocities

            if not np.any(active):
                # All done — freeze remaining timesteps
                for t_rem in range(t_idx + 1, self.T):
                    trajectories[:, t_rem, 0] = positions
                    trajectories[:, t_rem, 1] = velocities
                break

            # ── Build state matrix for active trajectories ──────────────────
            active_idx  = np.where(active)[0]           # indices of active trajectories
            dist_active = np.clip(
                detection_distances[active_idx] - positions[active_idx], 0.0, None
            )                                             # (N_active,)
            vel_active  = velocities[active_idx]         # (N_active,)
            time_col    = np.full(len(active_idx), t, dtype=np.float32)

            states = np.stack([vel_active, dist_active, time_col], axis=1)  # (N_active, 3)

            # ── MLP forward pass ─────────────────────────────────────────────
            # Returns braking_force ∈ [0, 1] for each active trajectory.
            braking_forces = self.controller.predict(states)  # (N_active,)

            # ── Physics update ───────────────────────────────────────────────
            # applied_decel = force * max_decel  (continuous; not thresholded)
            applied_decels = braking_forces * max_decels[active_idx]  # (N_active,)

            new_v = np.maximum(
                0.0,
                vel_active - applied_decels * self.dt,
            )                                              # (N_active,)
            new_p = positions[active_idx] + new_v * self.dt

            positions[active_idx]  = new_p
            velocities[active_idx] = new_v

            # ── Deactivate stopped or crashed trajectories ───────────────────
            active[active_idx[new_p >= detection_distances[active_idx]]] = False
            active[active_idx[new_v == 0.0]] = False

        return trajectories

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter bounds (identical to physics simulator)
    # ──────────────────────────────────────────────────────────────────────────

    def ParamBounds(self) -> dict:
        return {
            "lower": np.array([5.0,  0.3,  10.0, 0.05]),
            "upper": np.array([50.0, 1.0, 100.0, 0.50]),
        }
