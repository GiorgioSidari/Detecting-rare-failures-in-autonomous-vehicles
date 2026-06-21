"""
Behavioral cloning trainer for the Cut-In LSTM controller.

TODO (Blocco 2b): implement after CutInSimulator is complete.

Dataset generation
------------------
For each trajectory from CutInSimulator:
  - At each timestep t, build the state window:
      window = states[max(0, t-SEQUENCE_LEN):t+1]   (padded if t < SEQUENCE_LEN)
  - Optimal action:
      braking_force      = 1.0 if cutter is in lane AND t >= reaction_delay else 0.0
      steering_correction = direction away from cutter, proportional to lateral_gap
"""

import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cut_in_lstm.keras")

N_TRAIN = 5_000
SEED    = 42


def generate_dataset(n: int = N_TRAIN, seed: int = SEED):
    """Returns X (M, SEQUENCE_LEN, 3), y (M, 2)."""
    raise NotImplementedError


def train(n: int = N_TRAIN, seed: int = SEED) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = generate_dataset(n=n, seed=seed)
    # TODO: build EvasiveLSTM, fit, save
    raise NotImplementedError


if __name__ == "__main__":
    train()
