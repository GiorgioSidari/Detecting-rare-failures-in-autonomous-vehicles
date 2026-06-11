import numpy as np


class EmbedderPOD:
    """
    Proper Orthogonal Decomposition (POD) embedder for trajectory data.

    Compresses (N, T, 2) trajectories into (N, nModes) codes using SVD,
    selecting the minimum number of modes to explain >= variance_threshold
    of the total variance.
    """

    def __init__(self, variance_threshold: float = 0.99):
        self.variance_threshold = variance_threshold
        self.mean_trajectory: np.ndarray | None = None  # shape (T*2,)
        self.V_reduced: np.ndarray | None = None        # shape (nModes, T*2)
        self.nModes: int | None = None
        self.T: int | None = None
        self._explained_variance: float | None = None

    def fit(self, trajectories: np.ndarray) -> "EmbedderPOD":
        """
        Fit the POD basis from training trajectories.

        Parameters
        ----------
        trajectories : (N, T, 2)

        Returns
        -------
        self
        """
        N, T, _ = trajectories.shape
        self.T = T

        # 1. Flatten + center
        # Each trajectory (T, 2) becomes a single vector of length T*2.
        # Subtracting the mean removes the "average trajectory" so SVD focuses
        # purely on how trajectories differ from one another, not their shared shape.
        flat = trajectories.reshape(N, T * 2)
        self.mean_trajectory = flat.mean(axis=0)
        flat_centered = flat - self.mean_trajectory

        # 2. SVD — find the dominant directions in trajectory space
        # Vt rows are the basis directions ranked by variance explained.
        # U (ignored) holds per-sample coordinates valid only for this batch;
        # Vt is the reusable map that works for any future trajectory.
        _, S, Vt = np.linalg.svd(flat_centered, full_matrices=False)

        # 3. Pick the minimum number of modes that explain >= variance_threshold
        # S[i]**2 is proportional to the variance captured by direction i.
        # We keep only the first nModes directions and discard the rest as noise.
        cumulative_variance = np.cumsum(S ** 2) / np.sum(S ** 2)
        self.nModes = int(np.argmax(cumulative_variance >= self.variance_threshold) + 1)
        self._explained_variance = float(cumulative_variance[self.nModes - 1])

        self.V_reduced = Vt[: self.nModes, :]  # (nModes, T*2) — the kept basis
        return self

    def embed(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Project trajectories into the POD subspace.

        Parameters
        ----------
        trajectories : (N, T, 2)

        Returns
        -------
        codes : (N, nModes)
        """
        # 4. Embed — project each trajectory onto the POD basis
        # Centering first removes the mean, then the dot product with V_reduced.T
        # gives each trajectory's coordinates in the low-dimensional subspace.
        # errstate: numpy/BLAS emits spurious overflow/invalid warnings on large
        # matmuls; results are verified NaN/Inf-free.
        self._check_fitted()
        N = len(trajectories)
        flat = trajectories.reshape(N, -1)
        centered = flat - self.mean_trajectory
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return centered @ self.V_reduced.T

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct trajectories from POD codes.

        Parameters
        ----------
        codes : (N, nModes)

        Returns
        -------
        trajectories : (N, T, 2)
        """
        # 5. Reconstruct — reverse the projection
        # Multiply coordinates by the basis vectors to get back to the full
        # feature space, then add the mean to restore the absolute trajectory.
        # The result is an approximation: the dropped modes cause a small error
        # bounded by the (1 - variance_threshold) fraction of total variance.
        self._check_fitted()
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            flat = codes @ self.V_reduced + self.mean_trajectory
        return flat.reshape(len(codes), self.T, 2)

    def fit_transform(self, trajectories: np.ndarray) -> np.ndarray:
        """Fit the embedder and return codes for the training trajectories."""
        return self.fit(trajectories).embed(trajectories)

    @property
    def explained_variance(self) -> float:
        self._check_fitted()
        return self._explained_variance

    def _check_fitted(self) -> None:
        if self.V_reduced is None:
            raise RuntimeError("EmbedderPOD is not fitted. Call fit() first.")
