"""
Sampling strategies, shared by every pipeline module.

Why this module exists
----------------------
The pipeline currently seeds its exploration with a Latin Hypercube (LHS): a
*stratified* design that guarantees every 1-D projection of the parameter space
is covered exactly once per stratum. The claim we want to validate empirically
is that stratification is what lets the search reach the rare failure regions:
plain uniform random sampling leaves holes, and with a small simulation budget
those holes are exactly where the rare failures hide.

To test that claim we need the two designs behind one interface, so that the
*algorithm* (active boundary, cross-entropy) stays byte-for-byte identical and
the *only* thing that changes is how points are drawn. That is what
:class:`BaseSampler` provides.

    from pipeline.samplers import get_sampler
    s = get_sampler("lhs")        # stratified
    s = get_sampler("random")     # plain uniform random search

    u     = s.unit(n=64, d=9, seed=0)                 # (64, 9) in [0, 1)
    theta = s.bounded(64, lower, upper, seed=0)       # scaled to the bounds
    theta = s.from_dists(64, dists, seed=0)           # ppf of the ODD (realistic)

All three helpers take an explicit ``seed`` so a run is reproducible, and so a
multi-seed comparison can vary only the seed while keeping everything else fixed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

# ppf(0) = -inf and ppf(1) = +inf for unbounded distributions: clip the unit
# samples just inside the open interval before any inverse-CDF transform.
_PPF_EPS = 1e-12


class BaseSampler(ABC):
    """
    A design of experiments over the d-dimensional unit cube.

    Subclasses only implement :meth:`unit`; the mapping to physical bounds
    (:meth:`bounded`) and to the operational distribution (:meth:`from_dists`)
    is shared, which is what keeps the LHS/random comparison honest.
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    def unit(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        """Return (n, d) points in [0, 1)."""

    def bounded(self, n: int, lower, upper, seed: int | None = None) -> np.ndarray:
        """(n, d) points linearly scaled onto [lower, upper] — a uniform ODD."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        return scale(self.unit(n, len(lower), seed), lower, upper)

    def from_dists(self, n: int, dists, seed: int | None = None) -> np.ndarray:
        """
        (n, d) points drawn from a product of frozen scipy distributions via the
        inverse CDF — the "realistic" ODD mapping used by the orchestrator.

        With an LHS design this is a stratified draw from the ODD; with the
        random design it degenerates to plain i.i.d. sampling from the ODD.
        """
        u = np.clip(self.unit(n, len(dists), seed), _PPF_EPS, 1.0 - _PPF_EPS)
        out = np.empty_like(u)
        for j, dist in enumerate(dists):
            out[:, j] = dist.ppf(u[:, j])
        return out

    def __repr__(self) -> str:      # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r})"


class LHSSampler(BaseSampler):
    """
    Latin Hypercube design (the current pipeline default).

    Each dimension is split into n equal-probability strata and every stratum is
    used exactly once, so the marginal coverage is guaranteed by construction and
    no region of any single axis is left unexplored. This is what gives the small
    budget a chance to land in the narrow parameter combinations that fail.
    """

    name = "lhs"
    description = "Latin Hypercube — stratified, one sample per stratum per axis"

    def __init__(self, scramble: bool = True):
        self.scramble = scramble

    def unit(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        return LatinHypercube(d=d, scramble=self.scramble, seed=seed).random(n=n)


class RandomSampler(BaseSampler):
    """
    Plain uniform random search — the baseline we compare LHS against.

    No stratification: with n points in d dimensions the design clusters and
    leaves gaps, and the probability of landing inside a failure region of
    volume v is 1 - (1 - v)^n. When v is small this is the design that misses
    the region, while LHS still probes every stratum of every axis.
    """

    name = "random"
    description = "Uniform i.i.d. random search — unstratified baseline"

    def unit(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        return np.random.default_rng(seed).random((n, d))


SAMPLERS: dict = {
    "lhs": LHSSampler,
    "random": RandomSampler,
}


def get_sampler(spec) -> BaseSampler:
    """
    Resolve a sampler from a name ("lhs" / "random"), a class, or an instance.

    Passing an instance through unchanged means callers can accept
    ``sampler=`` arguments without caring which form the user supplied.
    """
    if isinstance(spec, BaseSampler):
        return spec
    if isinstance(spec, type) and issubclass(spec, BaseSampler):
        return spec()
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in SAMPLERS:
            raise KeyError(f"Unknown sampler '{spec}'. Available: {sorted(SAMPLERS)}")
        return SAMPLERS[key]()
    raise TypeError(f"Cannot build a sampler from {spec!r}")


def discrepancy(u: np.ndarray) -> float:
    """
    Centered L2 discrepancy of a unit-cube design: lower = more uniform coverage.

    Reported by the comparison harness as the quantitative statement of "LHS
    covers the space better than random search", independent of any simulation.
    """
    from scipy.stats.qmc import discrepancy as _disc
    return float(_disc(np.asarray(u, dtype=float), method="CD"))
