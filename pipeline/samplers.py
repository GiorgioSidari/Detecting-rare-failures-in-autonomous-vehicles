"""
Sampling designs behind one interface, shared by every pipeline module.

Two designs are available. **LHS** (Latin Hypercube) is stratified: to draw n
points it splits each axis into n equiprobable strata and takes exactly one
value per stratum, permuting the strata independently across axes. **random**
draws i.i.d. uniform points. Both expose the same three helpers, so an algorithm
can be run under either design without changing a line of it.

    from pipeline.samplers import get_sampler
    s = get_sampler("lhs")        # stratified
    s = get_sampler("random")     # i.i.d. uniform

    u     = s.unit(n=64, d=9, seed=0)                 # (64, 9) in [0, 1)
    theta = s.bounded(64, lower, upper, seed=0)       # scaled to the bounds
    theta = s.from_dists(64, dists, seed=0)           # through the ODD marginals

All three take an explicit ``seed``, so a run is reproducible and a multi-seed
comparison can vary the seed alone.
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

    Subclasses implement :meth:`unit` only; the mapping to physical bounds
    (:meth:`bounded`) and through the operational marginals (:meth:`from_dists`)
    is shared by every design.
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    def unit(self, n: int, d: int, seed: int | None = None) -> np.ndarray:
        """Return (n, d) points in [0, 1)."""

    def bounded(self, n: int, lower: np.ndarray, upper: np.ndarray,
                seed: int | None = None) -> np.ndarray:
        """(n, d) points linearly scaled onto [lower, upper] — a uniform ODD."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        return scale(self.unit(n, len(lower), seed), lower, upper)

    def from_dists(self, n: int, dists: list, seed: int | None = None) -> np.ndarray:
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
