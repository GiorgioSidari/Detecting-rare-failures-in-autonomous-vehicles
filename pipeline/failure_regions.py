"""
Failure regions: extraction from a cloud of evaluated points, and the cost of
hitting one with a blind draw.

Input is an array of parameter points, each carrying its safety margin. The
failing points are normalised to the unit cube and clustered with DBSCAN, which
groups points of arbitrary shape without being told how many groups there are;
a point with no neighbour becomes a region of its own. The clustering radius
`eps` defaults to None, in which case :func:`estimate_eps` derives it from the
data with the k-distance heuristic.

Each region carries a bounding box, the axes it is localised on, and the
probability that a draw from the ODD falls inside it:

    p_hit_n = 1 - (1 - p)^n        probability that n blind samples hit it
    n_50    = log(0.5)/log(1-p)    samples for a 50/50 chance

The product runs only over the axes on which the region is LOCALISED -- those
whose observed span is clearly narrower than the span the same number of draws
would produce by chance (see `_constraining_axes`). Axes the region does not
constrain contribute a factor of 1. A region holding a single point constrains
nothing measurable and its `p_hit` is NaN.

The box containing a region is an outer approximation, so `p_hit` is an upper
bound and `n_50` a lower bound.

:func:`structure_score` answers a prior question: whether the failure cloud has
any joint structure at all, by comparing the observed neighbourhood density
with the one obtained after shuffling each axis independently.

Comparing several methods against one shared region map lives in
`pipeline.region_comparison`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# None = estimate the DBSCAN radius from the data. An explicit value only makes
# sense when you know the scale of your parameter space in the normalised cube.
DEFAULT_EPS = None
DEFAULT_MIN_SAMPLES = 2
# Half-width added to every bounding box so a small region is not zero-volume.
DEFAULT_PAD = 0.02
# An axis is REPORTED AS CONSTRAINING when the group's span on it is below this
# fraction of what n points would typically span. Scaling by n matters: three
# points cover on average half of an axis, so a flat "span < 0.5" rule is met by
# chance on about half the axes and a 3-point group comes out "constrained" on
# six dimensions out of nine. See _constraining_axes.
DEFAULT_MAX_SPAN = 0.5
# Below this many points a group gets no reported bounds at all.
DEFAULT_MIN_POINTS = 4
# Permutations used to test whether the failure cloud has any joint structure
# at all (see structure_score). 20 is plenty for a z-score.
DEFAULT_N_PERM = 20


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
@dataclass
class FailureRegion:
    """One connected region of the parameter space where the scenario fails."""

    region_id: int
    n_points: int                       # failing samples that define it
    centroid: np.ndarray                # (d,) in PHYSICAL units
    box_lower: np.ndarray               # (d,) bounding box, physical units
    box_upper: np.ndarray               # (d,)
    worst_margin: float                 # most negative margin inside
    mean_margin: float
    volume_fraction: float              # box volume / total box volume (unit cube)
    p_hit_uniform: float                # P(a uniform draw satisfies the constraints)
    p_hit_odd: float                    # P(an ODD draw satisfies them); NaN if n < 2
    is_singleton: bool                  # a lone failure, not a dense cluster
    param_names: list = field(default_factory=list)
    found_by: dict = field(default_factory=dict)   # method label -> n points it put here
    # Axes on which the region is genuinely localised — the ones its P(hit) is
    # computed over. Empty means the region spans the ODD and constrains nothing.
    constraining_axes: list = field(default_factory=list)

    # ── derived, blind-draw economics ───────────────────────────────────────
    def p_hit(self, weighting: str = "odd") -> float:
        p = self.p_hit_odd if weighting == "odd" else self.p_hit_uniform
        return float(p)

    @property
    def is_estimable(self) -> bool:
        """False when the region has too few points to measure any localisation."""
        return bool(np.isfinite(self.p_hit_odd))

    def p_hit_in_n(self, n: int, weighting: str = "odd") -> float:
        """Probability that n independent blind draws hit this region at least once."""
        p = self.p_hit(weighting)
        if not np.isfinite(p):
            return float("nan")
        if p <= 0.0:
            return 0.0
        return float(1.0 - (1.0 - p) ** int(n))

    def n_for_50pct(self, weighting: str = "odd") -> float:
        """Blind draws needed for an even chance of hitting the region."""
        p = self.p_hit(weighting)
        if not np.isfinite(p):
            return float("nan")
        if p <= 0.0:
            return float("inf")
        if p >= 1.0:
            return 1.0
        return float(np.log(0.5) / np.log(1.0 - p))

    def describe(self) -> str:
        """Centroid on every axis (the full location of the region)."""
        names = self.param_names or [f"p{j}" for j in range(len(self.centroid))]
        parts = [f"{n}={c:.2f}" for n, c in zip(names, self.centroid)]
        return ", ".join(parts)

    def describe_constraints(self) -> str:
        """
        The conditions that DEFINE the region: the interval on each constraining
        axis. This is the re-runnable characterisation of the failure — the other
        axes were free to take any value and the scenario still failed.
        """
        if not self.constraining_axes:
            return "no axis constrained (the region spans the ODD)"
        names = self.param_names or [f"p{j}" for j in range(len(self.centroid))]
        return ", ".join(
            f"{names[j]} in [{self.box_lower[j]:.2f}, {self.box_upper[j]:.2f}]"
            for j in self.constraining_axes)


def _to_unit(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = np.where((upper - lower) > 0, upper - lower, 1.0)
    return (np.asarray(theta, float) - lower) / span


def estimate_eps(unit_pts: np.ndarray, min_samples: int = DEFAULT_MIN_SAMPLES,
                 quantile: float = 0.5) -> float:
    """
    DBSCAN radius estimated from the data with the k-distance heuristic.

    For each point the distance to its k-th nearest neighbour is computed
    (k = `min_samples`), and the median of those distances is returned. That is
    the scale at which about half the points reach the density DBSCAN needs to
    seed a cluster, and it scales with the dimension of the space.

    Returns a radius in the normalised cube. With fewer than `min_samples + 1`
    points the estimate is not defined and sqrt(d)/4 is returned instead.
    """
    n, d = np.shape(unit_pts)
    if n <= min_samples:
        return float(np.sqrt(d) / 4.0)
    try:
        from sklearn.neighbors import NearestNeighbors
        k = min(min_samples, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(unit_pts)
        dist, _ = nn.kneighbors(unit_pts)
        kdist = dist[:, k]
    except Exception:                                        # pragma: no cover
        from scipy.spatial.distance import squareform, pdist
        D = squareform(pdist(unit_pts))
        np.fill_diagonal(D, np.inf)
        kdist = np.sort(D, axis=1)[:, min(min_samples, n - 1) - 1]
    eps = float(np.quantile(kdist, quantile))
    return eps if eps > 0 else float(np.sqrt(d) / 4.0)


def _cluster(unit_pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    DBSCAN labels, with every noise point promoted to its own singleton cluster.

    Density-based clustering suits failure regions: their number is unknown and
    their shape is rarely spherical. Promoting noise matters because an isolated
    failure at the edge of the ODD is precisely the rare event we are hunting.
    """
    if len(unit_pts) == 0:
        return np.empty(0, dtype=int)
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(unit_pts)
    except Exception:                                        # pragma: no cover
        labels = _greedy_cluster(unit_pts, eps)
    next_id = int(labels.max()) + 1 if (labels >= 0).any() else 0
    for i in np.where(labels < 0)[0]:
        labels[i] = next_id
        next_id += 1
    return labels


def _greedy_cluster(unit_pts: np.ndarray, eps: float) -> np.ndarray:
    """Single-link fallback used only when scikit-learn is unavailable."""
    n = len(unit_pts)
    labels = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack, labels[i] = [i], cid
        while stack:
            j = stack.pop()
            d = np.linalg.norm(unit_pts - unit_pts[j], axis=1)
            for k in np.where((d <= eps) & (labels < 0))[0]:
                labels[k] = cid
                stack.append(int(k))
        cid += 1
    return labels


def structure_score(unit_pts: np.ndarray, eps: float, min_samples: int = 2,
                    min_points: int = DEFAULT_MIN_POINTS,
                    n_perm: int = DEFAULT_N_PERM, seed: int = 0) -> dict:
    def _share(P):
        """
    Compare this cloud's clustering against the same data with shuffled axes.

    The statistic is the share of points that land in a group of at least
    `min_points`. It is computed on the real cloud and on copies whose axes have
    been shuffled independently, which preserves each parameter's marginal
    distribution and removes any joint structure.

    Returns {"observed", "null_mean", "null_std", "z", "has_structure"}, where `z`
    is the standardised difference between the observed share and the shuffled
    ones, and `has_structure` is z > 2.
    """
        lab = _cluster(P, eps=eps, min_samples=min_samples)
        _, counts = np.unique(lab, return_counts=True)
        return float(counts[counts >= min_points].sum() / max(len(P), 1))

    unit_pts = np.asarray(unit_pts, float)
    if len(unit_pts) < 2 * min_points:
        return {"observed": 0.0, "null_mean": 0.0, "null_std": 0.0,
                "z": 0.0, "has_structure": False}

    observed = _share(unit_pts)
    rng = np.random.default_rng(seed)
    d = unit_pts.shape[1]
    null = [_share(np.column_stack([rng.permutation(unit_pts[:, j])
                                    for j in range(d)]))
            for _ in range(max(2, n_perm))]
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
    z = (observed - mu) / sd if sd > 1e-12 else (0.0 if observed <= mu else 99.0)
    return {"observed": observed, "null_mean": mu, "null_std": sd,
            "z": float(z), "has_structure": bool(z > 2.0)}


def _constraining_axes(side_unit: np.ndarray, n_points: int,
                       max_span: float = DEFAULT_MAX_SPAN,
                       min_points: int = DEFAULT_MIN_POINTS) -> list:
    """
    The axes on which the group is localised.

    An axis qualifies when the group's points occupy less of the parameter's
    range than the threshold ``max_span * (n-1)/(n+1)``, where (n-1)/(n+1) is the
    expected span of n points spread uniformly over an axis. The scaling makes
    the requirement depend on the group's size: about 0.25 of the range for a
    3-point group, about 0.41 for a 10-point one.

    The returned axes are the ones :func:`_box_probability` multiplies over, and
    they describe where the observed failures sit; they are not established as
    necessary conditions for failure. DBSCAN produces compact groups by
    construction, so the narrowness of a group is not by itself evidence of
    structure. The uncircular measurements are the per-axis spread of the whole
    failure cloud and the overlap between arms, both reported separately.
    """
    if n_points < max(2, min_points):
        return []
    expected = (n_points - 1) / (n_points + 1)
    return [int(j) for j in np.where(np.asarray(side_unit) < max_span * expected)[0]]


def _box_probability(box_lo: np.ndarray, box_hi: np.ndarray, lower: np.ndarray,
                     upper: np.ndarray, dists: list, n_points: int,
                     max_span: float = DEFAULT_MAX_SPAN,
                     min_points: int = DEFAULT_MIN_POINTS) -> tuple:
    """
    (p_uniform, p_odd, constraining_axes) for one region.

    P(hit) is the probability that a blind draw falls inside the region's
    interval on every axis it is localised on, those axes being the ones
    :func:`_constraining_axes` returns. Axes the region does not constrain
    contribute a factor of 1.

    Both probabilities are analytic: the uniform one multiplies the box side
    lengths, the ODD one multiplies the spans in probability units read off the
    marginals. With fewer than 2 points no span is measurable and both are
    NaN.
    """
    span = np.where((upper - lower) > 0, upper - lower, 1.0)
    side_lin = np.clip((box_hi - box_lo) / span, 0.0, 1.0)

    # Span in probability units. Without an ODD the two coincide (uniform ODD).
    if dists is None:
        side_cdf = side_lin
    else:
        side_cdf = np.empty_like(side_lin)
        for j, dist in enumerate(dists):
            try:
                side_cdf[j] = float(dist.cdf(box_hi[j]) - dist.cdf(box_lo[j]))
            except Exception:                                # pragma: no cover
                side_cdf[j] = side_lin[j]
        side_cdf = np.clip(side_cdf, 0.0, 1.0)

    if n_points < max(2, min_points):
        return float("nan"), float("nan"), []

    axes = _constraining_axes(side_lin, n_points, max_span, min_points)
    if not axes:
        # Nothing about this region is statistically distinguishable from "n
        # points drawn from the ODD": every draw matches it.
        return 1.0, 1.0, []

    return float(np.prod(side_lin[axes])), float(np.prod(side_cdf[axes])), axes


@dataclass(frozen=True)
class _ClusterInputs:
    """
    Everything `_region_from_cluster` needs beyond the cluster itself.

    `idx` indexes the failing points inside `theta` / `margins`; `unit` holds
    those same points normalised to the unit cube; `span` is `upper - lower`
    with zero-width axes replaced by 1. The rest are the extraction options.
    """

    idx: np.ndarray
    unit: np.ndarray
    theta: np.ndarray
    margins: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    span: np.ndarray
    pad: float
    dists: object
    max_span: float
    min_points: int
    param_names: object
    owners: object


def _region_from_cluster(cid: int, sel: np.ndarray, inp: _ClusterInputs) -> FailureRegion:
    """One DBSCAN cluster turned into a FailureRegion (box, rarity, owners)."""
    members = inp.idx[sel]
    pts_u = inp.unit[sel]
    pts_p = inp.theta[members]
    m = inp.margins[members]

    box_lo_u = np.clip(pts_u.min(axis=0) - inp.pad, 0.0, 1.0)
    box_hi_u = np.clip(pts_u.max(axis=0) + inp.pad, 0.0, 1.0)
    box_lo = inp.lower + box_lo_u * inp.span
    box_hi = inp.lower + box_hi_u * inp.span

    p_uni, p_odd, axes = _box_probability(
        box_lo, box_hi, inp.lower, inp.upper, inp.dists, int(sel.sum()),
        inp.max_span, inp.min_points)

    found_by: dict = {}
    if inp.owners is not None:
        own = np.asarray(inp.owners)[members]
        for o in np.unique(own):
            found_by[str(o)] = int((own == o).sum())

    return FailureRegion(
        region_id=int(cid),
        n_points=int(sel.sum()),
        centroid=pts_p.mean(axis=0),
        box_lower=box_lo,
        box_upper=box_hi,
        worst_margin=float(m.min()),
        mean_margin=float(m.mean()),
        volume_fraction=float(np.prod(np.clip(box_hi_u - box_lo_u, 0.0, 1.0))),
        p_hit_uniform=p_uni,
        p_hit_odd=p_odd,
        is_singleton=bool(sel.sum() == 1),
        param_names=list(inp.param_names) if inp.param_names is not None else [],
        found_by=found_by,
        constraining_axes=axes,
    )


def extract_failure_regions(
    theta: np.ndarray,
    margins: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    threshold: float = 0.0,
    param_names=None,
    dists=None,
    eps: float | None = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    pad: float = DEFAULT_PAD,
    max_span: float = DEFAULT_MAX_SPAN,
    min_points: int = DEFAULT_MIN_POINTS,
    owners=None,
) -> list:
    """
    Cluster the failing points of one (or several, concatenated) run into regions.

    Parameters
    ----------
    theta, margins : (M, d) evaluated points and (M,) their safety margins.
    lower, upper   : parameter bounds used for normalisation.
    threshold      : margin below which a point is a failure.
    dists          : the scenario's ODD marginals (param_distributions), used for
                     the p_hit_odd column. Omit for a uniform-only analysis.
    eps            : DBSCAN radius in the normalised cube. None (the default)
                     estimates it from the data — the right choice unless you
                     know the scale of your space, since a fixed radius silently
                     breaks when the dimension changes.
    min_samples    : DBSCAN min_samples.
    pad            : half-width added to each box side (unit-cube fraction).
    max_span       : an axis is reported as constraining when the group covers
                     less than this fraction of its range (descriptive filter).
    min_points     : groups smaller than this get no reported bounds at all.
    owners         : optional (M,) array of method labels, so each region records
                     which methods reached it.

    Returns
    -------
    list[FailureRegion], sorted by rarity (rarest first).
    """
    theta = np.asarray(theta, float)
    margins = np.asarray(margins, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    span = np.where((upper - lower) > 0, upper - lower, 1.0)

    ok = np.isfinite(margins)
    fail_mask = ok & (margins < threshold)
    idx = np.where(fail_mask)[0]
    if len(idx) == 0:
        return []

    unit = _to_unit(theta[idx], lower, upper)
    if eps is None:
        eps = estimate_eps(unit, min_samples=min_samples)
    labels = _cluster(unit, eps=eps, min_samples=min_samples)

    inputs = _ClusterInputs(
        idx=idx, unit=unit, theta=theta, margins=margins,
        lower=lower, upper=upper, span=span, pad=pad, dists=dists,
        max_span=max_span, min_points=min_points,
        param_names=param_names, owners=owners)
    regions = [_region_from_cluster(cid, labels == cid, inputs)
               for cid in np.unique(labels)]

    # Rarest first; regions whose rarity is not estimable (single point) go last,
    # so the top of the report is always the part that carries information.
    regions.sort(key=lambda r: (not np.isfinite(r.p_hit_odd), r.p_hit_odd))
    for new_id, r in enumerate(regions):
        r.region_id = new_id
    return regions


