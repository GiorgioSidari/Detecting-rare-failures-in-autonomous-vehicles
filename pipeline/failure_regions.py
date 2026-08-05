"""
Failure REGIONS: extract them, compare them across methods, and price them.

Every search method in this repo returns a cloud of evaluated points with their
safety margins. Counting failures is not enough to answer the question that
matters — *do the different methods find the same failure regions?* — because
two methods can report the same failure rate while exploring two disjoint
corners of the parameter space.

This module answers three things:

1. **Where are the failure regions?**  Failing points are normalised to the unit
   cube and clustered (DBSCAN, density-based: it finds regions of arbitrary
   shape and does not need the number of clusters up front). A point that is
   isolated still becomes a region of its own — an isolated rare failure is
   exactly the thing we do not want to discard as noise.

2. **Do the methods agree?**  Regions are extracted ONCE from the union of every
   method's failing points, so all methods are scored against the same map.
   Then, per pair of methods: which regions each discovered, the Jaccard index
   of the discovered sets, and a clustering-free point-coverage measure (the
   fraction of A's failures that sit within tau of some failure of B).

3. **How likely is a blind draw to hit a region?**  For each region we estimate
   P(theta in region) by Monte Carlo, both under a uniform ODD and under the
   scenario's operational distribution. From that:

       p_hit_n = 1 - (1 - p)^n     probability that n blind samples hit it
       n_50    = log(0.5)/log(1-p) samples needed for a 50/50 chance

   This is the quantitative form of "random search does not reliably see this
   region": if a region has p = 3e-4, a 200-sample random design finds it about
   6% of the time, while a stratified design that probes every stratum of the
   driving axes has a structurally better chance.

Two things about high dimension are easy to get wrong, and this module handles
both explicitly.

**Clustering radius.** A fixed eps does not survive a change of dimension. In a
d-dimensional unit cube the distance between two uniform points concentrates
around sqrt(d/6): about 0.6 in 2-D but about 1.2 in 9-D. An eps tuned on a 2-D
toy problem gives every point its own cluster on a 9-D one, and the comparison
then reports "no method shares any region" as an artefact of the radius rather
than a finding. So eps defaults to None and is estimated from the data with the
k-distance heuristic (see :func:`estimate_eps`).

**Region probability.** A bounding box in d dimensions is a product of d side
lengths, so P(hit) collapses towards zero whether or not the region is genuinely
rare — the padding on the axes that do NOT matter dominates the product. We
therefore multiply only over the axes on which the region is actually LOCALISED:
those where its observed span is clearly narrower than the span the same number
of ODD draws would produce by chance. That answers the question you want
answered — "how unlikely is a draw satisfying the conditions that define this
region?" — and lets an axis the region does not constrain contribute a factor of
1, as it should.

A region of a single point constrains nothing measurable, so its P(hit) is
reported as NaN rather than as a fabricated tiny number.

Within the constraining axes the box is still an OPTIMISTIC approximation: the
true region is contained in its box, so the reported p_hit is an upper bound and
n_50 a lower bound. Conclusions of the form "a blind design would need at least
N samples" therefore stay valid.
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


@dataclass
class RegionComparison:
    """Result of comparing several methods against one shared region map."""

    regions: list                        # list[FailureRegion]
    method_labels: list                  # list[str], in report order
    discovery: dict                      # label -> set of region_ids it discovered
    per_method: dict                     # label -> summary dict (failures, worst margin, ...)
    pairwise: dict                       # (a, b) -> {"jaccard", "coverage_a_in_b", ...}
    param_names: list = field(default_factory=list)
    weighting: str = "odd"
    notes: list = field(default_factory=list)
    eps: float = float("nan")            # radius actually used (possibly estimated)
    # Whether the failure cloud has any joint structure at all, tested against
    # axis-shuffled data (see structure_score). This is the uncircular signal.
    structure: dict = field(default_factory=dict)
    # Per-axis spread of the pooled failure cloud, as a fraction of the ODD range.
    # Close to 1 on every axis = failures are everywhere, not in a region.
    axis_spread: dict = field(default_factory=dict)

    # ── convenience ────────────────────────────────────────────────────────
    def exclusive_regions(self, label: str) -> list:
        """Regions this method found and NO other method did — the crux of the claim."""
        others = set().union(*[self.discovery[m] for m in self.method_labels if m != label]) \
            if len(self.method_labels) > 1 else set()
        return [r for r in self.regions if r.region_id in self.discovery[label] - others]

    def rarest_regions(self, k: int = 5) -> list:
        """The k rarest regions whose rarity is actually estimable."""
        est = [r for r in self.regions if r.is_estimable]
        return sorted(est, key=lambda r: r.p_hit(self.weighting))[:k]

    def localised_regions(self) -> list:
        """Regions that constrain at least one parameter — the interpretable ones."""
        return [r for r in self.regions if r.constraining_axes]


    # ── the question the whole feature exists to answer ────────────────────
    def sampler_verdict(self) -> dict:
        """
        Pair each method family's LHS arm against its random arm and decide,
        per family, whether stratification reached failures the other missed.

        The claim "LHS is better than random search" is only supportable if the
        LHS arm reaches LOCALISED failure regions the random arm does not. More
        failures alone does not prove it (the two arms may simply be sampling a
        domain that fails everywhere), and neither does a low Jaccard on regions
        that are single points.

        Returns {family: {...}} with a "verdict" in each entry, one of:
          "lhs"        stratification found localised regions random missed
          "random"     the opposite
          "tie"        both found the same localised regions
          "n/a"        no localised region on either side — nothing to compare
        """
        families: dict = {}
        for lab in self.method_labels:
            if "[" not in lab:
                continue
            fam, design = lab.split("[", 1)
            design = design.rstrip("]").split("+")[0]
            if design in ("lhs", "random"):
                families.setdefault(fam, {})[design] = lab

        out: dict = {}
        localised = {r.region_id for r in self.regions if r.constraining_axes}
        for fam, arms in families.items():
            if set(arms) != {"lhs", "random"}:
                continue                      # only one design ran: nothing to pair
            a, b = arms["lhs"], arms["random"]
            loc_a = self.discovery[a] & localised
            loc_b = self.discovery[b] & localised
            only_a, only_b = loc_a - loc_b, loc_b - loc_a

            if not loc_a and not loc_b:
                verdict = "n/a"
            else:
                # A one-region difference between two arms that each found a
                # handful is noise. Require the gap to be both absolute (>= 2
                # regions) and relative (>= 30% of what the family found at all)
                # before calling a winner; otherwise it is a tie.
                gap = len(only_a) - len(only_b)
                pool = max(len(loc_a | loc_b), 1)
                decisive = abs(gap) >= 2 and abs(gap) / pool >= 0.3
                verdict = ("tie" if not decisive else
                           "lhs" if gap > 0 else "random")
            out[fam] = {
                "lhs_arm": a, "random_arm": b, "verdict": verdict,
                "failures_lhs": self.per_method[a]["n_failures"],
                "failures_random": self.per_method[b]["n_failures"],
                "worst_lhs": self.per_method[a]["worst_margin"],
                "worst_random": self.per_method[b]["worst_margin"],
                "localised_lhs": len(loc_a), "localised_random": len(loc_b),
                "only_lhs": sorted(only_a), "only_random": sorted(only_b),
                "decisive": verdict in ("lhs", "random"),
                "jaccard": self.pairwise.get((a, b), self.pairwise.get((b, a), {}))
                              .get("jaccard", float("nan")),
            }
        return out

    def is_rare_regime(self) -> bool:
        """
        True when there is failure structure for the designs to differ about.

        Two independent ways to qualify, either is enough:
          * at least one parameter is constrained across the WHOLE failure cloud
            (min axis spread < 0.9), or
          * the clustering concentrates points more than axis-shuffled data does
            (structure_score z > 2) — several separated regions look spread out
            on every axis individually, so the first test alone would miss them.

        Both are uncircular: neither asks a cluster to justify its own
        compactness. If both fail, the failures are a scatter over the ODD and
        nothing below is measuring the sampling designs.
        """
        if self.structure.get("has_structure"):
            return True
        if not self.axis_spread:
            return False
        return min(self.axis_spread.values()) < 0.9

    def localised_share(self) -> float:
        """Fraction of all failures that sit inside a localised region."""
        total = sum(s["n_failures"] for s in self.per_method.values())
        if not total:
            return 0.0
        return sum(r.n_points for r in self.regions if r.constraining_axes) / total

    def to_dict(self) -> dict:
        return {
            "weighting": self.weighting,
            "eps": self.eps,
            "axis_spread": dict(self.axis_spread),
            "param_names": list(self.param_names),
            "methods": list(self.method_labels),
            "notes": list(self.notes),
            "regions": [
                {
                    "region_id": r.region_id,
                    "n_points": r.n_points,
                    "centroid": [float(v) for v in r.centroid],
                    "box_lower": [float(v) for v in r.box_lower],
                    "box_upper": [float(v) for v in r.box_upper],
                    "worst_margin": r.worst_margin,
                    "mean_margin": r.mean_margin,
                    "volume_fraction": r.volume_fraction,
                    "p_hit_uniform": r.p_hit_uniform,
                    "p_hit_odd": r.p_hit_odd,
                    "n_for_50pct": r.n_for_50pct(self.weighting),
                    "is_singleton": r.is_singleton,
                    "constraining_axes": [int(j) for j in r.constraining_axes],
                    "constraints": r.describe_constraints(),
                    "found_by": dict(r.found_by),
                }
                for r in self.regions
            ],
            "sampler_verdict": self.sampler_verdict(),
            "rare_regime": self.is_rare_regime(),
            "discovery": {m: sorted(ids) for m, ids in self.discovery.items()},
            "per_method": self.per_method,
            "pairwise": {f"{a} vs {b}": v for (a, b), v in self.pairwise.items()},
        }

    def report(self, verbose: bool = False) -> str:
        """
        The answer first, the evidence second, the raw region dump last.

        verbose=True adds the per-region detail and the full pairwise matrix.
        The default is deliberately short: on a 9-parameter scenario the region
        list runs to hundreds of lines and buries the one line that matters.
        """
        w = 78
        L: list = []
        rare = self.is_rare_regime()
        verdict = self.sampler_verdict()
        localised = [r for r in self.regions if r.constraining_axes]

        # ── 1. THE ANSWER ──────────────────────────────────────────────────
        L.append("=" * w)
        L.append(" LHS vs RANDOM SEARCH — DOES STRATIFICATION REACH FAILURES")
        L.append(" THAT RANDOM SAMPLING MISSES?")
        L.append("=" * w)

        if not rare:
            L.append("")
            L.append("   ANSWER:  NOT ANSWERABLE FROM THIS RUN")
            L.append("")
            L.append("   The failures have no structure: they are scattered across the")
            L.append("   whole operational domain, so there is no rare region for either")
            L.append("   design to miss and the comparison has nothing to measure.")
            L.append("   That is a real finding about the MODEL — it fails throughout")
            L.append("   this ODD — but it says nothing about the DESIGNS.")
            if self.structure:
                st = self.structure
                L.append("")
                L.append(f"   Evidence: clustering puts {st['observed']:.0%} of the failures into")
                L.append(f"   groups; shuffling the parameters independently — which destroys")
                L.append(f"   any structure — puts {st['null_mean']:.0%} there. z = {st['z']:+.1f}, i.e. the")
                L.append("   real failures group no more tightly than pure scatter.")
            L.append("")
            L.append("   To get an answer: narrow the ODD until the failure rate of the")
            L.append("   plain_sampling arms lands between about 2% and 10%, then re-run.")
        elif not verdict:
            L.append("")
            L.append("   ANSWER:  NO PAIRED ARMS")
            L.append("   Run an lhs and a random arm of the same method to compare them.")
        else:
            wins = [f for f, v in verdict.items() if v["verdict"] == "lhs"]
            loses = [f for f, v in verdict.items() if v["verdict"] == "random"]
            ties = [f for f, v in verdict.items() if v["verdict"] == "tie"]
            na = [f for f, v in verdict.items() if v["verdict"] == "n/a"]
            L.append("")
            if wins and not loses:
                L.append(f"   ANSWER:  YES, in {len(wins)}/{len(verdict)} method families")
            elif loses and not wins:
                L.append(f"   ANSWER:  NO — random search reached regions LHS missed "
                         f"({len(loses)}/{len(verdict)} families)")
            elif wins and loses:
                L.append("   ANSWER:  MIXED — the two designs win in different families")
            else:
                L.append("   ANSWER:  NO DIFFERENCE — both designs found the same regions")
            L.append("")
            L.append(f"   {'family':<22}{'verdict':<10}{'localised regions':>19}"
                     f"{'only LHS':>10}{'only rnd':>10}")
            for fam, v in verdict.items():
                tag = {"lhs": "LHS", "random": "random", "tie": "tie",
                       "n/a": "n/a"}[v["verdict"]]
                L.append(f"   {fam:<22}{tag:<10}"
                         f"{v['localised_lhs']:>9} vs{v['localised_random']:>7}"
                         f"{len(v['only_lhs']):>10}{len(v['only_random']):>10}")
            if ties or na:
                L.append("")
                L.append("   'n/a' = neither arm found a localised region in that family.")
        L.append("")

        # ── 2. THE EVIDENCE ────────────────────────────────────────────────
        L.append("-" * w)
        L.append(" EVIDENCE")
        L.append("-" * w)
        L.append(f"   failures analysed        : "
                 f"{sum(s['n_failures'] for s in self.per_method.values())}")
        L.append(f"   groups after clustering  : {len(self.regions)}  "
                 f"(eps={self.eps:.3f}, {len(self.param_names)}-D)")
        L.append(f"   of which LOCALISED       : {len(localised)}   "
                 f"<- only these are failure REGIONS")
        L.append(f"   failures inside them     : {self.localised_share():.0%}")
        if self.structure:
            st = self.structure
            L.append(f"   structure vs shuffled    : {st['observed']:.0%} vs "
                     f"{st['null_mean']:.0%} grouped, z={st['z']:+.1f}"
                     f"{'  <- real structure' if st['has_structure'] else '  <- no structure'}")
        L.append("")
        L.append("   LOCALISED = the group sits in a much narrower band than its size")
        L.append("   would give by chance on at least one parameter. Read those")
        L.append("   bounds as a description of the failures found, not as proven")
        L.append("   conditions: clustering makes groups compact by construction. The")
        L.append("   line above (structure vs shuffled) is the claim you can defend.")
        L.append("")

        if self.axis_spread:
            L.append("   How much of each parameter's range the failures cover")
            L.append("   (1.00 = failures everywhere on that axis, no localisation):")
            for name, sp in self.axis_spread.items():
                bar = "#" * int(round(sp * 28))
                flag = "" if sp > 0.9 else "   <- localised"
                L.append(f"     {name:<24}{sp:>5.2f} {bar}{flag}")
            L.append("")

        L.append(f"   {'arm':<28}{'failures':>10}{'regions':>9}{'localised':>11}"
                 f"{'only it':>9}{'worst':>9}")
        for m in self.method_labels:
            st = self.per_method[m]
            loc = len({r.region_id for r in localised} & self.discovery[m])
            excl = len([r for r in self.exclusive_regions(m) if r.constraining_axes])
            L.append(f"   {m:<28}{st['n_failures']:>10}{len(self.discovery[m]):>9}"
                     f"{loc:>11}{excl:>9}{st['worst_margin']:>9.3f}")
        L.append("")
        L.append("   failures   how many runs crashed        regions    groups it reached")
        L.append("   localised  of those, real regions       only it    localised regions")
        L.append("   worst      deepest safety margin                   no other arm found")
        L.append("")

        # ── 3. THE REGIONS ─────────────────────────────────────────────────
        if localised:
            L.append("-" * w)
            L.append(" THE FAILURE REGIONS THAT EXIST, rarest first")
            L.append("-" * w)
            L.append("   P(hit) = chance that one realistic scenario lands in the region")
            L.append("   n50    = scenarios you would have to draw blindly for a coin-flip")
            L.append("            chance of stumbling on it")
            L.append("")
            for r in sorted(localised, key=lambda x: x.p_hit(self.weighting)):
                n50 = r.n_for_50pct(self.weighting)
                n50_s = ("n/a" if not np.isfinite(n50)
                         else "inf" if np.isinf(n50) else f"{n50:,.0f}")
                L.append(f"   R{r.region_id:<4} {r.n_points:>3} failures   "
                         f"worst margin {r.worst_margin:+.3f}   "
                         f"P(hit)={r.p_hit(self.weighting):.1e}   n50={n50_s}")
                L.append(f"          happens when: {r.describe_constraints()}")
                L.append(f"          found by    : {', '.join(sorted(r.found_by)) or '-'}")
                if len(r.constraining_axes) > len(self.param_names) / 2:
                    L.append(f"          NOTE: constrains {len(r.constraining_axes)} of "
                             f"{len(self.param_names)} parameters on {r.n_points} points —")
                    L.append("          the bounds are over-fitted, read the region as a")
                    L.append("          location, not as a set of necessary conditions.")
                L.append("")

        # ── 4. DETAIL (opt-in) ─────────────────────────────────────────────
        if verbose:
            L.append("-" * w)
            L.append(" DETAIL")
            L.append("-" * w)
            for n in self.notes:
                L.append(f"   note: {n}")
            L.append("")
            if self.pairwise:
                width = max(len(f"{a} vs {b}") for a, b in self.pairwise) + 2

                def _fmt(x):
                    return "   n/a" if not np.isfinite(x) else f"{x:6.2f}"

                L.append(f"   {'pair':<{width}}{'Jaccard':>9}{'A in B':>9}{'B in A':>9}")
                for (a, b), v in self.pairwise.items():
                    L.append(f"   {a + ' vs ' + b:<{width}}   {_fmt(v['jaccard'])}"
                             f"   {_fmt(v['coverage_a_in_b'])}   {_fmt(v['coverage_b_in_a'])}")
                L.append("")
                L.append("   Jaccard = shared groups / groups found by either.")
                L.append("   'A in B' = fraction of A's failures within eps of one of B's.")
            n_single = sum(1 for r in self.regions if r.is_singleton)
            n_free = len(self.regions) - len(localised) - n_single
            L.append("")
            L.append(f"   {n_single} single-point groups (rarity not estimable) and "
                     f"{n_free} multi-point")
            L.append("   groups that constrain no parameter are omitted above.")
        else:
            L.append(f" (run with verbose=True for the pairwise matrix and the "
                     f"{len(self.regions) - len(localised)} non-localised groups)")

        L.append("=" * w)
        return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────
def _to_unit(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = np.where((upper - lower) > 0, upper - lower, 1.0)
    return (np.asarray(theta, float) - lower) / span


def estimate_eps(unit_pts: np.ndarray, min_samples: int = DEFAULT_MIN_SAMPLES,
                 quantile: float = 0.5) -> float:
    """
    Estimate the DBSCAN radius from the data (the k-distance heuristic).

    For each point take the distance to its k-th nearest neighbour (k =
    min_samples); the median of those distances is a scale at which about half
    the points have the density DBSCAN needs to seed a cluster. This adapts
    automatically to the dimension, which a hard-coded eps cannot: in a 9-D unit
    cube two uniform points sit ~1.2 apart, so any eps below ~0.5 declares every
    point isolated and the whole region analysis degenerates into one region per
    point.

    Returns a radius in the normalised cube; falls back to sqrt(d)/4 when there
    are too few points to estimate anything.
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
    """
    Does this failure cloud have joint structure, or is it just a scatter?

    The one question about clustering that can be answered without circularity.
    Instead of picking groups and then asking whether they look tight — which
    they always do, because the clustering made them tight — we compare the WHOLE
    clustering against the clustering of the same data with every axis shuffled
    independently. Shuffling preserves each parameter's marginal distribution and
    destroys any joint structure, so it is exactly the "no regions here" null.

    The statistic is the share of points that land in a group of at least
    `min_points`. Real structure concentrates points into groups; a scatter does
    not, whatever eps you choose.

    Returns {"observed", "null_mean", "null_std", "z", "has_structure"}.
    ``has_structure`` is z > 2, i.e. the real clustering concentrates points
    beyond what shuffled data of the same marginals produces.
    """
    def _share(P):
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
    Which axes to report as defining the group: those where its points occupy
    much less of the parameter's range than n points typically would.

    The threshold is ``max_span * (n-1)/(n+1)``, because (n-1)/(n+1) is the
    expected span of n points spread over an axis. Without that scaling the rule
    is trivially satisfied by small groups: three points span half an axis on
    average, so "span < 0.5" fires on roughly half the axes by chance and a
    3-point cluster is reported as constrained on six dimensions out of nine,
    with a P(hit) of 1e-8 that means nothing. Scaling makes the filter demand
    ~0.25 from a 3-point group and ~0.41 from a 10-point one.

    Read these as a DESCRIPTION, not as proven necessary conditions. There is no
    honest significance test available here: DBSCAN builds groups that are
    compact by construction, so asking "is this group surprisingly narrow?" of a
    group the clustering itself made narrow is circular. A permutation null does
    not rescue it either — re-clustering shuffled data changes the groups, so the
    comparison is between different things.

    What the filter is genuinely for is keeping P(hit) meaningful. In 9
    dimensions the axes a group does NOT constrain contribute padding factors
    that dominate any product, and every region comes out at 1e-13 whether it is
    rare or not. Multiplying only the narrow axes gives the interpretable
    quantity: how likely is a scenario that satisfies the conditions we observed.

    The uncircular evidence lives elsewhere in the report: the per-axis spread of
    the WHOLE failure cloud, and whether different search arms reach the same
    groups. Judge the run on those.
    """
    if n_points < max(2, min_points):
        return []
    expected = (n_points - 1) / (n_points + 1)
    return [int(j) for j in np.where(np.asarray(side_unit) < max_span * expected)[0]]


def _box_probability(box_lo, box_hi, lower, upper, dists, n_points: int,
                     max_span: float = DEFAULT_MAX_SPAN,
                     min_points: int = DEFAULT_MIN_POINTS) -> tuple:
    """
    (p_uniform, p_odd, constraining_axes) for one region.

    P(hit) is the probability that a blind draw satisfies the conditions that
    DEFINE the region, i.e. falls inside its interval on each axis the region is
    significantly localised on. Axes it does not constrain contribute a factor of
    1 — multiplying them in is what made every region of a 9-D space look
    impossibly rare regardless of whether it was.

    Which axes count is decided on the geometry (:func:`_constraining_axes`,
    calibrated by permutation); the probability itself is then read off the ODD,
    where the per-axis factor is the span in probability units. Both terms are
    analytic; no Monte-Carlo noise enters a column whose values are small by
    construction.

    Fewer than 2 points means no measurable span on any axis, so both
    probabilities are NaN: we cannot tell a rare region from one lucky draw.
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


def extract_failure_regions(
    theta: np.ndarray,
    margins: np.ndarray,
    lower,
    upper,
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
    regions: list = []
    for cid in np.unique(labels):
        sel = labels == cid
        members = idx[sel]
        pts_u = unit[sel]
        pts_p = theta[members]
        m = margins[members]

        box_lo_u = np.clip(pts_u.min(axis=0) - pad, 0.0, 1.0)
        box_hi_u = np.clip(pts_u.max(axis=0) + pad, 0.0, 1.0)
        box_lo = lower + box_lo_u * span
        box_hi = lower + box_hi_u * span

        p_uni, p_odd, axes = _box_probability(
            box_lo, box_hi, lower, upper, dists, int(sel.sum()),
            max_span, min_points)

        found_by: dict = {}
        if owners is not None:
            own = np.asarray(owners)[members]
            for o in np.unique(own):
                found_by[str(o)] = int((own == o).sum())

        regions.append(FailureRegion(
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
            param_names=list(param_names) if param_names is not None else [],
            found_by=found_by,
            constraining_axes=axes,
        ))

    # Rarest first; regions whose rarity is not estimable (single point) go last,
    # so the top of the report is always the part that carries information.
    regions.sort(key=lambda r: (not np.isfinite(r.p_hit_odd), r.p_hit_odd))
    for new_id, r in enumerate(regions):
        r.region_id = new_id
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Cross-method comparison
# ─────────────────────────────────────────────────────────────────────────────
def _point_coverage(A: np.ndarray, B: np.ndarray, tau: float) -> float:
    """Fraction of A's rows within distance tau of some row of B (unit cube)."""
    if len(A) == 0:
        return float("nan")
    if len(B) == 0:
        return 0.0
    hit = 0
    for a in A:                       # loop keeps memory flat for large clouds
        if np.min(np.linalg.norm(B - a, axis=1)) <= tau:
            hit += 1
    return float(hit / len(A))


def compare_failure_regions(
    runs: dict,
    lower,
    upper,
    *,
    threshold: float = 0.0,
    param_names=None,
    dists=None,
    eps: float | None = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    pad: float = DEFAULT_PAD,
    max_span: float = DEFAULT_MAX_SPAN,
    min_points: int = DEFAULT_MIN_POINTS,
    tau: float | None = None,
    weighting: str = "odd",
) -> RegionComparison:
    """
    Compare the failure regions reached by several methods.

    Parameters
    ----------
    runs : {label: (theta, margins)} — one entry per method/arm. Concatenate the
           seeds of a multi-seed campaign before passing them in if you want the
           region map of the whole campaign.
    eps  : DBSCAN radius in the normalised cube; None estimates it from the data.
           Leave it None: a radius that works in 2-D declares every point isolated
           in 9-D, and the comparison then reports Jaccard 0 everywhere as an
           artefact rather than a result.
    tau  : radius (unit cube) for the clustering-free point-coverage metric.
           Defaults to the resolved `eps`, so "A's failure is covered by B" means
           B reached a point DBSCAN would consider part of the same neighbourhood.

    Returns
    -------
    RegionComparison — call .report() for the printable summary, .to_dict() to
    serialise.

    Note on interpretation: a low Jaccard together with a high failure count on
    both sides means the two methods are exploring DIFFERENT parts of the failure
    set. That, not the raw failure rate, is the evidence that one design reaches
    rare regions the other misses.
    """
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    labels_order = list(runs.keys())

    # One shared region map, built from the union of every method's failures, so
    # no method gets a home-field advantage from its own clustering.
    all_theta, all_margin, owners = [], [], []
    for label, (th, mg) in runs.items():
        th = np.asarray(th, float)
        mg = np.asarray(mg, float)
        all_theta.append(th)
        all_margin.append(mg)
        owners.append(np.array([label] * len(mg), dtype=object))
    theta_u = np.vstack(all_theta) if all_theta else np.empty((0, len(lower)))
    margin_u = np.concatenate(all_margin) if all_margin else np.empty(0)
    owner_u = np.concatenate(owners) if owners else np.empty(0, dtype=object)

    # Resolve eps ONCE, on the pooled failure cloud, so every arm and the
    # point-coverage metric are judged at the same scale.
    ok_u = np.isfinite(margin_u) & (margin_u < threshold)
    eps_auto = eps is None
    if eps_auto:
        eps = (estimate_eps(_to_unit(theta_u[ok_u], lower, upper),
                            min_samples=min_samples)
               if ok_u.any() else float(np.sqrt(len(lower)) / 4.0))
    tau = eps if tau is None else float(tau)

    regions = extract_failure_regions(
        theta_u, margin_u, lower, upper, threshold=threshold,
        param_names=param_names, dists=dists, eps=eps,
        min_samples=min_samples, pad=pad, max_span=max_span,
        min_points=min_points, owners=owner_u,
    )

    discovery = {lab: {r.region_id for r in regions if lab in r.found_by}
                 for lab in labels_order}

    per_method, fail_clouds = {}, {}
    for lab, (th, mg) in runs.items():
        th = np.asarray(th, float)
        mg = np.asarray(mg, float)
        ok = np.isfinite(mg)
        f = ok & (mg < threshold)
        fail_clouds[lab] = _to_unit(th[f], lower, upper)
        per_method[lab] = {
            "n_evaluations": int(ok.sum()),
            "n_failures": int(f.sum()),
            "failure_rate": float(f.sum() / ok.sum()) if ok.sum() else float("nan"),
            "worst_margin": float(mg[ok].min()) if ok.any() else float("nan"),
            "n_regions": len(discovery[lab]),
        }

    pairwise = {}
    for i, a in enumerate(labels_order):
        for b in labels_order[i + 1:]:
            ra, rb = discovery[a], discovery[b]
            union = ra | rb
            pairwise[(a, b)] = {
                "jaccard": float(len(ra & rb) / len(union)) if union else float("nan"),
                "shared_regions": sorted(ra & rb),
                "only_a": sorted(ra - rb),
                "only_b": sorted(rb - ra),
                "coverage_a_in_b": _point_coverage(fail_clouds[a], fail_clouds[b], tau),
                "coverage_b_in_a": _point_coverage(fail_clouds[b], fail_clouds[a], tau),
            }

    d = len(lower)
    n_single = sum(1 for r in regions if r.is_singleton)
    notes = [
        f"clustered with DBSCAN(eps={eps:.3f}{' auto' if eps_auto else ''}, "
        f"min_samples={min_samples}) in the {d}-D normalised cube "
        f"(two uniform points sit ~{np.sqrt(d / 6.0):.2f} apart at this dimension)",
        "P(hit) multiplies only the axes each region is genuinely localised on; "
        "it is an upper bound, so n50 is a lower bound on the blind-draw cost",
    ]
    if n_single:
        notes.append(
            f"{n_single}/{len(regions)} regions hold a single point: their rarity is "
            "not estimable (one draw measures no span) and reads n/a")
    if n_single > 0.8 * max(len(regions), 1):
        notes.append(
            "MOST regions are single points — the failures are spread out rather "
            "than clustered. Either the budget is too small to see structure, or "
            "there is no localised failure region in this ODD (check the per-axis "
            "spread before reading anything into the Jaccard column)")
    if dists is None:
        notes.append("no ODD supplied — p_hit_odd falls back to the uniform estimate")

    struct = (structure_score(_to_unit(theta_u[ok_u], lower, upper), eps=eps,
                              min_samples=min_samples, min_points=min_points)
              if ok_u.sum() >= 2 * min_points else {})

    # Per-axis spread of the pooled failure cloud. This is the first thing to
    # look at: if every axis spans the whole ODD, the failures are not in a
    # region at all and no clustering parameter will make one appear.
    names = list(param_names) if param_names is not None else [
        f"p{j}" for j in range(len(lower))]
    axis_spread: dict = {}
    if ok_u.any():
        U = _to_unit(theta_u[ok_u], lower, upper)
        for j, nm in enumerate(names):
            axis_spread[nm] = float(U[:, j].max() - U[:, j].min())

    return RegionComparison(
        regions=regions,
        method_labels=labels_order,
        discovery=discovery,
        per_method=per_method,
        pairwise=pairwise,
        param_names=names,
        weighting=weighting,
        notes=notes,
        eps=float(eps),
        axis_spread=axis_spread,
        structure=struct,
    )
