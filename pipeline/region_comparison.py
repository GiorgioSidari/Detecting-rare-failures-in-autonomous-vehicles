"""
Comparison of several search methods against one shared map of failure regions.

The input is one cloud of evaluated points per method, as
`{label: (theta, margins)}`. Regions are extracted once, by
`pipeline.failure_regions.extract_failure_regions`, from the union of every
method's failing points, so all methods are scored against the same map.

For each pair of methods :func:`compare_failure_regions` reports the regions
each one discovered, the Jaccard index of the two sets, and a clustering-free
point coverage: the fraction of A's failures lying within `tau` of some failure
of B. Per method it reports the number of failures, the worst margin and the
regions found; over the pooled cloud it reports the per-axis spread and the
structure score.

:class:`RegionComparison` holds that outcome and renders it as a report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.failure_regions import (
    DEFAULT_EPS,
    DEFAULT_MAX_SPAN,
    DEFAULT_MIN_POINTS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_PAD,
    _to_unit,
    estimate_eps,
    extract_failure_regions,
    structure_score,
)



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
    # Output of structure_score on the pooled failure cloud: observed share,
    # shuffled null and z. Empty when there were too few points to compute it.
    structure: dict = field(default_factory=dict)
    # Per-axis spread of the pooled failure cloud, as a fraction of the ODD
    # range: 1.0 means the failures cover that parameter end to end.
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
        Per family, compare the LHS arm against the random arm on localised regions.

        Two arms belong to the same family when their labels differ only by the
        sampling design. For each family the comparison counts the localised regions
        each arm reached and those only one of them reached; single-point regions and
        non-localised groups do not enter the verdict.

        Returns {family: {...}} with a "verdict" in each entry, one of:
          "lhs"        localised regions reached only by the LHS arm
          "random"     localised regions reached only by the random arm
          "tie"        the same localised regions on both sides
          "n/a"        no localised region on either side
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
        True when the failure cloud shows structure, by either of two criteria.

        Either is enough:
          * at least one parameter is constrained across the WHOLE failure cloud
            (minimum axis spread below 0.9), or
          * the clustering concentrates points more than axis-shuffled data does
            (structure_score z above 2), which catches several separated regions that
            individually look spread out on every axis.

        Neither criterion measures a cluster against itself: the first uses the whole
        cloud, the second an external null.
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

    def _answer_lines(self, w: int, rare: bool, verdict: dict) -> list:
        """Section 1: the verdict, stated before any evidence."""
        L: list = []
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
                L.append("   groups; shuffling the parameters independently — which destroys")
                L.append(f"   any structure — puts {st['null_mean']:.0%} there. "
                      f"z = {st['z']:+.1f}, i.e. the")
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

        return L

    def _evidence_lines(self, w: int, localised: list) -> list:
        """Section 2: the numbers the verdict rests on."""
        L: list = []
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

        return L

    def _regions_lines(self, w: int, localised: list) -> list:
        """Section 3: the localised regions, rarest first."""
        L: list = []
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

        return L

    def _detail_lines(self, w: int, localised: list, verbose: bool) -> list:
        """Section 4: opt-in detail (pairwise matrix, omitted groups)."""
        L: list = []
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

        return L

    def report(self, verbose: bool = False) -> str:
        """
        The answer first, the evidence second, the raw region dump last.

        verbose=True adds the per-region detail and the full pairwise matrix.
        The default is deliberately short: on a 9-parameter scenario the region
        list runs to hundreds of lines and buries the one line that matters.
        """
        w = 78
        rare = self.is_rare_regime()
        verdict = self.sampler_verdict()
        localised = [r for r in self.regions if r.constraining_axes]

        L = self._answer_lines(w, rare, verdict)
        L += self._evidence_lines(w, localised)
        L += self._regions_lines(w, localised)
        L += self._detail_lines(w, localised, verbose)
        L.append("=" * w)
        return "\n".join(L)

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


def _pool_runs(runs: dict, n_params: int) -> tuple:
    """
    Stack every arm's points into one cloud, remembering who produced each.

    One shared region map is built from the union of all failures, so no arm
    gets a home-field advantage from its own clustering.
    """
    all_theta, all_margin, owners = [], [], []
    for label, (th, mg) in runs.items():
        th = np.asarray(th, float)
        mg = np.asarray(mg, float)
        all_theta.append(th)
        all_margin.append(mg)
        owners.append(np.array([label] * len(mg), dtype=object))
    theta_u = np.vstack(all_theta) if all_theta else np.empty((0, n_params))
    margin_u = np.concatenate(all_margin) if all_margin else np.empty(0)
    owner_u = np.concatenate(owners) if owners else np.empty(0, dtype=object)
    return theta_u, margin_u, owner_u


def _per_method_stats(runs: dict, discovery: dict, lower: np.ndarray, upper: np.ndarray,
                      threshold: float) -> tuple:
    """Each arm's counts, and its failure cloud in the unit cube."""
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
    return per_method, fail_clouds


def _pairwise_overlap(labels_order: list, discovery: dict, fail_clouds: dict,
                      tau: float) -> dict:
    """Region overlap (Jaccard) and point coverage for every pair of arms."""
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
    return pairwise


def _comparison_notes(regions: list, d: int, eps: float, eps_auto: bool,
                      min_samples: int, dists: list) -> list:
    """The caveats that must travel with the numbers."""
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
    return notes


def _axis_spread(theta_u: np.ndarray, ok_u: np.ndarray, lower: np.ndarray,
                 upper: np.ndarray, names: list) -> dict:
    """
    Fraction of each parameter's range the pooled failures cover.

    The first thing to look at: if every axis spans the whole ODD, the failures
    are not in a region at all and no clustering parameter will make one appear.
    """
    if not ok_u.any():
        return {}
    U = _to_unit(theta_u[ok_u], lower, upper)
    return {nm: float(U[:, j].max() - U[:, j].min()) for j, nm in enumerate(names)}


def compare_failure_regions(
    runs: dict,
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
    eps  : DBSCAN radius in the normalised cube. None estimates it from the
           pooled failure cloud with :func:`estimate_eps`, which scales with the
           dimension of the space.
    tau  : radius (unit cube) for the clustering-free point-coverage metric.
           Defaults to the resolved `eps`, so "A's failure is covered by B" means
           B reached a point within the same neighbourhood DBSCAN would use.

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

    theta_u, margin_u, owner_u = _pool_runs(runs, len(lower))

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

    per_method, fail_clouds = _per_method_stats(
        runs, discovery, lower, upper, threshold)
    pairwise = _pairwise_overlap(labels_order, discovery, fail_clouds, tau)
    notes = _comparison_notes(regions, len(lower), eps, eps_auto, min_samples, dists)

    struct = (structure_score(_to_unit(theta_u[ok_u], lower, upper), eps=eps,
                              min_samples=min_samples, min_points=min_points)
              if ok_u.sum() >= 2 * min_points else {})

    names = list(param_names) if param_names is not None else [
        f"p{j}" for j in range(len(lower))]
    axis_spread = _axis_spread(theta_u, ok_u, lower, upper, names)

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
