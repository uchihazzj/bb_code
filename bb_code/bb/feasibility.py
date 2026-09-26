"""Exact admission intervals under disjoint support constraints."""
import numpy as np

def intervals(order, old, groups, capacity, k):
    """Return per-candidate admission intervals and their rank-count witnesses.

    `capacity` counts all possible new leaves, including those outside this beam.
    `order` is a fixed total order, independent of the chosen subcatalog. Each
    required group is a disjoint nonempty set of nonold terminal candidate indices.
    No model or catalog enumeration occurs here. Old candidates get possible=False.
    """
    order, old = np.asarray(order, dtype=np.int64), np.asarray(old, dtype=bool)
    n = len(old)
    if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("order must be a permutation of all terminal indices")
    if k < 1 or capacity < int((~old).sum()):
        raise ValueError("invalid rank cutoff or optional capacity")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    grouped = np.zeros(n, dtype=bool)
    end_counts = np.zeros(n, dtype=np.int64)
    for group in groups:
        ids = np.asarray(list(group), dtype=np.int64)
        if (ids.ndim != 1 or not len(ids) or np.any((ids < 0) | (ids >= n)) or
                len(np.unique(ids)) != len(ids) or old[ids].any() or grouped[ids].any()):
            raise ValueError("required groups must be disjoint nonempty nonold subsets")
        grouped[ids] = True
        end_counts[int(ranks[ids].max())] += 1
    old_before_sorted = np.cumsum(old[order]) - old[order]
    forced_before_sorted = np.cumsum(end_counts) - end_counts
    a, g = old_before_sorted[ranks], forced_before_sorted[ranks]
    lower = len(groups) + (~grouped).astype(np.int64)
    upper = int(capacity) - np.maximum(0, ranks + 1 - k)
    possible = (~old) & (a + g < k) & (lower <= upper)
    return dict(possible=possible, lower=lower, upper=upper, rank=ranks + 1,
                old_before=a, forced_groups_before=g)


def construct_witness(universe, order, old, groups, target, admissions, k):
    """Materialize one witness in a small finite universe; None if infeasible.

    Uses direct support and padding sets, separate from the vectorized interval
    calculation, so local tests can validate the constructive proof too.
    """
    universe, old = set(universe), set(old)
    ordered = list(order)
    if target not in ordered or target in old or not old <= universe:
        raise ValueError("invalid target or old catalog")
    before = set(ordered[:ordered.index(target)])
    optional = universe - old
    high = before & optional
    low = optional - high - {target}
    chosen = {target}
    for group in map(set, groups):
        if chosen & group:
            continue
        pool = group & low or group & high
        if not pool:
            return None
        chosen.add(min(pool))
    if len(chosen) > admissions or admissions > len(optional):
        return None
    for pool in (low, high):
        for leaf in sorted(pool - chosen):
            if len(chosen) == admissions:
                break
            chosen.add(leaf)
    catalog = old | chosen
    returned = [leaf for leaf in ordered if leaf in catalog][:k]
    return catalog if len(chosen) == admissions and target in returned else None
