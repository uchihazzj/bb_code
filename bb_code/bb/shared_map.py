"""Training-context selection over feasible proposals and compatible compositions.

The final map is selected from modified maps. The original map is the fallback
only when the candidate set is empty. This is finite candidate selection, not
optimization over all legal catalog assignments.
"""
import numpy as np
from . import generator
from .construction import generate, signature, changed, subset, key
from .io import load


def select_shared_map(model, train, mapping, output, protocol, seed=17):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed + 14000)
    new_ids = mapping['item_ids'][~np.isin(mapping['item_ids'], mapping['old_item_ids'])]
    new = np.flatnonzero(np.isin(train['target_items'], new_ids))
    old = np.flatnonzero(np.isin(train['target_items'], mapping['old_item_ids']))
    rng.shuffle(new)
    rng.shuffle(old)
    limit = protocol['training_selection']['candidate_audit_rows']
    na = min(len(new) // 2, limit // 2)
    no = min(len(old), limit - na)
    audit_rows = np.sort(np.r_[new[:na], old[:no]])
    fit = new[na:]
    first = [int(rng.choice(fit[train['target_items'][fit] == item]))
             for item in np.unique(train['target_items'][fit])]
    rng.shuffle(first)
    remaining = np.setdiff1d(fit, first)
    rng.shuffle(remaining)
    sample = np.r_[first, remaining][:protocol['candidate_generation']['sample_contexts']].astype(int)
    assert not set(sample) & set(audit_rows)
    if not len(audit_rows) or not len(sample):
        return mapping['paths'].copy(), {'candidates': 0, 'fallback': True}
    audit = subset(train, audit_rows)
    model.eval()
    generator.native_branch.V = model.config.vocab_size
    proposals = generate(model, train, mapping, sample, output, protocol, lambda *a, **k: None)['minimum_edit']
    cache = {}

    def evaluate(paths):
        sig = signature(paths)
        if sig not in cache:
            cache[sig] = generator.evaluate(model, audit, mapping, paths)[1]
        return cache[sig]

    paths_by_proposal = [load(output / f'proposal_minimum_edit_{i}.npz')['paths']
                         for i in range(len(proposals))]
    # Every retained proposal belongs to the final candidate pool, even if no
    # greedy composition improves the original map on the audit subset.
    for paths in paths_by_proposal:
        evaluate(paths)
    accepted = {signature(p): p for p in paths_by_proposal}
    current = mapping['paths'].copy()
    current_changed, used = set(), set()
    for _ in range(protocol['training_selection']['shared_steps']):
        best_key = key(evaluate(current), changed(current, mapping))
        best = None
        for index, paths in enumerate(paths_by_proposal):
            if index in used:
                continue
            edits = set(np.flatnonzero(np.any(paths != mapping['paths'], axis=1)).tolist())
            if edits & current_changed:
                continue
            merged = current.copy()
            ix = sorted(edits)
            merged[ix] = paths[ix]
            if (changed(merged, mapping) > protocol['candidate_generation']['identity_edit_budget']
                    or len(np.unique(generator.pack(merged))) != len(merged)):
                continue
            metrics = evaluate(merged)
            candidate_key = key(metrics, changed(merged, mapping))
            if candidate_key > best_key:
                best_key = candidate_key
                best = index, merged, edits
        if best is None:
            break
        index, current, edits = best
        accepted[signature(current)] = current.copy()
        used.add(index)
        current_changed |= edits
    # Historical selection includes individual proposals and accepted chain
    # members, rather than every trial composition evaluated during the search.
    if not accepted:
        return mapping['paths'].copy(), {'candidates': 0, 'fallback': True}
    selected_sig = max(accepted, key=lambda s: (*key(cache[s], changed(accepted[s], mapping)), s))
    selected = accepted[selected_sig]
    old = np.isin(mapping['item_ids'], mapping['old_item_ids'])
    assert np.array_equal(selected[old], mapping['paths'][old])
    return selected, {'candidates': len(accepted), 'fallback': False,
                      'changed_identities': changed(selected, mapping),
                      'training_metrics': cache[selected_sig]}
