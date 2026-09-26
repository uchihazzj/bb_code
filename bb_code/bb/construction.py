"""BB-only proposal generation over the preset states, with a proposal cap.

The experimental driver coupled early stopping to ablation-control success.
This portable entry point removes that dependency and visits the preset states.
It retains the fixed-state solver and native witness checks.
"""
import hashlib, time
import numpy as np
import torch
from . import generator as base
from .repair import minimum_replacement, stable_identity_mapping
from .feasibility import intervals
METHODS = ['minimum_edit']

def subset(data, rows):
    return {k: v[rows] for (k, v) in data.items()}

def signature(paths):
    return hashlib.sha256(paths.tobytes()).hexdigest()

def changed(paths, m):
    return int(np.any(paths != m['paths'], axis=1).sum())

def key(metrics, edits):
    return (metrics['all']['ndcg10'], metrics['all']['recall10'], -edits)

def compact_universe(info, original, old, m):
    forbidden = info['forbidden']
    codes = info['codes']
    terminal = set(codes.tolist())
    universe = {int(c) for c in original | terminal if c not in old and c // 256 not in forbidden}
    fillers = 0
    leaf = 0
    while fillers < m and leaf < 256 ** 4:
        if leaf not in terminal and leaf not in original and (leaf not in old) and (leaf // 256 not in forbidden):
            universe.add(leaf)
            fillers += 1
        leaf += 1
    groups = [set(codes[g].tolist()) for g in info['groups']]
    return (universe, groups)

def generate(model, train, m, sample, out, protocol, progress):
    p = protocol['candidate_generation']
    budget = p['identity_edit_budget']
    limit = p['proposals_per_method']
    newrows = np.flatnonzero(~np.isin(m['item_ids'], m['old_item_ids']))
    n = len(newrows)
    original = base.pack(m['paths'][newrows])
    originalset = set(original.tolist())
    old = set(base.pack(m['old_paths']).tolist())
    oldpref = np.unique(base.pack(m['old_paths'][:, :3]))
    x = base.encoded(train, m, m['paths'])[0]
    with torch.inference_mode():
        initial_predictions = base.decode(model, x[sample], m['paths'])
    counters = {f: {'solver_calls': 0, 'native_decode_calls': 0, 'solver_seconds': 0.0, 'native_decode_seconds': 0.0} for f in METHODS}
    proposals = {method: [] for method in METHODS}
    seen = {method: set() for method in METHODS}
    attempts = []
    for (index, row) in enumerate(sample):
        targetrow = int(np.flatnonzero(m['item_ids'] == train['target_items'][row])[0])
        target_index = int(np.flatnonzero(newrows == targetrow)[0])
        if int(original[target_index]) in initial_predictions[index, :10]:
            attempts.append({'row': int(row), 'status': 'already_hit10'})
            continue
        with torch.inference_mode():
            cap = base.capture_parent(model, torch.as_tensor(x[row:row + 1], device=model.device), m['old_paths'])
            ref = base.transition(model, cap)
            ref_scores = base.final_step(model, cap, ref)
        shared = bool(torch.stack(cap.flags, dim=1)[0, :2].all())
        state_pairs = []
        if shared:
            (_, third, mandatory, children) = base.third_candidates(cap, 0, m['old_paths'])
            plan = base.plan_states(third.reshape(-1).tolist(), mandatory.tolist(), base.W, p['max_q'], p['max_states_enumerated'], reject_ties=True)
            if plan['status'] == 'READY':
                full = tuple(np.argsort(-third.reshape(-1), kind='stable')[:base.W])
                state_pairs = sorted(zip(plan['states'], plan['representatives']), key=lambda z: (tuple(z[0]) != full, tuple(z[0])))[:p['states_per_context']]
            else:
                attempts.append({'row': int(row), 'status': plan['status']})
        else:
            attempts.append({'row': int(row), 'status': 'prefix_unknown'})

        def retain(method, chosen, y, meta):
            if len(proposals[method]) >= limit:
                return False
            candidate = m['paths'].copy()
            candidate[newrows] = base.unpack(stable_identity_mapping(original, target_index, y, chosen))
            edits = changed(candidate, m)
            sig = signature(candidate)
            if not 0 < edits <= budget or sig in seen[method]:
                return False
            seen[method].add(sig)
            native_start = time.monotonic()
            counters[method]['native_decode_calls'] += 2
            with torch.inference_mode():
                fixed = base.decode(model, x[row:row + 1], candidate)[0]
                one = subset(train, np.array([row]))
                (actual, _) = base.evaluate(model, one, m, candidate)
            counters[method]['native_decode_seconds'] += time.monotonic() - native_start
            optimization = meta.pop('_optimization', None)
            fixedhit = np.flatnonzero(fixed[:10] == y)
            actualhit = np.flatnonzero(actual[0, :10] == y)
            record = {'method': method, 'row': int(row), 'target_item': int(train['target_items'][row]), 'target_code': int(y), 'identity_changes': edits, 'leaf_replacements': len(set(chosen) - originalset), 'mapping_sha': sig, 'fixed_rank': int(fixedhit[0]) + 1 if len(fixedhit) else None, 'reencoded_rank': int(actualhit[0]) + 1 if len(actualhit) else None, **meta}
            if not len(fixedhit) or not len(actualhit):
                attempts.append({**record, 'status': 'native_witness_failed'})
                return False
            idx = len(proposals[method])
            if optimization is not None:
                base.write(out / f'optimization_{method}_{idx}.json', optimization)
            np.savez_compressed(out / f'proposal_{method}_{idx}.npz', paths=candidate, fixed_predictions=fixed, reencoded_predictions=actual)
            proposals[method].append(record)
            return True
        for (state_index, (stateidx, subset_codes)) in enumerate(state_pairs):
            if all((len(proposals[f]) >= limit for f in ['minimum_edit'])):
                break
            with torch.inference_mode():
                if tuple(stateidx) == full:
                    (st, sc) = (ref, ref_scores)
                else:
                    st = base.transition(model, cap, 0, plan['critical'], subset_codes)
                    sc = base.final_step(model, cap, st)
            info = base.state_data(st, sc, 0, old, oldpref, children, third, mandatory)
            interval = intervals(info['order'], info['oldmask'], info['groups'], info['capacity'], 10)
            ok = interval['possible'] & (interval['lower'] <= n) & (n <= interval['upper'])
            eligible_targets = [int(i) for i in info['order'] if ok[i] and np.count_nonzero(info['values'] == info['values'][i]) == 1]
            original_targets = [i for i in eligible_targets if int(info['codes'][i]) == int(original[target_index])]
            target_indices = (original_targets + [i for i in eligible_targets if i not in original_targets])[:p['targets_per_state']]
            if not target_indices:
                continue
            (universe, groups) = compact_universe(info, originalset, old, n)
            success = {f: False for f in ['minimum_edit']}
            for target in target_indices:
                y = int(info['codes'][target])
                before = set(info['codes'][info['order'][:int(interval['rank'][target]) - 1]].tolist())
                high = before - old
                h = 9 - len(before & old)
                for (method, solver) in [('minimum_edit', minimum_replacement)]:
                    if success[method] or len(proposals[method]) >= limit:
                        continue
                    solve_start = time.monotonic()
                    counters[method]['solver_calls'] += 1
                    chosen = solver(universe, originalset, groups, high, y, n, h)
                    counters[method]['solver_seconds'] += time.monotonic() - solve_start
                    if chosen is not None:
                        success[method] = retain(method, chosen, y, {'state_index': state_index, 'q': int(plan['q']), 'state_indices': list(map(int, stateidx)), '_optimization': {'universe': sorted(universe), 'original': sorted(originalset), 'groups': [sorted(g) for g in groups], 'high': sorted(high), 'target': y, 'count': n, 'high_limit': h, 'chosen': sorted(chosen)}, 'guarantee': 'fixed_state_target_leaf_optimum'})
        progress('proposal_generation', contexts=index + 1, counts={k: len(v) for (k, v) in proposals.items()})
        base.write(out / 'proposals.json', {'methods': proposals, 'attempts': attempts, 'sample': sample.tolist(), 'counters': counters})
        if all((len(proposals[f]) >= limit for f in METHODS)):
            break
    base.write(out / 'proposals.json', {'methods': proposals, 'attempts': attempts, 'sample': sample.tolist(), 'counters': counters})
    return proposals
