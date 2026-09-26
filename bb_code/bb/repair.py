"""Integral minimum-replacement repair; identity changes are a separate quantity."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog

def feasible(chosen, universe, groups, high, target, count, high_limit):
    return (len(chosen)==count and target in chosen and chosen<=universe
            and len(chosen&high)<=high_limit and all(chosen&g for g in groups))


def minimum_replacement(universe, original, groups, high, target, count, high_limit):
    universe=set(universe); original=set(original); high=set(high)&universe
    groups=[set(g)&universe for g in groups]
    if target not in universe or count<1 or high_limit<0 or len(universe)<count:
        return None
    owner={}
    for index,group in enumerate(groups):
        for leaf in group:
            if leaf in owner:raise ValueError('Support groups must be disjoint')
            owner[leaf]=index
    required=[i for i,g in enumerate(groups) if target not in g]
    buckets=defaultdict(list)
    for leaf in sorted(universe-{target}):
        buckets[(owner.get(leaf,-1),int(leaf in high),int(leaf not in original))].append(leaf)
    if count==1:
        chosen={target}
        return chosen if feasible(chosen,universe,groups,high,target,count,high_limit) else None
    keys=sorted(buckets);n=len(keys)
    if not n:return None
    aub=[[int(k[1]) for k in keys]];bub=[high_limit-int(target in high)]
    for group in required:
        aub.append([-int(k[0]==group) for k in keys]);bub.append(-1)
    result=linprog([k[2] for k in keys],A_ub=np.array(aub,dtype=float),b_ub=bub,
                   A_eq=np.ones((1,n)),b_eq=[count-1],bounds=[(0,len(buckets[k])) for k in keys],
                   method='highs-ds',options={'time_limit':10.0})
    if result.status==2:return None
    if not result.success:raise RuntimeError(f'LP did not solve: {result.message}')
    integers=np.rint(result.x).astype(int)
    if not np.allclose(result.x,integers,atol=1e-7,rtol=0):
        raise RuntimeError('Expected integral network vertex')
    chosen={target}
    for k,number in zip(keys,integers):chosen.update(buckets[k][:number])
    assert feasible(chosen,universe,groups,high,target,count,high_limit)
    assert len(chosen-original)==round(result.fun)+int(target not in original)
    return chosen


def stable_identity_mapping(original_codes, target_index, target_code, chosen):
    """Pin target, preserve every other compatible identity, fill displaced IDs."""
    chosen=set(chosen);codes=np.asarray(original_codes,dtype=np.int64)
    assert len(chosen)==len(codes) and target_code in chosen
    result=np.full(len(codes),-1,dtype=np.int64);result[target_index]=target_code
    used={int(target_code)}
    for i,code in enumerate(codes):
        if i!=target_index and int(code) in chosen and int(code) not in used:
            result[i]=code;used.add(int(code))
    remaining=iter(sorted(chosen-used))
    for i in np.flatnonzero(result<0):result[i]=next(remaining)
    assert set(result.tolist())==chosen and result[target_index]==target_code
    return result
