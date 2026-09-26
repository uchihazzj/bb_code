import numpy as np
from .catalog import pack
from .feasibility import intervals
B=256

def plan_states(scores, mandatory, width, max_critical=8, max_states=256,
                reject_ties=True):
    """Return READY or explicit UNKNOWN; never use truncated state enumeration.

    Integer indices identify candidates and supply the toy's fixed lexical tie order.
    Native use rejects any selected/internal boundary tie instead of assuming that order.
    """
    if width < 1 or len(scores) != len(mandatory):
        raise ValueError("invalid candidate dimensions or width")
    old = [i for i, required in enumerate(mandatory) if required]
    result = {"status": "UNKNOWN", "reason": None, "candidate_count": len(scores),
              "mandatory_count": len(old), "q": None, "states": [], "representatives": []}
    if len(old) < width:
        return result | {"reason": "insufficient_mandatory_children"}
    old.sort(key=lambda i: (-scores[i], i))
    tau = scores[old[width - 1]]
    critical = [i for i, required in enumerate(mandatory) if not required and scores[i] >= tau]
    result.update(tau=tau, critical=critical, q=len(critical), subset_count=2 ** len(critical))
    if len(critical) > max_critical:
        return result | {"reason": "critical_count_cap"}
    seen = {}
    for mask in range(2 ** len(critical)):
        enabled = old + [c for j, c in enumerate(critical) if mask & (1 << j)]
        enabled.sort(key=lambda i: (-scores[i], i))
        relevant = enabled[:width + 1]
        if reject_ties and any(scores[a] == scores[b] for a, b in zip(relevant, relevant[1:])):
            return result | {"reason": "intermediate_tie", "tie_subset": mask}
        state = tuple(enabled[:width])
        seen.setdefault(state, mask)
        if len(seen) > max_states:
            return result | {"reason": "state_count_cap"}
    return result | {"status": "READY", "reason": None,
                     "states": list(seen), "representatives": list(seen.values()),
                     "state_count": len(seen)}

def unpack(c):
 c=np.asarray(c,dtype=np.int64)
 return np.stack([(c//B**(3-d))%B+1+d*B for d in range(4)],axis=-1)


def state_data(state,score,slot,old,old_prefixes,children,third,mandatory):
 tokens=state[0][slot,:,1:4].cpu().numpy();prefix=pack(tokens)
 codes=(prefix[:,None]*B+np.arange(B)).reshape(-1)
 values=score[slot].cpu().numpy().reshape(-1)
 order=np.argsort(-values,kind='stable')
 selected=set(prefix.tolist());cutoff=float(state[1][slot].min().item())
 forbidden=set(children[(~mandatory)&(third.reshape(-1)>cutoff)].tolist())-selected
 groups=[np.arange(i*B,(i+1)*B) for i in np.flatnonzero(~np.isin(prefix,old_prefixes))]
 oldmask=np.isin(codes,list(old));capacity=B**4-len(old)-B*len(forbidden)
 interval=intervals(order,oldmask,groups,capacity,20)
 return dict(tokens=tokens,prefix=prefix,codes=codes,values=values,order=order,
             forbidden=forbidden,groups=groups,oldmask=oldmask,capacity=capacity,interval=interval)
