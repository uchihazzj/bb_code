"""Bound-guided Top-20 completion; certificate uses a 1e-3 numerical allowance.

The budget counts additional four-token identifier evaluations after E0.
E0 includes every terminal leaf scored by the beam, not just its returned leaves.
Inputs require at least 20 catalog items and finite prefix bounds/corrections.
"""
import numpy as np
ALLOWANCE = 1e-3

def stable_topk(values,k):
    values=np.asarray(values);one=values.ndim==1
    x=values[None,:] if one else values
    assert x.ndim==2 and 0<k<=x.shape[1]
    if np.isnan(x).any():
        chosen=np.argsort(-x,axis=1,kind='stable')[:,:k].copy()
        return chosen[0] if one else chosen
    chosen=np.argpartition(-x,k-1,axis=1)[:,:k].copy()
    scores=np.take_along_axis(x,chosen,axis=1)
    order=np.lexsort((chosen,-scores),axis=1);chosen=np.take_along_axis(chosen,order,axis=1);scores=np.take_along_axis(scores,order,axis=1)
    threshold=scores[:,-1]
    ties_all=(x==threshold[:,None]).sum(1);ties_kept=(scores==threshold[:,None]).sum(1)
    for q in np.flatnonzero(ties_all>ties_kept):
        higher=np.flatnonzero(x[q]>threshold[q]);tied=np.flatnonzero(x[q]==threshold[q])[:k-len(higher)]
        ix=np.concatenate([higher,tied]);chosen[q]=ix[np.lexsort((ix,-x[q,ix]))]
    return chosen[0] if one else chosen


def complete(bounds,depth,offset,scorer,priority='bound',budget=80):
 known=depth==4;values=np.where(known,bounds,-np.inf).copy();upper=bounds+offset;n=len(bounds);added=np.full((n,budget),-1,np.int32);lp=np.full((n,budget),np.nan,np.float32);count=np.zeros(n,np.int32);stopped=np.zeros(n,bool)
 for step in range(0,budget,20):
  score=values+offset;kth=np.sort(score,axis=1)[:,-20];unseen=np.where(~known,upper,-np.inf).max(1);stopped|=kth-ALLOWANCE>unseen+ALLOWANCE
  qs=[];its=[]
  for q in np.flatnonzero(~stopped):
   p=upper[q] if priority=='bound' else offset[q]
   ix=stable_topk(np.where(~known[q],p,-np.inf),min(20,budget-step));ix=ix[~known[q,ix]];qs.extend([int(q)]*len(ix));its.extend(ix.tolist())
  if not qs:break
  qs=np.array(qs);its=np.array(its);vals=scorer(qs,its);assert np.isfinite(vals).all();assert np.all(vals<=bounds[qs,its]+ALLOWANCE)
  for q,i,v in zip(qs,its,vals):j=count[q];added[q,j]=i;lp[q,j]=v;count[q]+=1
  values[qs,its]=vals;known[qs,its]=True
 score=values+offset;ix=stable_topk(score,20);kth=np.take_along_axis(score,ix,1)[:,-1];unseen=np.where(~known,upper,-np.inf).max(1)
 return {'indices':ix,'score':np.take_along_axis(score,ix,1),'added_indices':added,'added_logp':lp,'added_count':count,'initial_known':(depth==4).sum(1),'certified':kth-ALLOWANCE>unseen+ALLOWANCE,'gap':kth-unseen}

def calibrated_offset(prior,new,weight,gamma=0.,bias=0.):
 logq=np.log(np.maximum(prior,1e-8));offset=weight*logq
 if gamma==1. and bias==0.:return offset
 c=-np.log(prior.shape[1]);offset[:,new]=weight*c+gamma*weight*(logq[:,new]-c)+bias;return offset
