"""Retained prefix bounds and exact teacher-forced identifier likelihoods."""
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
from .catalog import CatalogMask, pack
from .native import SETTINGS

class PrefixBounds:
 def __init__(self,paths,batch,device,width=40):
  self.paths=torch.as_tensor(paths,device=device);self.batch=batch;self.width=width;self.mask=CatalogMask(paths,device=device);self.bounds=torch.zeros((batch,len(paths)),device=device);self.depth=torch.zeros_like(self.bounds,dtype=torch.int8);self.q=torch.arange(batch,device=device)[:,None];self.parents=[]
  for d in range(4):
   code=pack(paths[:,:d]);keys,first=np.unique(code,return_index=True);self.parents.append((torch.as_tensor(code,device=device),torch.as_tensor(keys,device=device),torch.as_tensor(first,device=device)))
 def __call__(self,input_ids,scores):
  d=input_ids.shape[1]-1;q=self.batch;w=self.width;prefix=input_ids[:,1:].reshape(q,w,d);code=torch.zeros((q,w),dtype=torch.long,device=scores.device)
  for j in range(d):code=code*256+prefix[:,:,j]-1-256*j
  item_parent,keys,first=self.parents[d];index=torch.searchsorted(keys,code);assert (index<len(keys)).all() and torch.equal(keys[index],code)
  parent=self.bounds[self.q,first[index]] if d else torch.zeros((q,w),device=scores.device)
  ordered,slots=torch.sort(code,dim=1,stable=True);pos=torch.searchsorted(ordered.contiguous(),item_parent[None,:].expand(q,-1).contiguous());safe=pos.clamp(max=w-1);live=(pos<w)&(ordered.gather(1,safe)==item_parent[None,:]);beam=slots.gather(1,safe)
  raw=scores.reshape(q,w,-1);candidate=parent.gather(1,beam)+raw[self.q,beam,self.paths[:,d][None,:]]
  self.bounds=torch.where(live,candidate,self.bounds);self.depth=torch.where(live,d+1,self.depth)
  return self.mask(input_ids,scores)


@torch.inference_mode()
def trace(model,x,paths,batch_size=64):
 model.eval();pred=[];upper=[];depth=[]
 for lo in range(0,len(x),batch_size):
  ids=torch.as_tensor(x[lo:lo+batch_size],device=model.device);observer=PrefixBounds(paths,len(ids),model.device)
  output=model.generate(input_ids=ids,attention_mask=ids.ne(0),logits_processor=[observer],**SETTINGS)
  pred.append(pack(output[:,1:5].reshape(len(ids),40,4).cpu().numpy()));upper.append(observer.bounds.cpu().numpy());depth.append(observer.depth.cpu().numpy())
 return {'beam_codes':np.concatenate(pred),'upper':np.concatenate(upper),'depth':np.concatenate(depth)}


class ExactScorer:
 def __init__(self,model,x,paths):
  self.model=model;self.paths=torch.as_tensor(paths,device=model.device);self.x=torch.as_tensor(x,device=model.device);self.att=self.x.ne(0)
  with torch.inference_mode():self.enc=model.encoder(input_ids=self.x,attention_mask=self.att,return_dict=True).last_hidden_state
 @torch.inference_mode()
 def __call__(self,queries,items):
  if not len(queries):return np.empty(0,np.float32)
  values=[]
  for lo in range(0,len(queries),256):
   qr=torch.as_tensor(queries[lo:lo+256],device=self.model.device);it=torch.as_tensor(items[lo:lo+256],device=self.model.device);labels=self.paths[it]
   out=self.model(encoder_outputs=BaseModelOutput(last_hidden_state=self.enc[qr]),attention_mask=self.att[qr],labels=labels,use_cache=False)
   lp=out.logits.log_softmax(-1).gather(-1,labels[:,:,None]).squeeze(-1).sum(-1);values.append(lp.cpu().numpy())
  return np.concatenate(values)
