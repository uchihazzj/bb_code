"""Four position-specific 256-token blocks; raw full-vocabulary scores."""
from types import MethodType
import numpy as np
import torch
import transformers

def pack(paths, base=256, bias=1):
    value=np.zeros(paths.shape[:-1],dtype=np.int64)
    for d in range(paths.shape[-1]):
        value=value*base+paths[...,d]-bias-d*base
    return value


class CatalogMask:
    def __init__(self, paths, base=256, bias=1, device='cuda', full=False):
        if paths.ndim!=2 or not len(paths):raise ValueError('Nonempty SID matrix required')
        raw=paths-bias-np.arange(paths.shape[1])*base
        if not np.issubdtype(paths.dtype,np.integer) or np.any((raw<0)|(raw>=base)):
            raise ValueError('SID tokens are outside their position-specific codebooks')
        self.base,self.bias,self.length,self.full=base,bias,paths.shape[1],full
        self.nodes=[];self.children=[]
        for depth in range(self.length):
            prefix=pack(paths[:,:depth],base,bias)
            keys,indices=np.unique(prefix,return_inverse=True)
            children=np.zeros((len(keys),base),dtype=bool)
            children[indices,paths[:,depth]-bias-depth*base]=True
            self.nodes.append(torch.as_tensor(keys,device=device))
            self.children.append(torch.as_tensor(children,device=device))

    def __call__(self,input_ids,scores):
        depth=input_ids.shape[1]-1
        if not 0<=depth<self.length:raise ValueError('Unexpected SID length')
        start=self.bias+depth*self.base
        result=torch.full_like(scores,-torch.inf)
        if self.full:
            result[:,start:start+self.base]=scores[:,start:start+self.base]
        else:
            code=torch.zeros(len(input_ids),device=scores.device,dtype=torch.long)
            for d in range(depth):code=code*self.base+input_ids[:,1+d]-self.bias-d*self.base
            keys=self.nodes[depth];index=torch.searchsorted(keys,code)
            valid=(index<len(keys))&(keys[index.clamp(max=len(keys)-1)]==code)
            allowed=self.children[depth][index.clamp(max=len(keys)-1)]&valid[:,None]
            result[:,start:start+self.base]=scores[:,start:start+self.base].masked_fill(~allowed,-torch.inf)
        return result


class ClosureObserver:
    def __init__(self,model,mask,old_paths,batch,width,device='cuda'):
        if transformers.__version__!='5.9.0':
            raise RuntimeError('Private native hook is validated only for Transformers 5.9.0')
        self.model,self.mask,self.batch,self.width=model,mask,batch,width
        self.old=[torch.as_tensor(np.unique(pack(old_paths[:,:d],mask.base,mask.bias)),device=device)
                  for d in range(1,mask.length+1)]
        self.live=torch.full((batch,width),-1e9,device=device);self.live[:,0]=0
        self.flags=[];self.margins=[];self.scores=[];self.prefixes=[];self.original=None

    def __call__(self,input_ids,scores):
        d=input_ids.shape[1]-1;w=self.width;b=self.mask.base;bias=self.mask.bias
        if d!=len(self.flags):raise RuntimeError('Unexpected native score call order')
        start=bias+d*b
        cumulative=scores[:,start:start+b].reshape(self.batch,w,b)+self.live[:,:,None]
        values,indices=torch.topk(cumulative.flatten(1),w+1,sorted=True)
        prefix=input_ids[:,1:].reshape(self.batch,w,d)
        encoded=torch.zeros((self.batch,w),dtype=torch.long,device=scores.device)
        for k in range(d):encoded=encoded*b+prefix[:,:,k]-bias-k*b
        top=torch.gather(encoded,1,indices[:,:w]//b)*b+indices[:,:w]%b
        old=self.old[d];where=torch.searchsorted(old,top)
        belongs=(where<len(old))&(old[where.clamp(max=len(old)-1)]==top)
        margin=(values[:,:w]-values[:,1:w+1]).amin(dim=1)
        valid=torch.isfinite(values).all(dim=1)&(values[:,w-1]>-1e8)
        self.flags.append(belongs.all(dim=1)&(margin>0)&valid)
        self.margins.append(margin);self.scores.append(values);self.prefixes.append(top)
        return self.mask(input_ids,scores)

    def install(self):
        if self.original is not None or '_get_running_beams_for_next_iteration' in self.model.__dict__:
            raise RuntimeError('Native observer already installed')
        self.original=self.model._get_running_beams_for_next_iteration
        def running(module,*args,**kwargs):
            result=self.original(*args,**kwargs)
            self.live=result[1].detach().clone()
            return result
        self.model._get_running_beams_for_next_iteration=MethodType(running,self.model)

    def remove(self):
        if self.original is not None:
            delattr(self.model,'_get_running_beams_for_next_iteration');self.original=None

    def arrays(self):
        if len(self.flags)!=self.mask.length:raise RuntimeError('Incomplete native trace')
        return {name:torch.stack(value,dim=1).cpu().numpy() for name,value in
                [('flags',self.flags),('margins',self.margins),('top_scores',self.scores),
                 ('top_prefixes',self.prefixes)]}
