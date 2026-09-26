"""Encoding, native beam decoding, ranking metrics and fixed-map training."""
import random, time, json
import numpy as np
import torch
from .io import load, write
from .catalog import CatalogMask, pack
from .native import SETTINGS, W
from . import native as native_branch
from .native import capture_parent, third_candidates, transition, final_step
from .states import unpack, state_data, plan_states

def decode(model,x,paths):
 mask=CatalogMask(paths,device=model.device);pred=[]
 for lo in range(0,len(x),64):
  ids=torch.as_tensor(x[lo:lo+64],device=model.device)
  y=model.generate(input_ids=ids,attention_mask=ids.ne(0),logits_processor=[mask],**SETTINGS)
  pred.append(y[:,1:5].reshape(len(ids),W,4).cpu().numpy())
 return pack(np.concatenate(pred))

def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def encoded(data,m,paths):
    lookup=m['all_paths'].copy();lookup[m['item_ids']]=paths
    x=lookup[data['history_items']].reshape(len(data['history_items']),-1)
    y=lookup[data['target_items']]
    return x,y


def ranks_metrics(pred,m,paths,data):
    codes=pack(paths);order=np.argsort(codes);pos=np.searchsorted(codes[order],pred)
    assert (pos<len(order)).all() and np.array_equal(codes[order][pos],pred)
    items=m['item_ids'][order][pos];hit=items==data['target_items'][:,None]
    ranks=np.where(hit.any(1),hit.argmax(1)+1,0)
    old=np.isin(data['target_items'],m['old_item_ids']);current=np.isin(data['target_items'],m['item_ids'])
    masks={'all':np.ones(len(ranks),bool),'current':current,'old':old,'new':current&~old,'future':~current,'primary':current&~old&np.isin(data['history_items'],np.r_[0,m['old_item_ids']]).all(1)}
    result={}
    for group,mask in masks.items():
        r=ranks[mask];z={'n':int(mask.sum())}
        for k in [10,20]:
            good=(r>0)&(r<=k);z[f'hits{k}']=int(good.sum());z[f'recall{k}']=float(good.mean()) if len(r) else None
            gains=np.zeros(len(r));gains[good]=1/np.log2(r[good]+1);z[f'ndcg{k}']=float(gains.mean()) if len(r) else None
        result[group]=z
    return result,ranks,items


def evaluate(model,data,m,paths):
    model.eval()
    with torch.inference_mode(): pred=decode(model,encoded(data,m,paths)[0],paths)
    return pred,ranks_metrics(pred,m,paths,data)[0]


def train_epochs(model,data,m,paths,epochs,seed,log,callback=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    x,y=encoded(data,m,paths); x=torch.as_tensor(x,device=model.device);y=torch.as_tensor(y,device=model.device)
    opt=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.01)
    start=time.monotonic();rows=[];training_seconds=0.
    for epoch in range(1,epochs+1):
        epoch_started=time.monotonic();model.train();order=torch.randperm(len(x),device=model.device);total=0.
        for lo in range(0,len(x),256):
            ix=order[lo:lo+256];batch=x[ix];opt.zero_grad(set_to_none=True)
            loss=model(input_ids=batch,attention_mask=batch.ne(0),labels=y[ix],use_cache=False).loss
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=float(loss.detach())*len(ix)
        training_seconds+=time.monotonic()-epoch_started
        row={'epoch':epoch,'train_loss':total/len(x),'seconds':time.monotonic()-start,'optimizer_seconds':training_seconds};rows.append(row);write(log,rows)
        print(json.dumps({'phase':log.stem,**row}),flush=True)
        if callback:callback(epoch,training_seconds)
    model.eval();return time.monotonic()-start
