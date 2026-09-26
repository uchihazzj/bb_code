"""Optional CUDA decoding/scoring adapter for an already prepared LC-Rec model.

Caller supplies model/tokenizer, sets model._inference_autocast, and uses raw
0..255 SID digits (T5's position offsets do not apply here).
"""
import hashlib, copy
import numpy as np
import torch
from transformers import LogitsProcessor
TOKEN_CACHE = {}
DELIMITER = '|start_of_answer|'

def pack(paths):
    return np.asarray(paths, dtype=np.int64) @ (256 ** np.arange(3, -1, -1))


def code_text(paths):
    return [''.join(f'<{letter}_{int(value)}>' for letter, value in zip('abcd', row)) for row in paths]


def prompt_rows(histories, item_ids, paths, tokenizer):
    lookup = dict(zip(item_ids.tolist(), code_text(paths)))
    rows = []
    for history in histories:
        text = ', '.join(lookup[int(item)] for item in history if item != 0)
        rows.append(tokenizer.encode('What would user be likely to purchase next after buying items '+text+' ?'+DELIMITER, add_special_tokens=False))
    return rows

class CatalogTrace(LogitsProcessor):
    def __init__(self, paths, token_paths, prompt_width, batch, beam, vocab):
        self.width,self.batch,self.beam=prompt_width,batch,beam
        self.paths=torch.as_tensor(paths,device='cuda',dtype=torch.long)
        self.tokens=torch.as_tensor(token_paths,device='cuda',dtype=torch.long)
        self.bounds=torch.zeros((batch,len(paths)),device='cuda')
        self.depth=torch.zeros((batch,len(paths)),device='cuda',dtype=torch.int8)
        self.parents=[]
        self.lookup=torch.full((vocab,),-1,device='cuda',dtype=torch.long)
        unique, first_token = np.unique(token_paths.reshape(-1), return_index=True)
        self.lookup[torch.as_tensor(unique,device='cuda')]=torch.as_tensor(paths.reshape(-1)[first_token],device='cuda')
        for d in range(4):
            parents=np.zeros(len(paths),np.int64) if d==0 else paths[:,:d]@(256**np.arange(d-1,-1,-1))
            keys,first=np.unique(parents,return_index=True)
            self.parents.append(tuple(torch.as_tensor(z,device='cuda') for z in (parents,keys,first)))
        self.last=None

    def __call__(self, input_ids, scores):
        d=input_ids.shape[1]-self.width
        assert 0<=d<4
        prefix=input_ids[:,self.width:]
        digits=self.lookup[prefix]
        assert (digits>=0).all()
        code=(digits*(256**torch.arange(d-1,-1,-1,device='cuda'))).sum(1).reshape(self.batch,self.beam)
        parents,keys,first=self.parents[d]
        ix=torch.searchsorted(keys,code)
        assert (ix<len(keys)).all() and torch.equal(keys[ix],code)
        q=torch.arange(self.batch,device='cuda')[:,None]
        previous=self.bounds[q,first[ix]] if d else torch.zeros_like(code,dtype=scores.dtype)
        ordered,slots=torch.sort(code,dim=1,stable=True)
        pos=torch.searchsorted(ordered.contiguous(),parents[None,:].expand(self.batch,-1).contiguous())
        safe=pos.clamp(max=self.beam-1)
        live=(pos<self.beam)&(ordered.gather(1,safe)==parents[None,:])
        beam=slots.gather(1,safe)
        raw=scores.reshape(self.batch,self.beam,-1)
        values=previous.gather(1,beam)+raw[q,beam,self.tokens[:,d][None,:]]
        self.bounds=torch.where(live,values,self.bounds)
        self.depth=torch.where(live,d+1,self.depth)
        if d==3:
            self.last=(code.detach().cpu().numpy(),previous.detach().cpu().numpy(),raw.detach())
        masked=torch.full_like(scores,-torch.inf)
        query,slot,item=torch.nonzero(code[:,:,None]==parents[None,None,:],as_tuple=True)
        rows=query*self.beam+slot
        children=self.tokens[item,d]
        masked[rows,children]=scores[rows,children]
        return masked


def token_paths(tokenizer, paths):
    key=(tokenizer.name_or_path,hashlib.sha256(paths.tobytes()).hexdigest())
    if key in TOKEN_CACHE:return TOKEN_CACHE[key]
    tokens=np.asarray([tokenizer.encode(t,add_special_tokens=False) for t in code_text(paths)])
    assert tokens.shape==paths.shape
    if len(TOKEN_CACHE)>8:TOKEN_CACHE.clear()
    TOKEN_CACHE[key]=tokens
    return tokens


def padded(prompts, pad):
    width=max(map(len,prompts))
    ids=torch.tensor([[pad]*(width-len(x))+x for x in prompts],device='cuda')
    att=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in prompts],device='cuda')
    return ids,att


@torch.inference_mode()
def beam_batch(model,tokenizer,prompts,paths,beam=40,terminal=False,share_prompt_cache=True):
    ids,att=padded(prompts,tokenizer.pad_token_id)
    tokens=token_paths(tokenizer,paths)
    observer=CatalogTrace(paths,tokens,ids.shape[1],len(ids),beam,len(tokenizer))
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model._inference_autocast):
        kwargs={}
        if share_prompt_cache:
            # https://huggingface.co/docs/transformers/kv_cache
            # In Transformers5.9 _expand_inputs_for_generation expands tensors
            # but not Cache objects, so repeat the cache explicitly for beams.
            position=att.long().cumsum(-1)-1;position.masked_fill_(att==0,0)
            prefill=model(input_ids=ids[:,:-1],attention_mask=att[:,:-1],position_ids=position[:,:-1],use_cache=True,logits_to_keep=1)
            cache=prefill.past_key_values
            cache.batch_repeat_interleave(beam)
            kwargs['past_key_values']=cache
        output=model.generate(input_ids=ids,attention_mask=att,num_beams=beam,num_return_sequences=beam,
            do_sample=False,max_new_tokens=4,min_new_tokens=4,early_stopping=False,length_penalty=1.,
            use_cache=True,logits_to_keep=1,renormalize_logits=False,repetition_penalty=1.,
            eos_token_id=None,pad_token_id=tokenizer.pad_token_id,logits_processor=[observer],**kwargs)
    seq=output[:,ids.shape[1]:].reshape(len(ids),beam,4)
    codes=(observer.lookup[seq]*(256**torch.arange(3,-1,-1,device='cuda'))).sum(-1).cpu().numpy()
    index={int(c):i for i,c in enumerate(pack(paths))}
    indices=np.asarray([[index[int(c)] for c in row] for row in codes])
    assert all(len(set(row))==beam for row in indices)
    result={'indices':indices,'upper':observer.bounds.cpu().numpy(),'depth':observer.depth.cpu().numpy()}
    assert (np.take_along_axis(result['depth'],indices,axis=1)==4).all()
    if terminal:
        prefixes,previous,raw=observer.last
        d_tokens=torch.tensor([tokenizer.convert_tokens_to_ids(f'<d_{i}>') for i in range(256)],device='cuda')
        result['terminal_codes']=(prefixes[:,:,None]*256+np.arange(256)).reshape(len(ids),-1)
        result['terminal_scores']=(previous[:,:,None]+raw[:,:,d_tokens].cpu().numpy()).reshape(len(ids),-1)
    return result


@torch.inference_mode()
def exact_scores(model,tokenizer,prompts,paths,queries,items,batch_size=32,use_prompt_cache=True):
    paths_tokens=token_paths(tokenizer,paths)
    result=[]
    if not len(queries):return np.empty(0,np.float32)
    if use_prompt_cache:
        prompt_ids,prompt_att=padded(prompts,tokenizer.pad_token_id)
        position=prompt_att.long().cumsum(-1)-1;position.masked_fill_(prompt_att==0,0)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model._inference_autocast):
            prefill=model(input_ids=prompt_ids[:,:-1],attention_mask=prompt_att[:,:-1],position_ids=position[:,:-1],use_cache=True,logits_to_keep=1)
    for lo in range(0,len(queries),batch_size):
        kwargs={}
        if use_prompt_cache:
            q=torch.tensor(queries[lo:lo+batch_size],device='cuda',dtype=torch.long)
            rows=[[prompts[int(q)][-1]]+paths_tokens[int(i)].tolist()+[tokenizer.eos_token_id] for q,i in zip(queries[lo:lo+batch_size],items[lo:lo+batch_size])]
            ids=torch.tensor(rows,device='cuda')
            att=torch.cat([prompt_att[q,:-1],torch.ones((len(q),6),device='cuda',dtype=prompt_att.dtype)],dim=1)
            cache=copy.deepcopy(prefill.past_key_values);cache.batch_select_indices(q)
            kwargs['past_key_values']=cache
        else:
            rows=[prompts[int(q)]+paths_tokens[int(i)].tolist()+[tokenizer.eos_token_id] for q,i in zip(queries[lo:lo+batch_size],items[lo:lo+batch_size])]
            ids,att=padded(rows,tokenizer.pad_token_id)
        position=att.long().cumsum(-1)-1;position.masked_fill_(att==0,0)
        position=position[:,-ids.shape[1]:]
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model._inference_autocast):
            out=model(input_ids=ids,attention_mask=att,position_ids=position,use_cache=use_prompt_cache,logits_to_keep=6,**kwargs)
        logits=out.logits[:,:4,:].float()
        value=logits.log_softmax(-1).gather(-1,ids[:,-5:-1,None]).squeeze(-1).sum(-1)
        result.append(value.cpu().numpy())
    return np.concatenate(result) if result else np.empty(0,np.float32)

def response_loss(model, batch):
    # Qwen logits_to_keep is documented on its forward API. The final five
    # input tokens are SID[0:4],EOS; predictions begin one position earlier.
    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   use_cache=False, logits_to_keep=6)
    logits = output.logits[:, :4, :].float()
    target = batch['input_ids'][:, -5:-1]
    assert torch.equal(target, batch['labels'][:, -5:-1])
    return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
