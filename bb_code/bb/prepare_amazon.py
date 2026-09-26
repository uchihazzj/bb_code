"""Build a frozen chronological dataset and fit an old-only content tokenizer."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.cluster import KMeans


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(protocol_path, out):
    start=time.monotonic()
    p=json.loads(protocol_path.read_text())
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_bytes(protocol_path.read_bytes())
    inputs={k:{'path':p[k],'sha256':sha(p[k])} for k in
            ('raw_reviews','feature_id_mapping','content_embeddings')}
    (out/'source_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    mapping=json.loads(Path(p['feature_id_mapping']).read_text())['item2id']
    events=[]
    with gzip.open(p['raw_reviews'],'rt') as f:
        for order,line in enumerate(f):
            row=json.loads(line)
            events.append((int(row['unixReviewTime']),order,row['reviewerID'],mapping[row['asin']]))
    events.sort()
    cuts=np.quantile([r[0] for r in events],p['time_quantiles'],method='inverted_cdf').astype(np.int64)
    old_ids=np.array(sorted({i for t,_,_,i in events if t<=cuts[0]}),dtype=np.int64)
    current_ids=np.array(sorted({i for t,_,_,i in events if t<=cuts[1]}),dtype=np.int64)
    emb=np.fromfile(p['content_embeddings'],dtype=np.float32).reshape(-1,768)
    assert len(emb)==len(mapping)-1 and np.isfinite(emb).all()
    residual=emb/np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    raw_codes=np.zeros((len(mapping),4),dtype=np.int64)
    centers=[]
    for d in range(3):
        # Official API: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        km=KMeans(n_clusters=64,n_init=1,max_iter=100,random_state=p['tokenizer']['random_seed']+d)
        km.fit(residual[old_ids-1])
        labels=km.predict(residual)
        raw_codes[1:,d]=labels
        centers.append(km.cluster_centers_)
        residual=residual-km.cluster_centers_[labels]
    occupied=defaultdict(set)
    old_set=set(old_ids.tolist())
    for item in list(old_ids)+[i for i in current_ids if i not in old_set]:
        prefix=tuple(raw_codes[item,:3])
        suffix=next((j for j in range(256) if j not in occupied[prefix]),None)
        if suffix is None:raise RuntimeError('Frozen suffix capacity exhausted')
        raw_codes[item,3]=suffix
        occupied[prefix].add(suffix)
    paths=raw_codes+1+np.arange(4)*256
    paths[0]=0
    assert len(np.unique(paths[current_ids],axis=0))==len(current_ids)
    np.savez_compressed(out/'mapping.npz',item_ids=current_ids,paths=paths[current_ids],
                        old_item_ids=old_ids,old_paths=paths[old_ids],all_paths=paths,
                        centers=np.stack(centers),cutoffs=cuts)
    users=defaultdict(list)
    for t,order,u,item in events:users[u].append((t,order,item))
    def arrays(rows):
        histories=np.array([[0]*(20-len(h))+h for _,h,_ in rows],dtype=np.int64)
        targets=np.array([t for _,_,t in rows],dtype=np.int64)
        # Only local ordinal user indices are released, no raw account IDs.
        return dict(history_items=histories,target_items=targets,
                    input_ids=paths[histories].reshape(len(rows),80),
                    labels=paths[targets],user_indices=np.array([u for u,_,_ in rows],dtype=np.int64))
    train_users=set(); valid_users=set(); counts={}
    for role,cut in zip(('parent','update'),cuts):
        training=[];validation=[]
        for ui,(user,seq) in enumerate(sorted(users.items())):
            ids=[i for t,_,i in seq if t<=cut]
            candidates=[(ui,ids[max(0,j-20):j],ids[j]) for j in range(3,len(ids))][-4:]
            is_valid=int(hashlib.sha256(user.encode()).hexdigest(),16)%10==0
            if is_valid and candidates:
                validation.append(candidates[-1]);valid_users.add(ui)
            elif candidates:
                training.extend(candidates);train_users.add(ui)
        for split,rows in [('train',training),('valid',validation)]:
            assert rows
            np.savez_compressed(out/f'{role}_{split}.npz',**arrays(rows))
            counts[f'{role}_{split}']=len(rows)
    assert not train_users & valid_users
    test=[]
    for ui,(_,seq) in enumerate(sorted(users.items())):
        history=[i for t,_,i in seq if t<=cuts[1]][-20:]
        later=[i for t,_,i in seq if t>cuts[1]]
        if len(history)>=3 and later:test.append((ui,history,later[0]))
    arr=arrays(test)
    arr['pure_old']=np.all(np.isin(arr['history_items'],np.r_[0,old_ids]),axis=1)
    arr['new_target']=np.isin(arr['target_items'],current_ids)&~np.isin(arr['target_items'],old_ids)
    arr['old_target']=np.isin(arr['target_items'],old_ids)
    arr['future_target']=~np.isin(arr['target_items'],current_ids)
    np.savez_compressed(out/'test.npz',**arr)
    counts.update(test=len(test),old_items=len(old_ids),current_items=len(current_ids),
                  new_items=len(current_ids)-len(old_ids),pure_old=int(arr['pure_old'].sum()),
                  new_target=int(arr['new_target'].sum()),
                  primary=int((arr['pure_old']&arr['new_target']).sum()))
    assert counts['primary']>0,'Primary cohort empty; frozen protocol cannot be replaced'
    summary=dict(status='prepared',prepared_at=datetime.now(timezone.utc).isoformat(),
                 counts=counts,cutoffs=cuts.tolist(),source_inputs=inputs,
                 user_split_disjoint=True,content_fit_old_items_only=True,
                 code_identity_preserved=True,seconds=time.monotonic()-start,
                 outputs={f.name:sha(f) for f in out.glob('*.npz')})
    assert summary['seconds'] < p.get('preprocessing_cpu_seconds_cap', float('inf'))
    (out/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.protocol,a.output)
