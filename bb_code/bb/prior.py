"""Training-only supervised history-to-next-item ridge prediction.

This uses established ridge regression, not a newly proposed loss.
The item-to-item matrix has quadratic memory cost in catalog size.
"""
import numpy as np
import torch
from scipy import sparse

def history_matrix(hist,ids,decay):
 lookup=np.full(int(ids.max())+1,-1,np.int64);lookup[ids]=np.arange(len(ids));valid=hist>0;assert hist.max()<=ids.max();assert (lookup[hist[valid]]>=0).all()
 rows,cols=np.nonzero(valid);values=np.exp(-decay*(hist.shape[1]-1-cols));den=np.bincount(rows,weights=values,minlength=len(hist));values/=den[rows]
 return sparse.csr_matrix((values,(rows,lookup[hist[rows,cols]])),shape=(len(hist),len(ids)))

def fit_ridge(parent, update, item_ids, decay=0.05, ridge=100., device='cpu'):
    if ridge <= 0:
        raise ValueError('ridge must be positive')
    rows = np.unique(np.concatenate([
        np.column_stack((z['user_indices'], z['history_items'], z['target_items']))
        for z in (parent, update)]), axis=0)
    lookup = {int(v): i for i, v in enumerate(item_ids)}
    targets = [lookup[int(v)] for v in rows[:, -1]]
    y = sparse.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), targets)),
                         shape=(len(rows), len(item_ids)))
    x = history_matrix(rows[:, 1:-1], item_ids, decay)
    gram = torch.tensor((x.T @ x).toarray(), dtype=torch.float64, device=device)
    cross = torch.tensor((x.T @ y).toarray(), dtype=torch.float64, device=device)
    diagonal = torch.arange(len(item_ids), device=device)
    gram[diagonal, diagonal] += ridge
    matrix = torch.cholesky_solve(cross, torch.linalg.cholesky(gram))
    return matrix.float().cpu().numpy()
