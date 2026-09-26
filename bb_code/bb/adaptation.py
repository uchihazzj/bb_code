"""Fixed-map weighted token cross-entropy and validation checkpoint selection.

Recipe values are supplied explicitly. This module runs one trajectory; it
does not search recipes, select seeds or inspect test metrics.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
from .generator import encoded, evaluate, ranks_metrics, cpu_state


def adapt(model, train, valid, mapping, paths, recipe, training,
          primary_mask, train_gap_days, seed=17):
    primary_mask = np.asarray(primary_mask, dtype=bool)
    assert primary_mask.shape == (len(valid['target_items']),) and primary_mask.any()
    gaps = np.asarray(train_gap_days)
    assert gaps.shape == (len(train['target_items']),) and (gaps >= 0).all()
    random.seed(seed + 1000)
    np.random.seed(seed + 1000)
    torch.manual_seed(seed + 1000)
    torch.cuda.manual_seed_all(seed + 1000)
    ema = AveragedModel(model, avg_fn=get_ema_avg_fn(recipe['ema'])) if recipe['ema'] is not None else None
    if ema is not None:
        ema.update_parameters(model)
    x, y = encoded(train, mapping, paths)
    x = torch.as_tensor(x, device=model.device)
    y = torch.as_tensor(y, device=model.device)
    weights = np.where(gaps > 0, 2., .5) if recipe['cross_day_weight'] else np.ones(len(x))
    weights /= weights.mean()
    weights = torch.as_tensor(weights, dtype=torch.float32, device=model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=training['weight_decay'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training['max_epochs'], eta_min=recipe['eta_min'])
        if recipe['schedule'] == 'cosine' else None)
    history, best_key, best_state, best_epoch = [], None, None, 0
    best_primary, patience = -float('inf'), 0

    def check(epoch):
        nonlocal best_key, best_state, best_epoch, best_primary, patience
        evaluated = ema.module if ema is not None else model
        pred, metrics = evaluate(evaluated, valid, mapping, paths)
        sliced = {k: v[primary_mask] for k, v in valid.items()}
        primary = ranks_metrics(pred[primary_mask], mapping, paths, sliced)[0]['all']
        key = (primary['ndcg10'], primary['recall10'], metrics['all']['ndcg10'], -epoch)
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch = key, cpu_state(evaluated), epoch
        if primary['ndcg10'] > best_primary:
            best_primary, patience = primary['ndcg10'], 0
        elif epoch > 0:
            patience += 1
        history.append({'epoch': epoch, 'primary': primary, 'full': metrics['all']})

    check(0)
    for epoch in range(1, training['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(x), device=model.device)
        for lo in range(0, len(x), training['batch']):
            ix = order[lo:lo + training['batch']]
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=x[ix], attention_mask=x[ix].ne(0), labels=y[ix], use_cache=False).logits
            loss_by_row = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[ix].reshape(-1),
                                          reduction='none').reshape(len(ix), -1).mean(1)
            loss = (loss_by_row * weights[ix]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adaptation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training['clip'])
            optimizer.step()
            if ema is not None:
                ema.update_parameters(model)
        if scheduler is not None:
            scheduler.step()
        if epoch % training['check_every'] == 0:
            check(epoch)
            if epoch >= training['minimum_epochs'] and patience >= training['patience_checks']:
                break
    return best_state, {'selected_adaptation_epoch': best_epoch,
                        'selected_update_epoch_including_warmup': best_epoch + 5,
                        'epochs_run': epoch, 'validation': history}
