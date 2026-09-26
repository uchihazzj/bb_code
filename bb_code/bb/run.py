"""Portable entry points for one fixed BB model/configuration."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration
from .io import read, load, write
from . import generator, native
from .shared_map import select_shared_map
from .adaptation import adapt
from .prior import fit_ridge, history_matrix
from .scoring import trace, ExactScorer
from .completion import calibrated_offset, complete


def main(a):
    # A new output directory is required; original input files are never updated.
    a.output.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = load(a.data / 'mapping.npz')
    if a.command == 'prior':
        matrix = fit_ridge(load(a.data / 'parent_train.npz'), load(a.data / 'update_train.npz'),
                           m['item_ids'], a.decay, a.ridge, a.device)
        np.save(a.output / 'prior.npy', matrix)
        write(a.output / 'prior_config.json', {'decay': a.decay, 'ridge': a.ridge})
        return
    model = T5ForConditionalGeneration(T5Config(**read(a.model_config))).to(a.device)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    native.V = model.config.vocab_size
    paths = load(a.mapping)['paths'] if a.mapping else m['paths']
    if a.command == 'warmup':
        generator.train_epochs(model, load(a.data / 'update_train.npz'), m, paths, 5,
                               a.seed, a.output / 'training.json')
        torch.save(generator.cpu_state(model), a.output / 'warmup.pt')
    elif a.command == 'construct':
        paths, report = select_shared_map(model, load(a.data / 'update_train.npz'), m,
                                         a.output / 'construction', read(a.construction_config), a.seed)
        np.savez_compressed(a.output / 'mapping.npz', paths=paths)
        write(a.output / 'selection.json', report)
    elif a.command == 'adapt':
        config = read(a.adaptation_config)
        tr, va = load(a.data / 'update_train.npz'), load(a.data / 'update_valid.npz')
        if a.timing:
            timing = load(a.timing)
        elif a.full_validation:
            timing = {'train_gap_days': np.ones(len(tr['target_items'])),
                      'validation_primary_mask': np.ones(len(va['target_items']), bool)}
            if config['recipes'][a.recipe]['cross_day_weight']:
                raise ValueError('Day-weighted adaptation requires --timing')
        else:
            raise ValueError('Supply --timing (Beauty/Toys) or --full-validation (Tools)')
        state, report = adapt(model, tr, va, m, paths, config['recipes'][a.recipe], config['training'],
                              timing['validation_primary_mask'], timing['train_gap_days'], a.seed)
        torch.save(state, a.output / 'model.pt')
        np.savez_compressed(a.output / 'mapping.npz', paths=paths)
        write(a.output / 'selection.json', report)
    elif a.command == 'recommend':
        data = load(a.queries)
        x = generator.encoded(data, m, paths)[0]
        prior = np.load(a.prior, allow_pickle=False)
        scores = np.asarray(history_matrix(data['history_items'], m['item_ids'], a.decay) @ prior, dtype=np.float32)
        offset = calibrated_offset(scores, ~np.isin(m['item_ids'], m['old_item_ids']), a.weight, a.gamma, a.bias)
        batches = []
        for lo in range(0, len(x), 64):
            bx = x[lo:lo+64]
            cap = trace(model, bx, paths)
            result = complete(cap['upper'], cap['depth'], offset[lo:lo+64],
                              ExactScorer(model, bx, paths), budget=a.budget)
            result['items'] = m['item_ids'][result['indices']]
            batches.append(result)
        if not batches:
            raise ValueError('Queries must be nonempty')
        np.savez_compressed(a.output / 'recommendations.npz',
                            **{key: np.concatenate([r[key] for r in batches]) for key in batches[0]})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['warmup', 'construct', 'adapt', 'prior', 'recommend'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--model-config', type=Path, default=Path('configs/t5.json'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--mapping', type=Path)
    p.add_argument('--construction-config', type=Path, default=Path('configs/construction.json'))
    p.add_argument('--adaptation-config', type=Path, default=Path('configs/adaptation.json'))
    p.add_argument('--recipe', choices=['long', 'smooth', 'cross_day'], default='long')
    p.add_argument('--timing', type=Path)
    p.add_argument('--full-validation', action='store_true')
    p.add_argument('--queries', type=Path)
    p.add_argument('--prior', type=Path)
    p.add_argument('--decay', type=float, default=.05)
    p.add_argument('--ridge', type=float, default=100.)
    p.add_argument('--weight', type=float, default=1.)
    p.add_argument('--gamma', type=float, default=1.)
    p.add_argument('--bias', type=float, default=0.)
    p.add_argument('--budget', type=int, default=80)
    return p


if __name__ == '__main__':
    p = parser()
    a = p.parse_args()
    if a.command != 'prior' and a.checkpoint is None:
        p.error('--checkpoint is required for this command')
    if a.command == 'recommend' and (a.queries is None or a.prior is None):
        p.error('recommend requires --queries and --prior')
    if a.budget < 0:
        p.error('--budget must be nonnegative')
    main(a)
