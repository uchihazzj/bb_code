"""Convert supplied chronological Tools parquet data and retained SID codes.

Input code arrays contain four raw 0..255 digits; rows follow item IDs 1..N.
The old catalog is the initial contiguous block of the current catalog.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def prepare(old_codes, current_codes, train, valid, test, output, parent_train=None):
    old = np.load(old_codes, allow_pickle=False)
    raw = np.load(current_codes, allow_pickle=False)
    assert raw.ndim == 2 and raw.shape[1] == 4
    assert np.issubdtype(raw.dtype, np.integer) and ((raw >= 0) & (raw < 256)).all()
    assert np.array_equal(raw[:len(old)], old)
    assert len(np.unique(raw, axis=0)) == len(raw)
    paths = raw + np.array([1, 257, 513, 769])
    oldpaths = paths[:len(old)]
    allpaths = np.vstack([np.zeros((1, 4), int), paths])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'mapping.npz', item_ids=np.arange(1, len(paths)+1),
                        paths=paths, old_item_ids=np.arange(1, len(old)+1),
                        old_paths=oldpaths, all_paths=allpaths)
    events = {}
    sources = [('update_train', train), ('update_valid', valid), ('test', test)]
    if parent_train is not None:
        sources.insert(0, ('parent_train', parent_train))
    for filename, source in sources:
        frame = pd.read_parquet(source)
        hist = np.zeros((len(frame), 20), np.int64)
        for i, history in enumerate(frame.history):
            values = np.asarray(history[-20:], dtype=np.int64)
            if len(values):
                hist[i, -len(values):] = values
        targets = frame.target.to_numpy(dtype=np.int64)
        assert (hist >= 0).all() and (hist <= len(paths)).all()
        assert ((targets > 0) & (targets <= len(paths))).all()
        if filename == 'parent_train':
            assert (hist <= len(old)).all() and (targets <= len(old)).all()
        new = targets > len(old)
        users = frame.user.to_numpy(dtype=np.int64)
        events[filename] = set((int(u), tuple(h), int(t)) for u, h, t in zip(users, hist, targets))
        np.savez_compressed(output / f'{filename}.npz', history_items=hist,
                            target_items=targets, input_ids=allpaths[hist].reshape(len(frame), 80),
                            labels=allpaths[targets], user_indices=users,
                            pure_old=(hist <= len(old)).all(1), new_target=new,
                            old_target=~new, future_target=np.zeros(len(frame), bool))
    assert not events['update_train'] & events['update_valid'], 'Duplicate train/validation records'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['old-codes', 'current-codes', 'train', 'valid', 'test', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parent-train', type=Path,
                        help='Optional original 0.7 training parquet, needed to fit the collaborative prior')
    prepare(**vars(parser.parse_args()))
