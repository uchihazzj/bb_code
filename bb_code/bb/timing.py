"""Reconstruct training day gaps and validation masks from chronological inputs."""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import read, load


def prepare(protocol_file, data_dir, output):
    protocol = read(protocol_file)
    item_ids = read(protocol['feature_id_mapping'])['item2id']
    events = []
    with gzip.open(protocol['raw_reviews'], 'rt') as stream:
        for order, line in enumerate(stream):
            row = json.loads(line)
            events.append((int(row['unixReviewTime']), order, row['reviewerID'], item_ids[row['asin']]))
    events.sort()
    users = defaultdict(list)
    for time, order, user, item in events:
        users[user].append((time, order, item))
    cutoff = load(data_dir / 'mapping.npz')['cutoffs'][1]
    records = {'train': [], 'valid': []}
    for ordinal, (user, seq) in enumerate(sorted(users.items())):
        seq = [row for row in seq if row[0] <= cutoff]
        selected = list(range(3, len(seq)))[-4:]
        is_valid = int(hashlib.sha256(user.encode()).hexdigest(), 16) % 10 == 0
        if is_valid:
            selected = selected[-1:]
        for j in selected:
            records['valid' if is_valid else 'train'].append((ordinal,
                [r[2] for r in seq[max(0, j-20):j]], seq[j][2], (seq[j][0]-seq[j-1][0])/86400))
    gaps = {}
    for split, rows in records.items():
        actual = load(data_dir / f'update_{split}.npz')
        assert len(rows) == len(actual['target_items'])
        for (user, hist, target, gap), u, h, t in zip(rows, actual['user_indices'], actual['history_items'], actual['target_items']):
            assert user == u and hist == h[h != 0].tolist() and target == t and gap >= 0
        gaps[split] = np.array([r[3] for r in rows])
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, train_gap_days=gaps['train'], validation_primary_mask=gaps['valid'] > 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    prepare(a.protocol, a.data_dir, a.output)
