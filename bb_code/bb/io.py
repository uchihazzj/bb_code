from pathlib import Path
import hashlib, json
import numpy as np

def read(path):
    return json.loads(Path(path).read_text())

def load(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
