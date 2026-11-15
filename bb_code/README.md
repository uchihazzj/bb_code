# Beyond the Beam (BB)

Implementation of **Beyond the Beam: Constructive Repair and Candidate Completion
for Generative Recommendation**, including data preprocessing, semantic identifier
repair, model adaptation, and candidate completion.

## Code

| Modules | Purpose |
| --- | --- |
| `bb/prepare_amazon.py`, `bb/prepare_tools.py`, `bb/timing.py` | Prepare datasets and timing metadata |
| `bb/feasibility.py`, `bb/states.py`, `bb/repair.py` | Construct and repair semantic identifiers |
| `bb/construction.py`, `bb/shared_map.py` | Generate proposals and select a shared identifier map |
| `bb/generator.py`, `bb/adaptation.py` | Warm up and adapt the recommendation model |
| `bb/prior.py` | Fit the next-item ridge prior |
| `bb/catalog.py`, `bb/native.py`, `bb/scoring.py`, `bb/completion.py` | Score and complete recommendation candidates |
| `bb/lcrec.py` | LC-Rec encoding, generation, and loss helpers |
| `bb/run.py` | Command-line entry point |

## Environment

Use Python 3.12 with PyTorch 2.6.0, Transformers 5.9.0, NumPy 2.5.1,
SciPy 1.18.0, and scikit-learn 1.9.0. Dependencies are listed in `requirements.txt`.

Run the following commands from this directory (`bb_code/` inside the repository):

```bash
python -m pip install -r requirements.txt
python -m bb.run --help
```

## Data preparation

### Review data

Set the paths in `configs/amazon.example.json` to the review file (`.json.gz`),
item mapping (`id_mapping.json` with an `item2id` field), and raw float32 content
embeddings with 768 dimensions. Item IDs start at 1, with 0 used for padding.

```bash
python -m bb.prepare_amazon --protocol configs/amazon.example.json --output data/prepared
python -m bb.timing --protocol configs/amazon.example.json --data-dir data/prepared --output data/prepared/timing.npz
```

### Parquet data

Prepare chronological parquet splits with `user`, `history`, and `target`
columns, plus old and current four-digit code arrays in item-ID order.

```bash
python -m bb.prepare_tools --old-codes data/tools_old.npy --current-codes data/tools_current.npy --parent-train data/train_0.7.parquet --train data/train_0.8.parquet --valid data/valid_0.8.parquet --test data/test_0.8.parquet --output data/tools
```

The prepared data includes `mapping.npz`, `parent_train.npz`, `update_train.npz`,
`update_valid.npz`, and `test.npz`.

## Running

Place the parent T5 checkpoint (a PyTorch state dictionary) at `inputs/parent.pt`.
The default model configuration is `configs/t5.json`. To use
another model configuration, pass `--model-config path/to/config.json`.

Run the following stages in order. These commands use GPU `cuda:0`; use
`--device cpu` for CPU execution.

```bash
# 1. Warm up the model
python -m bb.run warmup --data data/prepared --checkpoint inputs/parent.pt --output runs/warmup --device cuda:0

# 2. Construct the identifier map
python -m bb.run construct --data data/prepared --checkpoint runs/warmup/warmup.pt --output runs/construction --device cuda:0

# 3. Adapt the model
python -m bb.run adapt --data data/prepared --checkpoint runs/warmup/warmup.pt --mapping runs/construction/mapping.npz --timing data/prepared/timing.npz --recipe long --output runs/adapted --device cuda:0

# 4. Fit the prior
python -m bb.run prior --data data/prepared --decay 0.05 --ridge 100 --output runs/prior --device cuda:0

# 5. Generate recommendations
python -m bb.run recommend --data data/prepared --checkpoint runs/adapted/model.pt --mapping runs/adapted/mapping.npz --queries data/prepared/test.npz --prior runs/prior/prior.npy --decay 0.05 --weight 1.125 --gamma 1.25 --bias 2.5 --output runs/predictions --device cuda:0
```

Recommendations are saved to `runs/predictions/recommendations.npz`.
For another run, choose new output paths. Use the same `--decay` value for prior
fitting and recommendation.

## Configuration

- `configs/t5.json`: T5 architecture configuration for the dataset.
- `configs/construction.json`: identifier construction settings.
- `configs/adaptation.json`: `long`, `smooth`, and `cross_day` training recipes.
- `configs/paper_t5_settings.json`: scoring parameters and checkpoint settings.
