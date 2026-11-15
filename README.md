# Beyond the Beam (BB): core code

This source-only package accompanies **Beyond the Beam: Constructive Repair and
Candidate Completion for Generative Recommendation**. It contains our data
processing and main method, with explicit inputs instead of paths into a private
experiment workspace.

## Contents

| Module | Purpose |
| --- | --- |
| `bb/prepare_amazon.py` | Chronological Beauty/Toys split; old-item-only residual KMeans; stable collision suffixes; history encoding |
| `bb/prepare_tools.py` | Convert supplied chronological Tools parquet splits and retained identifier arrays |
| `bb/timing.py` | Training day gaps and positive-gap validation mask for Beauty/Toys |
| `bb/feasibility.py`, `bb/states.py` | Feasible admission intervals and bounded state enumeration |
| `bb/repair.py` | Integral minimum leaf-replacement solver and stable identity assignment |
| `bb/construction.py`, `bb/shared_map.py` | Native proposal verification and training-context shared-map selection |
| `bb/generator.py`, `bb/adaptation.py` | History re-encoding, warm-up, weighted token loss and fixed-recipe adaptation |
| `bb/prior.py` | Supervised next-item ridge fit and normalized recency-weighted histories |
| `bb/catalog.py`, `bb/native.py`, `bb/scoring.py` | Native T5 catalog masks, state capture, retained prefix bounds and exact likelihoods |
| `bb/completion.py` | Calibrated score, stable candidate priority, budgeted completion and numerical stopping |
| `bb/lcrec.py` | Optional LC-Rec prompt encoding, beam/prefix capture, likelihood and response-loss functions |
| `bb/run.py` | Portable command-line entry points |

No datasets, embeddings, model weights, predictions, third-party baseline
implementations, experiment supervisors, hyperparameter-search drivers, plots,
development notebooks or historical experiment outputs are bundled.

## Environment and quick check

Use Python 3.12 or newer. The tested environment is Python 3.12.13, PyTorch 2.6.0,
Transformers 5.9.0, NumPy 2.5.1, SciPy 1.18.0 and scikit-learn 1.9.0. The native
beam hooks require **Transformers 5.9.0**; the source checks this version.
Install the matching PyTorch CPU/CUDA build for your system.

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python -m bb.run --help
```

The CPU smoke check uses a randomly initialized tiny T5 and synthetic inputs.
It checks 300 minimum-replacement cases against exhaustive search, retained
prefix bounds against full likelihoods, completed Top-20 against exact catalog
ranking, ridge fitting, and one training/adaptation epoch. It does not rerun
the paper's experiments. LC-Rec neural execution requires a supplied compatible
CUDA model/tokenizer and is not part of this CPU check.

## Data inputs

Beauty/Toys preprocessing consumes the original gzip JSONL reviews, a matching
`id_mapping.json` with an `item2id` field, and raw float32 content embeddings of
shape `(number_of_items, 768)`. IDs are contiguous, start at 1, and reserve 0 for
padding; the source mapping has a padding entry. Content embeddings are external
inputs; their encoder implementation is not bundled. The example configuration
uses relative paths; edit it to point to your own inputs.

```bash
python -m bb.prepare_amazon --protocol configs/amazon.example.json --output data/beauty
python -m bb.timing --protocol configs/amazon.example.json --data-dir data/beauty --output data/beauty/timing.npz
```

The same preprocessing code serves Toys with the corresponding review, mapping
and embedding files. It fits three 64-centroid residual KMeans levels only on
old items, appends a collision-resolving fourth token, and retains old codes.
The chronological cutoffs are the 60% and 80% event-time quantiles including
ties. Validation users are selected by the original SHA256 rule. Histories
contain at most 20 items and produce 80 T5 input tokens.

Tools uses the existing chronological parquet splits (columns `user`, `history`,
`target`) and raw 0..255 four-digit code arrays in contiguous item-ID order:

```bash
python -m bb.prepare_tools --old-codes data/tools_old.npy --current-codes data/tools_current.npy --parent-train data/train_0.7.parquet --train data/train_0.8.parquet --valid data/valid_0.8.parquet --test data/test_0.8.parquet --output data/tools
```

`--parent-train` is optional for conversion, but is required to create the
`parent_train.npz` consumed by the ridge-prior command. Tools' old IDs must be
the initial contiguous block of the current catalog, with identical old codes.

Prepared directories contain `mapping.npz`, `update_train.npz`,
`update_valid.npz`, `test.npz`, and (when supplied/prepared) `parent_train.npz`.
Mapping keys are `item_ids`, `paths`, `old_item_ids`, `old_paths`, `all_paths`.
Examples contain `history_items`, `target_items`, `user_indices`, `input_ids`
and `labels`; test arrays also describe target/history cohorts. Mapping paths
have shape `(N, 4)` and T5 position offsets `[1, 257, 513, 769]`.

## Run one T5 configuration

Start with a **compatible parent checkpoint and its exact model configuration**.
Parent pretraining and checkpoint distribution are outside this small package.
`configs/t5.json` describes the Beauty/Toys architecture; Tools has a different
vocabulary size, so supply its matching configuration instead. Checkpoints must
be PyTorch state dictionaries compatible with `weights_only=True`.

```bash
python -m bb.run warmup --data data/beauty --checkpoint inputs/parent.pt --output runs/warmup --device cuda:0
python -m bb.run construct --data data/beauty --checkpoint runs/warmup/warmup.pt --output runs/construction --device cuda:0
python -m bb.run adapt --data data/beauty --checkpoint runs/warmup/warmup.pt --mapping runs/construction/mapping.npz --timing data/beauty/timing.npz --recipe long --output runs/adapted --device cuda:0
python -m bb.run prior --data data/beauty --decay 0.05 --ridge 100 --output runs/prior --device cuda:0
python -m bb.run recommend --data data/beauty --checkpoint runs/adapted/model.pt --mapping runs/adapted/mapping.npz --queries data/beauty/test.npz --prior runs/prior/prior.npy --decay 0.05 --weight 1.125 --gamma 1.25 --bias 2.5 --output runs/predictions --device cuda:0
```

These are usage examples. Use the corresponding fixed settings for your run,
and keep `--decay` identical when fitting and applying the prior. For Tools,
use `--full-validation` instead of `--timing`; day-weighted recipes require
actual timing metadata. Every command requires a fresh output directory.

The warm-up runs five epochs. `configs/adaptation.json` contains the three
training settings and the validation stopping/checkpoint rules used by the
adaptation implementation. `configs/paper_t5_settings.json` records the nine
adopted T5 scoring settings and checkpoint coordinates; it is a settings
reference, not a new tuning algorithm or a claim that these example commands
reproduce the reported tables.

## Scope of the extraction

The numerical repair, prefix capture, scoring and completion kernels are
extracted from the research implementation. Imports and file interfaces are
made portable; the ridge fit, shared-map orchestration and adaptation loop are
factored into callable functions. Raw source provenance is retained separately
by the authors and is not included as development history in this package.

The original experimental proposal driver coupled an early-stop condition to
an ablation control. The BB-only driver here visits the preset states up to
the proposal cap without that control. Consequently, regenerating a shared
map can differ from the recorded experiment's candidate sequence. The local
minimum-replacement objective and feasibility/native witness checks are
unchanged. This package is intended to expose the core method, not to replay
the complete experimental pipeline or reproduce every reported table.

Shared-map selection considers retained individual proposals and accepted
compatible compositions. It selects the best modified map by training-audit
NDCG@10, Recall@10, fewer changed identities, then the original deterministic
mapping-hash tie rule. The original map is used when no proposal is available.
Optimality concerns occupied-leaf replacement for a fixed state/target, not
the number of changed identities or global shared-map quality.

The T5 completion kernel is specialized to four tokens, beam width 40, cutoff
20 and batches of 20; the default additional evaluation budget is 80. `E0`
includes every complete leaf scored at the terminal expansion. Full-vocabulary
log probabilities are used without trie renormalization. A returned
`certified=True` checks the score bound with a `1e-3` allowance on each side;
it is a numerical certificate, not a floating-point proof. Budget exhaustion
can return an uncertified ranking.

LC-Rec helpers use raw SID digits, a compatible prepared tokenizer and an
already loaded causal model. Set `model._inference_autocast` explicitly, supply
CUDA inputs through the helpers, and connect `beam_batch`/`exact_scores` to
the common calibrated correction and completion kernel. Pretrained backbones,
tokenizer artifacts, quantization/adapter setup and full LC-Rec training are
external to this core package.
