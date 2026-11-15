# Beyond the Beam (BB)

Code for **Beyond the Beam: Constructive Repair and Candidate Completion for
Generative Recommendation**.

## Code

- `bb_code/bb/`: data preprocessing, identifier construction, training, and recommendation.
- `bb_code/configs/`: model, construction, adaptation, and scoring settings.
- `bb_code/requirements.txt`: Python dependencies.

## Environment

Use Python 3.12. Main dependencies are PyTorch 2.6.0, Transformers 5.9.0,
NumPy 2.5.1, SciPy 1.18.0, and scikit-learn 1.9.0.

From the repository root:

```bash
cd bb_code
python -m pip install -r requirements.txt
python -m bb.run --help
```

## Running

The workflow is data preparation → warm-up → identifier construction → adaptation
→ prior fitting → recommendation.

See [the usage guide](bb_code/README.md) for input files and commands.
