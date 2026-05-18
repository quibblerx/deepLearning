# Semi-Supervised Learning for Drug Discovery

This project studies graph neural networks for molecular property prediction on QM9, with a focus on semi-supervised learning. We compare a supervised GCN baseline against Mean Teacher and a regression-friendly peer-consistency variant, using labeled and unlabeled molecular graphs.

## What’s Included

- Supervised GCN baseline for QM9 regression
- Mean Teacher semi-supervised learning
- Peer consistency on unlabeled graphs
- Low-label stress tests
- Reproducible experiment notebook

## Tech Stack

- Python
- PyTorch
- PyTorch Geometric
- Hydra
- Weights & Biases
- NumPy / tqdm

## Installation

The recommended workflow uses `uv`.

```bash
uv venv
source .venv/bin/activate
uv sync
```

If you need a GPU-enabled PyTorch build, install it first using the official PyTorch instructions for your CUDA version, then run `uv sync` or `uv pip install -e .`.

## Running Training

Run the main training entry point from the project root:

```bash
python src/run.py
```

Hydra config overrides can be passed on the command line:

```bash
python src/run.py model=gcn trainer.init.consistency_weight=1
```

## Reproducing Results

Open `run.ipynb` to reproduce the final experiments used in the report.
The notebook includes:

- environment setup
- W&B login
- main-split runs
- low-label runs
- ablation runs

## Project Structure

- `src/` - training, data, model, and logging code
- `configs/` - Hydra configs
- `run.ipynb` - reproduction notebook
