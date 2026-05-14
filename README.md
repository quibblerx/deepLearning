# GNN Introduction

This project provides an introduction to Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric on the dataset QM9.

## Installation

### Option 1: Using UV (Recommended - Much Faster!)

[UV](https://github.com/astral-sh/uv) is an extremely fast Python package installer (10-100x faster than pip).

**Install UV on Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Then install dependencies:**

```powershell
# Create and activate virtual environment
uv venv
.\.venv\Scripts\Activate.ps1

# Install all dependencies from pyproject.toml
uv sync

# Or install with GPU support (CUDA 11.8)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .
```

See [README_UV.md](README_UV.md) for detailed UV usage instructions.

### Option 2: Using pip (Traditional Method)

To run this project, you need to install the required Python packages. You can install them using pip:

```bash
# It is recommended to install PyTorch first, following the official instructions
# for your specific hardware (CPU or GPU with a specific CUDA version).
# See: https://pytorch.org/get-started/locally/

# For example, for a recent CUDA version:
# pip install torch torchvision torchaudio

# Or for CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# After installing PyTorch, install PyTorch Geometric.
# The exact command depends on your PyTorch and CUDA versions.
# Please refer to the PyTorch Geometric installation guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Example for PyTorch 2.7 and CUDA 11.8
# pip install torch_geometric

# Then, install the other required packages:
pip install hydra-core omegaconf wandb pytorch-lightning numpy tqdm

# Or install from pyproject.toml:
pip install -e .
```

## How to Run

The main entry point for this project is `src/run.py`. It uses `hydra` for configuration management. Hydra is a broadly used and highly respected so I recommend using it. You can find a guide to it here https://medium.com/@jennytan5522/introduction-to-hydra-configuration-for-python-646e1dd4d1e9.

To run the code, execute the following command from the root of the project:

```bash
python src/run.py
```

You can override the default configuration by passing arguments from the command line. For example, to use a different model configuration:

```bash
python src/run.py model=gcn
```

The configuration files are located in the `configs/` directory.

## Improving the predictive accuracy

There are many ways to improve the GNN. Please try to get the validation error (MSE) as low as possible. I have not implemented the code to run on the test data. That is for you to do, but please wait until you have the final model.
Here are some great resources:

- Try different GNN architectures and layers see (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)
- Try different optimizers and schedulers
- Tune hyperparameters (especially learning rate, layers, and hidden units)
- Use advanced regularization techniques such as https://openreview.net/forum?id=xkljKdGe4E#discussion
- You can try changing the generated features of the dataloader
