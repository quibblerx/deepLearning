# Using UV with GNN Introduction

This guide explains how to use `uv` for faster package management in this project.

## What is UV?

`uv` is an extremely fast Python package installer and resolver written in Rust. It's designed to be a drop-in replacement for `pip` and `pip-tools`, but with significantly better performance.

## Installation

### Installing UV

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**

```powershell
pip install uv
```

## Quick Start

### 1. Create a Virtual Environment with UV

```powershell
# Create a virtual environment
uv venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
# Install all project dependencies (much faster than pip!)
uv pip install -e .

# Or install from the pyproject.toml directly
uv sync
```

### 3. Install Specific Packages

```powershell
# Install a single package
uv pip install numpy

# Install multiple packages
uv pip install torch torchvision torchaudio

# Install with specific versions
uv pip install "torch>=2.0.0"
```

### 4. Install PyTorch with CUDA Support

For GPU support, you may need to specify the PyTorch index URL:

```powershell
# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## UV Commands Cheat Sheet

```powershell
# Create virtual environment
uv venv

# Install from pyproject.toml
uv sync

# Install package
uv pip install <package>

# Install from requirements.txt
uv pip install -r requirements.txt

# Install in editable mode
uv pip install -e .

# Uninstall package
uv pip uninstall <package>

# List installed packages
uv pip list

# Freeze dependencies
uv pip freeze

# Compile requirements
uv pip compile pyproject.toml -o requirements.txt
```

## Benefits of UV

- **Speed**: 10-100x faster than pip for most operations
- **Reliability**: Better dependency resolution
- **Compatibility**: Drop-in replacement for pip
- **Modern**: Written in Rust with excellent performance

## Running Your Project

After installing dependencies with UV, run your project as usual:

```powershell
python src/run.py
```

Or with custom configurations:

```powershell
python src/run.py model=gcn
```

## Troubleshooting

### UV not found after installation

Close and reopen your terminal, or add UV to your PATH manually.

### PyTorch installation issues

For PyTorch with specific CUDA versions, always use the official PyTorch index URLs as shown above.

### Virtual environment activation

Make sure to activate your virtual environment before running UV commands:

```powershell
.\.venv\Scripts\Activate.ps1
```
