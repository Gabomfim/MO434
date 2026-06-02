# Knowledge Distillation Environment Setup

## Environment Details
- **Virtual Environment**: `.venv`
- **Python Version**: 3.11
- **Package Manager**: `uv`

## Installation Summary

### All packages (via pyproject.toml + uv)
```bash
uv sync
```

### Main packages
- torch 2.2.2
- torchvision 0.17.2
- matplotlib 3.10.9
- seaborn 0.13.2
- plotly 6.7.0
- pandas 3.0.3
- numpy 1.26.4 (pinned as `<2.0` for PyTorch compatibility)
- jupyter 1.0.0+
- ipython 8.0.0+

## Activation

To activate the environment in your terminal:
```bash
source .venv/bin/activate
```

To use this environment in VS Code:
1. Select the `.venv` Python interpreter in VS Code
2. Or use the Jupyter kernel from `.venv`

## Why This Setup?

- **Single source of truth**: dependencies are defined in `pyproject.toml`
- **Reproducible setup**: `uv sync` installs from `pyproject.toml` and `uv.lock`

## Updating Packages

To add a new package:
```bash
uv add <package-name>
```

To re-sync all dependencies:
```bash
uv sync
```
