# Setup

Python ≥ 3.10, managed with **uv** (`pyproject.toml` + `uv.lock` are the single
source of truth).

```bash
uv sync            # creates .venv and installs everything
```

Run anything with `uv run <cmd>` (or `source .venv/bin/activate`). In VS Code,
select the `.venv` interpreter / Jupyter kernel.

## Core dependencies
- **torch 2.7.0**, **torchvision 0.22.0** (CUDA build) — teacher/student training
- **numpy < 2.0**, **scipy** — arrays; scipy loads Cars `.mat` annotations
- **wandb 0.27** — experiment logging (`gabomfim-unicamp/graph-rkd`)
- **tqdm** — progress bars
- **modal ≥ 1.5** — rented-GPU backend (`RKD/sm/run_modal.py`)
- **pandas**, **matplotlib** — analysis notebooks / figures (`RKD/analysis/`)
- **jupyter**, **ipython** — notebooks

`seaborn`, `plotly`, `h5py`, `kagglehub` are declared but not imported by the
trimmed campaign (legacy); safe to drop with `uv remove seaborn plotly h5py kagglehub`.

## Credentials (only for the parts you use)
- **W&B:** `export WANDB_API_KEY=...` (or `wandb login`).
- **AWS** (S3 data + SageMaker): profile `gabomfim` (`aws configure`); bucket
  `graph-rkd-832271495954` in `us-east-1`.
- **Modal:** `modal setup`, then `modal secret create wandb ...` and
  `modal secret create aws ...` (see `RKD/sm/README.md`).

## Data
Datasets are staged in S3 as archives and pulled automatically by the runners
(`RKD/sm/data_prep.py`); nothing to place by hand. To (re)stage:
`python RKD/sm/stage_data.py`. See `README.md` for the full run instructions and
`REPO_MAP.md` for where each concept lives.
