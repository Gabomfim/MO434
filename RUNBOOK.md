# RUNBOOK — running the Graph-RKD experiments on a GPU

For whoever runs the campaign on a GPU box (e.g. the RTX 5070). **One command**,
resumable, logs to **W&B `gabomfim-unicamp/graph-rkd`**. Datasets download
themselves from public S3 (no AWS credentials) and are cached. Analysis/charts
come **later** — this runbook is just about producing the runs.

## 1. One-time setup
```bash
git clone <this repo> && cd MO434       # or: git pull
uv sync                                  # installs torch (cu128 — works on Blackwell/5070), etc.
export WANDB_API_KEY=xxxxxxxx            # from wandb.ai/authorize  (or: uv run wandb login)
nvidia-smi                               # confirm the GPU is visible
```
No AWS setup, no data download by hand — `run_experiments.sh` fetches Cars-196
and CUB-200 from public S3 on first use and caches them under `data/`.

## 2. Run everything (the trimmed campaign)
```bash
./RKD/sm/run_experiments.sh
```
That runs **teachers + phases 2,3,4,5** (`--trimmed`: drop hybrid, λg∈{0.01,0.1,1},
N∈{3,4,8}). It **uses the GPU at 100%** — it auto-packs as many jobs as fit in
VRAM and round-robins across multiple GPUs. Expected: **~2–3 days** on a single
12 GB RTX 5070 (~196 GPU-hours; teachers first, then the student fan-out).

### Variants
```bash
PHASES="teachers phase2 phase3 phase4" ./RKD/sm/run_experiments.sh   # search phases only (~1 day)
PHASES="phase5" ./RKD/sm/run_experiments.sh                          # headline only
MAX_PARALLEL=3 ./RKD/sm/run_experiments.sh                           # fix concurrency (else auto by VRAM)
PER_JOB_GB=3   ./RKD/sm/run_experiments.sh                           # smaller VRAM/job -> more parallel
```
Pick the phase-5 headline config explicitly (after seeing phases 2–4) with, e.g.:
```bash
PHASES="phase5" ./RKD/sm/run_experiments.sh \
  ... # or call run_local.py directly:
uv run python RKD/sm/run_local.py --trimmed --phases phase5 \
  --headline-method mds --headline-norm per_graph --headline-objective regression \
  --headline-nodes 4 --headline-lambda 0.01 \
  --wandb-entity gabomfim-unicamp --wandb-project graph-rkd
```

## 3. Monitor
- **W&B:** https://wandb.ai/gabomfim-unicamp/graph-rkd — loss curves, val/test
  mAP@R per run. Teachers appear first, then the student runs.
- Terminal: a live line per running job (`[run] ...`, `[done] ...`).

## 4. Resuming (safe to interrupt)
Just **re-run the same command**. Finished jobs are skipped (ledger), and a job
interrupted mid-training resumes from its last checkpoint. So Ctrl-C / reboot /
power loss are fine — rerun and it continues.

## 5. If something's off
- **`no GPU CUDA visible`** → run on the GPU box; check `nvidia-smi` + drivers.
- **torch/CUDA mismatch on the 5070 (Blackwell)** → `uv sync` pulls the cu128
  build; if needed, update the NVIDIA driver.
- **W&B auth error** → `export WANDB_API_KEY=…` (must log to `gabomfim-unicamp`,
  the personal entity — *not* the org).
- **Out of VRAM** → lower packing: `PER_JOB_GB=6 ./RKD/sm/run_experiments.sh`.
- **Dataset download slow/failed** → it resumes on rerun; datasets cache under
  `data/` so it only downloads once.

## 6. When it's done
Tell Gabriel — the runs are in W&B. Charts, tables and the paper's Section 7 are
produced afterward from `RKD/analysis/` (notebooks `00`→`06`) into
`FINDINGS.md` + `PHASE_k_REPORT.md`. See `README.md` / `REPO_MAP.md` for details.
