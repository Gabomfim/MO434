#!/usr/bin/env python3
"""
Parte 1 — Conexão de dados (reusável).

Baixa da conta W&B (rodz-ralm-v-ai / convnextmicro-metric-distill) TODOS os runs
Graph-RKD de metric learning e gera dois parquets/csv limpos:

  analysis/data/runs_summary.parquet   -> 1 linha por run  (métricas finais + config)
  analysis/data/history.parquet        -> 1 linha por (run, época)  (curvas de convergência)

Escopo do trabalho do Gabriel: mode=regression, method in {profile, mds},
dataset in {cars196, cub200}, varredura de N (graph_rkd_nodes) x 3 seeds.
Também trazemos o baseline mode=off (sem grafo) p/ medir "quão melhor o N* é".

Uso:
  python analysis/pull_wandb.py                # usa entity/project default
  WANDB_ENTITY=... python analysis/pull_wandb.py --project convnextmicro-metric-distill

O Gabriel pode rodar isto com a chave dele (wandb login) e reusar os parquets.
"""
import argparse, os, re, sys
import pandas as pd

ENTITY_DEFAULT  = os.environ.get("WANDB_ENTITY", "rodz-ralm-v-ai")
PROJECT_DEFAULT = "convnextmicro-metric-distill"

# métricas que nos interessam (nomes exatos logados no W&B)
METRICS = ["mAP@R", "recall@1", "recall@2", "recall@4", "recall@8", "R_precision"]
HIST_KEYS = ["epoch"] + [f"test/{m}" for m in METRICS] + [f"val/{m}" for m in METRICS] \
            + ["train/graph_loss", "train/loss_loss", "lr"]

TAG_RE = re.compile(r"-(search|final)-")


def parse_tag(name: str) -> str:
    m = TAG_RE.search(name or "")
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity",  default=ENTITY_DEFAULT)
    ap.add_argument("--project", default=PROJECT_DEFAULT)
    ap.add_argument("--modes", nargs="+", default=["regression", "off"],
                    help="graph_rkd_mode a incluir (regression = foco; off = baseline)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=120)
    path = f"{args.entity}/{args.project}"
    print(f"[pull] conectando a {path} ...", flush=True)
    runs = api.runs(path, per_page=500)
    print(f"[pull] {len(runs)} runs no projeto. Filtrando modes={args.modes} ...", flush=True)

    summary_rows, hist_rows = [], []
    kept = 0
    for i, r in enumerate(runs):
        c = r.config
        mode = c.get("graph_rkd_mode")
        if mode not in args.modes:
            continue
        method = c.get("graph_rkd_method")
        N      = c.get("graph_rkd_nodes")
        ds     = c.get("dataset") or c.get("dataset_name")
        seed   = c.get("seed")
        tag    = parse_tag(r.name)
        base = dict(run_id=r.id, run_name=r.name, state=r.state, tag=tag,
                    dataset=ds, method=method, mode=mode, N=N, seed=seed,
                    teacher_arch=c.get("teacher_arch"))
        kept += 1

        # ---- summary (métricas finais) ----
        s = r.summary
        srow = dict(base)
        srow["best_val_score"] = s.get("best_val_score")
        srow["epochs_logged"]  = s.get("epoch")
        for split in ("test", "val", "train"):
            for m in METRICS:
                # há duas convenções no summary: 'final/test/mAP@R' e 'final_test_mAP@R'
                v = s.get(f"final/{split}/{m}")
                if v is None:
                    v = s.get(f"final_{split}_{m.replace('@','').replace('_precision','_R_precision') if m=='R_precision' else m}")
                srow[f"{split}/{m}"] = v
        summary_rows.append(srow)

        # ---- history (curva por época) ----
        try:
            h = r.history(keys=[k for k in HIST_KEYS if k != "epoch"], pandas=True)
        except Exception as e:
            print(f"   [warn] history falhou p/ {r.name}: {e}", flush=True)
            h = None
        if h is not None and len(h):
            if "epoch" not in h.columns and "_step" in h.columns:
                h = h.rename(columns={"_step": "epoch"})
            for _, hr in h.iterrows():
                hrow = dict(base)
                hrow["epoch"] = hr.get("epoch", hr.get("_step"))
                for k in HIST_KEYS:
                    if k in h.columns and k != "epoch":
                        hrow[k] = hr.get(k)
                hist_rows.append(hrow)

        if kept % 25 == 0:
            print(f"   ... {kept} runs processados (varridos {i+1})", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    sdf = pd.DataFrame(summary_rows)
    hdf = pd.DataFrame(hist_rows)
    # CSV é o formato canônico (portátil, sem dependência de pyarrow).
    sp = os.path.join(args.outdir, "runs_summary.csv")
    hp = os.path.join(args.outdir, "history.csv")
    sdf.to_csv(sp, index=False)
    hdf.to_csv(hp, index=False)
    # parquet best-effort (se pyarrow estiver disponível) — não derruba o run se faltar.
    for df, base in ((sdf, "runs_summary"), (hdf, "history")):
        try:
            df.to_parquet(os.path.join(args.outdir, base + ".parquet"))
        except Exception as e:
            print(f"   [info] parquet pulado ({base}): {type(e).__name__}", flush=True)
    print(f"\n[pull] OK. {len(sdf)} runs -> {sp}")
    print(f"[pull]    {len(hdf)} linhas de história -> {hp}")
    # sanity: cobertura regression
    reg = sdf[sdf["mode"] == "regression"]
    print("\n=== cobertura (mode=regression) — runs por (dataset, method, tag) ===")
    if len(reg):
        print(reg.groupby(["dataset", "method", "tag"]).size().to_string())
        print("\n=== N disponíveis por (dataset, method) [tag=search] ===")
        sr = reg[reg["tag"] == "search"]
        for (d, m), g in sr.groupby(["dataset", "method"]):
            print(f"  {d:8} {m:8}: N={sorted(g['N'].dropna().unique().tolist())}  "
                  f"seeds/N={g.groupby('N')['seed'].nunique().to_dict()}")


if __name__ == "__main__":
    main()
