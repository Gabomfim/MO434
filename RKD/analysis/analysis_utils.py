"""Utilidades de análise para a campanha Graph-RKD (preenche a Seção 7 do paper).

Puxa as runs do W&B (gabomfim-unicamp/graph-rkd), classifica cada run em um
dos 5 alunos / config Graph-RKD, agrega por seeds (média ± sem) e produz as
tabelas e figuras que a Seção 7 pede (H0–H5). Os notebooks só chamam estas
funções. `pandas`/`matplotlib` são dependências; `wandb` é importado sob demanda.

Regras de análise (§8 do paper):
  * noise floor: sem de seeds por célula; diferença < ~1 sem = indistinguível;
  * "vence" só se a média excede o melhor baseline por > 1 sem E o sinal é
    consistente entre teachers; reportar por célula (dataset × teacher).
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

ENTITY = "gabomfim-unicamp"
PROJECT = "graph-rkd"
METRICS = ["mAP@R", "R_precision", "recall@1", "recall@2", "recall@4", "recall@8"]
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# --------------------------------------------------------------------------- #
# fetch + classify                                                             #
# --------------------------------------------------------------------------- #
def _student(cfg, tags):
    """Classifica a run em um dos 5 alunos (ou teacher) a partir de config/tags."""
    tags = set(tags or [])
    if "teacher" in tags:
        return "teacher"
    if cfg.get("graph_rkd_mode", "off") not in ("off", None):
        return "graph-rkd"
    dist, angle = cfg.get("dist_ratio", 0) or 0, cfg.get("angle_ratio", 0) or 0
    if dist > 0 and angle > 0:
        return "rkd_both"
    if dist > 0:
        return "rkd_dist"
    if angle > 0:
        return "rkd_angle"
    return "triplet_only"


def _phase_of(name, tags):
    """Fase da run: tag ``phaseN`` se houver; senão derivada do prefixo do nome
    (runs antigas não têm a tag). smoke->phase0, floor/baseline gate->phase1."""
    for t in (tags or []):
        if t.startswith("phase"):
            return t
    n = name or ""
    if n.startswith("phase"):
        return n.split("-")[0]
    if n.startswith("smoke"):
        return "phase0"
    if n.startswith("floor"):
        return "phase1"
    if n.startswith("baseline"):
        return "phase5"
    return ""


def fetch_runs(entity=ENTITY, project=PROJECT):
    """DataFrame: uma linha por run W&B (config + summary achatados + classificação)."""
    import wandb
    api = wandb.Api()
    rows = []
    for r in api.runs(f"{entity}/{project}"):
        cfg = {k: v for k, v in r.config.items() if not k.startswith("_")}
        s = dict(r.summary)
        row = {
            "run": r.name, "id": r.id, "state": r.state,
            "group": r.group, "phase": _phase_of(r.name, r.tags),
            "student": _student(cfg, r.tags),
            "dataset": cfg.get("dataset"), "teacher": cfg.get("teacher_arch"),
            "seed": cfg.get("seed"),
            "norm": cfg.get("graph_rkd_norm"), "method": cfg.get("graph_rkd_method"),
            "objective": cfg.get("graph_rkd_mode"), "N": cfg.get("graph_rkd_nodes"),
            "lambda_g": cfg.get("graph_rkd_ratio"),
            "best_val": s.get("best_val_score"),
            "teacher_test_mAP@R": s.get("teacher_test_mAP@R"),
        }
        for split in ("test", "val"):
            for m in METRICS:
                row[f"{split}_{m}"] = s.get(f"final_{split}_{m}")
        rows.append(row)
    return pd.DataFrame(rows)


def agg(df, group_cols, metric="test_mAP@R"):
    """Média ± sem sobre seeds. Retorna colunas mean/sem/n."""
    g = df.dropna(subset=[metric]).groupby(group_cols)[metric]
    out = g.agg(["mean", "sem", "count"]).reset_index()
    return out.rename(columns={"count": "n"})


# --------------------------------------------------------------------------- #
# §7.1 — H1: 5 alunos por (dataset, teacher)                                   #
# --------------------------------------------------------------------------- #
STUDENT_ORDER = ["triplet_only", "rkd_dist", "rkd_angle", "rkd_both", "graph-rkd"]
STUDENT_LABEL = {"triplet_only": "triplet-only", "rkd_dist": "+RKD-D",
                 "rkd_angle": "+RKD-A", "rkd_both": "+RKD-D+RKD-A",
                 "graph-rkd": "+Graph-RKD"}


def headline_table(df, dataset, teacher, metric="test_mAP@R"):
    """Tabela dos 5 alunos (mean±sem) para uma célula (dataset, teacher)."""
    cell = df[(df.dataset == dataset) & (df.phase == "phase5")]
    cell = cell[(cell.teacher == teacher) | (cell.student == "triplet_only")]
    t = agg(cell, ["student"], metric)
    t["order"] = t.student.map(lambda s: STUDENT_ORDER.index(s) if s in STUDENT_ORDER else 9)
    t = t.sort_values("order").drop(columns="order")
    t["student"] = t.student.map(STUDENT_LABEL).fillna(t.student)
    return t


def h1_verdict(table, sem_slack=1.0):
    """Regra do §8-H1: Graph-RKD 'iguala' se o intervalo cobre o melhor baseline;
    'vence' só se média > melhor baseline + 1 sem. (Chama por célula.)"""
    base = table[table.student != "+Graph-RKD"]
    g = table[table.student == "+Graph-RKD"]
    if base.empty or g.empty:
        return "sem dados suficientes"
    best = base.loc[base["mean"].idxmax()]
    gm, gs = float(g["mean"].iloc[0]), float(g["sem"].iloc[0])
    if gm > best["mean"] + sem_slack * best["sem"]:
        return f"Graph-RKD VENCE {best.student} ({gm:.4f} > {best['mean']:.4f}+sem)"
    if gm + sem_slack * gs >= best["mean"]:
        return f"Graph-RKD IGUALA {best.student} (dentro do noise floor)"
    return f"Graph-RKD PERDE p/ {best.student} ({gm:.4f} < {best['mean']:.4f})"


def fig_h1_bar(df, metric="test_mAP@R", save=True):
    """Barras agrupadas de mAP@R por aluno, uma faceta por (dataset, teacher)."""
    cells = df[df.phase == "phase5"][["dataset", "teacher"]].dropna().drop_duplicates()
    cells = [(d, t) for d, t in cells.itertuples(index=False)]
    if not cells:
        print("sem runs de phase5 ainda"); return None
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.4), squeeze=False)
    for ax, (d, t) in zip(axes[0], cells):
        tab = headline_table(df, d, t, metric)
        ax.bar(tab.student, tab["mean"], yerr=tab["sem"], capsize=3,
               color=["#888"] * (len(tab) - 1) + ["#2b6cb0"])
        ax.set_title(f"{d} · {t}"); ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "fig_h1_headline.pdf"), bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# §7.2 — H0 (λg viability) and H2 (normalization)                              #
# --------------------------------------------------------------------------- #
def fig_h0_lambda(df, save=True):
    """val mAP@R vs λg na fatia do gate (phase1) + piso triplet-only."""
    g = df[df.phase == "phase1"]
    gg = g[g.student == "graph-rkd"].dropna(subset=["lambda_g", "val_mAP@R"])
    if gg.empty:
        print("sem runs de phase1 ainda"); return None
    curve = agg(gg, ["lambda_g"], "val_mAP@R").sort_values("lambda_g")
    floor = g[g.student == "triplet_only"]["val_mAP@R"].dropna()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(curve.lambda_g, curve["mean"], yerr=curve["sem"], marker="o", capsize=3)
    ax.set_xscale("log"); ax.set_xlabel(r"$\lambda_g$"); ax.set_ylabel("val mAP@R")
    if len(floor):
        ax.axhline(floor.mean(), ls="--", color="k", label="triplet-only (floor)")
        ax.legend()
    ax.set_title("H0: λg viability gate")
    fig.tight_layout()
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "fig_h0_lambda.pdf"), bbox_inches="tight")
    return fig


def fig_h2_norm(df, save=True):
    """Barras de mAP@R por esquema de normalização (melhor λg por esquema), phase2."""
    g = df[(df.phase == "phase2") & (df.student == "graph-rkd")].dropna(subset=["norm"])
    if g.empty:
        print("sem runs de phase2 ainda"); return None
    best = agg(g, ["norm"], "val_mAP@R")            # melhor λg embutido via max? usar mean
    # pega, por norm, a melhor média entre λg:
    per = agg(g, ["norm", "lambda_g"], "val_mAP@R")
    best = per.loc[per.groupby("norm")["mean"].idxmax()]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(best.norm, best["mean"], yerr=best["sem"], capsize=3, color="#2b6cb0")
    ax.set_ylabel("val mAP@R (melhor λg)"); ax.set_title("H2: normalization ablation")
    fig.tight_layout()
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "fig_h2_norm.pdf"), bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# §7.3 — H4 (objective robustness) + descriptor probe (H3 fidelity)            #
# --------------------------------------------------------------------------- #
def fig_h4_overlay(df, save=True):
    """Overlay val mAP@R vs λg p/ regression e contrastive (phase4). Quantifica a
    largura da banda que fica a <1 sem do melhor de cada objetivo."""
    g = df[df.phase == "phase4"].dropna(subset=["lambda_g", "objective", "val_mAP@R"])
    if g.empty:
        print("sem runs de phase4 ainda"); return None, {}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    widths = {}
    for obj, sub in g.groupby("objective"):
        c = agg(sub, ["lambda_g"], "val_mAP@R").sort_values("lambda_g")
        ax.errorbar(c.lambda_g, c["mean"], yerr=c["sem"], marker="o", capsize=3, label=obj)
        best = c["mean"].max(); sem_at_best = float(c.loc[c["mean"].idxmax(), "sem"])
        within = c[c["mean"] >= best - sem_at_best]
        widths[obj] = (within.lambda_g.min(), within.lambda_g.max())
    ax.set_xscale("log"); ax.set_xlabel(r"$\lambda_g$"); ax.set_ylabel("val mAP@R")
    ax.legend(); ax.set_title("H4: regression vs contrastive robustness")
    fig.tight_layout()
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "fig_h4_overlay.pdf"), bbox_inches="tight")
    return fig, widths


def load_probe(path=None):
    """Carrega o CSV da sonda offline de descritor (§7.3 / H3)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "descriptor_probe.csv")
    return pd.read_csv(path)


def fig_probe(save=True):
    """MDS degeneracy e profile tie rate vs N (evidência de mecanismo p/ H3)."""
    p = load_probe()
    md = p[p.method == "mds"].groupby("N")["mds_degenerate_rate"].max()
    pt = p[p.method == "profile"].groupby("N")["profile_tie_rate"].max()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(md.index, md.values * 100, "o-", label="MDS near-degenerate %")
    ax.plot(pt.index, pt.values * 100, "s-", label="profile tie %")
    ax.set_xlabel("relational order N"); ax.set_ylabel("rate (%)")
    ax.legend(); ax.set_title("H3 mechanism: descriptor fragility vs N")
    fig.tight_layout()
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "fig_h3_probe.pdf"), bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# §7.4 — H5: N=3 vs RKD-A                                                       #
# --------------------------------------------------------------------------- #
def n3_vs_rkda(df, metric="test_mAP@R"):
    """Tabela N=3 (profile e mds) vs RKD-A por (dataset, teacher)."""
    g3 = df[(df.student == "graph-rkd") & (df.N == 3)]
    rkda = df[df.student == "rkd_angle"]
    a = agg(g3, ["dataset", "teacher", "method"], metric)
    b = agg(rkda, ["dataset", "teacher"], metric); b["method"] = "RKD-A"
    return pd.concat([a, b], ignore_index=True).sort_values(["dataset", "teacher", "method"])
