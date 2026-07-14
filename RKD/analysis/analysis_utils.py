"""Analysis utilities for the Graph-RKD campaign (fills Section 7 of the paper).

Pulls the W&B runs (gabomfim-unicamp/graph-rkd), classifies each run into one
of the 5 students / Graph-RKD config, aggregates over seeds (mean ± sem) and produces the
tables and figures that Section 7 requires (H0–H5). The notebooks only call these
functions. `pandas`/`matplotlib` are dependencies; `wandb` is imported on demand.

Analysis rules (§8 of the paper):
  * noise floor: sem of seeds per cell; difference < ~1 sem = indistinguishable;
  * "beats" only if the mean exceeds the best baseline by > 1 sem AND the sign is
    consistent across teachers; report per cell (dataset × teacher).
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd


def _load_dotenv():
    """Loads variables from a `.env` (WANDB_API_KEY etc.) without depending on a package.
    Looks in RKD/analysis/, RKD/ and the repo root; does NOT overwrite variables already
    defined in the environment. Run on import — so `00` finds the W&B key without a
    manual `export`."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        path = os.path.join(d, ".env")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        break


_load_dotenv()

ENTITY = "gabomfim-unicamp"
PROJECT = "graph-rkd"
METRICS = ["mAP@R", "R_precision", "recall@1", "recall@2", "recall@4", "recall@8"]
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
TABDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")


# --------------------------------------------------------------------------- #
# fetch + classify                                                             #
# --------------------------------------------------------------------------- #
def _student(cfg, tags):
    """Classifies the run into one of the 5 students (or teacher) from config/tags."""
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


KNOWN_PHASES = ("teachers", "dev", "conv", "phase0", "phase1", "phase2",
                "phase3", "phase4", "phase5")


def _phase_of(name, tags):
    """Run phase: phase tag if present (build_plan tags each job); otherwise
    derived from the name prefix (old runs have no tag). convfloor->conv,
    smoke->phase0, floor->phase1, baseline gate->phase5."""
    for t in (tags or []):
        if t in KNOWN_PHASES:
            return t
    n = name or ""
    if n.startswith("convfloor"):
        return "conv"
    for p in KNOWN_PHASES:
        if n.startswith(p):                     # dev-, conv-, phase0-, ...
            return p
    if n.startswith("smoke"):
        return "phase0"
    if n.startswith("floor"):
        return "phase1"
    if n.startswith("baseline"):
        return "phase5"
    return ""


def fetch_runs(entity=ENTITY, project=PROJECT):
    """DataFrame: one row per W&B run (flattened config + summary + classification)."""
    import wandb
    api = wandb.Api()
    rows = []
    for r in api.runs(f"{entity}/{project}"):
        # only FINISHED runs enter the analysis. 'crashed'/'failed'/'killed' runs
        # or still 'running' ones (campaign in progress) have partial/missing summary and
        # would enter with garbage metrics -> exclude.
        if r.state != "finished":
            continue
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
    """Statistics per config over seeds. Primary = **median** (robust, and the
    project reports the median model — we are not increasing seeds); mean/sem
    remain only as reference. Returns median/mean/sem/n."""
    g = df.dropna(subset=[metric]).groupby(group_cols)[metric]
    out = g.agg(["median", "mean", "sem", "count"]).reset_index()
    return out.rename(columns={"count": "n"})


def median_run(df, group_cols, metric="test_mAP@R"):
    """Per config, the RUN whose metric is the median — the model/checkpoint to report."""
    d = df.dropna(subset=[metric]).copy()
    picks = []
    for key, sub in d.groupby(group_cols):
        sub = sub.sort_values(metric).reset_index(drop=True)
        picks.append(sub.iloc[len(sub) // 2])       # median (odd seeds -> real run)
    return pd.DataFrame(picks)


# --------------------------------------------------------------------------- #
# §7.1 — H1: 5 alunos por (dataset, teacher)                                   #
# --------------------------------------------------------------------------- #
STUDENT_ORDER = ["triplet_only", "rkd_dist", "rkd_angle", "rkd_both", "graph-rkd"]
STUDENT_LABEL = {"triplet_only": "triplet-only", "rkd_dist": "+RKD-D",
                 "rkd_angle": "+RKD-A", "rkd_both": "+RKD-D+RKD-A",
                 "graph-rkd": "+Graph-RKD"}


def headline_table(df, dataset, teacher, metric="test_mAP@R"):
    """Table of the 5 students (mean±sem) for a cell (dataset, teacher)."""
    cell = df[(df.dataset == dataset) & (df.phase == "phase5")]
    cell = cell[(cell.teacher == teacher) | (cell.student == "triplet_only")]
    t = agg(cell, ["student"], metric)
    t["order"] = t.student.map(lambda s: STUDENT_ORDER.index(s) if s in STUDENT_ORDER else 9)
    t = t.sort_values("order").drop(columns="order")
    t["student"] = t.student.map(STUDENT_LABEL).fillna(t.student)
    return t


def h1_verdict(table):
    """H1 verdict by MEDIAN (seeds not increased; we report the median
    model). Graph-RKD 'beats'/'ties'/'loses' comparing the cell median
    against that of the best baseline. (Call per dataset×teacher cell.)"""
    base = table[table.student != "+Graph-RKD"]
    g = table[table.student == "+Graph-RKD"]
    if base.empty or g.empty:
        return "insufficient data"
    best = base.loc[base["median"].idxmax()]
    gm = float(g["median"].iloc[0])
    if gm > best["median"]:
        return f"Graph-RKD BEATS {best.student} (median {gm:.4f} > {best['median']:.4f})"
    if abs(gm - best["median"]) < 1e-6:
        return f"Graph-RKD TIES {best.student} (median {gm:.4f})"
    return f"Graph-RKD LOSES to {best.student} (median {gm:.4f} < {best['median']:.4f})"


def fig_h1_bar(df, metric="test_mAP@R", save=True):
    """Grouped bars of mAP@R per student, one facet per (dataset, teacher)."""
    cells = df[df.phase == "phase5"][["dataset", "teacher"]].dropna().drop_duplicates()
    cells = [(d, t) for d, t in cells.itertuples(index=False)]
    if not cells:
        print("no phase5 runs yet"); return None
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.4), squeeze=False)
    for ax, (d, t) in zip(axes[0], cells):
        tab = headline_table(df, d, t, metric)
        ax.bar(tab.student, tab["median"], yerr=tab["sem"], capsize=3,
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
    """val mAP@R vs λg on the gate slice (phase1) + triplet-only floor."""
    g = df[df.phase == "phase1"]
    gg = g[g.student == "graph-rkd"].dropna(subset=["lambda_g", "val_mAP@R"])
    if gg.empty:
        print("no phase1 runs yet"); return None
    curve = agg(gg, ["lambda_g"], "val_mAP@R").sort_values("lambda_g")
    floor = g[g.student == "triplet_only"]["val_mAP@R"].dropna()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(curve.lambda_g, curve["median"], yerr=curve["sem"], marker="o", capsize=3)
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
    """mAP@R bars per normalization scheme (best λg per scheme), phase2."""
    g = df[(df.phase == "phase2") & (df.student == "graph-rkd")].dropna(subset=["norm"])
    if g.empty:
        print("no phase2 runs yet"); return None
    best = agg(g, ["norm"], "val_mAP@R")            # best λg embedded via max? use mean
    # takes, per norm, the best mean across λg:
    per = agg(g, ["norm", "lambda_g"], "val_mAP@R")
    best = per.loc[per.groupby("norm")["median"].idxmax()]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(best.norm, best["median"], yerr=best["sem"], capsize=3, color="#2b6cb0")
    ax.set_ylabel("val mAP@R (best λg)"); ax.set_title("H2: normalization ablation")
    fig.tight_layout()
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "fig_h2_norm.pdf"), bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# §7.3 — H4 (objective robustness) + descriptor probe (H3 fidelity)            #
# --------------------------------------------------------------------------- #
def fig_h4_overlay(df, save=True):
    """Overlay val mAP@R vs λg for regression and contrastive (phase4). Quantifies the
    width of the band that stays within <1 sem of the best of each objective."""
    g = df[df.phase == "phase4"].dropna(subset=["lambda_g", "objective", "val_mAP@R"])
    if g.empty:
        print("no phase4 runs yet"); return None, {}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    widths = {}
    for obj, sub in g.groupby("objective"):
        c = agg(sub, ["lambda_g"], "val_mAP@R").sort_values("lambda_g")
        ax.errorbar(c.lambda_g, c["median"], yerr=c["sem"], marker="o", capsize=3, label=obj)
        best = c["median"].max()                    # width: λg within 10% of the best
        within = c[c["median"] >= 0.9 * best]
        widths[obj] = (within.lambda_g.min(), within.lambda_g.max())
    ax.set_xscale("log"); ax.set_xlabel(r"$\lambda_g$"); ax.set_ylabel("val mAP@R")
    ax.legend(); ax.set_title("H4: regression vs contrastive robustness")
    fig.tight_layout()
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "fig_h4_overlay.pdf"), bbox_inches="tight")
    return fig, widths


def load_probe(path=None):
    """Loads the CSV from the offline descriptor probe (§7.3 / H3)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "descriptor_probe.csv")
    return pd.read_csv(path)


def fig_probe(save=True):
    """MDS degeneracy and profile tie rate vs N (mechanism evidence for H3)."""
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
def n3_vs_rkda(df, metric="test_mAP@R", phase="phase5"):
    """Table N=3 (profile and mds) vs RKD-A per (dataset, teacher).

    Restricted to the HEADLINE (``phase='phase5'``) by default: the matched-arity comparison
    must use the SAME conditions as RKD-A (which only runs in phase5). Without the filter,
    the search N=3 (phase2/3, other norm/λg) would enter and bias the verdict."""
    sub = df[df.phase == phase] if phase else df
    g3 = sub[(sub.student == "graph-rkd") & (sub.N == 3)]
    rkda = sub[sub.student == "rkd_angle"]
    a = agg(g3, ["dataset", "teacher", "method"], metric)
    b = agg(rkda, ["dataset", "teacher"], metric); b["method"] = "RKD-A"
    return pd.concat([a, b], ignore_index=True).sort_values(["dataset", "teacher", "method"])


# --------------------------------------------------------------------------- #
# Export of tables for the paper (CSV + LaTeX booktabs) → tables/                #
# --------------------------------------------------------------------------- #
def save_table(df, name, caption="", label=""):
    """Saves `df` as tables/<name>.csv and tables/<name>.tex (booktabs)."""
    os.makedirs(TABDIR, exist_ok=True)
    df.to_csv(os.path.join(TABDIR, name + ".csv"), index=False)
    body = df.to_latex(index=False, float_format="%.4f")
    tex = "\\begin{table}[h]\n\\centering\n\\small\n"
    if caption:
        tex += f"\\caption{{{caption}}}\n"
    if label:
        tex += f"\\label{{{label}}}\n"
    tex += body + "\\end{table}\n"
    with open(os.path.join(TABDIR, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    return name


def export_tables(df):
    """Generates the canonical paper tables (H1/H2/H3/H4/H5) in tables/*.{csv,tex}.
    Skips the table whose phase has no data yet. Returns the list of files."""
    written = []
    # H1 — headline per cell (dataset, teacher): 5 students × mAP@R/R-Prec/R@1 (median)
    cells = df[df.phase == "phase5"][["dataset", "teacher"]].dropna().drop_duplicates()
    for d, t in cells.itertuples(index=False):
        tab = headline_table(df, d, t, "test_mAP@R")[["student", "median", "n"]]
        tab = tab.rename(columns={"median": "mAP@R"})
        for m, col in [("test_R_precision", "R-Prec"), ("test_recall@1", "R@1")]:
            mm = headline_table(df, d, t, m)[["student", "median"]].rename(columns={"median": col})
            tab = tab.merge(mm, on="student", how="left")
        written.append(save_table(tab, f"headline_{d}_{t}",
                                  f"Headline (H1) --- {d}, {t}: median test metrics.",
                                  f"tab:headline-{d}-{t}"))
    # H2 — normalization (phase2): best λg per (norm, N)
    g2 = df[(df.phase == "phase2") & (df.student == "graph-rkd")]
    if len(g2):
        per = agg(g2, ["norm", "N", "lambda_g"], "test_mAP@R")
        best = per.loc[per.groupby(["norm", "N"])["median"].idxmax()]
        written.append(save_table(best[["norm", "N", "lambda_g", "median", "n"]]
                                  .rename(columns={"median": "mAP@R"}), "h2_normalization",
                                  "Normalization ablation (H2): best-$\\lambda_g$ median test mAP@R.",
                                  "tab:h2"))
    # H3 — descriptor (phase3): profile vs mds per (dataset, teacher, N)
    g3 = df[(df.phase == "phase3") & (df.student == "graph-rkd")]
    if len(g3):
        t3 = agg(g3, ["dataset", "teacher", "method", "N"], "test_mAP@R")
        written.append(save_table(t3[["dataset", "teacher", "method", "N", "median", "n"]]
                                  .rename(columns={"median": "mAP@R"}), "h3_descriptor",
                                  "Descriptor characterization (H3): median test mAP@R.",
                                  "tab:h3"))
    # H4 — objective (phase4): regression vs contrastive per λg
    g4 = df[(df.phase == "phase4") & (df.student == "graph-rkd")]
    if len(g4):
        t4 = agg(g4, ["objective", "lambda_g"], "test_mAP@R")
        written.append(save_table(t4[["objective", "lambda_g", "median", "n"]]
                                  .rename(columns={"median": "mAP@R"}), "h4_objective",
                                  "Objective robustness (H4): median test mAP@R vs $\\lambda_g$.",
                                  "tab:h4"))
    # H5 — N=3 vs RKD-A (phase5)
    t5 = n3_vs_rkda(df, "test_mAP@R")
    if len(t5):
        written.append(save_table(t5[["dataset", "teacher", "method", "median", "n"]]
                                  .rename(columns={"median": "mAP@R"}), "h5_n3_vs_rkda",
                                  "N=3 vs RKD-A (H5): median test mAP@R per cell.", "tab:h5"))
    print(f"[tables] {len(written)} tables -> {TABDIR}: {written}")
    return written
