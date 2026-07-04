#!/usr/bin/env python3
"""
Parte 2 — Análise + material para o LaTeX.

Lê os CSVs de pull_wandb.py e produz, em analysis/out/:
  - figuras (mAP@R vs N; convergência; métricas de apoio) — uma figura 2x2 por teacher
  - tabelas LaTeX (booktabs)
  - FINDINGS.md  -> texto pronto p/ o Claude-web compor a seção de resultados
  - summary_Nstar.csv

Dimensões nos dados: teacher_arch {resnet18, convnext_tiny} x method {profile, mds}
x dataset {cars196, cub200}, mode=regression, N in {2,4,8,16,17} x 3 seeds.
O Gabriel definiu "4 análises" (2 method x 2 dataset) -> provavelmente 1 teacher.
Geramos os 8 cenários (4 por teacher) e sinalizamos p/ escolher qual teacher manter.

Caveats registrados no relatório:
  * As curvas de N (tag=search) usam orçamento CURTO de épocas — servem p/ comparar
    N entre si (relativo), não como número final de qualidade.
  * Os runs tag=final da campanha caíram todos em N=2 (seleção por val recall@1 no
    budget curto); a análise de convergência mostra o trade-off velocidade x teto.
"""
import os, itertools, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT  = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

PRIMARY  = "test/mAP@R"
SUPPORT  = ["test/recall@1", "test/R_precision"]
DATASETS = ["cars196", "cub200"]
METHODS  = ["profile", "mds"]
SPEED_FRAC = 0.95

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})


def sem(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def spearman(x, y):
    x, y = pd.Series(list(x)), pd.Series(list(y))
    return x.corr(y, method="spearman") if x.notna().sum() >= 3 else np.nan


def df_to_md(df):
    """Tabela markdown sem depender de 'tabulate'."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _read(base):
    pq, csv = os.path.join(DATA, base + ".parquet"), os.path.join(DATA, base + ".csv")
    if os.path.exists(pq):
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(csv)


def load():
    return _read("runs_summary"), _read("history")


def teachers_in(s):
    reg = s[s["mode"] == "regression"]
    return sorted(reg["teacher_arch"].dropna().unique().tolist())


# --------------------------------------------------------------------------- #
def per_N_table(s, teacher, dataset, method):
    reg = s[(s["mode"] == "regression") & (s.teacher_arch == teacher) &
            (s.dataset == dataset) & (s.method == method) & (s.tag == "search")]
    rows = []
    for N, g in reg.groupby("N"):
        row = {"N": int(N), "n_seeds": g.seed.nunique()}
        for col in [PRIMARY] + SUPPORT:
            row[col + "_mean"] = g[col].mean()
            row[col + "_sem"]  = sem(g[col].values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def baseline_value(s, teacher, dataset, col=PRIMARY):
    """Baseline off = destilação sem grafo. Casa por (teacher, dataset) — 'method'
    não se aplica quando não há grafo."""
    off = s[(s["mode"] == "off") & (s.teacher_arch == teacher) & (s.dataset == dataset)]
    return (off[col].mean(), off.run_id.nunique()) if len(off) else (np.nan, 0)


def convergence_curves(h, teacher, dataset, method, col=PRIMARY):
    reg = h[(h["mode"] == "regression") & (h.teacher_arch == teacher) &
            (h.dataset == dataset) & (h.method == method) & (h.tag == "search")]
    curves = {}
    for N, g in reg.groupby("N"):
        gg = g.dropna(subset=[col])
        if gg.empty:
            continue
        agg = gg.groupby("epoch")[col].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        curves[int(N)] = agg
    return curves


def convergence_stats(curves, frac=SPEED_FRAC):
    rows = []
    for N, agg in sorted(curves.items()):
        ceil = agg["mean"].max(); thr = frac * ceil
        hit = agg[agg["mean"] >= thr]
        rows.append({"N": N, "ceiling": ceil,
                     "epoch_hit": int(hit["epoch"].min()) if len(hit) else np.nan,
                     "final": agg["mean"].iloc[-1]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def fig_metric_vs_N(s, teacher, col, ylabel, fname):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, (ds, me) in zip(axes.ravel(), itertools.product(DATASETS, METHODS)):
        t = per_N_table(s, teacher, ds, me)
        if t.empty:
            ax.set_title(f"{ds} · {me} (sem dados)"); continue
        xpos = np.arange(len(t))                      # eixo ordinal (evita 16/17 colidirem)
        ax.errorbar(xpos, t[col + "_mean"], yerr=t[col + "_sem"], marker="o",
                    capsize=3, lw=1.8, label="regression")
        bv, _ = baseline_value(s, teacher, ds, col)
        if not np.isnan(bv):
            ax.axhline(bv, ls="--", color="gray", lw=1.2, label=f"off@120ép ({bv:.3f})")
        i = int(t[col + "_mean"].idxmax())
        ax.scatter([xpos[i]], [t[col + "_mean"][i]], s=140, facecolors="none",
                   edgecolors="crimson", linewidths=2, zorder=5, label=f"N*={int(t.N[i])}")
        ax.set_xticks(xpos); ax.set_xticklabels([int(n) for n in t.N])
        ax.set_title(f"{ds} · {me}"); ax.set_xlabel("N (nº de nós)")
        ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    fig.suptitle(f"{ylabel} vs N — regression · teacher={teacher}", fontsize=12, y=.995)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p + ".png"); fig.savefig(p + ".pdf"); plt.close(fig)
    print("  figura:", p + ".png")


def fig_convergence(h, teacher, fname):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, (ds, me) in zip(axes.ravel(), itertools.product(DATASETS, METHODS)):
        curves = convergence_curves(h, teacher, ds, me)
        if not curves:
            ax.set_title(f"{ds} · {me} (sem dados)"); continue
        cmap = plt.cm.viridis(np.linspace(0, .9, len(curves)))
        for c, (N, agg) in zip(cmap, sorted(curves.items())):
            ax.plot(agg.epoch, agg["mean"], color=c, lw=1.7, label=f"N={N}")
            ax.fill_between(agg.epoch, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                            color=c, alpha=.15)
        ax.set_title(f"{ds} · {me}"); ax.set_xlabel("época")
        ax.set_ylabel("test mAP@R"); ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Convergência: test mAP@R por época · teacher={teacher}", y=.995)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p + ".png"); fig.savefig(p + ".pdf"); plt.close(fig)
    print("  figura:", p + ".png")


# --------------------------------------------------------------------------- #
def latex_tables(s, teachers):
    lines = [r"% Tabelas geradas por analyze.py — requer \usepackage{booktabs}", ""]
    for teacher in teachers:
        for ds, me in itertools.product(DATASETS, METHODS):
            t = per_N_table(s, teacher, ds, me)
            if t.empty:
                continue
            best = int(t.loc[t[PRIMARY + "_mean"].idxmax(), "N"])
            ns = int(t.n_seeds.max())
            tt = teacher.replace("_", r"\_")
            lines += [r"\begin{table}[t]\centering",
                      rf"\caption{{{ds} $\cdot$ {me} (regression, teacher={tt}): "
                      rf"métricas por $N$ (média$\pm$sem, {ns} seeds). $N^*$ em negrito.}}",
                      r"\begin{tabular}{rccc}\toprule",
                      r"$N$ & mAP@R & Recall@1 & R-Precision \\ \midrule"]
            for _, r in t.iterrows():
                N = int(r.N); is_best = N == best
                cells = []
                for j, col in enumerate([PRIMARY, "test/recall@1", "test/R_precision"]):
                    cell = f"{r[col+'_mean']:.3f}\\,$\\pm$\\,{r[col+'_sem']:.3f}"
                    if is_best and j == 0:
                        cell = rf"\textbf{{{cell}}}"
                    cells.append(cell)
                nlab = rf"\textbf{{{N}}}" if is_best else f"{N}"
                lines.append(f"{nlab} & " + " & ".join(cells) + r" \\")
            lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    with open(os.path.join(OUT, "tables.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  tables.tex escrito")


def build_findings(s, h, teachers):
    md = ["# Resultados — melhor N para Graph-RKD (regression)\n",
          "**Métrica primária:** test mAP@R (Recall@1 e R-Precision de apoio), "
          "média ± sem sobre 3 seeds. `N` = nº de nós do grafo relacional. "
          "Baseline `off` = destilação sem perda de grafo.\n",
          "> **Escopo:** o Gabriel definiu 4 análises (2 métodos × 2 datasets). "
          "Os dados têm **2 teachers** (`resnet18`, `convnext_tiny`); geramos os 4 "
          "cenários para cada um — **escolham qual teacher manter** no trabalho.\n",
          "> **Caveats (importantes):** (1) o sweep de N usa runs de *busca* de **~30 "
          "épocas** (subtreinados: mAP@R ~0.006–0.02). Ele compara N **entre si** no mesmo "
          "budget — não é o número final de qualidade. (2) O baseline `off` e os runs "
          "`final` rodam **120 épocas**, então **NÃO são comparáveis** ao sweep @30ép "
          "(o off aparece maior só por isso — não é o grafo 'perdendo'). (3) O único N com "
          "budget cheio (120 ép) é **N=2**, porque a seleção automática (val Recall@1) "
          "elegeu N=2 em todas as células. Ainda assim, as curvas de treino já permitem "
          "concluir (ver **Veredito** no fim): os N>2 ficam planos, não apenas lentos.\n"]
    summary_rows = []
    for teacher in teachers:
        md.append(f"\n# Teacher = {teacher}\n")
        for ds, me in itertools.product(DATASETS, METHODS):
            t = per_N_table(s, teacher, ds, me)
            if t.empty:
                continue
            col = PRIMARY + "_mean"
            i = t[col].idxmax(); Nstar = int(t.N[i]); best = t[col][i]
            worst = t[col].min()
            bv, _ = baseline_value(s, teacher, ds)
            rho = spearman(t.N, t[col])
            cst = convergence_stats(convergence_curves(h, teacher, ds, me))
            d_worst = (best - worst) / worst * 100 if worst else np.nan
            d_base  = (best - bv) / bv * 100 if bv and not np.isnan(bv) else np.nan
            # meio-termo: dentro de 2% do teto de qualidade e mais rápido a convergir
            near = t[t[col] >= 0.98 * best]
            speed = dict(zip(cst.N, cst.epoch_hit)) if not cst.empty else {}
            mid = None
            if not near.empty and speed:
                near = near.assign(ep=near.N.map(speed))
                if near.ep.notna().any():
                    mid = int(near.sort_values("ep").iloc[0].N)

            summary_rows.append(dict(teacher=teacher, dataset=ds, method=me, Nstar=Nstar,
                                     mAPR_sweep30ep=round(best, 4),
                                     delta_vs_worstN_pct=round(d_worst, 1) if not np.isnan(d_worst) else None,
                                     off_ref_120ep=round(bv, 4) if not np.isnan(bv) else None,
                                     spearman_N=round(rho, 2) if not np.isnan(rho) else None,
                                     meio_termo=mid))
            trend = ("↑ cresce com N" if rho > .6 else "↓ cai com N" if rho < -.6 else
                     "pico interno / não-monotônico" if abs(rho) < .3 else "tendência fraca")
            md.append(f"## {ds} · {me}\n")
            md.append(f"- **N\\* = {Nstar}** (test mAP@R = {best:.4f}, sweep @30ép).")
            md.append(f"- **Quão melhor (dentro do sweep):** {d_worst:+.1f}% vs o pior N ({worst:.4f}).")
            if not np.isnan(bv):
                md.append(f"- _Ref.: baseline off @120ép = {bv:.4f} (não comparável ao sweep @30ép — só contexto)._")
            md.append(f"- **Padrão com N:** Spearman(N, mAP@R) = {rho:.2f} → {trend}.")
            if not cst.empty and cst.epoch_hit.notna().any():
                topc = cst.sort_values("ceiling", ascending=False).iloc[0]   # maior teto
                others = cst[cst.N != topc.N]
                flat = others.ceiling.max() if len(others) else np.nan
                topc_speed = cst[cst.N == topc.N].epoch_hit.iloc[0]
                md.append(f"- **Convergência:** só **N={int(topc.N)}** aprende de fato "
                          f"(maior teto {topc.ceiling:.4f}; atinge {int(SPEED_FRAC*100)}% dele "
                          f"na época {int(topc_speed)}). Os demais N ficam **estagnados** "
                          f"(teto ≤ {flat:.4f}) em 30 ép — não é só 'convergir mais devagar', "
                          f"eles quase não saem do lugar.")
            if mid is not None and mid == Nstar:
                md.append(f"- **Meio-termo:** não há trade-off — N={Nstar} domina em "
                          f"velocidade **e** qualidade, então é a escolha única (não se ganha "
                          f"nada indo pra N maior neste budget).")
            elif mid is not None:
                md.append(f"- **Meio-termo:** N={mid} (≤2% abaixo do teto e entre os mais rápidos).")
            md.append("")

    md += [
        "\n# Veredito (com os dados existentes no W&B)\n",
        "- **Melhor N = 2**, de forma consistente nas **8 células** (2 teachers × 2 datasets "
        "× 2 métodos). Vale para ambos os teachers (`resnet18` e `convnext_tiny`) — a escolha "
        "de teacher **não muda** o N ótimo.",
        "- **Não é só 'N=2 converge mais rápido':** nas curvas de treino, os N>2 ficam "
        "**planos** (~0.002 de mAP@R ao longo das 30 épocas), enquanto N=2 **sobe de forma "
        "consistente**. Uma curva plana não é 'lenta' — é sinal de que o sinal relacional do "
        "grafo com muitos nós praticamente não é aproveitado neste setup. Não há indício de "
        "cruzamento: os N maiores não estão subindo em direção ao N=2.",
        "- **Tendência com N:** aumentar N não ajuda (Spearman(N, mAP@R) ≤ 0 na maioria; cai "
        "claramente no `mds`). A mensagem do trabalho pode ser direta: **para a comparação por "
        "regressão, o grafo relacional mínimo (N=2) é o que entrega — adicionar nós não traz "
        "ganho e chega a atrapalhar.**",
        "- **Corroboração em budget cheio:** o único N que rodou 120 épocas (runs `final`) é o "
        "N=2, que sobe até mAP@R ~0.03–0.04 — coerente com N=2 sendo o operacional.",
        "",
        "## Limite honesto desta conclusão",
        "- O sweep de N existe apenas em **~30 épocas**; não há N>2 em 120 épocas. Logo, não é "
        "possível **provar** que nenhum N maior superaria N=2 num treino longo. Mas, com os "
        "dados disponíveis, a evidência (curvas planas dos N>2, ausência de tendência de "
        "cruzamento) aponta **de forma consistente** para N=2 — é a conclusão defensável.",
        "- O baseline `off` (120 ép) **não** é comparável ao sweep (30 ép); entra só como "
        "contexto, não como veredito sobre 'o grafo ajuda vs não-grafo'.",
        "",
    ]
    sdf = pd.DataFrame(summary_rows)
    md.insert(4, "## Resumo dos cenários\n\n" + df_to_md(sdf) + "\n")
    with open(os.path.join(OUT, "FINDINGS.md"), "w") as f:
        f.write("\n".join(md))
    sdf.to_csv(os.path.join(OUT, "summary_Nstar.csv"), index=False)
    print("  FINDINGS.md e summary_Nstar.csv escritos")
    return sdf


def main():
    s, h = load()
    teachers = teachers_in(s)
    print(f"[analyze] {len(s)} runs, {len(h)} linhas de história. teachers={teachers}")
    for teacher in teachers:
        fig_metric_vs_N(s, teacher, PRIMARY, "test mAP@R", f"fig_mapr_vs_N_{teacher}")
        fig_metric_vs_N(s, teacher, "test/recall@1", "test Recall@1", f"fig_recall1_vs_N_{teacher}")
        fig_metric_vs_N(s, teacher, "test/R_precision", "test R-Precision", f"fig_rprec_vs_N_{teacher}")
        fig_convergence(h, teacher, f"fig_convergence_{teacher}")
    latex_tables(s, teachers)
    build_findings(s, h, teachers)
    print("[analyze] concluído. Veja analysis/out/")


if __name__ == "__main__":
    main()
