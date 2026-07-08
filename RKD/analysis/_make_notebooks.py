"""Gera os notebooks de análise (.ipynb) que preenchem a Seção 7 do paper.

Os notebooks são finos: chamam ``analysis_utils`` (a lógica real e testável).
Rode: python analysis/_make_notebooks.py  ->  escreve os .ipynb em analysis/.
Requer que as runs estejam no W&B (gabomfim-unicamp-org/graph-rkd) para produzir
números; sem runs, as células de figura avisam "sem dados ainda" e não quebram.
"""

import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(*cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md"
               else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}}
    return n


def write(name, notebook):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print("wrote", name)


SETUP = ("code", "import sys, os\n"
         "sys.path.insert(0, os.path.dirname(os.getcwd()))  # RKD/ on path\n"
         "sys.path.insert(0, os.getcwd())\n"
         "import analysis_utils as au\n"
         "import pandas as pd\n"
         "pd.set_option('display.float_format', lambda x: f'{x:.4f}')")

# --------------------------------------------------------------------------- #
# 00 — aggregate                                                               #
# --------------------------------------------------------------------------- #
write("00_aggregate_results.ipynb", nb(
    ("md", "# 00 · Aggregate W&B runs → `results.csv`\n\n"
           "Pulls every run from **gabomfim-unicamp-org/graph-rkd**, classifies each "
           "into one of the 5 students / Graph-RKD config, flattens config+summary, "
           "and writes `results.csv` used by the other notebooks. Needs `WANDB_API_KEY`."),
    SETUP,
    ("code", "df = au.fetch_runs()               # entity/project default to the campaign\n"
             "print(len(df), 'runs')\n"
             "df.to_csv('results.csv', index=False)\n"
             "df[['run','phase','student','dataset','teacher','seed','test_mAP@R']].head(20)"),
    ("md", "Sanity: how many finished runs per phase / student."),
    ("code", "df.groupby(['phase','student']).size().unstack(fill_value=0)"),
))

# --------------------------------------------------------------------------- #
# 01 — §7.1 quantitative (H1)                                                  #
# --------------------------------------------------------------------------- #
write("01_quantitative_H1.ipynb", nb(
    ("md", "# 01 · §7.1 Quantitative Results (H1)\n\n"
           "Per-(dataset, teacher) tables of test **mAP@R / R-Precision / Recall@K** "
           "(mean ± sem over ≥3 seeds) for the 5 students, plus the grouped mAP@R bar "
           "chart. Verdict by the §8 rule: *matches* if the interval overlaps the best "
           "baseline; *beats* only if mean > best baseline + 1 sem **and** the sign is "
           "consistent across teachers — reported per cell, never averaged over a sign flip."),
    SETUP,
    ("code", "df = pd.read_csv('results.csv')\n"
             "cells = df[df.phase=='phase5'][['dataset','teacher']].dropna().drop_duplicates()\n"
             "cells = list(cells.itertuples(index=False, name=None)); cells"),
    ("md", "### Headline tables (one per cell)"),
    ("code", "for d,t in cells:\n"
             "    print(f'=== {d} · {t} ===')\n"
             "    tab = au.headline_table(df, d, t, 'test_mAP@R')\n"
             "    display(tab)\n"
             "    print('H1:', au.h1_verdict(tab), '\\n')"),
    ("md", "### R-Precision and Recall@1 (secondary)"),
    ("code", "for m in ['test_R_precision','test_recall@1']:\n"
             "    print('---', m, '---')\n"
             "    for d,t in cells: display(au.headline_table(df, d, t, m))"),
    ("md", "### Grouped mAP@R bar chart (Figure for §7.1)"),
    ("code", "au.fig_h1_bar(df);"),
))

# --------------------------------------------------------------------------- #
# 02 — §7.2 order + normalization (H0, H2)                                     #
# --------------------------------------------------------------------------- #
write("02_order_normalization_H0_H2.ipynb", nb(
    ("md", "# 02 · §7.2 Relational Order N and Normalization (H0, H2)\n\n"
           "**H0 gate:** val mAP@R vs λg on the gate slice + triplet-only floor — accept "
           "if some λg clears the floor within noise. **H2:** normalization ablation. "
           "Plus the per-order characterization (quality vs N)."),
    SETUP,
    ("code", "df = pd.read_csv('results.csv')"),
    ("md", "### H0 — λg viability gate (phase1)"),
    ("code", "au.fig_h0_lambda(df);\n"
             "g = df[(df.phase=='phase1')&(df.student=='graph-rkd')]\n"
             "floor = df[(df.phase=='phase1')&(df.student=='triplet_only')]['val_mAP@R'].mean()\n"
             "band = au.agg(g,['lambda_g'],'val_mAP@R')\n"
             "print('floor val mAP@R =', floor)\n"
             "print('viable λg band (mean ≥ floor):')\n"
             "display(band[band['mean']>=floor])"),
    ("md", "### H2 — normalization ablation (phase2)"),
    ("code", "au.fig_h2_norm(df);\n"
             "g2 = df[(df.phase=='phase2')&(df.student=='graph-rkd')]\n"
             "per = au.agg(g2,['norm','lambda_g'],'val_mAP@R')\n"
             "display(per.loc[per.groupby('norm')['mean'].idxmax()])"),
    ("md", "### Per-order quality (phase3): mAP@R vs N by descriptor"),
    ("code", "g3 = df[(df.phase=='phase3')&(df.student=='graph-rkd')]\n"
             "if len(g3):\n"
             "    display(au.agg(g3,['method','N'],'test_mAP@R'))\n"
             "else:\n"
             "    print('sem runs de phase3 ainda')"),
))

# --------------------------------------------------------------------------- #
# 03 — §7.3 descriptor + objective (H3, H4)                                    #
# --------------------------------------------------------------------------- #
write("03_descriptor_objective_H3_H4.ipynb", nb(
    ("md", "# 03 · §7.3 Descriptor & Objective (H3, H4)\n\n"
           "**H3:** profile-vs-MDS accuracy crossed with the offline fidelity/stability "
           "probe (`descriptor_probe.csv`) — recommend a descriptor only where the "
           "accuracy gap clears noise **and** is explained by a measured mechanism. "
           "**H4:** regression-vs-contrastive λg-robustness overlay + band width."),
    SETUP,
    ("code", "df = pd.read_csv('results.csv')"),
    ("md", "### H3 — descriptor fidelity/stability probe (runs now, no training)\n"
           "MDS near-degeneracy and profile tie rate vs N; this is the *mechanism* "
           "evidence the paper's §7.3 asks to cross with accuracy."),
    ("code", "probe = au.load_probe(); display(probe)\n"
             "au.fig_probe();"),
    ("md", "### H3 — profile vs MDS accuracy by (N, dataset, teacher) [needs runs]"),
    ("code", "g3 = df[(df.phase=='phase3')&(df.student=='graph-rkd')]\n"
             "if len(g3):\n"
             "    display(au.agg(g3,['dataset','teacher','method','N'],'test_mAP@R'))\n"
             "else:\n"
             "    print('sem runs de phase3 ainda — a sonda acima já dá o mecanismo (H3)')"),
    ("md", "### H4 — objective robustness overlay (phase4)"),
    ("code", "fig, widths = au.fig_h4_overlay(df)\n"
             "print('largura da banda λg (dentro de 1 sem do melhor) por objetivo:')\n"
             "for k,v in (widths or {}).items(): print(f'  {k}: {v}')"),
))

# --------------------------------------------------------------------------- #
# 04 — §7.4 N=3 vs RKD-A (H5)                                                   #
# --------------------------------------------------------------------------- #
write("04_n3_vs_rkda_H5.ipynb", nb(
    ("md", "# 04 · §7.4 N=3 vs RKD-A (H5)\n\n"
           "Matched-arity comparison: Graph-RKD at N=3 (profile **and** MDS) vs RKD-A, "
           "per (dataset, teacher). If RKD-A beats Graph-RKD-N3 beyond noise, the angular "
           "signal carries information the distance descriptor misses → motivates angular "
           "descriptors (future work)."),
    SETUP,
    ("code", "df = pd.read_csv('results.csv')\n"
             "tab = au.n3_vs_rkda(df, 'test_mAP@R'); display(tab)"),
    ("md", "Per-cell verdict: compare each method's mean±sem against RKD-A within the "
           "same (dataset, teacher)."),
))

# --------------------------------------------------------------------------- #
# 05 — §7.5 qualitative retrieval                                              #
# --------------------------------------------------------------------------- #
write("05_qualitative_retrieval.ipynb", nb(
    ("md", "# 05 · §7.5 Qualitative retrieval panels\n\n"
           "Top-k retrieval panels (successes + failures) for Graph-RKD vs the strongest "
           "classic baseline. Loads a trained student checkpoint and the test split; "
           "green border = same class as query, red = different. Caveat (paper): panels "
           "show error *modes*, not ranking quality — that is the metrics' job."),
    ("code", "import os, sys, torch\n"
             "sys.path.insert(0, os.path.dirname(os.getcwd()))\n"
             "import torchvision.transforms as T\n"
             "from metric_common import DATASETS, build_metric_loaders, embed\n"
             "from model import ConvNextMicro\n"
             "CKPT = os.environ.get('STUDENT_CKPT', '')   # set to a student_last.pth / best\n"
             "DATASET = os.environ.get('QUAL_DATASET', 'cub200')\n"
             "DATA = os.environ.get('DATA', 'data')\n"
             "assert CKPT and os.path.exists(CKPT), 'defina STUDENT_CKPT p/ um checkpoint treinado'"),
    ("code", "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
             "mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]\n"
             "test_tf = T.Compose([T.Resize((256,256)), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean,std)])\n"
             "cls, marker = DATASETS[DATASET]\n"
             "loaders, info = build_metric_loaders(cls, DATA, test_tf, test_tf, 128, 5, 100, 4, 0.2, 0, False)\n"
             "student = ConvNextMicro(num_classes=2, dims=(24,48,96,192), depths=(1,1,3,1), apply_softmax=False).to(dev)\n"
             "sd = torch.load(CKPT, map_location=dev); student.load_state_dict(sd.get('best_state') or sd['model']); student.eval()"),
    ("code", "import torch.nn.functional as F\n"
             "embs, labels, imgs = [], [], []\n"
             "with torch.no_grad():\n"
             "    for x,y in loaders['test']:\n"
             "        e,_ = embed(student, x.to(dev), True); embs.append(e.cpu()); labels.append(y); imgs.append(x)\n"
             "embs=torch.cat(embs); labels=torch.cat(labels); imgs=torch.cat(imgs)\n"
             "sim = embs @ embs.T; sim.fill_diagonal_(-1)\n"
             "topk = sim.topk(5, dim=1).indices\n"
             "print('test embeddings:', embs.shape)"),
    ("code", "import matplotlib.pyplot as plt, numpy as np\n"
             "def show(qs, title):\n"
             "    fig, ax = plt.subplots(len(qs), 6, figsize=(11, 1.9*len(qs)))\n"
             "    inv = lambda t: (t*torch.tensor(std)[:,None,None]+torch.tensor(mean)[:,None,None]).clamp(0,1).permute(1,2,0).numpy()\n"
             "    for r,q in enumerate(qs):\n"
             "        ax[r,0].imshow(inv(imgs[q])); ax[r,0].set_title('query'); ax[r,0].axis('off')\n"
             "        for c,j in enumerate(topk[q]):\n"
             "            a=ax[r,c+1]; a.imshow(inv(imgs[j])); a.axis('off')\n"
             "            ok = labels[j]==labels[q]\n"
             "            for s in a.spines.values(): s.set_visible(True); s.set_color('green' if ok else 'red'); s.set_linewidth(3)\n"
             "    fig.suptitle(title); fig.tight_layout()\n"
             "# alguns acertos e erros\n"
             "correct = [i for i in range(len(labels)) if labels[topk[i,0]]==labels[i]]\n"
             "wrong = [i for i in range(len(labels)) if labels[topk[i,0]]!=labels[i]]\n"
             "show(correct[:4], 'Graph-RKD student — successes')\n"
             "show(wrong[:4], 'Graph-RKD student — failures')"),
))

# --------------------------------------------------------------------------- #
# 06 — findings (§8 verdicts)                                                  #
# --------------------------------------------------------------------------- #
write("06_findings.ipynb", nb(
    ("md", "# 06 · Findings — H0–H5 verdicts (feeds §8 / FINDINGS)\n\n"
           "Consolidates each hypothesis to an explicit verdict using the §8 rules "
           "(noise floor first; per-cell, no sign-flip averaging). Prints text ready to "
           "paste into the paper's Discussion."),
    SETUP,
    ("code", "df = pd.read_csv('results.csv')\n"
             "lines = []\n"
             "# H0\n"
             "g1 = df[(df.phase=='phase1')&(df.student=='graph-rkd')]\n"
             "floor = df[(df.phase=='phase1')&(df.student=='triplet_only')]['val_mAP@R'].mean()\n"
             "if len(g1):\n"
             "    band = au.agg(g1,['lambda_g'],'val_mAP@R'); viable = band[band['mean']>=floor]\n"
             "    lines.append(f\"H0: {'ACCEPT' if len(viable) else 'REJECT'} — viable λg: {list(viable.lambda_g)} (floor={floor:.4f})\")\n"
             "else: lines.append('H0: sem dados (phase1)')\n"
             "# H1 por célula\n"
             "for d,t in df[df.phase=='phase5'][['dataset','teacher']].dropna().drop_duplicates().itertuples(index=False):\n"
             "    lines.append(f'H1 [{d}·{t}]: '+au.h1_verdict(au.headline_table(df,d,t)))\n"
             "print('\\n'.join(lines) if lines else 'sem runs ainda')"),
    ("md", "### H3 mechanism (available now from the probe)"),
    ("code", "p = au.load_probe()\n"
             "print('MDS near-degenerate rate by N:')\n"
             "print(p[p.method=='mds'].groupby('N')['mds_degenerate_rate'].max())\n"
             "print('\\nprofile tie rate by N:')\n"
             "print(p[p.method=='profile'].groupby('N')['profile_tie_rate'].max())"),
))

print("done")
