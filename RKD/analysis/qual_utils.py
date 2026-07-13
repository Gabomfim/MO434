"""Lógica do notebook qualitativo (§7.5) — painéis de retrieval top-k.

Separado do `analysis_utils` de propósito: importa torch/torchvision (pesado), então
só é carregado pelo `05`. Mantém o `analysis_utils` livre de torch (para os notebooks
00–04/06 rodarem em máquina sem torchvision).

Fluxo: escolhe o run (do results.csv) → resolve o checkpoint (local
`experiments_local/` ou artefato W&B) → embute o test split → top-k retrieval →
figura (query + vizinhos, borda verde=mesma classe, vermelha=classe diferente).
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_RKD = os.path.dirname(_HERE)
for _p in (_RKD, os.path.join(_RKD, "sm"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis_utils as au          # noqa: E402  (carrega o .env -> WANDB_API_KEY)
import data_prep                     # noqa: E402
from metric_common import DATASETS, build_metric_loaders, embed  # noqa: E402
from model import ConvNextMicro      # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(_RKD, "data")


# --------------------------------------------------------------------------- #
# seleção de run + resolução de checkpoint                                     #
# --------------------------------------------------------------------------- #
def pick_run(df, dataset, teacher, student, method=None, N=None, seed=0, phase="phase5"):
    """Escolhe UM run (linha do results.csv) para (dataset, teacher, student).
    triplet-only não tem teacher; graph-rkd filtra por method/N. Prefere `seed`,
    senão o primeiro disponível. Retorna a linha (Series) ou None."""
    q = df[(df.phase == phase) & (df.dataset == dataset) & (df.student == student)]
    if student != "triplet_only":
        q = q[q.teacher == teacher]
    if student == "graph-rkd":
        if method is not None:
            q = q[q.method == method]
        if N is not None:
            q = q[q.N == N]
    if q.empty:
        return None
    pref = q[q.seed == seed]
    return (pref if not pref.empty else q).iloc[0]


def resolve_ckpt(run_row):
    """Caminho local p/ o checkpoint do run: procura em experiments_local/<name>/,
    senão baixa o artefato de modelo logado por aquele run no W&B."""
    name = run_row["run"]
    for base in (os.path.join(_RKD, "experiments_local"), "experiments_local",
                 os.path.join(_RKD, "sm", "experiments_local")):
        for fn in ("student_last.pth", "best.pth"):
            p = os.path.join(base, name, fn)
            if os.path.exists(p):
                return p
    # W&B: artefato de modelo logado por este run (o student; o teacher é
    # use_artifact, não logged -> não aparece aqui). Prefere o nome 'distill'.
    import wandb
    run = wandb.Api().run(f"{au.ENTITY}/{au.PROJECT}/{run_row['id']}")
    arts = [a for a in run.logged_artifacts() if a.type == "model"]
    arts.sort(key=lambda a: 0 if "distill" in a.name else 1)
    for art in arts:
        d = art.download()
        for fn in ("best.pth", "student_last.pth"):
            if os.path.exists(os.path.join(d, fn)):
                return os.path.join(d, fn)
        pths = sorted(glob.glob(os.path.join(d, "*.pth")))
        if pths:
            return pths[0]
    raise FileNotFoundError(f"sem checkpoint p/ run '{name}' (local ou artefato W&B)")


def load_student(ckpt):
    m = ConvNextMicro(num_classes=2, dims=(24, 48, 96, 192), depths=(1, 1, 3, 1),
                      apply_softmax=False).to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE)
    m.load_state_dict(sd.get("best_state") or sd["model"])
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# dados + embeddings + figura                                                  #
# --------------------------------------------------------------------------- #
def test_loader(dataset):
    """Test split do dataset (puxa do S3 e cacheia se preciso)."""
    data_prep.ensure(DATA_DIR, [dataset])
    cls, _ = DATASETS[dataset]
    tf = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    loaders, info = build_metric_loaders(cls, DATA_DIR, tf, tf, 128, 5, 100, 4,
                                         0.2, 0, False)
    print(f"[{dataset}] test: {info}")
    return loaders["test"]


def embed_test(model, loader):
    E, Y, I = [], [], []
    with torch.no_grad():
        for x, y in loader:
            e, _ = embed(model, x.to(DEVICE), True)
            E.append(e.cpu()); Y.append(y); I.append(x)
    return torch.cat(E), torch.cat(Y), torch.cat(I)


def _unnorm(t):
    m = torch.tensor(IMAGENET_STD)[:, None, None]
    b = torch.tensor(IMAGENET_MEAN)[:, None, None]
    return (t * m + b).clamp(0, 1).permute(1, 2, 0).numpy()


def _topk(E, k):
    sim = E @ E.T
    sim.fill_diagonal_(-1.0)
    return sim.topk(k, dim=1).indices


def compare_panels(Eg, Eb, Y, I, queries, title, path, topk=5,
                   labels=("Graph-RKD", "baseline")):
    """Figura comparando dois modelos nas MESMAS queries: cada linha = uma query;
    colunas = [query | top-k do modelo A | top-k do modelo B]. Borda verde = mesma
    classe da query; vermelha = classe diferente."""
    tg, tb = _topk(Eg, topk), _topk(Eb, topk)
    ncol = 1 + 2 * topk
    fig, ax = plt.subplots(len(queries), ncol, figsize=(1.5 * ncol, 1.7 * len(queries)),
                           squeeze=False)
    for r, q in enumerate(queries):
        a0 = ax[r][0]; a0.imshow(_unnorm(I[q])); a0.axis("off")
        a0.set_ylabel(f"q{r}", rotation=0, labelpad=12)
        if r == 0:
            a0.set_title("query", fontsize=8)
        for c, (tk, off) in enumerate([(tg, 1), (tb, 1 + topk)]):
            for j in range(topk):
                a = ax[r][off + j]; idx = int(tk[q, j])
                a.imshow(_unnorm(I[idx])); a.axis("off")
                ok = bool(Y[idx] == Y[q])
                for s in a.spines.values():
                    s.set_visible(True); s.set_color("green" if ok else "red"); s.set_linewidth(3)
                if r == 0 and j == topk // 2:
                    a.set_title(labels[c], fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return fig


def run_cell(df, dataset, teacher, baseline_student, graph_method="mds", graph_N=4,
             topk=5, n_success=3, n_fail=3, outdir="figures"):
    """Produz a figura qualitativa de uma célula (dataset, teacher): Graph-RKD
    headline vs o baseline mais forte, nas mesmas queries (acertos + erros do
    Graph-RKD). Salva em outdir/fig_qual_<dataset>_<teacher>.pdf."""
    gr = pick_run(df, dataset, teacher, "graph-rkd", graph_method, graph_N)
    br = pick_run(df, dataset, teacher, baseline_student)
    if gr is None or br is None:
        print(f"[skip] {dataset}/{teacher}: run ausente (graph={gr is not None}, "
              f"baseline={br is not None})")
        return None
    loader = test_loader(dataset)
    Eg, Y, I = embed_test(load_student(resolve_ckpt(gr)), loader)
    Eb, _, _ = embed_test(load_student(resolve_ckpt(br)), loader)
    tg = _topk(Eg, 1).squeeze(1)
    correct = [i for i in range(len(Y)) if Y[tg[i]] == Y[i]]
    wrong = [i for i in range(len(Y)) if Y[tg[i]] != Y[i]]
    queries = correct[:n_success] + wrong[:n_fail]
    path = os.path.join(outdir, f"fig_qual_{dataset}_{teacher}.pdf")
    compare_panels(Eg, Eb, Y, I, queries, f"{dataset} · {teacher}", path, topk=topk,
                   labels=("Graph-RKD", baseline_student.replace("_", "-")))
    print(f"[ok] {dataset}/{teacher} -> {path} "
          f"(acertos top-1 do Graph-RKD: {len(correct)}/{len(Y)})")
    return path
