"""Logic for the qualitative notebook (§7.5) — top-k retrieval panels.

Kept separate from `analysis_utils` on purpose: it imports torch/torchvision (heavy), so
it is only loaded by `05`. Keeps `analysis_utils` torch-free (so notebooks
00–04/06 run on a machine without torchvision).

Flow: picks the run (from results.csv) → resolves the checkpoint (local
`experiments_local/` or W&B artifact) → embeds the test split → top-k retrieval →
figure (query + neighbors, green border=same class, red=different class).
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from tqdm.auto import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_RKD = os.path.dirname(_HERE)
for _p in (_RKD, os.path.join(_RKD, "sm"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis_utils as au          # noqa: E402  (loads the .env -> WANDB_API_KEY)
import data_prep                     # noqa: E402
from metric_common import DATASETS, build_metric_loaders, embed  # noqa: E402
from model import ConvNextMicro      # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(_RKD, "data")


# --------------------------------------------------------------------------- #
# run selection + checkpoint resolution                                        #
# --------------------------------------------------------------------------- #
def pick_run(df, dataset, teacher, student, method=None, N=None, seed=0, phase="phase5"):
    """Picks ONE run (row of results.csv) for (dataset, teacher, student).
    triplet-only has no teacher; graph-rkd filters by method/N. Prefers `seed`,
    otherwise the first available. Returns the row (Series) or None."""
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
    """Local path to the run's checkpoint: looks in experiments_local/<name>/,
    otherwise downloads the model artifact logged by that run on W&B."""
    name = run_row["run"]
    for base in (os.path.join(_RKD, "experiments_local"), "experiments_local",
                 os.path.join(_RKD, "sm", "experiments_local")):
        for fn in ("student_last.pth", "best.pth"):
            p = os.path.join(base, name, fn)
            if os.path.exists(p):
                return p
    # W&B: model artifact logged by this run (the student; the teacher is
    # use_artifact, not logged -> does not appear here). The run logs ONE version per
    # epoch (epochs 5,10,...); the "best"/"last" aliases are not reliable because
    # the artifact NAME is shared across runs (the alias migrates to the last run
    # that logged). We take the version with the HIGHEST epoch: its student_last.pth loads the
    # GLOBAL best_state of the run (accumulated over training).
    import wandb
    run = wandb.Api().run(f"{au.ENTITY}/{au.PROJECT}/{run_row['id']}")
    arts = [a for a in run.logged_artifacts()
            if a.type == "model" and "distill" in a.name]
    if not arts:
        arts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not arts:
        raise FileNotFoundError(f"no model artifact for run '{name}'")
    art = max(arts, key=lambda a: a.metadata.get("epoch", -1))
    d = art.download()
    for fn in ("student_last.pth", "best.pth"):
        if os.path.exists(os.path.join(d, fn)):
            return os.path.join(d, fn)
    pths = sorted(glob.glob(os.path.join(d, "*.pth")))
    if pths:
        return pths[0]
    raise FileNotFoundError(f"no .pth in artifact {art.name} (run '{name}')")


def load_student(ckpt):
    m = ConvNextMicro(num_classes=2, dims=(24, 48, 96, 192), depths=(1, 1, 3, 1),
                      apply_softmax=False).to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE)
    m.load_state_dict(sd.get("best_state") or sd["model"])
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# data + embeddings + figure                                                   #
# --------------------------------------------------------------------------- #
def test_loader(dataset, workers=0):
    """Test split of the dataset (pulls from S3 and caches if needed).

    workers=0 by default: we only do embedding (inference), so DataLoader
    workers do not help and they avoid the "bus error / insufficient shared memory (shm)"
    common on WSL/Docker, where /dev/shm is small. Increase it if your machine has
    enough shm and you want parallel I/O."""
    data_prep.ensure(DATA_DIR, [dataset], progress=True)
    cls, _ = DATASETS[dataset]
    tf = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    loaders, info = build_metric_loaders(cls, DATA_DIR, tf, tf, 128, 5, 100,
                                         workers, 0.2, 0, False)
    print(f"[{dataset}] test: {info}")
    return loaders["test"]


def embed_test(model, loader, desc="embedding"):
    """Embeddings + labels of the split. Does NOT keep the images (that blew up RAM
    and crashed WSL: ~8k imgs x 3x224x224 float32 ~ 5 GB per model). The few
    panel images are re-read by dataset index in run_cell."""
    E, Y = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, unit="batch", leave=False):
            e, _ = embed(model, x.to(DEVICE), True)
            E.append(e.cpu()); Y.append(y)
    return torch.cat(E), torch.cat(Y)


def _unnorm(t):
    m = torch.tensor(IMAGENET_STD)[:, None, None]
    b = torch.tensor(IMAGENET_MEAN)[:, None, None]
    return (t * m + b).clamp(0, 1).permute(1, 2, 0).numpy()


def _topk(E, k, chunk=2048):
    """Top-k neighbors (excludes the query itself) by BLOCKS of rows, so as to never
    allocate the entire N x N similarity matrix (avoids RAM spikes)."""
    N = E.size(0)
    out = torch.empty(N, k, dtype=torch.long)
    for s in range(0, N, chunk):
        block = E[s:s + chunk] @ E.T                 # (<=chunk, N)
        rows = torch.arange(block.size(0))
        block[rows, s + rows] = -1.0                 # remove self-match
        out[s:s + chunk] = block.topk(k, dim=1).indices
    return out


def compare_panels(tg, tb, Y, imgs, queries, title, path, topk=5,
                   labels=("Graph-RKD", "baseline")):
    """Figure comparing two models on the SAME queries: each row = one query;
    columns = [query | top-k of model A | top-k of model B]. Green border = same
    class as the query; red = different class. `tg`/`tb` are top-k indices already
    computed; `imgs` is a dict {index -> tensor} with only the images used."""
    ncol = 1 + 2 * topk
    fig, ax = plt.subplots(len(queries), ncol, figsize=(1.5 * ncol, 1.7 * len(queries)),
                           squeeze=False)
    for r, q in enumerate(queries):
        a0 = ax[r][0]; a0.imshow(_unnorm(imgs[q])); a0.axis("off")
        if r == 0:
            a0.set_title("query", fontsize=8)
        for c, (tk, off) in enumerate([(tg, 1), (tb, 1 + topk)]):
            for j in range(topk):
                a = ax[r][off + j]; idx = int(tk[q, j])
                a.imshow(_unnorm(imgs[idx])); a.axis("off")
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
    """Produces the qualitative figure of a cell (dataset, teacher): Graph-RKD
    headline vs the strongest baseline, on the same queries (hits + misses of
    Graph-RKD). Saves to outdir/fig_qual_<dataset>_<teacher>.pdf."""
    gr = pick_run(df, dataset, teacher, "graph-rkd", graph_method, graph_N)
    br = pick_run(df, dataset, teacher, baseline_student)
    if gr is None or br is None:
        print(f"[skip] {dataset}/{teacher}: run missing (graph={gr is not None}, "
              f"baseline={br is not None})")
        return None
    loader = test_loader(dataset)
    ds = loader.dataset                              # test split (order = order of the embeddings)
    Eg, Y = embed_test(load_student(resolve_ckpt(gr)), loader,
                       desc=f"{dataset}/{teacher} embedding graph-rkd")
    Eb, _ = embed_test(load_student(resolve_ckpt(br)), loader,
                       desc=f"{dataset}/{teacher} embedding {baseline_student}")
    tg, tb = _topk(Eg, topk), _topk(Eb, topk)
    top1 = tg[:, 0]
    correct = [i for i in range(len(Y)) if Y[top1[i]] == Y[i]]
    wrong = [i for i in range(len(Y)) if Y[top1[i]] != Y[i]]
    queries = correct[:n_success] + wrong[:n_fail]
    # re-reads only the images that appear in the panel (queries + neighbors shown)
    need = set(queries)
    for q in queries:
        need.update(int(i) for i in tg[q])
        need.update(int(i) for i in tb[q])
    imgs = {i: ds[i][0] for i in need}
    path = os.path.join(outdir, f"fig_qual_{dataset}_{teacher}.pdf")
    compare_panels(tg, tb, Y, imgs, queries, f"{dataset} · {teacher}", path, topk=topk,
                   labels=("Graph-RKD", baseline_student.replace("_", "-")))
    print(f"[ok] {dataset}/{teacher} -> {path} "
          f"(Graph-RKD top-1 hits: {len(correct)}/{len(Y)})")
    return path
