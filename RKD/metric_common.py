"""Helpers compartilhados para os experimentos de METRIC LEARNING (retrieval).

Datasets usam o split DISJUNTO (*Metric): classes de treino e teste não se
sobrepõem; avaliação por recall@K. Política comum (espelha a de classificação,
mas para retrieval):

* Mesma métrica (recall@K) em train/val/test.
* Validação SEMPRE existe: como o split disjunto não traz val, separamos um
  subconjunto de CLASSES de treino como val (disjunto de treino e teste).
* Seleção do modelo final pelo melhor recall@1 de VALIDAÇÃO (nunca teste).

Reutiliza NPairs (amostragem por classe p/ triplet), recall() e os datasets
*Metric do repo.
"""

import copy
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from metric.batchsampler import NPairs
from metric.utils import pdist

__all__ = ["DATASETS", "build_metric_loaders", "embed", "evaluate_recall_splits",
           "recall_log_dict", "retrieval_metrics", "score_of", "SELECT_METRICS",
           "RECALL_K"]

# --select_metric -> métrica de validação p/ seleção (primária = mAP@R, Musgrave
# et al. 2020). Chaves "fixas" mapeiam direto; ndcg/prec/f1 resolvem para o
# maior K disponível (ver score_of).
SELECT_METRICS = {"mapr": "mAP@R", "rprec": "R_precision", "mrr": "MRR",
                  "recall1": "recall@1", "ndcg": "nDCG@", "prec": "precision@",
                  "f1": "F1@"}

# chave -> (classe *Metric, marcador de download sob --data)
DATASETS = {
    "cars196": (dataset.Cars196Metric, os.path.join("Cars196", "cars_annos.mat")),
    "cub200": (dataset.CUB2011Metric, os.path.join("CUB_200_2011", "images.txt")),
}
RECALL_K = [1, 2, 4, 8]   # K padrão p/ CUB/Cars


def _restrict(ds, allowed_classes):
    """Cópia rasa do dataset restrita a um conjunto de classes (mantém transform)."""
    allowed = set(allowed_classes)
    d = copy.copy(ds)
    d.samples = [(p, c) for (p, c) in ds.samples if c in allowed]
    d.imgs = d.samples
    d.class_to_idx = {k: v for k, v in ds.class_to_idx.items() if v in allowed}
    if hasattr(ds, "classes"):
        d.classes = [k for k in ds.classes if ds.class_to_idx.get(k) in allowed]
    return d


def build_metric_loaders(dataset_cls, data_root, train_tf, test_tf, batch,
                         num_image_per_class, iter_per_epoch, workers,
                         val_class_frac, seed, download):
    """Loaders de metric learning. Retorna (loaders, info).

    loaders: 'train' (NPairs, classes de treino menos val), 'train_eval', 'val'
    (classes de treino separadas), 'test' (classes de teste oficiais).
    A divisão treino/val é por CLASSE (disjunta), com semente fixa.
    """
    train_aug = dataset_cls(data_root, train=True, transform=train_tf, download=download)
    train_ev = dataset_cls(data_root, train=True, transform=test_tf, download=False)
    test_set = dataset_cls(data_root, train=False, transform=test_tf, download=False)

    classes = sorted(set(train_aug.class_to_idx.values()))
    g = torch.Generator().manual_seed(int(seed))
    perm = [classes[i] for i in torch.randperm(len(classes), generator=g).tolist()]
    n_val = max(1, int(round(val_class_frac * len(classes))))
    val_classes, train_classes = set(perm[:n_val]), set(perm[n_val:])

    train_tr = _restrict(train_aug, train_classes)
    train_ev_tr = _restrict(train_ev, train_classes)
    val_ev = _restrict(train_ev, val_classes)

    def plain(ds):
        return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                          pin_memory=True,
                          persistent_workers=workers > 0)

    loaders = {
        "train": DataLoader(
            train_tr, num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0,
            batch_sampler=NPairs(train_tr, batch, m=num_image_per_class,
                                 iter_per_epoch=iter_per_epoch)),
        "train_eval": plain(train_ev_tr),
        "val": plain(val_ev),
        "test": plain(test_set),
    }
    info = {"train_classes": len(train_classes), "val_classes": len(val_classes),
            "test_images": len(test_set), "train_images": len(train_tr.samples)}
    return loaders, info


def embed(net, images, l2=True):
    """forward_features -> (embedding [L2-norm opcional], stage2)."""
    feats = net.forward_features(images)
    emb = feats["embedding"]
    if l2:
        emb = F.normalize(emb, dim=1)
    return emb, feats["stage2"]


@torch.no_grad()
def retrieval_metrics(embeddings, labels, K):
    """Métricas de retrieval/reranking a partir dos embeddings + rótulos.

    Relevância = mesma classe (binária). Retorna dict com:
      * recall@k (para cada k em K) — há vizinho relevante no top-k;
      * R_precision e mAP@R — sensíveis à ORDENAÇÃO de todos os relevantes
        (R_i = nº de relevantes da query); recomendados por Musgrave et al. (2020);
      * nDCG (no maior K) — ganho descontado por posição;
      * MRR — recíproco da posição do 1º relevante.
    """
    labels = labels.view(-1)
    N = embeddings.size(0)
    D = pdist(embeddings, squared=True)
    D.fill_diagonal_(float("inf"))                      # exclui a própria query

    _, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    R = (counts[inv] - 1).clamp(min=0)                  # relevantes por query
    kmax = min(N - 1, max(int(R.max().item()), max(K)))
    knn = D.topk(kmax, dim=1, largest=False, sorted=True).indices
    correct = (labels[knn] == labels.unsqueeze(1)).float()      # (N, kmax)
    pos = torch.arange(1, kmax + 1, dtype=torch.float)
    valid = R > 0
    Rf = R.float().clamp(min=1)
    rank_mask = (torch.arange(kmax).unsqueeze(0) < R.unsqueeze(1)).float()

    disc = 1.0 / torch.log2(pos + 1.0)              # desconto por posição (nDCG)
    cdisc = torch.cumsum(disc, dim=0)
    out = {}
    for k in K:
        kc = min(k, kmax)
        hits_k = correct[:, :kc].sum(1)
        # recall@k estilo DML = taxa de acerto (>=1 relevante no top-k)
        out[f"recall@{k}"] = (hits_k > 0).float().mean().item()
        # precision@k (IR) = relevantes no top-k / k
        out[f"precision@{k}"] = (hits_k / kc)[valid].mean().item()
        # F1@k = média harmônica de precision@k e IR-recall@k (= rel no top-k / R)
        p = hits_k / kc
        r = (hits_k / Rf).clamp(max=1.0)
        f1 = torch.where((p + r) > 0, 2 * p * r / (p + r), torch.zeros(N))
        out[f"F1@{k}"] = f1[valid].mean().item()
        # nDCG@k (relevância binária)
        dcg = (correct[:, :kc] * disc[:kc].unsqueeze(0)).sum(1)
        ideal_n = R.clamp(max=kc)
        idcg = torch.where(ideal_n > 0, cdisc[(ideal_n - 1).clamp(min=0)], torch.ones(N))
        out[f"nDCG@{k}"] = torch.where(idcg > 0, dcg / idcg,
                                       torch.zeros(N))[valid].mean().item()

    # métricas sensíveis à ordenação de TODOS os relevantes
    out["R_precision"] = ((correct * rank_mask).sum(1) / Rf)[valid].mean().item()
    prec_at = torch.cumsum(correct, dim=1) / pos.unsqueeze(0)
    out["mAP@R"] = ((prec_at * correct * rank_mask).sum(1) / Rf)[valid].mean().item()

    has = correct.bool().any(1)
    first = correct.float().argmax(1) + 1
    mrr = torch.where(has, 1.0 / first.float(), torch.zeros(N))
    sel = valid & has
    out["MRR"] = mrr[sel].mean().item() if sel.any() else 0.0
    return out


def score_of(split_metrics, select_metric):
    """Escalar de seleção a partir do dict de um split.

    'mapr'->mAP@R, 'rprec'->R_precision, 'mrr'->MRR, 'recall1'->recall@1;
    'ndcg'/'prec'/'f1' resolvem para o MAIOR K disponível (ex.: nDCG@8).
    """
    target = SELECT_METRICS[select_metric]
    if target in split_metrics:
        return split_metrics[target]
    # prefixo "<nome>@" -> escolhe o maior K presente
    ks = [int(key.split("@")[1]) for key in split_metrics if key.startswith(target)]
    if not ks:
        raise KeyError(f"métrica '{select_metric}' ({target}) ausente")
    return split_metrics[f"{target}{max(ks)}"]


@torch.no_grad()
def _retrieval_on(net, loader, device, l2, K, desc):
    net.eval()
    embs, labels = [], []
    for images, target in tqdm(loader, ncols=80, desc=desc):
        e, _ = embed(net, images.to(device), l2)
        embs.append(e.cpu())
        labels.append(target)
    return retrieval_metrics(torch.cat(embs), torch.cat(labels), K)


def evaluate_recall_splits(net, loaders, device, l2, K, tag="",
                           splits=("train", "val", "test")):
    """Mesmas métricas de retrieval em cada split.
    Retorna {split: {recall@k, R_precision, mAP@R, nDCG, MRR}}."""
    key = {"train": "train_eval", "val": "val", "test": "test"}
    return {s: _retrieval_on(net, loaders[key.get(s, s)], device, l2, K, f"[{tag}{s}]")
            for s in splits}


def recall_log_dict(metrics, prefix=""):
    out = {}
    for split, m in metrics.items():
        for name, value in m.items():
            out[f"{prefix}{split}/{name}"] = value
    return out
