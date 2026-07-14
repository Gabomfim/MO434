"""Shared helpers for the METRIC LEARNING (retrieval) experiments.

Datasets use the DISJOINT split (*Metric): train and test classes do not
overlap; evaluation by recall@K. Common policy (mirrors the classification
one, but for retrieval):

* Same metric (recall@K) on train/val/test.
* Validation ALWAYS exists: since the disjoint split provides no val, we set
  aside a subset of training CLASSES as val (disjoint from train and test).
* Final model selection by the best VALIDATION recall@1 (never test).

Reuses NPairs (per-class sampling for triplet), recall() and the repo's
*Metric datasets.
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

# Metric set chosen for DML retrieval: Recall@K (conventional curve),
# R-Precision and mAP@R (sensitive to the ordering of all relevant items). Primary =
# mAP@R (Musgrave et al. 2020). --select_metric maps to the metric key.
SELECT_METRICS = {"mapr": "mAP@R", "rprec": "R_precision", "recall1": "recall@1"}

# key -> (*Metric class, download marker under --data)
DATASETS = {
    "cars196": (dataset.Cars196Metric, os.path.join("Cars196", "cars_annos.mat")),
    "cub200": (dataset.CUB2011Metric, os.path.join("CUB_200_2011", "images.txt")),
}
RECALL_K = [1, 2, 4, 8]   # default K for CUB/Cars


def _restrict(ds, allowed_classes):
    """Shallow copy of the dataset restricted to a set of classes (keeps transform)."""
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
    """Metric learning loaders. Returns (loaders, info).

    loaders: 'train' (NPairs, training classes minus val), 'train_eval', 'val'
    (held-out training classes), 'test' (official test classes).
    The train/val division is per CLASS (disjoint), with a fixed seed.
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
    """forward_features -> (embedding [optional L2-norm], stage2)."""
    feats = net.forward_features(images)
    emb = feats["embedding"]
    if l2:
        emb = F.normalize(emb, dim=1)
    return emb, feats["stage2"]


@torch.no_grad()
def retrieval_metrics(embeddings, labels, K):
    """Retrieval/reranking metrics from the embeddings + labels.

    Relevance = same class (binary). Set chosen for DML retrieval:
      * recall@k (for each k in K) — conventional curve (there is a relevant
        neighbor in the top-k); comparability with the literature;
      * R_precision and mAP@R — sensitive to the ORDERING of all R relevant items
        of the query; recommended by Musgrave et al. (2020), with mAP@R as primary.
    """
    labels = labels.view(-1)
    N = embeddings.size(0)
    D = pdist(embeddings, squared=True)
    D.fill_diagonal_(float("inf"))                      # excludes the query itself

    _, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    R = (counts[inv] - 1).clamp(min=0)                  # relevant items per query
    kmax = min(N - 1, max(int(R.max().item()), max(K)))
    knn = D.topk(kmax, dim=1, largest=False, sorted=True).indices
    correct = (labels[knn] == labels.unsqueeze(1)).float()      # (N, kmax)
    pos = torch.arange(1, kmax + 1, dtype=torch.float)
    valid = R > 0
    Rf = R.float().clamp(min=1)
    rank_mask = (torch.arange(kmax).unsqueeze(0) < R.unsqueeze(1)).float()

    # conventional recall@k (hit rate: >=1 relevant item in the top-k)
    out = {f"recall@{k}": (correct[:, :min(k, kmax)].sum(1) > 0).float().mean().item()
           for k in K}
    # sensitive to the ordering of ALL R relevant items of the query
    out["R_precision"] = ((correct * rank_mask).sum(1) / Rf)[valid].mean().item()
    prec_at = torch.cumsum(correct, dim=1) / pos.unsqueeze(0)
    out["mAP@R"] = ((prec_at * correct * rank_mask).sum(1) / Rf)[valid].mean().item()
    return out


def score_of(split_metrics, select_metric):
    """Selection scalar: 'mapr'->mAP@R (primary), 'rprec'->R_precision,
    'recall1'->recall@1."""
    return split_metrics[SELECT_METRICS[select_metric]]


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
    """Same retrieval metrics on each split.
    Returns {split: {recall@k, R_precision, mAP@R, nDCG, MRR}}."""
    key = {"train": "train_eval", "val": "val", "test": "test"}
    return {s: _retrieval_on(net, loaders[key.get(s, s)], device, l2, K, f"[{tag}{s}]")
            for s in splits}


def recall_log_dict(metrics, prefix=""):
    out = {}
    for split, m in metrics.items():
        for name, value in m.items():
            out[f"{prefix}{split}/{name}"] = value
    return out
