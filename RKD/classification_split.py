"""Helpers compartilhados para experimentos de classificação.

Política comum a TODOS os experimentos de classificação (fine-tune e
destilação), para que sejam comparáveis:

* **As mesmas métricas** (top-1 e top-5) são calculadas em **todos** os splits
  (train, val e test) -- via `evaluate_splits`.
* **Um split de validação sempre existe**: como Cars-196 e CUB-200 têm apenas
  train/test oficiais, derivamos um `val` **estratificado por classe** a partir
  do treino (`stratified_train_val_split`), com semente fixa.
* **O modelo final é o que maximiza a generalização**: a seleção é feita pelo
  melhor top-1 de **validação** (nunca de teste); as métricas finais são
  computadas recarregando esse checkpoint.
"""

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

SPLITS = ("train", "val", "test")


def stratified_train_val_split(labels, val_fraction, seed):
    """Divide índices em (train, val) mantendo a proporção por classe.

    Garante pelo menos 1 amostra de validação por classe (quando a classe tem
    mais de uma amostra) e nunca esvazia o treino de uma classe.
    """
    g = torch.Generator().manual_seed(int(seed))
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    train_idx, val_idx = [], []
    for y in sorted(by_class):
        idxs = torch.tensor(by_class[y])
        idxs = idxs[torch.randperm(len(idxs), generator=g)]
        if len(idxs) <= 1:
            n_val = 0
        else:
            n_val = min(len(idxs) - 1, max(1, int(round(len(idxs) * val_fraction))))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def build_classification_loaders(dataset_cls, data_root, train_tf, test_tf,
                                 batch, workers, val_fraction, seed, download):
    """Cria os DataLoaders de train/val/test + um loader de train com transform
    de avaliação (para medir métricas de treino sem augmentation).

    Retorna (loaders, sizes), onde loaders tem as chaves:
      'train'      -> treino com augmentation (shuffle, drop_last) p/ otimizar
      'train_eval' -> mesmas amostras de treino, transform de teste (p/ métricas)
      'val'        -> validação estratificada (transform de teste)
      'test'       -> teste oficial (transform de teste)
    """
    train_full = dataset_cls(data_root, train=True, transform=train_tf,
                             download=download)
    train_eval_full = dataset_cls(data_root, train=True, transform=test_tf,
                                  download=False)
    test_set = dataset_cls(data_root, train=False, transform=test_tf, download=False)

    labels = [lbl for _, lbl in train_full.samples]
    train_idx, val_idx = stratified_train_val_split(labels, val_fraction, seed)

    def loader(ds, idx=None, shuffle=False, drop_last=False):
        subset = Subset(ds, idx) if idx is not None else ds
        return DataLoader(subset, batch_size=batch, shuffle=shuffle,
                          num_workers=workers, pin_memory=True, drop_last=drop_last)

    loaders = {
        "train": loader(train_full, train_idx, shuffle=True, drop_last=True),
        "train_eval": loader(train_eval_full, train_idx),
        "val": loader(train_eval_full, val_idx),
        "test": loader(test_set),
    }
    sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_set)}
    return loaders, sizes


@torch.no_grad()
def eval_topk(logits_fn, loader, device, desc=""):
    """top-1/top-5 dado um callable que mapeia imagens (no device) -> logits."""
    top1 = top5 = total = 0
    for images, targets in tqdm(loader, ncols=80, desc=desc):
        logits = logits_fn(images.to(device))
        targets = targets.to(device)
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5 == targets.unsqueeze(1)
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        total += targets.numel()
    return top1 / max(1, total), top5 / max(1, total)


def evaluate_splits(logits_fn, loaders, device, tag="", splits=SPLITS):
    """Calcula a MESMA métrica (top-1/top-5) em cada split.

    Retorna {split: {'top1': x, 'top5': y}}. Use a chave 'train_eval' do dict
    de loaders para o split 'train'.
    """
    key = {"train": "train_eval", "val": "val", "test": "test"}
    out = {}
    for s in splits:
        top1, top5 = eval_topk(logits_fn, loaders[key.get(s, s)], device,
                               f"[{tag}{s}]")
        out[s] = {"top1": top1, "top5": top5}
    return out


def log_dict(metrics, prefix=""):
    """Achata {split: {'top1','top5'}} em {f'{prefix}{split}/top1': ...}."""
    flat = {}
    for split, m in metrics.items():
        for name, value in m.items():
            flat[f"{prefix}{split}/{name}"] = value
    return flat
