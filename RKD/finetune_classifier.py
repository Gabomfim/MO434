"""Fine-tune an ImageNet-1k pre-trained classifier on Cars-196 or CUB-200.

Standard classification fine-tuning (NOT the metric-learning split): uses the
*Classification* dataset variants, which keep all classes and the dataset's
official train/test split, and is evaluated with top-1 / top-5 accuracy.

Backbone is chosen with --arch {resnet18, convnext_tiny} (ImageNet-1k weights);
the head is replaced by a fresh N-way linear layer (N inferred from --dataset).
By default the backbone and the new head use different learning rates
(--head_lr_mult), since the head trains from scratch while the backbone adapts.
The per-arch default optimizer/lr (SGD for resnet18, AdamW for convnext_tiny)
come from teacher_models.ARCHS and can be overridden with --opt/--lr.

Both the metrics policy (stratified val split, same top-1/top-5 on
train/val/test, selection by best val) and the saved model are shared with the
distillation script via classification_split / teacher_models.
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
from classification_split import (
    build_classification_loaders,
    evaluate_splits,
    log_dict,
)
from teacher_models import ARCHS, build_classifier
from wandb_artifacts import log_model_artifact

import dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# dataset key -> (Classification dataset class, num_classes, marker file under --data
# that signals the dataset is already downloaded)
DATASETS = {
    "cars196": (dataset.Cars196Classification, 196,
                os.path.join("Cars196", "cars_annos.mat")),
    "cub200": (dataset.CUB2011Classification, 200,
               os.path.join("CUB_200_2011", "images.txt")),
}


def build_parser():
    p = argparse.ArgumentParser(
        description="Fine-tune a classifier (resnet18/convnext_tiny) on Cars/CUB")

    p.add_argument("--arch", choices=sorted(ARCHS), default="resnet18")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="cars196")
    p.add_argument("--data", default="data")
    p.add_argument("--num_classes", type=int, default=None,
                   help="override; inferred from --dataset when omitted")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="fração do treino reservada como validação (estratificada)")
    p.add_argument("--eval_every", type=int, default=1,
                   help="periodicidade (em épocas) p/ avaliar train/val/test")

    # optimization (opt/lr default per --arch via teacher_models.ARCHS)
    p.add_argument("--opt", choices=["sgd", "adamw"], default=None)
    p.add_argument("--lr", type=float, default=None, help="backbone base lr")
    p.add_argument("--head_lr_mult", type=float, default=10.0,
                   help="head lr = lr * this (head trains from scratch)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=0.0)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="train only the new head (linear probe)")

    # runtime
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--resume", default=None)

    # wandb
    p.add_argument("--wandb_project", default="classifier-finetune")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_id", default=None,
                   help="id estável da run W&B (resume='allow' p/ retomar)")
    p.add_argument("--wandb_group", default=None,
                   help="defaults to '<arch>-<dataset>'")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb_tags", nargs="*", default=None)
    return p


def _params_to_cli_args(params):
    out = []
    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                out.append(flag)
        elif isinstance(value, (list, tuple)):
            out.append(flag)
            out.extend(str(v) for v in value)
        else:
            out.extend([flag, str(value)])
    return out


def build_optimizer(model, opts):
    """Separate param groups: backbone at lr, fresh head at lr * head_lr_mult."""
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]

    groups = [{"params": head_params, "lr": opts.lr * opts.head_lr_mult}]
    if backbone_params:  # empty when --freeze_backbone
        groups.append({"params": backbone_params, "lr": opts.lr})

    if opts.opt == "sgd":
        return torch.optim.SGD(groups, lr=opts.lr, momentum=opts.momentum,
                               weight_decay=opts.weight_decay, nesterov=True)
    return torch.optim.AdamW(groups, lr=opts.lr, weight_decay=opts.weight_decay)


def build_scheduler(optimizer, opts, iters_per_epoch):
    """Linear warmup then cosine decay to min_lr, per-iteration, per-group."""
    warmup = opts.warmup_epochs * iters_per_epoch
    total = opts.epochs * iters_per_epoch
    floors = [opts.min_lr / g["lr"] if g["lr"] > 0 else 0.0
              for g in optimizer.param_groups]

    def make(floor):
        def fn(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            prog = (step - warmup) / max(1, total - warmup)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return fn

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [make(f) for f in floors])


def run_experiment(opts):
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_cls, default_classes, marker = DATASETS[opts.dataset]
    if opts.num_classes is None:
        opts.num_classes = default_classes
    # opt/lr default per arch (SGD/0.01 resnet18, AdamW/1e-4 convnext_tiny)
    if opts.opt is None:
        opts.opt = ARCHS[opts.arch]["opt"]
    if opts.lr is None:
        opts.lr = ARCHS[opts.arch]["lr"]
    if opts.wandb_group is None:
        opts.wandb_group = "%s-%s" % (opts.arch, opts.dataset)

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    download = not os.path.exists(os.path.join(os.path.abspath(opts.data), marker))
    loaders, sizes = build_classification_loaders(
        dataset_cls, opts.data, train_tf, test_tf, opts.batch, opts.workers,
        opts.val_fraction, opts.seed, download)
    train_loader = loaders["train"]
    print(f"dataset={opts.dataset} train={sizes['train']} val={sizes['val']} "
          f"test={sizes['test']} classes={opts.num_classes}")

    model = build_classifier(opts.arch, opts.num_classes, pretrained=True,
                             freeze_backbone=opts.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    optimizer = build_optimizer(model, opts)
    scheduler = build_scheduler(optimizer, opts, len(train_loader))

    # logits do classificador (mesma assinatura usada por evaluate_splits)
    def logits_fn(images):
        return model(images)
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")

    tags = ["finetune", opts.arch, opts.dataset]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags, id=opts.wandb_id,
                     resume=("allow" if opts.wandb_id else None),
                     config={**vars(opts), "device": device})

    start_epoch, best_val_top1, best_state = 0, 0.0, None
    art_name = "%s-%s" % (opts.arch, opts.dataset)
    if opts.resume and os.path.exists(opts.resume):
        ckpt = torch.load(opts.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_top1 = ckpt.get("best_val_top1", 0.0)
        best_state = ckpt.get("best_state", None)
        print(f"resumed from {opts.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, opts.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, ncols=90, desc=f"[Train {epoch}]")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=opts.amp and device == "cuda"):
                loss = criterion(model(images), targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if opts.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n+1):.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        train_loss = running / max(1, len(train_loader))
        is_eval_epoch = (epoch % opts.eval_every == 0) or (epoch == opts.epochs - 1)
        if not is_eval_epoch:
            run.log({"epoch": epoch, "train/loss": train_loss,
                     "lr/head": optimizer.param_groups[0]["lr"]}, step=epoch)
            continue

        # Mesmas métricas (top-1/top-5) em train, val e test.
        model.eval()
        metrics = evaluate_splits(logits_fn, loaders, device, tag=f"E{epoch} ")
        val_top1 = metrics["val"]["top1"]
        # Seleção pela melhor generalização: melhor top-1 de VALIDAÇÃO.
        improved = val_top1 > best_val_top1
        if improved:
            best_val_top1 = val_top1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[Epoch {epoch}] loss={train_loss:.4f} "
              f"train@1={metrics['train']['top1']*100:.2f} "
              f"val@1={val_top1*100:.2f} test@1={metrics['test']['top1']*100:.2f} "
              f"best_val@1={best_val_top1*100:.2f}")

        run.log({"epoch": epoch, "train/loss": train_loss,
                 "lr/head": optimizer.param_groups[0]["lr"],
                 "val/best_top1": best_val_top1,
                 **log_dict(metrics)}, step=epoch)

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch,
                     "best_val_top1": best_val_top1, "best_state": best_state}
            last_path = os.path.join(opts.save_dir, "last.pth")
            torch.save(state, last_path)
            # Local toda época (p/ --resume), mas só envia ao W&B quando melhora
            # (best) ou na última época (last) -- sem alias epoch-N e com TTL --
            # para não estourar o storage do W&B.
            if improved:
                best_path = os.path.join(opts.save_dir, "best.pth")
                torch.save(state, best_path)
                log_model_artifact(run, best_path, art_name, aliases=["best"],
                                   ttl_days=30,
                                   metadata={"epoch": epoch, "val_top1": best_val_top1})
            if epoch == opts.epochs - 1:
                log_model_artifact(run, last_path, art_name, aliases=["last"],
                                   ttl_days=30,
                                   metadata={"epoch": epoch, **log_dict(metrics)})

    # Métricas finais com o modelo que MAXIMIZA A GENERALIZAÇÃO (melhor val).
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    final = evaluate_splits(logits_fn, loaders, device, tag="final ")
    run.log(log_dict(final, prefix="final/"))
    for split, m in final.items():
        run.summary[f"final_{split}_top1"] = m["top1"]
        run.summary[f"final_{split}_top5"] = m["top5"]
    run.summary["best_val_top1"] = best_val_top1
    print("Done. métricas finais (modelo de melhor validação): "
          + " | ".join(f"{s} top1={m['top1']*100:.2f}" for s, m in final.items()))
    run.finish()
    return final["test"]["top1"]


def run_with_cli_args(cli_args):
    return run_experiment(build_parser().parse_args([str(a) for a in cli_args]))


def run_with_params(params):
    return run_with_cli_args(_params_to_cli_args(params))


def main(argv=None):
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
