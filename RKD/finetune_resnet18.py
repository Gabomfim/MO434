"""Fine-tune an ImageNet-1k pre-trained ResNet-18 on Cars-196 or CUB-200.

Standard classification fine-tuning (NOT the metric-learning split): uses the
*Classification* dataset variants, which keep all classes and the dataset's
official train/test split, and is evaluated with top-1 / top-5 accuracy.

The ImageNet head is replaced by a fresh N-way linear layer (N inferred from
the dataset). By default the backbone and the new head use different learning
rates (--head_lr_mult), since the head trains from scratch while the backbone
only adapts.

Select the dataset with --dataset {cars196,cub200}.
"""

import argparse
import math
import os

import dataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    p = argparse.ArgumentParser(description="Fine-tune ResNet-18 on Cars-196 / CUB-200")

    p.add_argument("--dataset", choices=sorted(DATASETS), default="cars196")
    p.add_argument("--data", default="data")
    p.add_argument("--num_classes", type=int, default=None,
                   help="override; inferred from --dataset when omitted")
    p.add_argument("--workers", type=int, default=8)

    # optimization
    p.add_argument("--opt", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=0.01, help="backbone base lr")
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
    p.add_argument("--wandb_project", default="resnet18-finetune")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_group", default=None,
                   help="defaults to 'resnet18-<dataset>'")
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


def build_model(num_classes, freeze_backbone):
    """ImageNet-pretrained ResNet-18 with a fresh num_classes head."""
    try:  # torchvision >= 0.13 weights API
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    except AttributeError:  # older fallback
        model = torchvision.models.resnet18(pretrained=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer(model, opts):
    """Separate param groups: backbone at lr, fresh head at lr * head_lr_mult."""
    head_params = list(model.fc.parameters())
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


@torch.no_grad()
def evaluate(model, loader, device, desc):
    model.eval()
    top1 = top5 = total = 0
    for images, targets in tqdm(loader, ncols=80, desc=desc):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5 == targets.unsqueeze(1)
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        total += targets.numel()
    return top1 / max(1, total), top5 / max(1, total)


def run_experiment(opts):
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_cls, default_classes, marker = DATASETS[opts.dataset]
    if opts.num_classes is None:
        opts.num_classes = default_classes
    if opts.wandb_group is None:
        opts.wandb_group = "resnet18-%s" % opts.dataset

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
    train_set = dataset_cls(opts.data, train=True, transform=train_tf,
                            download=download)
    test_set = dataset_cls(opts.data, train=False, transform=test_tf, download=False)
    print(f"dataset={opts.dataset} train={len(train_set)} test={len(test_set)} "
          f"classes={opts.num_classes}")

    train_loader = DataLoader(train_set, batch_size=opts.batch, shuffle=True,
                              num_workers=opts.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=opts.batch, shuffle=False,
                             num_workers=opts.workers, pin_memory=True)

    model = build_model(opts.num_classes, opts.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    optimizer = build_optimizer(model, opts)
    scheduler = build_scheduler(optimizer, opts, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")

    tags = ["finetune", "resnet18", opts.dataset]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags,
                     config={**vars(opts), "device": device})

    start_epoch, best_top1 = 0, 0.0
    if opts.resume:
        ckpt = torch.load(opts.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)
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
        top1, top5 = evaluate(model, test_loader, device, f"[Val {epoch}]")
        improved = top1 > best_top1
        best_top1 = max(best_top1, top1)
        print(f"[Epoch {epoch}] loss={train_loss:.4f} top1={top1*100:.2f} "
              f"top5={top5*100:.2f} best_top1={best_top1*100:.2f}")

        run.log({"epoch": epoch, "train/loss": train_loss,
                 "val/top1": top1, "val/top5": top5, "val/best_top1": best_top1,
                 "val/error": 1.0 - top1,
                 "lr/head": optimizer.param_groups[0]["lr"]}, step=epoch)

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch,
                     "best_top1": best_top1}
            torch.save(state, os.path.join(opts.save_dir, "last.pth"))
            if improved:
                torch.save(state, os.path.join(opts.save_dir, "best.pth"))

    run.summary["best_top1"] = best_top1
    print(f"Done. best top1 = {best_top1*100:.2f}")
    run.finish()
    return best_top1


def run_with_cli_args(cli_args):
    return run_experiment(build_parser().parse_args([str(a) for a in cli_args]))


def run_with_params(params):
    return run_with_cli_args(_params_to_cli_args(params))


def main(argv=None):
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
