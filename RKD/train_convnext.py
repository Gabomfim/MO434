"""Train ConvNextMicro following the ConvNeXt paper recipe.

Recipe (ConvNeXt, ImageNet-1K, Tab. of training settings):
    weight init        trunc. normal (0.2)
    optimizer          AdamW
    base lr            4e-3   (for batch size 4096)
    weight decay       0.05
    betas              (0.9, 0.999)
    batch size         4096
    epochs             300
    lr schedule        cosine decay
    warmup             20 epochs, linear
    layer-wise lr decay  none
    RandAugment        (9, 0.5)
    mixup              0.8
    cutmix             1.0
    random erasing     0.25
    label smoothing    0.1
    stochastic depth   0.1
    head init scale    none
    grad clip          none
    EMA                0.9999

Data is read as an ImageFolder with ``train/`` and ``val/`` subdirectories
(pass --data). Use --fake for a quick synthetic smoke test.

Notes / unavoidable deviations from the paper, documented inline:
  * Paper *table* says trunc-normal std 0.2; the official ConvNeXt *code* uses
    0.02. We follow the value you gave (0.2) via --init_std.
  * RandAugment "(9, 0.5)" is timm's magnitude=9, magnitude_std=0.5.
    torchvision's RandAugment has no magnitude-std, so only magnitude=9 is
    applied (the 0.5 jitter cannot be reproduced with torchvision).
  * batch 4096 / 300 epochs is ImageNet-scale. Use --grad_accum to emulate the
    4096 global batch on smaller GPUs; lr is auto-scaled to the global batch.
"""

import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from model import ConvNextMicro

try:
    import wandb
except ImportError:  # wandb is optional for a bare run
    wandb = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_parser():
    p = argparse.ArgumentParser(description="Train ConvNextMicro (ConvNeXt recipe)")

    # data
    p.add_argument("--data", default="data/imagenet",
                   help="ImageFolder root with train/ and val/ subdirs")
    p.add_argument("--fake", action="store_true",
                   help="use synthetic FakeData for a smoke test")
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)

    # model
    p.add_argument("--dims", type=int, nargs=4, default=[24, 48, 96, 192])
    p.add_argument("--depths", type=int, nargs=4, default=[1, 1, 3, 1])
    p.add_argument("--init_std", type=float, default=0.2,
                   help="trunc-normal std (paper table: 0.2; official code: 0.02)")

    # optimization (paper recipe)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4096,
                   help="global batch size targeted by the recipe")
    p.add_argument("--device_batch", type=int, default=128,
                   help="per-step batch that fits in memory; grad-accum fills the rest")
    p.add_argument("--lr", type=float, default=4e-3, help="base lr at batch 4096")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--min_lr", type=float, default=1e-6, help="cosine floor")
    p.add_argument("--no_scale_lr", action="store_true",
                   help="do NOT linearly scale lr by global_batch/4096")

    # regularization (paper recipe)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--rand_erase", type=float, default=0.25)
    p.add_argument("--randaug_magnitude", type=int, default=9)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--clip_grad", type=float, default=0.0, help="0 = no clip (recipe)")

    # runtime
    p.add_argument("--amp", action="store_true", help="mixed-precision training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--resume", default=None)

    # wandb
    p.add_argument("--wandb_project", default="convnext-micro")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="disabled")
    return p


def build_transforms(opts):
    """Train: RandomResizedCrop + flip + RandAugment + normalize + RandomErasing.
    Val: Resize(256)/CenterCrop(224) + normalize."""
    train_tf = v2.Compose([
        v2.RandomResizedCrop(opts.input_size, antialias=True),
        v2.RandomHorizontalFlip(),
        # timm "(9, 0.5)" -> magnitude=9; torchvision has no magnitude-std (0.5).
        v2.RandAugment(num_ops=2, magnitude=opts.randaug_magnitude),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.RandomErasing(p=opts.rand_erase),
    ])
    resize = int(opts.input_size * 256 / 224)
    val_tf = v2.Compose([
        v2.Resize(resize, antialias=True),
        v2.CenterCrop(opts.input_size),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_datasets(opts, train_tf, val_tf):
    if opts.fake:
        train_ds = datasets.FakeData(1024, (3, opts.input_size, opts.input_size),
                                     opts.num_classes, transform=train_tf)
        val_ds = datasets.FakeData(256, (3, opts.input_size, opts.input_size),
                                   opts.num_classes, transform=val_tf)
        return train_ds, val_ds
    train_ds = datasets.ImageFolder(os.path.join(opts.data, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(opts.data, "val"), transform=val_tf)
    return train_ds, val_ds


def split_param_groups(model, weight_decay):
    """No weight decay on biases, norms and the layer-scale gamma (ConvNeXt)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".gamma"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, opts, iters_per_epoch):
    """Linear warmup for warmup_epochs, then cosine decay to min_lr. Per-iteration."""
    warmup_iters = opts.warmup_epochs * iters_per_epoch
    total_iters = opts.epochs * iters_per_epoch
    base_lr = optimizer.param_groups[0]["lr"]
    min_ratio = opts.min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup_iters:
            return (step + 1) / max(1, warmup_iters)
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_ratio + (1.0 - min_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, desc):
    model.eval()
    correct = total = 0
    for images, targets in tqdm(loader, ncols=80, desc=desc):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.numel()
    return correct / max(1, total)


def run(opts):
    torch.manual_seed(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # global batch = device_batch * grad_accum; clamp accum >= 1
    grad_accum = max(1, round(opts.batch / opts.device_batch))
    global_batch = opts.device_batch * grad_accum
    # linear lr scaling rule relative to the recipe's 4096 reference.
    lr = opts.lr if opts.no_scale_lr else opts.lr * global_batch / 4096.0

    train_tf, val_tf = build_transforms(opts)
    train_ds, val_ds = build_datasets(opts, train_tf, val_tf)

    train_loader = DataLoader(train_ds, batch_size=opts.device_batch, shuffle=True,
                              num_workers=opts.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=opts.device_batch, shuffle=False,
                            num_workers=opts.workers, pin_memory=True)

    # mixup / cutmix applied per-batch; produces soft (one-hot mixed) targets.
    mix_transforms = []
    if opts.mixup > 0:
        mix_transforms.append(v2.MixUp(alpha=opts.mixup, num_classes=opts.num_classes))
    if opts.cutmix > 0:
        mix_transforms.append(v2.CutMix(alpha=opts.cutmix, num_classes=opts.num_classes))
    mixer = v2.RandomChoice(mix_transforms) if mix_transforms else None

    model = ConvNextMicro(
        num_classes=opts.num_classes, drop_path=opts.drop_path,
        dims=tuple(opts.dims), depths=tuple(opts.depths),
        apply_softmax=False,            # CrossEntropyLoss needs logits
        init_std=opts.init_std,
    ).to(device)

    # EMA of the weights (decay 0.9999).
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(opts.ema_decay))

    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    optimizer = torch.optim.AdamW(
        split_param_groups(model, opts.weight_decay),
        lr=lr, betas=(opts.beta1, opts.beta2))

    iters_per_epoch = max(1, len(train_loader) // grad_accum)
    scheduler = build_scheduler(optimizer, opts, iters_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} | params={n_params/1e6:.2f}M | global_batch={global_batch} "
          f"(device_batch={opts.device_batch} x grad_accum={grad_accum}) | lr={lr:.2e}")

    start_epoch = 0
    if opts.resume:
        ckpt = torch.load(opts.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {opts.resume} at epoch {start_epoch}")

    use_wandb = wandb is not None and opts.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                   name=opts.wandb_run_name, mode=opts.wandb_mode, config=vars(opts))

    best_acc = 0.0
    for epoch in range(start_epoch, opts.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        pbar = tqdm(train_loader, ncols=90, desc=f"[Train {epoch}]")
        for it, (images, targets) in enumerate(pbar):
            images, targets = images.to(device), targets.to(device)
            if mixer is not None:
                images, targets = mixer(images, targets)

            with torch.autocast(device_type=device.split(":")[0],
                                enabled=opts.amp and device == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets) / grad_accum

            scaler.scale(loss).backward()
            running += loss.item() * grad_accum

            if (it + 1) % grad_accum == 0:
                if opts.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_model.update_parameters(model)
                scheduler.step()
                pbar.set_postfix(loss=f"{running/(it+1):.3f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = running / max(1, len(train_loader))
        acc = evaluate(model, val_loader, device, f"[Val {epoch}]")
        ema_acc = evaluate(ema_model, val_loader, device, f"[EMA {epoch}]")
        best_acc = max(best_acc, acc, ema_acc)
        print(f"[Epoch {epoch}] loss={train_loss:.4f} acc={acc*100:.2f} "
              f"ema_acc={ema_acc*100:.2f} best={best_acc*100:.2f}")

        if use_wandb:
            wandb.log({"epoch": epoch, "train/loss": train_loss,
                       "val/acc": acc, "val/ema_acc": ema_acc, "val/best_acc": best_acc,
                       "lr": optimizer.param_groups[0]["lr"]})

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "best_acc": best_acc},
                       os.path.join(opts.save_dir, "last.pth"))

    print(f"Done. best acc = {best_acc*100:.2f}")
    if use_wandb:
        wandb.finish()
    return best_acc


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
