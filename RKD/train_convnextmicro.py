"""Baseline: treina a ConvNextMicro do zero (só cross-entropy) em Cars/CUB.

Serve de **comparação** para a destilação (distill_to_convnextmicro.py): mesma
arquitetura de aluno, mesmos datasets/split oficial, mesma política de métricas
(val estratificado, top-1/top-5 em train/val/test, seleção pelo melhor val) e os
mesmos hiperparâmetros de otimização do aluno destilado -- a ÚNICA diferença é a
ausência do sinal de destilação (KD/RKD/atenção). A diferença de top-1 entre
este baseline e o aluno destilado isola o ganho da destilação.

Usa --dataset {cars196,cub200}; registra o modelo no W&B como
``convnextmicro-baseline-<dataset>`` (mesmo projeto dos alunos destilados, p/
comparar lado a lado).
"""

import argparse
import math
import os

import dataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from model import ConvNextMicro
from tqdm import tqdm
from classification_split import (
    build_classification_loaders,
    evaluate_splits,
    log_dict,
)
from wandb_artifacts import log_model_artifact

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASETS = {
    "cars196": (dataset.Cars196Classification, 196,
                os.path.join("Cars196", "cars_annos.mat")),
    "cub200": (dataset.CUB2011Classification, 200,
               os.path.join("CUB_200_2011", "images.txt")),
}


def build_parser():
    p = argparse.ArgumentParser(
        description="Train ConvNextMicro from scratch (baseline) on Cars/CUB")

    p.add_argument("--dataset", choices=sorted(DATASETS), default="cars196")
    p.add_argument("--data", default="data")
    p.add_argument("--num_classes", type=int, default=None,
                   help="inferred from --dataset when omitted")
    p.add_argument("--workers", type=int, default=8)

    # student
    p.add_argument("--dims", type=int, nargs=4, default=[24, 48, 96, 192])
    p.add_argument("--depths", type=int, nargs=4, default=[1, 1, 3, 1])
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # optimization (iguais ao aluno destilado, p/ comparação justa)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--clip_grad", type=float, default=0.0)
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="fração do treino reservada como validação (estratificada)")
    p.add_argument("--eval_every", type=int, default=5,
                   help="periodicidade (em épocas) p/ avaliar train/val/test")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)

    # wandb (mesmo projeto dos alunos destilados, p/ comparar lado a lado)
    p.add_argument("--wandb_project", default="convnextmicro-distill")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_id", default=None,
                   help="id estável da run W&B (resume='allow' p/ retomar)")
    p.add_argument("--wandb_group", default=None,
                   help="defaults to baseline-<dataset>")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb_tags", nargs="*", default=None)
    p.add_argument("--resume", default=None,
                   help="caminho do baseline_last.pth p/ retomar (tolerante a ausência)")
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


def build_scheduler(optimizer, opts, iters_per_epoch):
    warmup = opts.warmup_epochs * iters_per_epoch
    total = opts.epochs * iters_per_epoch
    base_lr = optimizer.param_groups[0]["lr"]
    floor = opts.min_lr / base_lr if base_lr > 0 else 0.0

    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def run_experiment(opts):
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_cls, default_classes, marker = DATASETS[opts.dataset]
    if opts.num_classes is None:
        opts.num_classes = default_classes
    if opts.wandb_group is None:
        opts.wandb_group = "baseline-%s" % opts.dataset

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize])

    download = not os.path.exists(os.path.join(os.path.abspath(opts.data), marker))
    loaders, sizes = build_classification_loaders(
        dataset_cls, opts.data, train_tf, test_tf, opts.batch, opts.workers,
        opts.val_fraction, opts.seed, download)
    train_loader = loaders["train"]
    print(f"dataset={opts.dataset} train={sizes['train']} val={sizes['val']} "
          f"test={sizes['test']} classes={opts.num_classes}")

    tags = ["baseline", "convnextmicro", opts.dataset]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags, id=opts.wandb_id,
                     resume=("allow" if opts.wandb_id else None), config=vars(opts))

    model = ConvNextMicro(num_classes=opts.num_classes, drop_path=opts.drop_path,
                          dims=tuple(opts.dims), depths=tuple(opts.depths),
                          apply_softmax=False).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr,
                                  weight_decay=opts.weight_decay, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, opts, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"convnextmicro={n_params/1e6:.2f}M device={device}")

    def logits_fn(images):
        return model(images)  # apply_softmax=False -> logits

    art_name = "convnextmicro-baseline-%s" % opts.dataset
    start_epoch, best_val_top1, best_state = 0, 0.0, None
    if opts.resume and os.path.exists(opts.resume):
        ckpt = torch.load(opts.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val_top1 = ckpt.get("best_val_top1", 0.0)
        best_state = ckpt.get("best_state", None)
        print(f"resumed from {opts.resume} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch + 1, opts.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, ncols=90, desc=f"[Baseline {epoch}]")
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
        is_eval_epoch = (epoch % opts.eval_every == 0) or (epoch == opts.epochs)
        if not is_eval_epoch:
            run.log({"epoch": epoch, "train/loss": train_loss,
                     "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
            continue

        # Mesmas métricas (top-1/top-5) em train, val e test.
        model.eval()
        metrics = evaluate_splits(logits_fn, loaders, device, tag=f"E{epoch} ")
        val_top1 = metrics["val"]["top1"]
        improved = val_top1 > best_val_top1   # seleção pela melhor VALIDAÇÃO
        if improved:
            best_val_top1 = val_top1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[Epoch {epoch}] loss={train_loss:.4f} "
              f"train@1={metrics['train']['top1']*100:.2f} "
              f"val@1={val_top1*100:.2f} test@1={metrics['test']['top1']*100:.2f} "
              f"best_val@1={best_val_top1*100:.2f}")
        run.log({"epoch": epoch, "train/loss": train_loss,
                 "lr": optimizer.param_groups[0]["lr"],
                 "val/best_top1": best_val_top1, **log_dict(metrics)}, step=epoch)

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            last_path = os.path.join(opts.save_dir, "baseline_last.pth")
            # estado completo toda época (resume); ao W&B só best/last com TTL.
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "best_val_top1": best_val_top1, "best_state": best_state},
                       last_path)
            is_final = epoch == opts.epochs
            if improved or is_final:
                aliases = (["best"] if improved else []) + (["last"] if is_final else [])
                log_model_artifact(run, last_path, art_name, aliases=aliases,
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
    print("Done. baseline (melhor validação): " + " | ".join(
        f"{s} top1={m['top1']*100:.2f}" for s, m in final.items()))
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
