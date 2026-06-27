"""Fine-tune de um teacher de METRIC LEARNING (embedding) em Cars/CUB.

Treina resnet18/convnext_tiny (ImageNet) como rede de EMBEDDING com triplet loss
no split disjunto (*Metric), avaliando por recall@K. Seleção pelo melhor
recall@1 de VALIDAÇÃO (classes de treino separadas). O checkpoint vira o teacher
das destilações métricas (carregado por teacher_models.load_teacher, strict=False).
"""

import argparse
import math
import os

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from metric.loss import L2Triplet
import metric.pairsampler as pair
from metric_common import (DATASETS, RECALL_K, SELECT_METRICS, build_metric_loaders,
                           embed, evaluate_recall_splits, recall_log_dict, score_of)
from teacher_models import ARCHS, build_classifier
from tqdm import tqdm
from wandb_artifacts import log_model_artifact, stable_run_id

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SAMPLERS = {"random": pair.RandomNegative, "hard": pair.HardNegative,
            "semihard": pair.SemiHardNegative, "distance": pair.DistanceWeighted,
            "all": pair.AllPairs}


def build_parser():
    p = argparse.ArgumentParser(description="Fine-tune metric teacher (Cars/CUB)")
    p.add_argument("--arch", choices=sorted(ARCHS), default="resnet18")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="cars196")
    p.add_argument("--data", default="data")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val_class_frac", type=float, default=0.2,
                   help="fração de CLASSES de treino separadas como validação")

    # triplet / otimização
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument("--triplet_sample", choices=sorted(SAMPLERS), default="distance")
    p.add_argument("--l2normalize", choices=["true", "false"], default="true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num_image_per_class", type=int, default=5)
    p.add_argument("--iter_per_epoch", type=int, default=100)
    p.add_argument("--recall", type=int, nargs="+", default=RECALL_K)
    p.add_argument("--select_metric", choices=sorted(SELECT_METRICS), default="mapr",
                   help="métrica de validação p/ seleção (mapr=mAP@R, recomendado)")
    p.add_argument("--eval_every", type=int, default=5)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--resume", default=None)

    p.add_argument("--wandb_project", default="metric-teacher-finetune")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_id", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb_tags", nargs="*", default=None)
    return p


def _params_to_cli_args(params):
    out = []
    for k, v in params.items():
        if v is None:
            continue
        flag = "--" + k
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif isinstance(v, (list, tuple)):
            out.append(flag); out.extend(str(x) for x in v)
        else:
            out.extend([flag, str(v)])
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
    return optim.lr_scheduler.LambdaLR(optimizer, fn)


def run_experiment(opts):
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    l2 = opts.l2normalize == "true"
    dataset_cls, marker = DATASETS[opts.dataset]
    if opts.wandb_group is None:
        opts.wandb_group = "%s-%s" % (opts.arch, opts.dataset)

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize])

    download = not os.path.exists(os.path.join(os.path.abspath(opts.data), marker))
    loaders, info = build_metric_loaders(
        dataset_cls, opts.data, train_tf, test_tf, opts.batch,
        opts.num_image_per_class, opts.iter_per_epoch, opts.workers,
        opts.val_class_frac, opts.seed, download)
    print(f"dataset={opts.dataset} {info}")

    model = build_classifier(opts.arch, info["train_classes"], pretrained=True).to(device)
    triplet = L2Triplet(sampler=SAMPLERS[opts.triplet_sample](),
                        margin=opts.triplet_margin)
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr,
                            weight_decay=opts.weight_decay)
    scheduler = build_scheduler(optimizer, opts, opts.iter_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")
    art_name = "metric-%s-%s" % (opts.arch, opts.dataset)
    primary = opts.recall[0]

    tags = ["metric-teacher", opts.arch, opts.dataset]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags, id=opts.wandb_id,
                     resume=("allow" if opts.wandb_id else None),
                     config={**vars(opts), "info": info})

    start_epoch, best_val, best_state = 0, 0.0, None
    if opts.resume and os.path.exists(opts.resume):
        ckpt = torch.load(opts.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val", 0.0)
        best_state = ckpt.get("best_state", None)
        print(f"resumed from {opts.resume} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch + 1, opts.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loaders["train"], ncols=90, desc=f"[Teacher {epoch}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=opts.amp and device == "cuda"):
                emb, _ = embed(model, images, l2)
                loss = triplet(emb, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n+1):.3f}")

        if epoch % opts.eval_every and epoch != opts.epochs:
            run.log({"epoch": epoch, "train/triplet_loss": running / max(1, len(loaders['train'])),
                     "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
            continue

        metrics = evaluate_recall_splits(model, loaders, device, l2, opts.recall,
                                         tag=f"E{epoch} ")
        val_score = score_of(metrics["val"], opts.select_metric)
        improved = val_score > best_val
        if improved:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[Epoch {epoch}] val {opts.select_metric}={val_score*100:.2f} "
              f"(test mAP@R={metrics['test']['mAP@R']*100:.2f}, "
              f"r@{primary}={metrics['test'][f'recall@{primary}']*100:.2f}) "
              f"best_val={best_val*100:.2f}")
        run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                 "val/best_score": best_val, **recall_log_dict(metrics)}, step=epoch)

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            last_path = os.path.join(opts.save_dir, "last.pth")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "best_val": best_val, "best_state": best_state}, last_path)
            is_final = epoch == opts.epochs
            if improved or is_final:
                # salva best separado (carregado depois como teacher)
                if improved:
                    torch.save({"model": best_state}, os.path.join(opts.save_dir, "best.pth"))
                aliases = (["best"] if improved else []) + (["last"] if is_final else [])
                log_model_artifact(run, os.path.join(opts.save_dir, "best.pth"),
                                   art_name, aliases=aliases, ttl_days=30,
                                   metadata={"epoch": epoch, **recall_log_dict(metrics)})

    run.summary["best_val_score"] = best_val
    run.summary["select_metric"] = opts.select_metric
    run.finish()
    return best_val


def run_with_cli_args(cli_args):
    return run_experiment(build_parser().parse_args([str(a) for a in cli_args]))


def run_with_params(params):
    return run_with_cli_args(_params_to_cli_args(params))


def main(argv=None):
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
