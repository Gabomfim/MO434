"""Destilação de METRIC LEARNING: teacher (resnet18/convnext_tiny embedding) ->
ConvNextMicro (embedding), no split disjunto, recall@K.

Perda = triplet (tarefa, peso constante) + termo relacional com WARMUP que o
balanceia contra o triplet:

    loss = triplet_ratio*triplet(s, y)
         + rel_scale(epoch) * ( dist_ratio*RKD_dist + angle_ratio*RKD_angle
                                + graph_ratio*GraphLoss )

rel_scale sobe linearmente de 0->1 na fração inicial das épocas (--rel_warmup_frac):
cedo os embeddings do aluno são ruído, então a geometria do teacher entra aos
poucos. Configurações:
  * Graph-RKD (método): graph_ratio>0, dist=angle=0.
  * Baseline RKD-distance isolado: dist_ratio>0, angle=graph=0.
  * Baseline RKD-angle isolado:    angle_ratio>0, dist=graph=0.

Sem Hinton KD (metric learning não tem logits compartilhados). Seleção pelo
melhor recall@1 de VALIDAÇÃO; resumível (W&B + checkpoint).
"""

import argparse
import math
import os

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
import metric.pairsampler as pair
from metric.loss import L2Triplet, RKdAngle, RkdDistance
from metric_common import (DATASETS, RECALL_K, SELECT_METRICS, build_metric_loaders,
                           embed, evaluate_recall_splits, recall_log_dict, score_of)
from model import ConvNextMicro
from teacher_models import ARCHS, load_teacher as _load_teacher_model
from tqdm import tqdm
from wandb_artifacts import (log_model_artifact, resolve_teacher_checkpoint,
                             stable_run_id)
from graph_rkd import GraphContrastiveDistillLoss, GraphRKDLoss

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SAMPLERS = {"random": pair.RandomNegative, "hard": pair.HardNegative,
            "semihard": pair.SemiHardNegative, "distance": pair.DistanceWeighted,
            "all": pair.AllPairs}


def build_parser():
    p = argparse.ArgumentParser(description="Metric distillation -> ConvNextMicro")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="cars196")
    p.add_argument("--data", default="data")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val_class_frac", type=float, default=0.2)

    # student
    p.add_argument("--dims", type=int, nargs=4, default=[24, 48, 96, 192])
    p.add_argument("--depths", type=int, nargs=4, default=[1, 1, 3, 1])
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--l2normalize", choices=["true", "false"], default="true")

    # teacher
    p.add_argument("--teacher_arch", choices=sorted(ARCHS), default="resnet18")
    p.add_argument("--teacher_load", default=None)
    p.add_argument("--teacher_artifact", default=None)

    # task loss
    p.add_argument("--triplet_ratio", type=float, default=1.0)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument("--triplet_sample", choices=sorted(SAMPLERS), default="distance")

    # relational terms (RKD baselines) + warmup que balanceia contra o triplet
    p.add_argument("--dist_ratio", type=float, default=0.0)
    p.add_argument("--angle_ratio", type=float, default=0.0)
    p.add_argument("--rel_warmup_frac", type=float, default=0.1,
                   help="rampa 0->1 do peso relacional (dist/angle/graph) p/ "
                        "balancear contra o triplet; 0 = sem warmup")

    # Graph-RKD
    p.add_argument("--graph_rkd_mode", choices=["off", "regression", "contrastive"],
                   default="off")
    p.add_argument("--graph_rkd_method", choices=["profile", "mds"], default="profile")
    p.add_argument("--graph_rkd_norm",
                   choices=["per_graph", "minibatch", "none", "hybrid"],
                   default="per_graph",
                   help="esquema de normalização de escala do descritor (H2)")
    p.add_argument("--graph_rkd_nodes", type=int, default=8)
    p.add_argument("--graph_rkd_ratio", type=float, default=0.0)
    p.add_argument("--graph_rkd_sampling", choices=["partition", "random", "log"],
                   default="log")
    p.add_argument("--graph_rkd_alpha", type=float, default=0.5)
    p.add_argument("--graph_rkd_gmax", type=int, default=64)
    p.add_argument("--num_negatives", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.07)

    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num_image_per_class", type=int, default=5)
    p.add_argument("--iter_per_epoch", type=int, default=100)
    p.add_argument("--recall", type=int, nargs="+", default=RECALL_K)
    p.add_argument("--select_metric", choices=sorted(SELECT_METRICS), default="mapr")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--resume", default=None)

    p.add_argument("--wandb_project", default="convnextmicro-metric-distill")
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


def rel_scale_at(epoch, total_epochs, warmup_frac):
    """Rampa linear 0->1 do peso relacional na fração inicial das épocas."""
    if warmup_frac <= 0:
        return 1.0
    w = max(1, int(warmup_frac * total_epochs))
    return min(1.0, epoch / w)


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


def load_teacher(opts, run, device):
    ckpt = resolve_teacher_checkpoint(teacher_load=opts.teacher_load,
                                      teacher_artifact=opts.teacher_artifact, run=run)
    teacher = _load_teacher_model(opts.teacher_arch, 2, ckpt, device, strict=False)
    teacher.eval()
    print(f"Loaded {opts.teacher_arch} metric teacher from {ckpt}")
    return teacher


def run_experiment(opts):
    if (opts.teacher_load is None) == (opts.teacher_artifact is None):
        raise ValueError("Provide exactly one of --teacher_load or --teacher_artifact")
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    l2 = opts.l2normalize == "true"
    dataset_cls, marker = DATASETS[opts.dataset]
    if opts.wandb_group is None:
        opts.wandb_group = "distill-%s-%s" % (opts.teacher_arch, opts.dataset)

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

    tags = ["metric-distill", opts.teacher_arch, opts.dataset]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags, id=opts.wandb_id,
                     resume=("allow" if opts.wandb_id else None),
                     config={**vars(opts), "info": info})

    student = ConvNextMicro(num_classes=2, drop_path=opts.drop_path,
                            dims=tuple(opts.dims), depths=tuple(opts.depths),
                            apply_softmax=False).to(device)
    teacher = load_teacher(opts, run, device)

    triplet = L2Triplet(sampler=SAMPLERS[opts.triplet_sample](), margin=opts.triplet_margin)
    dist_criterion = RkdDistance()
    angle_criterion = RKdAngle()
    graph_criterion = None
    if opts.graph_rkd_mode != "off" and opts.graph_rkd_ratio > 0:
        common = dict(method=opts.graph_rkd_method, n_nodes=opts.graph_rkd_nodes,
                      sampling=opts.graph_rkd_sampling, alpha=opts.graph_rkd_alpha,
                      g_max=opts.graph_rkd_gmax, norm=opts.graph_rkd_norm)
        graph_criterion = (GraphRKDLoss(**common) if opts.graph_rkd_mode == "regression"
                           else GraphContrastiveDistillLoss(
                               num_negatives=opts.num_negatives,
                               temperature=opts.temperature, **common)).to(device)

    optimizer = optim.AdamW(student.parameters(), lr=opts.lr,
                            weight_decay=opts.weight_decay, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, opts, opts.iter_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")
    art_name = "convnextmicro-metric-distill-%s-%s" % (opts.teacher_arch, opts.dataset)
    primary = opts.recall[0]

    # referência: recall do teacher nos mesmos splits
    teacher_metrics = evaluate_recall_splits(teacher, loaders, device, l2, opts.recall,
                                             tag="teacher ")
    run.log(recall_log_dict(teacher_metrics, prefix="teacher/"), step=0)

    start_epoch, best_val, best_state = 0, 0.0, None
    if opts.resume and os.path.exists(opts.resume):
        ckpt = torch.load(opts.resume, map_location=device)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val", 0.0)
        best_state = ckpt.get("best_state", None)
        print(f"resumed from {opts.resume} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch + 1, opts.epochs + 1):
        student.train()
        rel = rel_scale_at(epoch, opts.epochs, opts.rel_warmup_frac)
        # Somatórios ponderados (o que entra na loss) e CRUS (não-ponderados, I7):
        # um termo silenciosamente-zerado deve ser detectável separadamente.
        sums = {"loss": 0, "triplet": 0, "dist": 0, "angle": 0, "graph": 0}
        raw_sums = {"triplet": 0, "dist": 0, "angle": 0, "graph": 0}
        pbar = tqdm(loaders["train"], ncols=100, desc=f"[MDistill {epoch}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                t_emb, _ = embed(teacher, images, l2)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=opts.amp and device == "cuda"):
                s_emb, _ = embed(student, images, l2)
                tri_raw = triplet(s_emb, labels)
                dist_raw = dist_criterion(s_emb, t_emb)
                angle_raw = angle_criterion(s_emb, t_emb)
                graph_raw = (graph_criterion(s_emb, t_emb)
                             if graph_criterion is not None else s_emb.new_zeros(()))
                tri = opts.triplet_ratio * tri_raw
                dist = rel * opts.dist_ratio * dist_raw
                angle = rel * opts.angle_ratio * angle_raw
                graph = rel * opts.graph_rkd_ratio * graph_raw
                loss = tri + dist + angle + graph
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for k, v in (("loss", loss), ("triplet", tri), ("dist", dist),
                         ("angle", angle), ("graph", graph)):
                sums[k] += float(v)
            for k, v in (("triplet", tri_raw), ("dist", dist_raw),
                         ("angle", angle_raw), ("graph", graph_raw)):
                raw_sums[k] += float(v)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        n = len(loaders["train"])
        loss_log = {f"train/{k}_loss": v / n for k, v in sums.items()}
        loss_log.update({f"train/{k}_loss_raw": v / n for k, v in raw_sums.items()})
        loss_log["train/rel_scale"] = rel
        if epoch % opts.eval_every and epoch != opts.epochs:
            run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                     **loss_log}, step=epoch)
            continue

        metrics = evaluate_recall_splits(student, loaders, device, l2, opts.recall,
                                         tag=f"E{epoch} ")
        val_score = score_of(metrics["val"], opts.select_metric)
        improved = val_score > best_val
        if improved:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        print(f"[Epoch {epoch}] val {opts.select_metric}={val_score*100:.2f} "
              f"(test mAP@R={metrics['test']['mAP@R']*100:.2f}) best_val={best_val*100:.2f}")
        run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                 "val/best_score": best_val, **loss_log,
                 **recall_log_dict(metrics)}, step=epoch)

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            last_path = os.path.join(opts.save_dir, "student_last.pth")
            torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "best_val": best_val, "best_state": best_state}, last_path)
            is_final = epoch == opts.epochs
            if improved or is_final:
                aliases = (["best"] if improved else []) + (["last"] if is_final else [])
                log_model_artifact(run, last_path, art_name, aliases=aliases,
                                   ttl_days=30,
                                   metadata={"epoch": epoch, **recall_log_dict(metrics)})

    if best_state is not None:
        student.load_state_dict(best_state)
    final = evaluate_recall_splits(student, loaders, device, l2, opts.recall, tag="final ")
    run.log(recall_log_dict(final, prefix="final/"))
    for s, m in final.items():
        for k, v in m.items():
            run.summary[f"final_{s}_{k}"] = v
    run.summary["best_val_score"] = best_val
    run.summary["select_metric"] = opts.select_metric
    run.summary["teacher_test_mAP@R"] = teacher_metrics["test"]["mAP@R"]
    print("Done. " + " | ".join(
        f"{s} mAP@R={m['mAP@R']*100:.2f}" for s, m in final.items()))
    run.finish()
    return {"best_val_score": best_val, "select_metric": opts.select_metric,
            "final": final, "teacher": teacher_metrics}


def run_with_cli_args(cli_args):
    return run_experiment(build_parser().parse_args([str(a) for a in cli_args]))


def run_with_params(params):
    return run_with_cli_args(_params_to_cli_args(params))


def main(argv=None):
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
