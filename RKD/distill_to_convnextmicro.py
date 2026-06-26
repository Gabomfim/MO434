"""Distill a fine-tuned classifier (teacher) into ConvNextMicro (student).

Teacher backbone is chosen with --teacher_arch {resnet18, convnext_tiny} and
loaded from a finetune_classifier.py checkpoint / W&B artifact.

Classification distillation on the *official* split (all classes), so teacher
and student share the label space and Hinton KD on logits is valid. Combines
four distillation signals:

  * Hinton KD    : KL on temperature-softened logits         (--kd_ratio, --kd_T)
  * RKD distance : relational distance on pooled embeddings  (--dist_ratio)
  * RKD angle    : relational angle on pooled embeddings     (--angle_ratio)
  * Attention map: activation-attention transfer             (--at_ratio)
  * (+ cross-entropy on true labels, --ce_ratio)

Default loss weights follow the source papers / RepDistiller conventions:
CE=1.0, KD=0.9 with T=4 (Hinton); RKD distance=25, angle=50 (Park et al. 2019);
attention=1000 (Zagoruyko & Komodakis). The student is trained from scratch, so
the schedule is long (300 epochs, 20-epoch linear warmup + cosine, AdamW
lr 1e-3, weight decay 0.05) -- distillation provides the main supervision.

Attention map placement: student side is the 2nd non-pointwise layer = stage 2
(28x28); teacher side is its own stage-2 map (ResNet-18 layer2 or ConvNeXt-Tiny
features[:4]), also 28x28. The attention map pools over channels, so the channel
mismatch is irrelevant -- only the 28x28 spatial size must match (it does).

Use --dataset {cars196,cub200}; per-dataset launchers are in examples/.
"""

import argparse
import math
import os

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb
from metric.loss import AttentionTransfer, RKdAngle, RkdDistance
from model import ConvNextMicro
from tqdm import tqdm
from classification_split import (
    build_classification_loaders,
    evaluate_splits,
    log_dict,
)
from teacher_models import ARCHS, load_teacher as _load_teacher_model
from wandb_artifacts import log_model_artifact, resolve_teacher_checkpoint
from graph_rkd import GraphContrastiveDistillLoss, GraphRKDLoss

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASETS = {
    "cars196": (dataset.Cars196Classification, 196,
                os.path.join("Cars196", "cars_annos.mat")),
    "cub200": (dataset.CUB2011Classification, 200,
               os.path.join("CUB_200_2011", "images.txt")),
}


def build_parser():
    p = argparse.ArgumentParser(description="Distill classifier -> ConvNextMicro")

    p.add_argument("--dataset", choices=sorted(DATASETS), default="cars196")
    p.add_argument("--data", default="data")
    p.add_argument("--num_classes", type=int, default=None,
                   help="inferred from --dataset when omitted")
    p.add_argument("--workers", type=int, default=8)

    # student
    p.add_argument("--dims", type=int, nargs=4, default=[24, 48, 96, 192])
    p.add_argument("--depths", type=int, nargs=4, default=[1, 1, 3, 1])
    p.add_argument("--drop_path", type=float, default=0.1)

    # teacher
    p.add_argument("--teacher_arch", choices=sorted(ARCHS), default="resnet18")
    p.add_argument("--teacher_load", default=None)
    p.add_argument("--teacher_artifact", default=None,
                   help="W&B ref (e.g. resnet18-cars196:best); xor --teacher_load")

    # distillation ratios (literature-grounded defaults; see module docstring)
    p.add_argument("--ce_ratio", type=float, default=1.0, help="CE on labels (gamma)")
    p.add_argument("--kd_ratio", type=float, default=0.9, help="Hinton KD (alpha)")
    p.add_argument("--kd_T", type=float, default=4.0, help="KD temperature")
    p.add_argument("--dist_ratio", type=float, default=25.0, help="RKD distance (paper)")
    p.add_argument("--angle_ratio", type=float, default=50.0, help="RKD angle (paper)")
    p.add_argument("--at_ratio", type=float, default=1000.0, help="attention transfer")
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # Graph-RKD (loss de grafo de N nós, distância euclidiana)
    p.add_argument("--graph_rkd_mode", choices=["off", "regression", "contrastive"],
                   default="off")
    p.add_argument("--graph_rkd_method", choices=["profile", "mds"], default="profile")
    p.add_argument("--graph_rkd_nodes", type=int, default=8, help="N (busca binária)")
    p.add_argument("--graph_rkd_ratio", type=float, default=0.0)
    p.add_argument("--graph_warmup_frac", type=float, default=0.0,
                   help="ramp linear do peso da loss de grafo (0->ratio) na fração "
                        "inicial das épocas; balanceia CE vs grafo. 0 = sem warmup")
    p.add_argument("--graph_rkd_sampling", choices=["partition", "random", "log"],
                   default="log")
    p.add_argument("--graph_rkd_graphs", type=int, default=None,
                   help="grafos/passo qdo sampling=random")
    p.add_argument("--graph_rkd_alpha", type=float, default=0.5,
                   help="sampling=log: G = alpha*log2(C(K,N))")
    p.add_argument("--graph_rkd_gmin", type=int, default=None,
                   help="sampling=log: piso de grafos (default ⌊K/N⌋)")
    p.add_argument("--graph_rkd_gmax", type=int, default=None,
                   help="sampling=log: teto de grafos (limita custo)")
    p.add_argument("--num_negatives", type=int, default=10, help="contrastive")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="temperatura inicial da InfoNCE contrastiva")
    # rotina de temperatura da destilação ao longo do treino (atua na InfoNCE)
    p.add_argument("--temp_schedule", choices=["constant", "linear", "cosine", "exp"],
                   default="cosine")
    p.add_argument("--temp_start", type=float, default=0.1)
    p.add_argument("--temp_end", type=float, default=0.05)

    # optimization
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

    # wandb
    p.add_argument("--wandb_project", default="convnextmicro-distill")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_id", default=None,
                   help="id estável da run W&B (resume='allow' p/ retomar)")
    p.add_argument("--wandb_group", default=None, help="defaults to distill-<dataset>")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb_tags", nargs="*", default=None)
    p.add_argument("--resume", default=None,
                   help="caminho do student_last.pth p/ retomar (tolerante a ausência)")
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


def hinton_kd(student_logits, teacher_logits, T):
    """Classic KD: KL(student||teacher) on softened logits, scaled by T^2."""
    return F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)


def load_teacher(opts, run, device):
    ckpt = resolve_teacher_checkpoint(teacher_load=opts.teacher_load,
                                      teacher_artifact=opts.teacher_artifact, run=run)
    teacher = _load_teacher_model(opts.teacher_arch, opts.num_classes, ckpt, device)
    teacher.eval()
    print(f"Loaded {opts.teacher_arch} teacher from {ckpt}")
    return teacher


def graph_ratio_at(epoch, total_epochs, target, warmup_frac):
    """Peso da loss de grafo na época `epoch`: rampa linear 0->target na fração
    inicial `warmup_frac` das épocas; depois constante. Balanceia CE vs grafo
    (cedo os embeddings do aluno são ruído -> matar a geometria do teacher cedo
    é sinal ruidoso, então entramos com ela aos poucos)."""
    if warmup_frac <= 0:
        return target
    w = max(1, int(warmup_frac * total_epochs))
    return target * min(1.0, epoch / w)


def temperature_at(epoch, total_epochs, t0, t1, shape="cosine"):
    """Temperatura da destilação na época ``epoch`` (1..total), de t0 -> t1."""
    p = (epoch - 1) / max(1, total_epochs - 1)
    if shape == "constant":
        return t0
    if shape == "linear":
        return t0 + (t1 - t0) * p
    if shape == "cosine":
        return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * p))
    if shape == "exp":
        t0 = max(t0, 1e-8)
        return t0 * (max(t1, 1e-8) / t0) ** p
    raise ValueError("temp_schedule inválido")


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


def student_outputs(student, images):
    feats = student.forward_features(images)
    return student.fc(feats["embedding"]), feats["embedding"], feats["stage2"]


def run_experiment(opts):
    if (opts.teacher_load is None) == (opts.teacher_artifact is None):
        raise ValueError("Provide exactly one of --teacher_load or --teacher_artifact")
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_cls, default_classes, marker = DATASETS[opts.dataset]
    if opts.num_classes is None:
        opts.num_classes = default_classes
    if opts.wandb_group is None:
        opts.wandb_group = "distill-%s-%s" % (opts.teacher_arch, opts.dataset)

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

    tags = ["distillation", opts.teacher_arch, "convnextmicro", opts.dataset]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags, id=opts.wandb_id,
                     resume=("allow" if opts.wandb_id else None), config=vars(opts))

    student = ConvNextMicro(num_classes=opts.num_classes, drop_path=opts.drop_path,
                            dims=tuple(opts.dims), depths=tuple(opts.depths),
                            apply_softmax=False).to(device)
    teacher = load_teacher(opts, run, device)

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    dist_criterion = RkdDistance()
    angle_criterion = RKdAngle()
    at_criterion = AttentionTransfer()

    # Graph-RKD: loss de grafo de N nós (euclidiana), somada à loss padrão.
    graph_criterion = None
    if opts.graph_rkd_mode != "off" and opts.graph_rkd_ratio > 0:
        if opts.graph_rkd_mode == "regression":
            graph_criterion = GraphRKDLoss(
                method=opts.graph_rkd_method, n_nodes=opts.graph_rkd_nodes,
                sampling=opts.graph_rkd_sampling,
                graphs_per_step=opts.graph_rkd_graphs,
                alpha=opts.graph_rkd_alpha, g_min=opts.graph_rkd_gmin,
                g_max=opts.graph_rkd_gmax).to(device)
        else:  # contrastive
            graph_criterion = GraphContrastiveDistillLoss(
                method=opts.graph_rkd_method, n_nodes=opts.graph_rkd_nodes,
                sampling=opts.graph_rkd_sampling,
                graphs_per_step=opts.graph_rkd_graphs,
                alpha=opts.graph_rkd_alpha, g_min=opts.graph_rkd_gmin,
                g_max=opts.graph_rkd_gmax,
                num_negatives=opts.num_negatives, temperature=opts.temperature).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=opts.lr,
                                  weight_decay=opts.weight_decay, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, opts, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")

    n_s = sum(p.numel() for p in student.parameters())
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"student={n_s/1e6:.2f}M teacher={n_t/1e6:.2f}M device={device}")

    # logits callables (mesma assinatura usada por evaluate_splits)
    def student_logits(images):
        return student(images)  # apply_softmax=False -> logits

    def teacher_logits(images):
        return teacher.forward_features(images)["logits"]

    art_name = "convnextmicro-distill-%s-%s" % (opts.teacher_arch, opts.dataset)

    # Professor avaliado nos MESMOS splits/métricas, como referência.
    teacher.eval()
    teacher_metrics = evaluate_splits(teacher_logits, loaders, device, tag="teacher ")
    run.log(log_dict(teacher_metrics, prefix="teacher/"), step=0)
    print("teacher: " + " | ".join(
        f"{s} top1={m['top1']*100:.2f}" for s, m in teacher_metrics.items()))

    is_contrastive = (opts.graph_rkd_mode == "contrastive") and graph_criterion is not None

    start_epoch, best_val_top1, best_state = 0, 0.0, None
    if opts.resume and os.path.exists(opts.resume):
        ckpt = torch.load(opts.resume, map_location=device)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val_top1 = ckpt.get("best_val_top1", 0.0)
        best_state = ckpt.get("best_state", None)
        print(f"resumed from {opts.resume} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch + 1, opts.epochs + 1):
        student.train()
        # rotina de temperatura da destilação (atua na InfoNCE contrastiva)
        cur_temp = temperature_at(epoch, opts.epochs, opts.temp_start,
                                  opts.temp_end, opts.temp_schedule)
        if is_contrastive:
            graph_criterion.nce.temperature = cur_temp
        # peso da loss de grafo nesta época (warmup p/ balancear CE vs grafo)
        cur_graph_ratio = graph_ratio_at(epoch, opts.epochs, opts.graph_rkd_ratio,
                                         opts.graph_warmup_frac)
        sums = {"loss": 0, "ce": 0, "kd": 0, "dist": 0, "angle": 0, "at": 0, "graph": 0}
        pbar = tqdm(train_loader, ncols=100, desc=f"[Distill {epoch}]")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            with torch.no_grad():
                t = teacher.forward_features(images)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=opts.amp and device == "cuda"):
                s_logits, s_emb, s_s2 = student_outputs(student, images)
                ce = opts.ce_ratio * ce_criterion(s_logits, targets)
                kd = opts.kd_ratio * hinton_kd(s_logits, t["logits"], opts.kd_T)
                dist = opts.dist_ratio * dist_criterion(s_emb, t["embedding"])
                angle = opts.angle_ratio * angle_criterion(s_emb, t["embedding"])
                at = opts.at_ratio * at_criterion(s_s2, t["stage2"])
                graph = (cur_graph_ratio * graph_criterion(s_emb, t["embedding"])
                         if graph_criterion is not None else s_logits.new_zeros(()))
                loss = ce + kd + dist + angle + at + graph

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if opts.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), opts.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k, v in (("loss", loss), ("ce", ce), ("kd", kd), ("dist", dist),
                         ("angle", angle), ("at", at), ("graph", graph)):
                sums[k] += float(v)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        n = len(train_loader)
        loss_log = {f"train/{k}_loss": v / n for k, v in sums.items()}
        is_eval_epoch = (epoch % opts.eval_every == 0) or (epoch == opts.epochs)
        if not is_eval_epoch:
            run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                     "train/graph_temperature": cur_temp,
                     "train/graph_ratio": cur_graph_ratio, **loss_log}, step=epoch)
            continue

        # Mesmas métricas (top-1/top-5) em train, val e test (aluno).
        student.eval()
        metrics = evaluate_splits(student_logits, loaders, device, tag=f"E{epoch} ")
        val_top1 = metrics["val"]["top1"]
        improved = val_top1 > best_val_top1   # seleção pela melhor VALIDAÇÃO
        if improved:
            best_val_top1 = val_top1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in student.state_dict().items()}
        print(f"[Epoch {epoch}] loss={sums['loss']/n:.4f} "
              f"train@1={metrics['train']['top1']*100:.2f} "
              f"val@1={val_top1*100:.2f} test@1={metrics['test']['top1']*100:.2f} "
              f"best_val@1={best_val_top1*100:.2f}")
        run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                 "val/best_top1": best_val_top1, "train/graph_temperature": cur_temp,
                 "train/graph_ratio": cur_graph_ratio,
                 **loss_log, **log_dict(metrics)}, step=epoch)

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            last_path = os.path.join(opts.save_dir, "student_last.pth")
            # estado completo toda época (resume); ao W&B só best/last com TTL.
            torch.save({"model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "best_val_top1": best_val_top1, "best_state": best_state},
                       last_path)
            # Local toda época; envia ao W&B só quando melhora (best) ou na
            # última época (last) -- sem alias epoch-N e com TTL -- para não
            # acumular uma versão por época e estourar o storage.
            is_final = epoch == opts.epochs
            if improved or is_final:
                aliases = (["best"] if improved else []) + (["last"] if is_final else [])
                log_model_artifact(run, last_path, art_name, aliases=aliases,
                                   ttl_days=30,
                                   metadata={"epoch": epoch, **log_dict(metrics)})

    # Métricas finais com o aluno que MAXIMIZA A GENERALIZAÇÃO (melhor val).
    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    final = evaluate_splits(student_logits, loaders, device, tag="final ")
    run.log(log_dict(final, prefix="final/"))
    for split, m in final.items():
        run.summary[f"final_{split}_top1"] = m["top1"]
        run.summary[f"final_{split}_top5"] = m["top5"]
    run.summary["best_val_top1"] = best_val_top1
    run.summary["teacher_test_top1"] = teacher_metrics["test"]["top1"]
    print("Done. aluno (melhor validação): " + " | ".join(
        f"{s} top1={m['top1']*100:.2f}" for s, m in final.items())
        + f" | teacher test top1={teacher_metrics['test']['top1']*100:.2f}")
    run.finish()
    # dict com os 3 splits + best_val (a busca binária usa val, nunca test).
    return {"final": final, "best_val_top1": best_val_top1,
            "teacher": teacher_metrics}


def run_with_cli_args(cli_args):
    return run_experiment(build_parser().parse_args([str(a) for a in cli_args]))


def run_with_params(params):
    return run_with_cli_args(_params_to_cli_args(params))


def main(argv=None):
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
