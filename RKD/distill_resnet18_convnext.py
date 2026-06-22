"""Distill a fine-tuned ResNet-18 (teacher) into ConvNextMicro (student).

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
(28x28); teacher side is ResNet-18 layer2 (28x28). The attention map pools over
channels, so the 48-vs-128 channel mismatch is irrelevant -- only the 28x28
spatial size must match (it does).

Teacher is a finetune_resnet18.py checkpoint / W&B artifact (state under "model").
Use --dataset {cars196,cub200}; the per-dataset launchers are
examples/distill_convnext_cars.sh and examples/distill_convnext_cub.sh.
"""

import argparse
import math
import os

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb
from metric.loss import AttentionTransfer, RKdAngle, RkdDistance
from model import ConvNextMicro
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb_artifacts import log_model_artifact, resolve_teacher_checkpoint

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASETS = {
    "cars196": (dataset.Cars196Classification, 196,
                os.path.join("Cars196", "cars_annos.mat")),
    "cub200": (dataset.CUB2011Classification, 200,
               os.path.join("CUB_200_2011", "images.txt")),
}


class ResNet18Teacher(nn.Module):
    """Fine-tuned ResNet-18 exposing logits, pooled embedding and layer2 map."""

    def __init__(self, num_classes):
        super().__init__()
        self.m = torchvision.models.resnet18(weights=None)
        self.m.fc = nn.Linear(self.m.fc.in_features, num_classes)

    def forward_features(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l1 = m.layer1(x)
        l2 = m.layer2(l1)          # 28x28 -> AT attachment (matches student stage2)
        l4 = m.layer4(m.layer3(l2))
        emb = torch.flatten(m.avgpool(l4), 1)
        return {"stage2": l2, "embedding": emb, "logits": m.fc(emb)}


def build_parser():
    p = argparse.ArgumentParser(description="Distill ResNet-18 -> ConvNextMicro")

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

    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--clip_grad", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)

    # wandb
    p.add_argument("--wandb_project", default="resnet18-to-convnext-distill")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_group", default=None, help="defaults to distill-<dataset>")
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


def hinton_kd(student_logits, teacher_logits, T):
    """Classic KD: KL(student||teacher) on softened logits, scaled by T^2."""
    return F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)


def load_teacher(opts, run, device):
    ckpt = resolve_teacher_checkpoint(teacher_load=opts.teacher_load,
                                      teacher_artifact=opts.teacher_artifact, run=run)
    blob = torch.load(ckpt, map_location=device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    teacher = ResNet18Teacher(opts.num_classes).to(device)
    teacher.m.load_state_dict(state)
    teacher.eval()
    print(f"Loaded teacher from {ckpt}")
    return teacher


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


@torch.no_grad()
def evaluate(net, loader, device, desc):
    net.eval()
    top1 = top5 = total = 0
    for images, targets in tqdm(loader, ncols=80, desc=desc):
        images, targets = images.to(device), targets.to(device)
        out = net.forward_features(images)
        logits = out["logits"] if "logits" in out else net.fc(out["embedding"])
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5 == targets.unsqueeze(1)
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        total += targets.numel()
    return top1 / max(1, total), top5 / max(1, total)


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
        opts.wandb_group = "distill-%s" % opts.dataset

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize])

    download = not os.path.exists(os.path.join(os.path.abspath(opts.data), marker))
    train_set = dataset_cls(opts.data, train=True, transform=train_tf, download=download)
    test_set = dataset_cls(opts.data, train=False, transform=test_tf, download=False)
    print(f"dataset={opts.dataset} train={len(train_set)} test={len(test_set)} "
          f"classes={opts.num_classes}")

    # persistent_workers evita recriar os workers a cada epoca (os loaders sao
    # reusados no loop de treino/eval). Sem isso, em runs longos (300 epocas) os
    # ciclos de teardown dos workers acumulam e deadlockam ao entrar no eval.
    persistent = opts.workers > 0
    train_loader = DataLoader(train_set, batch_size=opts.batch, shuffle=True,
                              num_workers=opts.workers, pin_memory=True, drop_last=True,
                              persistent_workers=persistent)
    test_loader = DataLoader(test_set, batch_size=opts.batch, shuffle=False,
                             num_workers=opts.workers, pin_memory=True,
                             persistent_workers=persistent)

    tags = ["distillation", "resnet18", "convnextmicro", opts.dataset]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags, config=vars(opts))

    student = ConvNextMicro(num_classes=opts.num_classes, drop_path=opts.drop_path,
                            dims=tuple(opts.dims), depths=tuple(opts.depths),
                            apply_softmax=False).to(device)
    teacher = load_teacher(opts, run, device)

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    dist_criterion = RkdDistance()
    angle_criterion = RKdAngle()
    at_criterion = AttentionTransfer()

    optimizer = torch.optim.AdamW(student.parameters(), lr=opts.lr,
                                  weight_decay=opts.weight_decay, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, opts, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp and device == "cuda")

    n_s = sum(p.numel() for p in student.parameters())
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"student={n_s/1e6:.2f}M teacher={n_t/1e6:.2f}M device={device}")

    t_top1, t_top5 = evaluate(teacher, test_loader, device, "[Teacher]")
    run.log({"teacher/top1": t_top1, "teacher/top5": t_top5}, step=0)
    print(f"teacher test top1={t_top1*100:.2f} top5={t_top5*100:.2f}")

    best_top1 = 0.0
    for epoch in range(1, opts.epochs + 1):
        student.train()
        sums = {"loss": 0, "ce": 0, "kd": 0, "dist": 0, "angle": 0, "at": 0}
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
                loss = ce + kd + dist + angle + at

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if opts.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), opts.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k, v in (("loss", loss), ("ce", ce), ("kd", kd), ("dist", dist),
                         ("angle", angle), ("at", at)):
                sums[k] += float(v)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        n = len(train_loader)
        top1, top5 = evaluate(student, test_loader, device, f"[Eval {epoch}]")
        improved = top1 > best_top1
        best_top1 = max(best_top1, top1)
        print(f"[Epoch {epoch}] loss={sums['loss']/n:.4f} top1={top1*100:.2f} "
              f"top5={top5*100:.2f} best={best_top1*100:.2f}")
        run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                 "val/top1": top1, "val/top5": top5, "val/best_top1": best_top1,
                 **{f"train/{k}_loss": v / n for k, v in sums.items()}}, step=epoch)

        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            last_path = os.path.join(opts.save_dir, "student_last.pth")
            torch.save(student.state_dict(), last_path)
            # Salva o checkpoint local toda epoca, mas so envia ao W&B quando
            # melhora (best) ou na ultima epoca (last) -- sem alias epoch-N e com
            # TTL -- para nao acumular uma versao por epoca e estourar o storage.
            is_final = epoch == opts.epochs
            if improved or is_final:
                aliases = (["best"] if improved else []) + (["last"] if is_final else [])
                log_model_artifact(run, last_path,
                                   "convnextmicro-distill-%s" % opts.dataset,
                                   aliases=aliases, ttl_days=30,
                                   metadata={"epoch": epoch, "top1": top1, "top5": top5,
                                             "best_top1": best_top1})

    run.summary["best_top1"] = best_top1
    run.summary["teacher_top1"] = t_top1
    print(f"Done. student best top1 = {best_top1*100:.2f} (teacher {t_top1*100:.2f})")
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
