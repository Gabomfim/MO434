"""Distill a teacher ConvNeXt into the student ConvNextMicro.

Cars196 (and the other datasets here) use the *metric-learning* split: train and
test classes are disjoint, so distillation happens in the feature/embedding space
and is evaluated with recall@k -- not with logit/softmax KD.

Losses combined (each gated by its ratio; set to 0 to disable):
  * RKD distance / angle  : relational KD on the pooled embeddings (dim-agnostic)
  * Dark rank             : HardDarkRank on the embeddings
  * Triplet               : student-only metric loss against true labels
  * Attention Transfer    : activation-attention map matching, applied ONLY at the
                            student's second non-pointwise layer.

Attention Transfer placement: following "Paying More Attention to Attention"
(Zagoruyko & Komodakis, 2017), attention maps are taken at *group/stage outputs*.
The "second" such point in ConvNextMicro is the output of stage 2 (28x28). The
attention map pools over channels, so the student/teacher channel mismatch is
irrelevant -- only the 28x28 spatial size must match (it does, since both share
the stem-stride-4 + one-downsample geometry).

The teacher is a (wider) ConvNextMicro loaded from a checkpoint or W&B artifact
(--teacher_load / --teacher_artifact). Checkpoints saved by train_convnext.py
({"model"/"ema": state_dict}) and raw state_dicts are both accepted.
"""

import argparse
import math
import os

import dataset
import metric.pairsampler as pair
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from metric.batchsampler import NPairs
from metric.loss import AttentionTransfer, HardDarkRank, L2Triplet, RKdAngle, RkdDistance
from metric.utils import recall
from model import ConvNextMicro
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb_artifacts import resolve_teacher_checkpoint

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LookupChoices(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.choices[values])


def build_parser():
    p = argparse.ArgumentParser(description="Distill ConvNeXt -> ConvNextMicro")

    p.add_argument("--dataset",
                   choices=dict(cub200=dataset.CUB2011Metric,
                                cars196=dataset.Cars196Metric,
                                stanford=dataset.StanfordOnlineProductsMetric),
                   default=dataset.Cars196Metric, action=LookupChoices)
    p.add_argument("--data", default="data")
    p.add_argument("--num_classes", type=int, default=98,
                   help="#train classes (teacher head size; unused for distill losses)")

    # student
    p.add_argument("--dims", type=int, nargs=4, default=[24, 48, 96, 192])
    p.add_argument("--depths", type=int, nargs=4, default=[1, 1, 3, 1])
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--l2normalize", choices=["true", "false"], default="true")

    # teacher
    p.add_argument("--teacher_dims", type=int, nargs=4, default=[96, 192, 384, 768])
    p.add_argument("--teacher_depths", type=int, nargs=4, default=[1, 1, 3, 1])
    p.add_argument("--teacher_load", default=None)
    p.add_argument("--teacher_artifact", default=None,
                   help="W&B teacher artifact ref (mutually exclusive w/ --teacher_load)")
    p.add_argument("--teacher_weights", choices=["auto", "model", "ema"], default="auto",
                   help="which weights to use from a train_convnext.py checkpoint")

    # distillation loss ratios
    p.add_argument("--dist_ratio", type=float, default=1.0)
    p.add_argument("--angle_ratio", type=float, default=2.0)
    p.add_argument("--dark_ratio", type=float, default=0.0)
    p.add_argument("--dark_alpha", type=float, default=2.0)
    p.add_argument("--dark_beta", type=float, default=3.0)
    p.add_argument("--triplet_ratio", type=float, default=0.0)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument("--triplet_sample",
                   choices=dict(random=pair.RandomNegative, hard=pair.HardNegative,
                                all=pair.AllPairs, semihard=pair.SemiHardNegative,
                                distance=pair.DistanceWeighted),
                   default=pair.DistanceWeighted, action=LookupChoices)
    p.add_argument("--at_ratio", type=float, default=0.0,
                   help="attention-transfer weight (stage-2 only)")

    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num_image_per_class", type=int, default=5)
    p.add_argument("--iter_per_epoch", type=int, default=100)
    p.add_argument("--recall", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--eval_every", type=int, default=5)

    # runtime
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None)

    # wandb
    p.add_argument("--wandb_project", default="convnext-distill")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_group", default="distill-convnextmicro")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb_tags", nargs="*", default=None)
    return p


def _name(x):
    return x.__name__ if hasattr(x, "__name__") else str(x)


def _params_to_cli_args(params):
    out = []
    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + key
        if isinstance(value, (list, tuple)):
            out.append(flag)
            out.extend(str(v) for v in value)
        else:
            out.extend([flag, str(value)])
    return out


def _dataset_available(dataset_cls, data_root):
    name = _name(dataset_cls).lower()
    root = os.path.abspath(data_root)
    if "cub" in name:
        return os.path.exists(os.path.join(root, "CUB_200_2011", "images.txt"))
    if "cars196" in name:
        return os.path.exists(os.path.join(root, "Cars196", "cars_annos.mat"))
    if "stanford" in name:
        return os.path.exists(os.path.join(root, "Stanford_Online_Products",
                                           "Ebay_train.txt"))
    return False


def load_teacher(opts, run, device):
    """Instantiate the wide teacher ConvNextMicro and load its weights."""
    ckpt_path = resolve_teacher_checkpoint(
        teacher_load=opts.teacher_load, teacher_artifact=opts.teacher_artifact, run=run)
    blob = torch.load(ckpt_path, map_location=device)

    # Accept raw state_dict or a train_convnext.py checkpoint dict.
    if isinstance(blob, dict) and any(k in blob for k in ("model", "ema")):
        if opts.teacher_weights == "ema" or (opts.teacher_weights == "auto"
                                             and "ema" in blob):
            state = blob["ema"]
        else:
            state = blob["model"]
    else:
        state = blob
    # EMA checkpoints (AveragedModel) prefix params with "module." and add n_averaged.
    state = {k.replace("module.", "", 1): v for k, v in state.items()
             if k != "n_averaged"}

    teacher = ConvNextMicro(num_classes=opts.num_classes, drop_path=0.0,
                            dims=tuple(opts.teacher_dims),
                            depths=tuple(opts.teacher_depths),
                            apply_softmax=False).to(device)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[teacher] missing={list(missing)} unexpected={list(unexpected)}")
    print(f"Loaded teacher from {ckpt_path}")
    return teacher


def embed(net, images, l2):
    feats = net.forward_features(images)
    emb = feats["embedding"]
    if l2:
        emb = F.normalize(emb, dim=1)
    return emb, feats["stage2"]


@torch.no_grad()
def evaluate(net, loader, device, l2, K, desc):
    net.eval()
    embs, labels = [], []
    for images, target in tqdm(loader, ncols=80, desc=desc):
        emb, _ = embed(net, images.to(device), l2)
        embs.append(emb.cpu())
        labels.append(target)
    return recall(torch.cat(embs), torch.cat(labels), K=K)


def build_scheduler(optimizer, opts):
    warmup = opts.warmup_epochs * opts.iter_per_epoch
    total = opts.epochs * opts.iter_per_epoch
    base_lr = optimizer.param_groups[0]["lr"]
    floor = opts.min_lr / base_lr if base_lr > 0 else 0.0

    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return optim.lr_scheduler.LambdaLR(optimizer, fn)


def run_experiment(opts):
    if (opts.teacher_load is None) == (opts.teacher_artifact is None):
        raise ValueError("Provide exactly one of --teacher_load or --teacher_artifact")

    for seed_fn in (torch.manual_seed, torch.cuda.manual_seed_all):
        seed_fn(opts.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    l2 = opts.l2normalize == "true"

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize])

    download = not _dataset_available(opts.dataset, opts.data)
    train_set = opts.dataset(opts.data, train=True, transform=train_tf, download=download)
    train_eval_set = opts.dataset(opts.data, train=True, transform=test_tf,
                                  download=False)
    test_set = opts.dataset(opts.data, train=False, transform=test_tf, download=False)
    print(f"train={len(train_set)} test={len(test_set)}")

    train_loader = DataLoader(
        train_set, pin_memory=True, num_workers=8,
        batch_sampler=NPairs(train_set, opts.batch, m=opts.num_image_per_class,
                             iter_per_epoch=opts.iter_per_epoch))
    train_eval_loader = DataLoader(train_eval_set, batch_size=opts.batch, shuffle=False,
                                   num_workers=8)
    test_loader = DataLoader(test_set, batch_size=opts.batch, shuffle=False,
                             num_workers=8, pin_memory=True)

    tags = ["distillation", "convnext", "distill_convnext.py"]
    if opts.wandb_tags:
        tags.extend(opts.wandb_tags)
    run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                     name=opts.wandb_run_name, group=opts.wandb_group,
                     mode=opts.wandb_mode, tags=tags,
                     config={**vars(opts), "dataset": _name(opts.dataset),
                             "triplet_sample": _name(opts.triplet_sample),
                             "device": device})

    student = ConvNextMicro(num_classes=opts.num_classes, drop_path=opts.drop_path,
                            dims=tuple(opts.dims), depths=tuple(opts.depths),
                            apply_softmax=False).to(device)
    teacher = load_teacher(opts, run, device)
    teacher.eval()

    n_student = sum(p.numel() for p in student.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"student={n_student/1e6:.2f}M  teacher={n_teacher/1e6:.2f}M  device={device}")

    dist_criterion = RkdDistance()
    angle_criterion = RKdAngle()
    dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)
    triplet_criterion = L2Triplet(sampler=opts.triplet_sample(),
                                  margin=opts.triplet_margin)
    at_criterion = AttentionTransfer()

    optimizer = optim.AdamW(student.parameters(), lr=opts.lr,
                            weight_decay=opts.weight_decay, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, opts)
    primary_k = opts.recall[0]

    def train_epoch(ep):
        student.train()
        sums = {"loss": 0, "dist": 0, "angle": 0, "dark": 0, "triplet": 0, "at": 0}
        pbar = tqdm(train_loader, ncols=100, desc=f"[Distill {ep}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                t_emb, t_s2 = embed(teacher, images, l2)
            s_emb, s_s2 = embed(student, images, l2)

            dist = opts.dist_ratio * dist_criterion(s_emb, t_emb)
            angle = opts.angle_ratio * angle_criterion(s_emb, t_emb)
            dark = opts.dark_ratio * dark_criterion(s_emb, t_emb)
            triplet = opts.triplet_ratio * triplet_criterion(s_emb, labels)
            # Attention transfer: ONLY at the student's 2nd non-pointwise layer.
            at = opts.at_ratio * at_criterion(s_s2, t_s2)
            loss = dist + angle + dark + triplet + at

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            for k, v in (("loss", loss), ("dist", dist), ("angle", angle),
                         ("dark", dark), ("triplet", triplet), ("at", at)):
                sums[k] += float(v)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        n = len(train_loader)
        return {k: v / n for k, v in sums.items()}

    def log_recall(prefix, rec, step):
        run.log({f"{prefix}_recall@{k}": r for k, r in zip(opts.recall, rec)}, step=step)

    teacher_rec = evaluate(teacher, test_loader, device, l2, opts.recall, "[Teacher]")
    log_recall("teacher/test", teacher_rec, 0)
    print("teacher test recall@%d: %.4f" % (primary_k, teacher_rec[0]))

    best = [0.0] * len(opts.recall)
    for ep in range(1, opts.epochs + 1):
        stats = train_epoch(ep)
        run.log({"epoch": ep, "lr": optimizer.param_groups[0]["lr"],
                 **{f"train/{k}_loss": v for k, v in stats.items()}}, step=ep)

        if ep % opts.eval_every == 0 or ep == opts.epochs:
            rec = evaluate(student, test_loader, device, l2, opts.recall, f"[Eval {ep}]")
            log_recall("eval/test", rec, ep)
            improved = rec[0] > best[0]
            best = [max(b, r) for b, r in zip(best, rec)]
            run.log({"best/test_recall@%d" % primary_k: best[0],
                     "eval/test_error@%d" % primary_k: 1.0 - rec[0]}, step=ep)
            print(f"[Epoch {ep}] loss={stats['loss']:.4f} "
                  f"recall@{primary_k}={rec[0]*100:.2f} best={best[0]*100:.2f}")

            if improved and opts.save_dir:
                os.makedirs(opts.save_dir, exist_ok=True)
                torch.save(student.state_dict(),
                           os.path.join(opts.save_dir, "student_best.pth"))

    run.summary["best_test_recall@%d" % primary_k] = best[0]
    for k, r in zip(opts.recall, best):
        run.summary["best_test_recall%d" % k] = r
    run.finish()
    return best


def run_with_cli_args(cli_args):
    return run_experiment(build_parser().parse_args([str(a) for a in cli_args]))


def run_with_params(params):
    return run_with_cli_args(_params_to_cli_args(params))


def main(argv=None):
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
