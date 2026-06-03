import argparse
import csv
import os
import random
import socket

import dataset
import metric.loss as loss
import metric.pairsampler as pair
import model.backbone as backbone
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from metric.batchsampler import NPairs
from metric.utils import pdist
from model.embedding import LinearEmbedding
from torch.utils.data import DataLoader
from tqdm import tqdm


class LookupChoices(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.choices[values])


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "eval"], default="train")

    parser.add_argument("--load", default=None)

    parser.add_argument(
        "--dataset",
        choices=dict(
            cub200=dataset.CUB2011Metric,
            cars196=dataset.Cars196Metric,
            stanford=dataset.StanfordOnlineProductsMetric,
        ),
        default=dataset.CUB2011Metric,
        action=LookupChoices,
    )

    parser.add_argument(
        "--base",
        choices=dict(
            googlenet=backbone.GoogleNet,
            inception_v1bn=backbone.InceptionV1BN,
            resnet18=backbone.ResNet18,
            resnet50=backbone.ResNet50,
        ),
        default=backbone.ResNet50,
        action=LookupChoices,
    )

    parser.add_argument(
        "--sample",
        choices=dict(
            random=pair.RandomNegative,
            hard=pair.HardNegative,
            all=pair.AllPairs,
            semihard=pair.SemiHardNegative,
            distance=pair.DistanceWeighted,
        ),
        default=pair.AllPairs,
        action=LookupChoices,
    )

    parser.add_argument(
        "--loss",
        choices=dict(
            l1_triplet=loss.L1Triplet,
            l2_triplet=loss.L2Triplet,
            contrastive=loss.ContrastiveLoss,
        ),
        default=loss.L2Triplet,
        action=LookupChoices,
    )

    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--l2normalize", choices=["true", "false"], default="true")

    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--lr_decay_epochs", type=int, default=[25, 30, 35], nargs="+")
    parser.add_argument("--lr_decay_gamma", default=0.5, type=float)

    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--num_image_per_class", default=5, type=int)

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--iter_per_epoch", type=int, default=100)
    parser.add_argument("--recall", default=[1], type=int, nargs="+")

    parser.add_argument("--seed", default=random.randint(1, 1000), type=int)
    parser.add_argument("--data", default="data")
    parser.add_argument("--save_dir", default=None)

    parser.add_argument("--wandb_project", default="rkd-metric-learning")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--wandb_group", default="teacher-runs")
    parser.add_argument("--wandb_job_type", default="teacher")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--max_confusion_classes", default=300, type=int)
    return parser


def _name(x):
    return x.__name__ if hasattr(x, "__name__") else str(x)


def _params_to_cli_args(params):
    cli_args = []
    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + key
        if isinstance(value, (list, tuple)):
            cli_args.append(flag)
            cli_args.extend([str(v) for v in value])
        else:
            cli_args.extend([flag, str(value)])
    return cli_args


def _alias_safe(value):
    text = str(value).lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "-")
    alias = "".join(out).strip("-")
    return alias or "na"


def _dataset_already_available(dataset_cls, data_root):
    dataset_name = _name(dataset_cls).lower()
    root = os.path.abspath(data_root)

    if "cub" in dataset_name:
        return os.path.exists(os.path.join(root, "CUB_200_2011", "images.txt"))

    if "cars196" in dataset_name:
        return os.path.exists(os.path.join(root, "Cars196", "cars_annos.mat"))

    if "stanford" in dataset_name:
        return os.path.exists(
            os.path.join(root, "Stanford_Online_Products", "Ebay_train.txt")
        )

    return False


def run_experiment(opts):
    if 1 not in opts.recall:
        opts.recall = [1] + opts.recall

    primary_recall_k = opts.recall[0]

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(opts.seed)

    base_model = opts.base(pretrained=True)
    if isinstance(base_model, backbone.InceptionV1BN) or isinstance(
        base_model, backbone.GoogleNet
    ):
        normalize = transforms.Compose(
            [
                transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
                transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
            ]
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    should_download = not _dataset_already_available(opts.dataset, opts.data)
    print(f"Dataset available locally: {not should_download}. download={should_download}")

    dataset_train = opts.dataset(
        opts.data, train=True, transform=train_transform, download=should_download
    )
    dataset_train_eval = opts.dataset(
        opts.data, train=True, transform=test_transform, download=should_download
    )
    dataset_eval = opts.dataset(
        opts.data, train=False, transform=test_transform, download=should_download
    )

    print("Number of images in Training Set: %d" % len(dataset_train))
    print("Number of images in Test set: %d" % len(dataset_eval))

    base_tags = ["rkd", "metric-learning", "run.py"]
    if opts.wandb_tags:
        base_tags.extend(opts.wandb_tags)

    run = wandb.init(
        project=opts.wandb_project,
        entity=opts.wandb_entity,
        name=opts.wandb_run_name,
        mode=opts.wandb_mode,
        group=opts.wandb_group,
        job_type=opts.wandb_job_type,
        config={
            "mode": opts.mode,
            "dataset": _name(opts.dataset),
            "base": _name(opts.base),
            "sample": _name(opts.sample),
            "loss": _name(opts.loss),
            "margin": opts.margin,
            "embedding_size": opts.embedding_size,
            "l2normalize": opts.l2normalize,
            "lr": opts.lr,
            "lr_decay_epochs": opts.lr_decay_epochs,
            "lr_decay_gamma": opts.lr_decay_gamma,
            "batch": opts.batch,
            "num_image_per_class": opts.num_image_per_class,
            "epochs": opts.epochs,
            "iter_per_epoch": opts.iter_per_epoch,
            "recall": opts.recall,
            "seed": opts.seed,
            "data": opts.data,
            "save_dir": opts.save_dir,
            "load": opts.load,
            "wandb_group": opts.wandb_group,
            "wandb_job_type": opts.wandb_job_type,
            "wandb_tags": opts.wandb_tags,
            "max_confusion_classes": opts.max_confusion_classes,
        },
        tags=base_tags,
    )

    run.summary["hostname"] = socket.gethostname()
    run.summary["train_size"] = len(dataset_train)
    run.summary["test_size"] = len(dataset_eval)

    dataset_artifact = wandb.Artifact(
        name="dataset-%s-%s" % (_name(opts.dataset).lower(), opts.seed),
        type="dataset",
        metadata={
            "dataset_class": _name(opts.dataset),
            "data_dir": os.path.abspath(opts.data),
            "train_size": len(dataset_train),
            "test_size": len(dataset_eval),
        },
    )
    run.log_artifact(dataset_artifact)

    loader_train_sample = DataLoader(
        dataset_train,
        batch_sampler=NPairs(
            dataset_train,
            opts.batch,
            m=opts.num_image_per_class,
            iter_per_epoch=opts.iter_per_epoch,
        ),
        pin_memory=True,
        num_workers=8,
    )
    loader_train_eval = DataLoader(
        dataset_train_eval,
        shuffle=False,
        batch_size=opts.batch,
        drop_last=False,
        pin_memory=False,
        num_workers=8,
    )
    loader_eval = DataLoader(
        dataset_eval,
        shuffle=False,
        batch_size=opts.batch,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
    )
    model = LinearEmbedding(
        base_model,
        output_size=base_model.output_size,
        embedding_size=opts.embedding_size,
        normalize=opts.l2normalize == "true",
    ).cuda()

    # Log parameter/gradient flow and model graph for richer W&B debugging.
    if opts.wandb_mode != "disabled":
        wandb.watch(model, log="all", log_graph=True, log_freq=100)

    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        print("Loaded Model from %s" % opts.load)

    criterion = opts.loss(sampler=opts.sample(), margin=opts.margin)
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma
    )

    def train(net, loader, ep):
        lr_scheduler.step()

        net.train()
        loss_all = []
        train_iter = tqdm(loader, ncols=80)
        for images, labels in train_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            loss = criterion(embedding, labels)
            loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_iter.set_description(
                "[Train][Epoch %d] Loss: %.5f" % (ep, loss.item())
            )
        mean_loss = torch.Tensor(loss_all).mean().item()
        print("[Epoch %d] Loss: %.5f\n" % (ep, mean_loss))
        return mean_loss

    def eval(net, loader, ep):
        K = opts.recall
        net.eval()
        test_iter = tqdm(loader, ncols=80)
        embeddings_all, labels_all = [], []

        test_iter.set_description("[Eval][Epoch %d]" % ep)
        with torch.no_grad():
            for images, labels in test_iter:
                images, labels = images.cuda(), labels.cuda()
                embedding = net(images)
                embeddings_all.append(embedding.data)
                labels_all.append(labels.data)

            embeddings_all = torch.cat(embeddings_all).cpu()
            labels_all = torch.cat(labels_all).cpu()
            metrics, _, _ = eval_metrics(embeddings_all, labels_all, K)

            for k in K:
                r = metrics["recall@%d" % k]
                print("[Epoch %d] Recall@%d: [%.4f]\n" % (ep, k, 100 * r))

        return metrics

    def eval_metrics(embeddings, labels, K):
        D = pdist(embeddings, squared=True)
        max_k = max(K)
        knn_inds = D.topk(1 + max_k, dim=1, largest=False, sorted=True)[1][:, 1:]

        selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
        correct_labels = labels.unsqueeze(1) == selected_labels

        metrics = {}
        for k in K:
            metrics["recall@%d" % k] = (
                (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
            )

        pred_top1 = selected_labels[:, 0]
        return metrics, labels, pred_top1

    def maybe_log_confusion(prefix, labels, preds):
        class_vals = sorted(set(labels.tolist()))
        class_count = len(class_vals)
        run.summary["%s_confusion_class_count" % prefix] = class_count

        if class_count > opts.max_confusion_classes:
            run.summary["%s_confusion_logged" % prefix] = False
            run.summary["%s_confusion_reason" % prefix] = "too_many_classes"
            return

        class_to_idx = {c: i for i, c in enumerate(class_vals)}
        y_true = [class_to_idx[int(x)] for x in labels.tolist()]
        y_pred = [class_to_idx[int(x)] for x in preds.tolist()]
        class_names = [str(c) for c in class_vals]

        run.log(
            {
                "%s/confusion_matrix"
                % prefix: wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            }
        )
        run.summary["%s_confusion_logged" % prefix] = True

    if opts.mode == "eval":
        train_metrics = eval(model, loader_train_eval, 0)
        test_metrics = eval(model, loader_eval, 0)

        train_error = 1.0 - train_metrics["recall@%d" % primary_recall_k]
        test_error = 1.0 - test_metrics["recall@%d" % primary_recall_k]

        eval_payload = {"epoch": 0}
        for k in opts.recall:
            eval_payload["eval/train_recall@%d" % k] = train_metrics["recall@%d" % k]
            eval_payload["eval/test_recall@%d" % k] = test_metrics["recall@%d" % k]
        eval_payload["eval/train_error@%d" % primary_recall_k] = train_error
        eval_payload["eval/test_error@%d" % primary_recall_k] = test_error
        run.log(eval_payload, step=0)

        model.eval()
        with torch.no_grad():
            emb_eval, labels_eval = [], []
            for images, labels in loader_eval:
                images = images.cuda()
                emb_eval.append(model(images).data.cpu())
                labels_eval.append(labels)
            emb_eval = torch.cat(emb_eval)
            labels_eval = torch.cat(labels_eval)
        _, cm_labels, cm_preds = eval_metrics(emb_eval, labels_eval, opts.recall)
        maybe_log_confusion("eval_test", cm_labels, cm_preds)

        for k in opts.recall:
            run.summary["final_train_recall@%d" % k] = train_metrics["recall@%d" % k]
            run.summary["final_test_recall@%d" % k] = test_metrics["recall@%d" % k]
        run.log(
            {
                "charts/train_val_error_curve": wandb.plot.line_series(
                    xs=[0],
                    ys=[[train_error], [test_error]],
                    keys=[
                        "train_error@%d" % primary_recall_k,
                        "val_error@%d" % primary_recall_k,
                    ],
                    title="Train vs Validation Error (Recall@%d)" % primary_recall_k,
                    xname="epoch",
                )
            }
        )
    else:
        history_rows = []
        history_table = wandb.Table(
            columns=["epoch", "train_loss", "lr", "best_recall@1"]
            + ["train_recall@%d" % k for k in opts.recall]
            + ["test_recall@%d" % k for k in opts.recall]
        )
        error_epochs = []
        train_errors = []
        val_errors = []

        train_metrics = eval(model, loader_train_eval, 0)
        val_metrics = eval(model, loader_eval, 0)
        best_rec = val_metrics["recall@%d" % primary_recall_k]

        log_payload = {
            "epoch": 0,
            "best_recall@%d" % primary_recall_k: best_rec,
            "best_error@%d" % primary_recall_k: 1.0 - best_rec,
            "eval/train_error@%d" % primary_recall_k: 1.0
            - train_metrics["recall@%d" % primary_recall_k],
            "eval/test_error@%d" % primary_recall_k: 1.0
            - val_metrics["recall@%d" % primary_recall_k],
            "lr": optimizer.param_groups[0]["lr"],
        }
        for k in opts.recall:
            log_payload["eval/train_recall@%d" % k] = train_metrics["recall@%d" % k]
            log_payload["eval/test_recall@%d" % k] = val_metrics["recall@%d" % k]
        run.log(log_payload, step=0)

        row = {
            "epoch": 0,
            "train_loss": float("nan"),
            "lr": optimizer.param_groups[0]["lr"],
            "best_recall@1": best_rec,
        }
        for k in opts.recall:
            row["train_recall@%d" % k] = train_metrics["recall@%d" % k]
            row["test_recall@%d" % k] = val_metrics["recall@%d" % k]
        history_rows.append(row)
        history_table.add_data(*[row[c] for c in history_table.columns])
        error_epochs.append(0)
        train_errors.append(1.0 - train_metrics["recall@%d" % primary_recall_k])
        val_errors.append(1.0 - val_metrics["recall@%d" % primary_recall_k])

        for epoch in range(1, opts.epochs + 1):
            train_loss = train(model, loader_train_sample, epoch)
            train_metrics = eval(model, loader_train_eval, epoch)
            val_metrics = eval(model, loader_eval, epoch)
            val_recall1 = val_metrics["recall@%d" % primary_recall_k]

            log_payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "best_recall@%d" % primary_recall_k: best_rec,
                "best_error@%d" % primary_recall_k: 1.0 - best_rec,
                "eval/train_error@%d" % primary_recall_k: 1.0
                - train_metrics["recall@%d" % primary_recall_k],
                "eval/test_error@%d" % primary_recall_k: 1.0
                - val_metrics["recall@%d" % primary_recall_k],
                "lr": optimizer.param_groups[0]["lr"],
            }
            for k in opts.recall:
                log_payload["eval/train_recall@%d" % k] = train_metrics["recall@%d" % k]
                log_payload["eval/test_recall@%d" % k] = val_metrics["recall@%d" % k]
            run.log(log_payload, step=epoch)

            if best_rec < val_recall1:
                best_rec = val_recall1
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(
                        model.state_dict(), "%s/%s" % (opts.save_dir, "best.pth")
                    )
                    best_path = "%s/%s" % (opts.save_dir, "best.pth")
                    best_artifact = wandb.Artifact(
                        name="model-best-%s-%s" % (_name(opts.base).lower(), opts.seed),
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "best_recall@%d" % primary_recall_k: best_rec,
                        },
                    )
                    best_artifact.add_file(best_path)
                    run.log_artifact(
                        best_artifact,
                        aliases=[
                            "best",
                            "epoch-%d" % epoch,
                            "dataset-%s" % _alias_safe(_name(opts.dataset)),
                            "group-%s" % _alias_safe(opts.wandb_group),
                            "job-%s" % _alias_safe(opts.wandb_job_type),
                        ],
                    )
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
                last_path = "%s/%s" % (opts.save_dir, "last.pth")
                last_artifact = wandb.Artifact(
                    name="model-last-%s-%s" % (_name(opts.base).lower(), opts.seed),
                    type="model",
                    metadata={
                        "epoch": epoch,
                        "final_recall@%d" % primary_recall_k: val_recall1,
                    },
                )
                last_artifact.add_file(last_path)
                run.log_artifact(
                    last_artifact,
                    aliases=[
                        "last",
                        "epoch-%d" % epoch,
                        "dataset-%s" % _alias_safe(_name(opts.dataset)),
                        "group-%s" % _alias_safe(opts.wandb_group),
                        "job-%s" % _alias_safe(opts.wandb_job_type),
                    ],
                )
                with open("%s/result.txt" % opts.save_dir, "w") as f:
                    f.write("Best Recall@1: %.4f\n" % (best_rec * 100))
                    f.write("Final Recall@1: %.4f\n" % (val_recall1 * 100))

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "best_recall@1": best_rec,
            }
            for k in opts.recall:
                row["train_recall@%d" % k] = train_metrics["recall@%d" % k]
                row["test_recall@%d" % k] = val_metrics["recall@%d" % k]
            history_rows.append(row)
            history_table.add_data(*[row[c] for c in history_table.columns])
            error_epochs.append(epoch)
            train_errors.append(1.0 - train_metrics["recall@%d" % primary_recall_k])
            val_errors.append(1.0 - val_metrics["recall@%d" % primary_recall_k])

            print("Best Recall@1: %.4f" % best_rec)

        run.log({"artifacts/epoch_history_table": history_table})
        run.log(
            {
                "charts/train_val_error_curve": wandb.plot.line_series(
                    xs=error_epochs,
                    ys=[train_errors, val_errors],
                    keys=[
                        "train_error@%d" % primary_recall_k,
                        "val_error@%d" % primary_recall_k,
                    ],
                    title="Train vs Validation Error (Recall@%d)" % primary_recall_k,
                    xname="epoch",
                )
            }
        )

        history_dir = opts.save_dir if opts.save_dir is not None else "."
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        history_csv_path = os.path.join(history_dir, "teacher_epoch_history.csv")
        with open(history_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_table.columns)
            writer.writeheader()
            for r in history_rows:
                writer.writerow(r)

        history_artifact = wandb.Artifact(
            name="teacher-history-%s-%s" % (_name(opts.base).lower(), opts.seed),
            type="metrics",
            metadata={"rows": len(history_rows), "recall_k": opts.recall},
        )
        history_artifact.add_file(history_csv_path)
        run.log_artifact(history_artifact, aliases=["latest"])

        model.eval()
        with torch.no_grad():
            emb_eval, labels_eval = [], []
            for images, labels in loader_eval:
                images = images.cuda()
                emb_eval.append(model(images).data.cpu())
                labels_eval.append(labels)
            emb_eval = torch.cat(emb_eval)
            labels_eval = torch.cat(labels_eval)
        _, cm_labels, cm_preds = eval_metrics(emb_eval, labels_eval, opts.recall)
        maybe_log_confusion("final_test", cm_labels, cm_preds)

        run.summary["best_recall@%d" % primary_recall_k] = best_rec
        run.summary["final_recall@%d" % primary_recall_k] = history_rows[-1][
            "test_recall@%d" % primary_recall_k
        ]

    run.finish()
    return run


def run_with_cli_args(cli_args):
    opts = build_parser().parse_args([str(a) for a in cli_args])
    return run_experiment(opts)


def run_with_params(params):
    return run_with_cli_args(_params_to_cli_args(params))


def main(argv=None):
    opts = build_parser().parse_args(argv)
    run_experiment(opts)


if __name__ == "__main__":
    main()
