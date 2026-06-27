"""Campanha completa de METRIC LEARNING (retrieval, recall@K), logando no W&B.
Resumível (ledger). Substitui a campanha de classificação.

Fases (em ordem de dependência):
  0a teachers : datasets × teachers   -> finetune_metric (embedding + triplet)
  0b baseline : datasets              -> train_metric_baseline (triplet puro)
  1  clássicos: datasets × teachers × {rkd_dist, rkd_angle}  (triplet + RKD,
               com warmup balanceando o triplet)              -> distill_metric
  2  Graph-RKD: datasets × teachers × embeddings × objectives, com busca de N
               (triplet + loss de grafo, warmup)              -> run_metric_search

Use --dry_run para ver o plano. --phases seleciona subconjuntos.
"""

import argparse
import os

import finetune_metric as finetune
import train_metric_baseline as baseline
import distill_metric as distill
import run_metric_search as msearch
from experiment_ledger import is_done, mark_done
from graph_rkd import find_best_n, log_spaced_orders
from wandb_artifacts import stable_run_id

CLASSIC = {"rkd_dist": ("dist_ratio", 25.0), "rkd_angle": ("angle_ratio", 50.0)}


def build_parser():
    p = argparse.ArgumentParser(description="Campanha completa de metric learning")
    p.add_argument("--data", default="data")
    p.add_argument("--save_root", default="experiments_metric")
    p.add_argument("--datasets", nargs="+", default=["cars196", "cub200"])
    p.add_argument("--teachers", nargs="+", default=["resnet18", "convnext_tiny"])
    p.add_argument("--embeddings", nargs="+", default=["profile", "mds"])
    p.add_argument("--objectives", nargs="+", default=["regression", "contrastive"])
    p.add_argument("--phases", nargs="+",
                   default=["teachers", "baseline", "classic", "graph"],
                   choices=["teachers", "baseline", "classic", "graph"])

    p.add_argument("--teacher_epochs", type=int, default=60)
    p.add_argument("--student_epochs", type=int, default=120)
    p.add_argument("--search_epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--edge_budget", type=float, default=1024.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--select", choices=["argmax", "1se"], default="argmax")
    p.add_argument("--select_metric", default="mapr",
                   help="métrica de validação p/ seleção (default mAP@R)")
    p.add_argument("--rel_warmup_frac", type=float, default=0.1)
    p.add_argument("--recall", type=int, nargs="+", default=[1, 2, 4, 8])

    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teachers_project", default="metric-teacher-finetune")
    p.add_argument("--students_project", default="convnextmicro-metric-distill")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--dry_run", action="store_true")
    return p


def _teacher_ckpt(opts, arch, ds):
    return os.path.join(opts.save_root, "teachers", f"{arch}-{ds}", "best.pth")


def _run(label, save_dir, fn, dry):
    if is_done(save_dir):
        print(f">> SKIP (concluído): {label}")
        return
    print(f">> {label}")
    if dry:
        return
    fn()
    mark_done(save_dir, {"label": label})


def phase_teachers(opts, dry):
    print("\n===== Fase 0a: teachers métricos (embedding + triplet) =====")
    for ds in opts.datasets:
        for arch in opts.teachers:
            s = os.path.join(opts.save_root, "teachers", f"{arch}-{ds}")
            _run(f"metric teacher {arch} @ {ds}", s,
                 lambda a=arch, d=ds, sd=s: finetune.run_with_params({
                     "arch": a, "dataset": d, "data": opts.data,
                     "epochs": opts.teacher_epochs, "batch": opts.batch,
                     "recall": opts.recall, "select_metric": opts.select_metric,
                     "amp": opts.amp, "seed": opts.seed,
                     "save_dir": sd, "resume": os.path.join(sd, "last.pth"),
                     "wandb_project": opts.teachers_project,
                     "wandb_entity": opts.wandb_entity, "wandb_mode": opts.wandb_mode,
                     "wandb_run_name": f"metric-{a}-{d}",
                     "wandb_id": stable_run_id(f"metric-teacher-{a}-{d}")}), dry)


def phase_baseline(opts, dry):
    print("\n===== Fase 0b: baseline métrico ConvNextMicro (triplet puro) =====")
    for ds in opts.datasets:
        s = os.path.join(opts.save_root, "baseline", ds)
        _run(f"metric baseline @ {ds}", s,
             lambda d=ds, sd=s: baseline.run_with_params({
                 "dataset": d, "data": opts.data, "epochs": opts.student_epochs,
                 "batch": opts.batch, "recall": opts.recall,
                 "select_metric": opts.select_metric, "amp": opts.amp,
                 "seed": opts.seed, "save_dir": sd,
                 "resume": os.path.join(sd, "baseline_last.pth"),
                 "wandb_project": opts.students_project,
                 "wandb_entity": opts.wandb_entity, "wandb_mode": opts.wandb_mode,
                 "wandb_run_name": f"baseline-{d}", "wandb_group": f"baseline-{d}",
                 "wandb_id": stable_run_id(f"metric-baseline-{d}"),
                 "wandb_tags": ["baseline", "metric", d]}), dry)


def phase_classic(opts, dry):
    print("\n===== Fase 1: baselines clássicos (RKD-dist / RKD-angle, triplet+warmup) =====")
    for ds in opts.datasets:
        for arch in opts.teachers:
            tck = _teacher_ckpt(opts, arch, ds)
            for name, (ratio_key, ratio_val) in CLASSIC.items():
                ratios = {"dist_ratio": 0.0, "angle_ratio": 0.0}
                ratios[ratio_key] = ratio_val
                s = os.path.join(opts.save_root, "classic", f"{name}-{arch}-{ds}")
                _run(f"classic {name} ({arch}->micro) @ {ds}", s,
                     lambda a=arch, d=ds, sd=s, r=dict(ratios), nm=name, t=tck:
                     distill.run_with_params({
                         "dataset": d, "data": opts.data, "teacher_arch": a,
                         "teacher_load": t, "graph_rkd_mode": "off",
                         "triplet_ratio": 1.0, "rel_warmup_frac": opts.rel_warmup_frac,
                         "epochs": opts.student_epochs, "batch": opts.batch,
                         "recall": opts.recall, "select_metric": opts.select_metric,
                         "amp": opts.amp, "seed": opts.seed, "save_dir": sd, "resume": os.path.join(sd, "student_last.pth"),
                         "wandb_project": opts.students_project,
                         "wandb_entity": opts.wandb_entity, "wandb_mode": opts.wandb_mode,
                         "wandb_run_name": f"classic-{nm}-{a}-{d}",
                         "wandb_group": f"classic-{a}-{d}",
                         "wandb_id": stable_run_id(f"metric-classic-{nm}-{a}-{d}"),
                         "wandb_tags": ["classic", "metric", nm, a, d], **r}), dry)


def phase_graph(opts, dry):
    print("\n===== Fase 2: Graph-RKD métrico (busca de N) =====")
    for ds in opts.datasets:
        for arch in opts.teachers:
            tck = _teacher_ckpt(opts, arch, ds)
            block = os.path.join(opts.save_root, "graph", f"{arch}-{ds}")
            args = [
                "--dataset", ds, "--teacher_arch", arch, "--teacher_load", tck,
                "--data", opts.data, "--batch", str(opts.batch),
                "--edge_budget", str(opts.edge_budget),
                "--search_epochs", str(opts.search_epochs),
                "--final_epochs", str(opts.student_epochs),
                "--seeds", str(opts.seeds), "--select", opts.select,
                "--select_metric", opts.select_metric,
                "--rel_warmup_frac", str(opts.rel_warmup_frac), "--seed", str(opts.seed),
                "--recall", *[str(k) for k in opts.recall],
                "--modes", *opts.objectives, "--methods", *opts.embeddings,
                "--save_root", block, "--wandb_project", opts.students_project,
                "--wandb_mode", opts.wandb_mode,
            ]
            if opts.wandb_entity:
                args += ["--wandb_entity", opts.wandb_entity]
            if opts.amp:
                args += ["--amp"]
            _run(f"graph-search ({arch}->micro) @ {ds} "
                 f"[{len(opts.objectives)}x{len(opts.embeddings)}]",
                 block, lambda a=list(args): msearch.main(a), dry)


def _print_plan(opts):
    D, T = len(opts.datasets), len(opts.teachers)
    Em, O, R = len(opts.embeddings), len(opts.objectives), opts.seeds
    n_max = find_best_n(opts.batch, edge_budget=opts.edge_budget) or 2
    C = len(log_spaced_orders(2, n_max))
    counts = {"teachers": D * T, "baseline": D, "classic": D * T * 2,
              "graph": D * T * Em * O * (C * R + 1)}
    print("\n================= PLANO (metric learning) =================")
    print(f"datasets={opts.datasets} teachers={opts.teachers}")
    print(f"embeddings={opts.embeddings} objectives={opts.objectives} seeds={R}")
    print(f"N_max={n_max} -> {C} candidatos log {log_spaced_orders(2, n_max)}")
    for ph in opts.phases:
        print(f"  fase {ph:10s}: {counts[ph]} experimentos")
    print(f"  TOTAL: {sum(counts[p] for p in opts.phases)} experimentos")
    print("W&B: teachers -> %s | alunos -> %s | recall@%s | mode=%s"
          % (opts.teachers_project, opts.students_project, opts.recall, opts.wandb_mode))


def main(argv=None):
    opts = build_parser().parse_args(argv)
    _print_plan(opts)
    dry = opts.dry_run
    if dry:
        print("\n(dry-run: nada será treinado)\n")
    if "teachers" in opts.phases:
        phase_teachers(opts, dry)
    if "baseline" in opts.phases:
        phase_baseline(opts, dry)
    if "classic" in opts.phases:
        phase_classic(opts, dry)
    if "graph" in opts.phases:
        phase_graph(opts, dry)
    print("\nCampanha métrica %s." % ("planejada (dry-run)" if dry else "concluída"))


if __name__ == "__main__":
    main()
