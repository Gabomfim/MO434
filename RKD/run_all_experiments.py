"""Roda TODA a campanha de experimentos, em ordem de dependência, logando tudo
no W&B. Reaproveita os entrypoints existentes (run_with_params / main).

Grade completa:
  Fase 0a  professores      : datasets × teachers           (finetune_classifier)
  Fase 0b  baseline CE      : datasets                      (train_convnextmicro)
  Fase 1   baselines clássicos: datasets × teachers × {hinton, rkd_dist, rkd_angle}
                                                            (distill_to_convnextmicro)
  Fase 2   Graph-RKD        : datasets × teachers × embeddings × objectives,
           com busca de N (varredura log)                  (run_graph_rkd_search)

Dependências: Fases 1 e 2 usam o checkpoint do professor (best.pth) produzido na
Fase 0a. O baseline CE (Fase 0b) isola o ganho da destilação. Todos os modelos
e métricas vão para o W&B.

Loss da destilação Graph-RKD: SÓ cross-entropy + a loss de grafo (sem KD/RKD/AT).
Baselines clássicos: cross-entropy + UMA técnica clássica por vez.

Use --dry_run para ver o plano e a contagem sem treinar.
"""

import argparse
import os

import finetune_classifier as finetune
import train_convnextmicro as baseline_ce
import distill_to_convnextmicro as distill
import run_graph_rkd_search as gsearch
from experiment_ledger import is_done, mark_done
from graph_rkd import find_best_n, log_spaced_orders
from wandb_artifacts import stable_run_id

# baseline clássico -> (flag de razão, valor) ; demais razões relacionais = 0
CLASSIC = {
    "hinton":    ("kd_ratio", 0.9),     # Hinton KD (T=4 default)
    "rkd_dist":  ("dist_ratio", 25.0),  # RKD distance (Park et al.)
    "rkd_angle": ("angle_ratio", 50.0),  # RKD angle (Park et al.)
}


def build_parser():
    p = argparse.ArgumentParser(description="Roda toda a campanha de experimentos")
    p.add_argument("--data", default="data")
    p.add_argument("--save_root", default="experiments")

    # grade
    p.add_argument("--datasets", nargs="+", default=["cars196", "cub200"])
    p.add_argument("--teachers", nargs="+", default=["resnet18", "convnext_tiny"])
    p.add_argument("--embeddings", nargs="+", default=["profile", "mds"])
    p.add_argument("--objectives", nargs="+", default=["regression", "contrastive"])
    p.add_argument("--phases", nargs="+",
                   default=["teachers", "ce_baseline", "classic", "graph"],
                   choices=["teachers", "ce_baseline", "classic", "graph"])

    # épocas / batch
    p.add_argument("--finetune_epochs", type=int, default=60)
    p.add_argument("--student_epochs", type=int, default=300)   # final (longa)
    p.add_argument("--search_epochs", type=int, default=30)     # busca de N (curta)
    p.add_argument("--batch_teacher", type=int, default=64)
    p.add_argument("--batch_student", type=int, default=128)

    # busca de N
    p.add_argument("--edge_budget", type=float, default=1024.0)
    p.add_argument("--seeds", type=int, default=3,
                   help="seeds por candidato N na busca (R)")
    p.add_argument("--select", choices=["argmax", "1se"], default="argmax")
    p.add_argument("--graph_warmup_frac", type=float, default=0.1,
                   help="warmup do peso da loss de grafo (fração das épocas); "
                        "0 = desliga. Balanceia CE vs grafo no início do treino.")

    # runtime / W&B
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teachers_project", default="classifier-finetune")
    p.add_argument("--students_project", default="convnextmicro-distill")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--dry_run", action="store_true",
                   help="imprime o plano e a contagem, sem treinar")
    return p


def _teacher_ckpt(opts, arch, ds):
    return os.path.join(opts.save_root, "teachers", f"{arch}-{ds}", "best.pth")


def _run(label, save_dir, fn, dry):
    """Roda uma etapa, pulando se já concluída (ledger). Resumível."""
    if is_done(save_dir):
        print(f">> SKIP (concluído): {label}")
        return
    print(f">> {label}")
    if dry:
        return
    result = fn()
    mark_done(save_dir, {"label": label})


def phase_teachers(opts, dry):
    print("\n===== Fase 0a: professores (fine-tune) =====")
    for ds in opts.datasets:
        for arch in opts.teachers:
            save_dir = os.path.join(opts.save_root, "teachers", f"{arch}-{ds}")
            _run(f"finetune {arch} @ {ds}", save_dir,
                 lambda a=arch, d=ds, s=save_dir: finetune.run_with_params({
                "arch": a, "dataset": d, "data": opts.data,
                "epochs": opts.finetune_epochs, "batch": opts.batch_teacher,
                "amp": opts.amp, "seed": opts.seed, "save_dir": s,
                "resume": os.path.join(s, "last.pth"),
                "wandb_project": opts.teachers_project, "wandb_entity": opts.wandb_entity,
                "wandb_mode": opts.wandb_mode, "wandb_run_name": f"{a}-{d}",
                "wandb_id": stable_run_id(f"finetune-{a}-{d}"),
            }), dry)


def phase_ce_baseline(opts, dry):
    print("\n===== Fase 0b: baseline ConvNextMicro (só CE, do zero) =====")
    for ds in opts.datasets:
        save_dir = os.path.join(opts.save_root, "baseline_ce", ds)
        _run(f"baseline-CE @ {ds}", save_dir,
             lambda d=ds, s=save_dir: baseline_ce.run_with_params({
            "dataset": d, "data": opts.data, "epochs": opts.student_epochs,
            "batch": opts.batch_student, "amp": opts.amp, "seed": opts.seed,
            "save_dir": s, "resume": os.path.join(s, "baseline_last.pth"),
            "wandb_project": opts.students_project,
            "wandb_entity": opts.wandb_entity, "wandb_mode": opts.wandb_mode,
            "wandb_run_name": f"baseline-{d}", "wandb_group": f"baseline-{d}",
            "wandb_id": stable_run_id(f"baseline-{d}"),
            "wandb_tags": ["baseline", "ce-only", d],
        }), dry)


def phase_classic(opts, dry):
    print("\n===== Fase 1: baselines clássicos (Hinton / RKD-dist / RKD-angle) =====")
    for ds in opts.datasets:
        for arch in opts.teachers:
            tck = _teacher_ckpt(opts, arch, ds)
            for name, (ratio_key, ratio_val) in CLASSIC.items():
                ratios = {"ce_ratio": 1.0, "kd_ratio": 0.0, "dist_ratio": 0.0,
                          "angle_ratio": 0.0, "at_ratio": 0.0}
                ratios[ratio_key] = ratio_val
                save_dir = os.path.join(opts.save_root, "classic", f"{name}-{arch}-{ds}")
                _run(f"classic {name} ({arch} -> micro) @ {ds}", save_dir,
                     lambda a=arch, d=ds, s=save_dir, r=dict(ratios), nm=name, t=tck:
                     distill.run_with_params({
                         "dataset": d, "data": opts.data, "teacher_arch": a,
                         "teacher_load": t, "graph_rkd_mode": "off",
                         "epochs": opts.student_epochs, "batch": opts.batch_student,
                         "amp": opts.amp, "seed": opts.seed, "save_dir": s,
                         "resume": os.path.join(s, "student_last.pth"),
                         "wandb_project": opts.students_project,
                         "wandb_entity": opts.wandb_entity, "wandb_mode": opts.wandb_mode,
                         "wandb_run_name": f"classic-{nm}-{a}-{d}",
                         "wandb_group": f"classic-{a}-{d}",
                         "wandb_id": stable_run_id(f"classic-{nm}-{a}-{d}"),
                         "wandb_tags": ["classic", nm, a, d], **r}), dry)


def phase_graph(opts, dry):
    print("\n===== Fase 2: Graph-RKD (busca de N por varredura log) =====")
    for ds in opts.datasets:
        for arch in opts.teachers:
            tck = _teacher_ckpt(opts, arch, ds)
            block_dir = os.path.join(opts.save_root, "graph", f"{arch}-{ds}")
            args = [
                "--dataset", ds, "--teacher_arch", arch, "--teacher_load", tck,
                "--data", opts.data, "--batch", str(opts.batch_student),
                "--edge_budget", str(opts.edge_budget),
                "--search_epochs", str(opts.search_epochs),
                "--final_epochs", str(opts.student_epochs),
                "--seeds", str(opts.seeds), "--select", opts.select,
                "--graph_warmup_frac", str(opts.graph_warmup_frac),
                "--seed", str(opts.seed),
                "--modes", *opts.objectives, "--methods", *opts.embeddings,
                "--save_root", block_dir,
                "--wandb_project", opts.students_project,
                "--wandb_mode", opts.wandb_mode,
            ]
            if opts.wandb_entity:
                args += ["--wandb_entity", opts.wandb_entity]
            if opts.amp:
                args += ["--amp"]
            # Marca o bloco só ao concluir TUDO; num crash no meio, gsearch
            # re-entra e pula os sub-runs já cacheados (result.json).
            _run(f"graph-search ({arch} -> micro) @ {ds} "
                 f"[{len(opts.objectives)}x{len(opts.embeddings)} combos]",
                 block_dir, lambda a=list(args): gsearch.main(a), dry)


def _print_plan(opts):
    D, T = len(opts.datasets), len(opts.teachers)
    Em, O, R = len(opts.embeddings), len(opts.objectives), opts.seeds
    n_max = find_best_n(opts.batch_student, edge_budget=opts.edge_budget) or 2
    C = len(log_spaced_orders(2, n_max))
    counts = {
        "teachers": D * T,
        "ce_baseline": D,
        "classic": D * T * 3,
        "graph": D * T * Em * O * (C * R + 1),
    }
    total = sum(counts[p] for p in opts.phases)
    print("\n================= PLANO =================")
    print(f"datasets={opts.datasets} teachers={opts.teachers}")
    print(f"embeddings={opts.embeddings} objectives={opts.objectives} seeds={R}")
    print(f"N_max(orçamento={opts.edge_budget}, K={opts.batch_student})={n_max} "
          f"-> {C} candidatos log: {log_spaced_orders(2, n_max)}")
    for ph in opts.phases:
        print(f"  fase {ph:12s}: {counts[ph]} experimentos")
    print(f"  TOTAL ({'+'.join(opts.phases)}): {total} experimentos")
    print("W&B: professores -> %s | alunos -> %s | mode=%s"
          % (opts.teachers_project, opts.students_project, opts.wandb_mode))


def main(argv=None):
    opts = build_parser().parse_args(argv)
    _print_plan(opts)
    dry = opts.dry_run
    if dry:
        print("\n(dry-run: nada será treinado)\n")
    if "teachers" in opts.phases:
        phase_teachers(opts, dry)
    if "ce_baseline" in opts.phases:
        phase_ce_baseline(opts, dry)
    if "classic" in opts.phases:
        phase_classic(opts, dry)
    if "graph" in opts.phases:
        phase_graph(opts, dry)
    print("\nCampanha %s." % ("planejada (dry-run)" if dry else "concluída"))


if __name__ == "__main__":
    main()
