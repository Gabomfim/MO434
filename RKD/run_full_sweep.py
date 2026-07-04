"""Sweep COMPLETO de N em 120 épocas (budget cheio), para responder 'melhor N'
de forma robusta — todos os N no mesmo orçamento, não só o best_n.

Grade: dataset {cub200, cars196} × teacher {resnet18, convnext_tiny}
       × method {profile, mds} × N {2,4,8,16} × seed {0,1,2}, mode=regression.

Ordem: seed-outer, cub200-first (mais rápido) → em ~12h fecha o sweep de 1 seed
do cub200 (resposta cedo); seeds 2/3 preenchem depois. Resumível: run_one usa o
ledger (skip do que terminou) + resume=student_last.pth. Um run que falha é logado
e re-tentado no próximo passe (não trava o resto).

Uso:
  WANDB_ENTITY=rodz-ralm-v-ai python run_full_sweep.py
  python run_full_sweep.py --datasets cub200 --seeds 1        # subconjunto p/ testar
"""
import argparse, itertools, time, traceback

import run_metric_search as rms

TEACHER_CKPT = "experiments_metric/teachers/{teacher}-{dataset}/best.pth"
SAVE_ROOT    = "experiments_metric/sweep120/{teacher}-{dataset}"


def build_opts(dataset, teacher, entity):
    """opts idêntico à campanha (batch 128, amp, edge_budget 1024, mapr, ...)."""
    argv = [
        "--dataset", dataset, "--teacher_arch", teacher,
        "--teacher_load", TEACHER_CKPT.format(teacher=teacher, dataset=dataset),
        "--data", "../data", "--batch", "128", "--edge_budget", "1024",
        "--final_epochs", "120", "--rel_warmup_frac", "0.1",
        "--select_metric", "mapr", "--recall", "1", "2", "4", "8",
        "--modes", "regression", "--methods", "profile", "mds",
        "--save_root", SAVE_ROOT.format(teacher=teacher, dataset=dataset),
        "--wandb_project", "convnextmicro-metric-distill",
        "--wandb_mode", "online", "--amp",
    ]
    if entity:
        argv += ["--wandb_entity", entity]
    return rms.build_parser().parse_args(argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cub200", "cars196"])
    ap.add_argument("--teachers", nargs="+", default=["resnet18", "convnext_tiny"])
    ap.add_argument("--methods",  nargs="+", default=["profile", "mds"])
    ap.add_argument("--Ns", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--tag", default="full", help="tag do run (use 'smoke' p/ teste)")
    ap.add_argument("--entity", default=None)  # senão usa WANDB_ENTITY do ambiente
    args = ap.parse_args()

    import os
    entity = args.entity or os.environ.get("WANDB_ENTITY")

    grid = [(seed, ds, te, me, N)
            for seed in range(args.seeds)          # seed-outer: 1 seed completo primeiro
            for ds in args.datasets                # cub200 primeiro (mais rápido)
            for te in args.teachers
            for me in args.methods
            for N in args.Ns]
    total = len(grid)
    print(f"[sweep] {total} runs @ {args.epochs}ép. ordem: seed→dataset→teacher→method→N", flush=True)

    opts_cache = {}
    done = fail = 0
    t0 = time.time()
    for i, (seed, ds, te, me, N) in enumerate(grid, 1):
        key = (ds, te)
        opts = opts_cache.get(key) or opts_cache.setdefault(key, build_opts(ds, te, entity))
        tag = args.tag
        label = f"{ds}/{te}/{me}/N{N}/s{seed}"
        print(f"\n[sweep {i}/{total}] {label}  (elapsed {(time.time()-t0)/3600:.1f}h)", flush=True)
        try:
            val = rms.run_one(opts, "regression", me, N, args.epochs, tag, seed)
            done += 1
            print(f"[sweep {i}/{total}] OK {label} val_mapr={val:.4f}", flush=True)
        except Exception:
            fail += 1
            print(f"[sweep {i}/{total}] FALHOU {label}:\n{traceback.format_exc()}", flush=True)
    print(f"\n[sweep] fim. ok={done} falhas={fail} de {total} em {(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
