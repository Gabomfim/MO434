"""Busca de N (varredura log) para a destilação MÉTRICA, por (modo × método),
selecionando pelo melhor recall@1 de VALIDAÇÃO. Espelha run_graph_rkd_search,
mas chama distill_metric e usa recall (não top-1). Resumível (ledger).
"""

import argparse
import statistics

import distill_metric as distill
from experiment_ledger import load_result, mark_done
from graph_rkd import find_best_n, log_spaced_orders, select_order
from wandb_artifacts import stable_run_id


def build_parser():
    p = argparse.ArgumentParser(description="Busca de N (metric distill)")
    p.add_argument("--dataset", choices=["cars196", "cub200"], default="cub200")
    p.add_argument("--data", default="data")
    p.add_argument("--teacher_arch", choices=["resnet18", "convnext_tiny"],
                   default="resnet18")
    p.add_argument("--teacher_load", default=None)
    p.add_argument("--teacher_artifact", default=None)
    p.add_argument("--batch", type=int, default=128)

    p.add_argument("--modes", nargs="+", default=["regression", "contrastive"],
                   choices=["regression", "contrastive"])
    p.add_argument("--methods", nargs="+", default=["profile", "mds"],
                   choices=["profile", "mds"])

    p.add_argument("--edge_budget", type=float, default=1024.0)
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--base", type=int, default=2)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--select", choices=["argmax", "1se"], default="argmax")
    p.add_argument("--graph_rkd_ratio", type=float, default=None)
    p.add_argument("--graph_rkd_sampling", choices=["partition", "random", "log"],
                   default="log")
    p.add_argument("--graph_rkd_alpha", type=float, default=0.5)
    p.add_argument("--graph_rkd_gmax", type=int, default=64)
    p.add_argument("--rel_warmup_frac", type=float, default=0.1)

    p.add_argument("--triplet_ratio", type=float, default=1.0)
    p.add_argument("--search_epochs", type=int, default=30)
    p.add_argument("--final_epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_root", default="metric_runs")
    p.add_argument("--recall", type=int, nargs="+", default=[1, 2, 4, 8])

    p.add_argument("--wandb_project", default="convnextmicro-metric-distill")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    return p


def _default_ratio(mode):
    return 1000.0 if mode == "regression" else 1.0


def run_one(opts, mode, method, n_nodes, epochs, tag, seed):
    save_dir = f"{opts.save_root}/{mode}-{method}/N{n_nodes}-{tag}-s{seed}"
    cached = load_result(save_dir)
    if cached is not None:
        print(f"   [skip|cache] {mode}/{method} N={n_nodes} {tag} s{seed} "
              f"-> val r@1={cached['best_val_recall1']*100:.2f}")
        return float(cached["best_val_recall1"])
    ratio = opts.graph_rkd_ratio if opts.graph_rkd_ratio is not None \
        else _default_ratio(mode)
    name = f"{opts.dataset}-{opts.teacher_arch}-{mode}-{method}-N{n_nodes}-{tag}-s{seed}"
    params = {
        "dataset": opts.dataset, "data": opts.data, "teacher_arch": opts.teacher_arch,
        "teacher_load": opts.teacher_load, "teacher_artifact": opts.teacher_artifact,
        "batch": opts.batch, "epochs": epochs, "lr": opts.lr, "seed": seed,
        "recall": opts.recall, "triplet_ratio": opts.triplet_ratio,
        "dist_ratio": 0.0, "angle_ratio": 0.0, "rel_warmup_frac": opts.rel_warmup_frac,
        "graph_rkd_mode": mode, "graph_rkd_method": method,
        "graph_rkd_nodes": n_nodes, "graph_rkd_ratio": ratio,
        "graph_rkd_sampling": opts.graph_rkd_sampling,
        "graph_rkd_alpha": opts.graph_rkd_alpha, "graph_rkd_gmax": opts.graph_rkd_gmax,
        "amp": opts.amp, "save_dir": save_dir, "resume": f"{save_dir}/student_last.pth",
        "wandb_project": opts.wandb_project, "wandb_entity": opts.wandb_entity,
        "wandb_mode": opts.wandb_mode, "wandb_group": f"{mode}-{method}-{opts.dataset}",
        "wandb_run_name": name, "wandb_id": stable_run_id(name),
        "wandb_tags": ["graph-rkd", "metric", mode, method, tag, f"N{n_nodes}"],
    }
    res = distill.run_with_params(params)
    val = float(res["best_val_recall1"])
    mark_done(save_dir, {"best_val_recall1": val})
    return val


def search_for(opts, mode, method):
    n_max = find_best_n(opts.batch, edge_budget=opts.edge_budget, n_min=opts.n_min) \
        or opts.n_min
    candidates = log_spaced_orders(opts.n_min, n_max, base=opts.base)
    print(f"\n### {mode}/{method}: N em {candidates} (teto={n_max}; {opts.seeds} seed/cand) ###")
    means, sems = [], []
    for N in candidates:
        vals = [run_one(opts, mode, method, N, opts.search_epochs, "search", s)
                for s in range(opts.seeds)]
        means.append(statistics.fmean(vals))
        sems.append((statistics.pstdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0)
        print(f"   [{mode}/{method}] N={N} -> val r@1={means[-1]*100:.2f} (+/-{sems[-1]*100:.2f})")
    best_n = select_order(candidates, means, sems, rule=opts.select)
    final_val = run_one(opts, mode, method, best_n, opts.final_epochs, "final", opts.seed)
    return {"best_N": best_n, "n_max": n_max, "candidates": candidates,
            "search_val_at_best": means[candidates.index(best_n)], "final_val": final_val}


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if (opts.teacher_load is None) == (opts.teacher_artifact is None):
        raise SystemExit("Forneça exatamente um de --teacher_load / --teacher_artifact")
    results = {}
    for mode in opts.modes:
        for method in opts.methods:
            results[(mode, method)] = search_for(opts, mode, method)
    print("\n==================== RESUMO (recall@1 val) ====================")
    for (mode, method), r in results.items():
        print("%-12s %-8s N*=%-3d val@search=%.2f%% val@final=%.2f%%" %
              (mode, method, r["best_N"], r["search_val_at_best"] * 100,
               r["final_val"] * 100))
    return results


if __name__ == "__main__":
    main()
