"""Orquestra experimentos Graph-RKD: escolhe N (nº de nós) por uma VARREDURA
LOG limitada por orçamento, para cada (modo de loss × método de embedding).

Seleção de N (substitui a antiga busca binária por qualidade):
  1) teto N_max por orçamento de compute — busca binária num predicado
     monotônico/determinístico (find_best_n), exata e sem treino;
  2) candidatos espaçados em log em [n_min, N_max] (log_spaced_orders) —
     ~log2(N_max) pontos, mesmo orçamento da busca binária, mas que revelam a
     FORMA da curva qualidade-N e não assumem monotonicidade;
  3) cada candidato é treinado com --seeds seed(s); a qualidade é o melhor
     top-1 de VALIDAÇÃO (nunca test), média sobre seeds;
  4) seleção por argmax ou pela regra do 1-erro-padrão (--select);
  5) run final longa (--final_epochs) no N escolhido.

Por que não busca binária na qualidade: ela exige que a curva qualidade-N seja
monótona/unimodal e sem ruído — nada disso é garantido (q(N) pode subir e
descer, e cada ponto é uma estimativa ruidosa). A varredura log tem o mesmo
custo e é robusta a isso.

Loss = só cross-entropy + a loss de grafo (RKD-D/RKD-A/AT/KD zerados por
padrão). Distância euclidiana (padrão de pairwise_distance_matrix).

Exemplo:
    python run_graph_rkd_search.py --teacher_arch resnet18 \
        --teacher_artifact me/classifier-finetune/resnet18-cub200:best \
        --dataset cub200 --batch 128 --search_epochs 30 --final_epochs 300
"""

import argparse
import statistics

import distill_to_convnextmicro as distill
from experiment_ledger import load_result, mark_done
from graph_rkd import find_best_n, log_spaced_orders, select_order
from wandb_artifacts import stable_run_id


def build_parser():
    p = argparse.ArgumentParser(description="Busca binária de N para Graph-RKD")
    p.add_argument("--dataset", choices=["cars196", "cub200"], default="cub200")
    p.add_argument("--data", default="data")
    p.add_argument("--teacher_arch", choices=["resnet18", "convnext_tiny"],
                   default="resnet18")
    p.add_argument("--teacher_load", default=None)
    p.add_argument("--teacher_artifact", default=None)
    p.add_argument("--batch", type=int, default=128)

    # quais experimentos
    p.add_argument("--modes", nargs="+", default=["regression", "contrastive"],
                   choices=["regression", "contrastive"])
    p.add_argument("--methods", nargs="+", default=["profile", "mds"],
                   choices=["profile", "mds"])

    # busca de N: teto por orçamento (binária, exata) + varredura log + seleção
    p.add_argument("--edge_budget", type=float, default=1024.0,
                   help="orçamento (arestas/passo) p/ o teto de N via find_best_n")
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--base", type=int, default=2,
                   help="base do espaçamento log dos candidatos N")
    p.add_argument("--seeds", type=int, default=1,
                   help="seeds por candidato N (média p/ reduzir ruído)")
    p.add_argument("--select", choices=["argmax", "1se"], default="argmax",
                   help="regra de seleção de N a partir do val top-1")
    p.add_argument("--graph_rkd_ratio", type=float, default=None,
                   help="peso da loss de grafo (default por modo: reg=1000, contr=1)")
    p.add_argument("--graph_rkd_sampling", choices=["partition", "random", "log"],
                   default="log", help="amostragem de grafos (log = adaptativa)")
    p.add_argument("--graph_rkd_alpha", type=float, default=0.5,
                   help="sampling=log: G = alpha*log2(C(K,N))")
    p.add_argument("--graph_rkd_gmax", type=int, default=64,
                   help="sampling=log: teto de grafos/passo (custo)")
    p.add_argument("--graph_warmup_frac", type=float, default=0.0,
                   help="warmup do peso da loss de grafo (fração das épocas)")

    # treino
    p.add_argument("--search_epochs", type=int, default=30)
    p.add_argument("--final_epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_root", default="graph_rkd_runs")

    # Loss = SÓ cross-entropy + a loss de grafo. KD/dist/angle/at desligados.
    p.add_argument("--ce_ratio", type=float, default=1.0)
    p.add_argument("--kd_ratio", type=float, default=0.0)
    p.add_argument("--dist_ratio", type=float, default=0.0)
    p.add_argument("--angle_ratio", type=float, default=0.0)
    p.add_argument("--at_ratio", type=float, default=0.0)

    # Temperatura da InfoNCE contrastiva. Default CONSTANTE (τ fixa tunada) — é o
    # baseline limpo. O agendamento (linear/cosine/exp) fica como ABLAÇÃO opcional.
    p.add_argument("--temp_schedule", choices=["constant", "linear", "cosine", "exp"],
                   default="constant")
    p.add_argument("--temp_start", type=float, default=0.07,
                   help="τ (valor fixo se temp_schedule=constant)")
    p.add_argument("--temp_end", type=float, default=0.05,
                   help="τ final (usado só se temp_schedule != constant)")

    # wandb
    p.add_argument("--wandb_project", default="graph-rkd-node-search")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    return p


def _default_ratio(mode):
    # AT/RKD-regressão precisam de peso grande (loss ~1e-4..1e-3); contraste ~O(1).
    return 1000.0 if mode == "regression" else 1.0


def run_one(opts, mode, method, n_nodes, epochs, tag, seed):
    """Roda uma destilação e devolve o melhor top-1 de VALIDAÇÃO.

    Resumível: se o save_dir já tem result.json, devolve o valor cacheado sem
    re-treinar.
    """
    save_dir = f"{opts.save_root}/{mode}-{method}/N{n_nodes}-{tag}-s{seed}"
    cached = load_result(save_dir)
    if cached is not None:
        print(f"   [skip|cache] {mode}/{method} N={n_nodes} {tag} s{seed} "
              f"-> val top1={cached['best_val_top1']*100:.2f}")
        return float(cached["best_val_top1"])
    ratio = opts.graph_rkd_ratio if opts.graph_rkd_ratio is not None \
        else _default_ratio(mode)
    params = {
        "dataset": opts.dataset, "data": opts.data,
        "teacher_arch": opts.teacher_arch,
        "teacher_load": opts.teacher_load,
        "teacher_artifact": opts.teacher_artifact,
        "batch": opts.batch, "epochs": epochs, "lr": opts.lr, "seed": seed,
        "ce_ratio": opts.ce_ratio, "kd_ratio": opts.kd_ratio,
        "dist_ratio": opts.dist_ratio, "angle_ratio": opts.angle_ratio,
        "at_ratio": opts.at_ratio,
        "graph_rkd_mode": mode, "graph_rkd_method": method,
        "graph_rkd_nodes": n_nodes, "graph_rkd_ratio": ratio,
        "graph_rkd_sampling": opts.graph_rkd_sampling,
        "graph_rkd_alpha": opts.graph_rkd_alpha,
        "graph_rkd_gmax": opts.graph_rkd_gmax,
        "graph_warmup_frac": opts.graph_warmup_frac,
        "temp_schedule": opts.temp_schedule, "temp_start": opts.temp_start,
        "temp_end": opts.temp_end,
        "amp": opts.amp,
        "save_dir": save_dir, "resume": f"{save_dir}/student_last.pth",
        "wandb_project": opts.wandb_project, "wandb_entity": opts.wandb_entity,
        "wandb_mode": opts.wandb_mode,
        "wandb_group": f"{mode}-{method}-{opts.dataset}",
        "wandb_run_name": f"{mode}-{method}-N{n_nodes}-{tag}-s{seed}",
        "wandb_id": stable_run_id(
            f"{opts.dataset}-{opts.teacher_arch}-{mode}-{method}-N{n_nodes}-{tag}-s{seed}"),
        "wandb_tags": ["graph-rkd", mode, method, tag, f"N{n_nodes}"],
    }
    result = distill.run_with_params(params)
    val = float(result["best_val_top1"])
    mark_done(save_dir, {"best_val_top1": val})
    return val


def search_for(opts, mode, method):
    # 1) Teto por orçamento de compute: busca binária (predicado monotônico, exato).
    n_max = find_best_n(opts.batch, edge_budget=opts.edge_budget, n_min=opts.n_min)
    n_max = n_max or opts.n_min
    # 2) Candidatos espaçados em log dentro de [n_min, n_max].
    candidates = log_spaced_orders(opts.n_min, n_max, base=opts.base)
    print(f"\n### {mode} / {method}: varrendo N em {candidates} "
          f"(teto={n_max} por orçamento; {opts.seeds} seed(s)/candidato) ###")

    # 3) Qualidade de validação de cada candidato, média sobre seeds.
    means, sems = [], []
    for N in candidates:
        vals = [run_one(opts, mode, method, N, opts.search_epochs, "search", s)
                for s in range(opts.seeds)]
        mean = statistics.fmean(vals)
        sem = (statistics.pstdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0
        means.append(mean)
        sems.append(sem)
        print(f"   [{mode}/{method}] N={N} -> val top1={mean*100:.2f} "
              f"(+/-{sem*100:.2f}, n={opts.seeds})")

    # 4) Seleção (argmax ou regra do 1-erro-padrão).
    best_n = select_order(candidates, means, sems, rule=opts.select)
    best_search_val = means[candidates.index(best_n)]
    # 5) Run final (longa) no N escolhido.
    final_val = run_one(opts, mode, method, best_n, opts.final_epochs, "final", opts.seed)
    return {"best_N": best_n, "n_max": n_max, "candidates": candidates,
            "search_val_at_best": best_search_val, "final_val": final_val}


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if (opts.teacher_load is None) == (opts.teacher_artifact is None):
        raise SystemExit("Forneça exatamente um de --teacher_load / --teacher_artifact")

    results = {}
    for mode in opts.modes:
        for method in opts.methods:
            results[(mode, method)] = search_for(opts, mode, method)

    print("\n==================== RESUMO ====================")
    print("%-12s %-8s %6s %6s %12s %10s" %
          ("modo", "metodo", "N*", "Nmax", "val@search", "val@final"))
    for (mode, method), r in results.items():
        print("%-12s %-8s %6d %6d %11.2f%% %9.2f%%" %
              (mode, method, r["best_N"], r["n_max"],
               r["search_val_at_best"] * 100, r["final_val"] * 100))
    return results


if __name__ == "__main__":
    main()
