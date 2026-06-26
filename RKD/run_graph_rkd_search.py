"""Orquestra experimentos Graph-RKD: busca binária sobre N (nº de nós) guiada
pela QUALIDADE DE VALIDAÇÃO, para cada combinação (modo de loss × método de
embedding). Cada candidato N dispara uma destilação real
(distill_to_convnextmicro) com a loss padrão (CE + Hinton KD) + a loss de grafo.

Loss padrão isolada: por padrão zeramos RKD-D/RKD-A/AT (--dist/--angle/--at = 0),
pois a loss de grafo É a componente relacional sob teste. Assim a comparação
mede o efeito da loss de grafo sobre uma base CE+KD limpa.

Busca de N: o teto vem da busca binária por orçamento (find_best_n, compute), e
dentro de [2, teto] achamos o "joelho" por qualidade (find_knee_n). A qualidade
de cada N é o melhor top-1 de VALIDAÇÃO (nunca test) de uma run curta
(--search_epochs); ao final roda-se uma run longa (--final_epochs) no N escolhido.

Distância euclidiana (padrão de pairwise_distance_matrix).

Exemplo:
    python run_graph_rkd_search.py --teacher_arch resnet18 \
        --teacher_artifact me/classifier-finetune/resnet18-cub200:best \
        --dataset cub200 --batch 128 --search_epochs 30 --final_epochs 300
"""

import argparse
import functools

import distill_to_convnextmicro as distill
from graph_rkd import find_best_n, find_knee_n


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

    # busca de N
    p.add_argument("--edge_budget", type=float, default=1024.0,
                   help="orçamento (arestas/passo) p/ o teto de N via find_best_n")
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--rel_tol", type=float, default=0.01,
                   help="ganho relativo mínimo de val top-1 ao dobrar N")
    p.add_argument("--graph_rkd_ratio", type=float, default=None,
                   help="peso da loss de grafo (default por modo: reg=1000, contr=1)")

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


def run_one(opts, mode, method, n_nodes, epochs, tag):
    """Roda uma destilação e devolve o melhor top-1 de VALIDAÇÃO."""
    ratio = opts.graph_rkd_ratio if opts.graph_rkd_ratio is not None \
        else _default_ratio(mode)
    params = {
        "dataset": opts.dataset, "data": opts.data,
        "teacher_arch": opts.teacher_arch,
        "teacher_load": opts.teacher_load,
        "teacher_artifact": opts.teacher_artifact,
        "batch": opts.batch, "epochs": epochs, "lr": opts.lr, "seed": opts.seed,
        "ce_ratio": opts.ce_ratio, "kd_ratio": opts.kd_ratio,
        "dist_ratio": opts.dist_ratio, "angle_ratio": opts.angle_ratio,
        "at_ratio": opts.at_ratio,
        "graph_rkd_mode": mode, "graph_rkd_method": method,
        "graph_rkd_nodes": n_nodes, "graph_rkd_ratio": ratio,
        "temp_schedule": opts.temp_schedule, "temp_start": opts.temp_start,
        "temp_end": opts.temp_end,
        "amp": opts.amp,
        "save_dir": f"{opts.save_root}/{mode}-{method}/N{n_nodes}-{tag}",
        "wandb_project": opts.wandb_project, "wandb_entity": opts.wandb_entity,
        "wandb_mode": opts.wandb_mode,
        "wandb_group": f"{mode}-{method}-{opts.dataset}",
        "wandb_run_name": f"{mode}-{method}-N{n_nodes}-{tag}",
        "wandb_tags": ["graph-rkd", mode, method, tag, f"N{n_nodes}"],
    }
    result = distill.run_with_params(params)
    return float(result["best_val_top1"])


def search_for(opts, mode, method):
    n_max = find_best_n(opts.batch, edge_budget=opts.edge_budget, n_min=opts.n_min)
    n_max = n_max or opts.n_min
    print(f"\n### {mode} / {method}: buscando N em [{opts.n_min}, {n_max}] "
          f"(teto por orçamento) ###")

    @functools.lru_cache(maxsize=None)
    def quality(n):
        q = run_one(opts, mode, method, n, opts.search_epochs, tag="search")
        print(f"   [{mode}/{method}] N={n} -> val top1={q*100:.2f}")
        return q

    best_n = find_knee_n(quality, lo=opts.n_min, hi=n_max, rel_tol=opts.rel_tol)
    # run final (longa) no N escolhido
    final_val = run_one(opts, mode, method, best_n, opts.final_epochs, tag="final")
    return {"best_N": best_n, "n_max": n_max,
            "search_val_at_best": quality(best_n), "final_val": final_val}


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
