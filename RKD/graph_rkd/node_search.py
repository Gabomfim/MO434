"""Graph-RKD: escolha do número de nós N por busca binária + análise combinatória.

Contexto do método (RKD generalizado): cada nó é um objeto (amostra do batch) e
cada aresta é a distância entre os embeddings de dois objetos. O grafo é completo
e não-direcionado, e seu embedding é **invariante a permutação** — se o grafo é o
mesmo (mesmo conjunto de nós), o embedding é o mesmo. Logo um grafo é definido por
um *conjunto* de N nós, não por uma tupla ordenada.

Este módulo cobre dois pedidos:

1. ``find_best_n`` — busca binária pelo MAIOR N viável para um dado tamanho de
   batch, sob uma restrição monotônica (orçamento de compute/memória, ou uma
   medição real injetada). Assume-se que a qualidade cresce com N (logo, dentro
   do orçamento, "maior é melhor"); a busca binária acha a fronteira de
   viabilidade em O(log B) avaliações.
2. ``unique_graphs`` / ``plot_unique_graphs`` — contagem de grafos únicos
   C(B, N) e comparação com as N-tuplas ordenadas P(B, N) do RKD tradicional.

Notas honestas:
* C(B, N) NÃO é "não-exponencial": é combinatória e tem pico em N=B/2. A vantagem
  da invariância a permutação é o fator N! (P(B,N) = N! · C(B,N)) e o fato de o
  custo POR PASSO ser controlável quando se amostra um nº fixo de grafos.
* Busca binária exige predicado monotônico; por isso ela busca o maior N sob
  restrição, e usamos um custo suave (estritamente crescente em N).
"""

import math
from typing import Callable, Optional

__all__ = [
    "unique_graphs", "ordered_tuples", "permutation_reduction",
    "step_edge_cost", "largest_feasible_n", "find_best_n",
    "find_knee_n", "derive_scaling_rule", "plot_unique_graphs",
    "adaptive_num_graphs",
]


# --------------------------------------------------------------------------- #
# Combinatória                                                                #
# --------------------------------------------------------------------------- #
def unique_graphs(B: int, N: int) -> int:
    """Nº de grafos completos DISTINTOS de N nós a partir de um batch de B
    amostras = C(B, N) (a ordem dos nós/arestas não importa)."""
    if N < 0 or N > B:
        return 0
    return math.comb(B, N)


def ordered_tuples(B: int, N: int) -> int:
    """Nº de N-tuplas ORDENADAS (relação N-ária do RKD tradicional)
    = B!/(B-N)! = P(B, N). Vale: P(B,N) == N! * C(B,N)."""
    if N < 0 or N > B:
        return 0
    return math.perm(B, N)


def permutation_reduction(N: int) -> int:
    """Fator de redução set-vs-tupla = N! (quantas ordenações colapsam em 1 grafo)."""
    return math.factorial(N)


def adaptive_num_graphs(batch_size: int, n_nodes: int, alpha: float = 0.5,
                        g_min: Optional[int] = None,
                        g_max: Optional[int] = None) -> int:
    """Nº de grafos a amostrar que CRESCE com a linha do triângulo de Pascal.

    O valor da linha C(K, N) é gigante (até ~1e37), então escalamos com
    ``log2(C(K,N))`` — o *description length* (bits para especificar um grafo) do
    espaço de grafos. É limitado, cresce com a linha e tem pico em N = K/2:

        G(N) = clamp( round(alpha · log2 C(K,N)) , g_min , g_max )

    g_min default = ⌊K/N⌋ (pelo menos uma partição completa do batch), garantindo
    cobertura ≥ ``partition``. ``g_max`` limita o custo (None = sem teto).
    """
    if g_min is None:
        g_min = max(1, batch_size // n_nodes)
    bits = math.log2(unique_graphs(batch_size, n_nodes) + 1)
    g = max(g_min, round(alpha * bits))
    if g_max is not None:
        g = min(g, g_max)
    return max(1, g)


# --------------------------------------------------------------------------- #
# Modelo de custo por passo                                                   #
# --------------------------------------------------------------------------- #
def step_edge_cost(B: int, N: int, scheme: str = "partition",
                   graphs_per_step: int = 1) -> float:
    """Custo de um passo em nº de arestas processadas (proxy de compute/memória).

    * ``partition``: cada amostra do batch é usada UMA vez -> ~B/N grafos
      disjuntos de N nós. Custo = (B/N)·C(N,2) = B·(N-1)/2 (exato quando N|B;
      estritamente crescente em N -> predicado monotônico, seguro p/ busca
      binária). Dá a regra elegante N*·B ≈ const.
    * ``sample``: ``graphs_per_step`` grafos de N nós -> G·C(N,2) = G·N(N-1)/2
      (independe de B).
    """
    if N < 2 or N > B:
        return float("inf")
    if scheme == "partition":
        return B * (N - 1) / 2.0           # = (B/N) * C(N,2), exato p/ N|B
    if scheme == "sample":
        return graphs_per_step * N * (N - 1) / 2.0
    raise ValueError("scheme deve ser 'partition' ou 'sample'")


# --------------------------------------------------------------------------- #
# Busca binária                                                               #
# --------------------------------------------------------------------------- #
def largest_feasible_n(predicate: Callable[[int], bool],
                       lo: int = 2, hi: int = 128) -> Optional[int]:
    """Maior N em [lo, hi] com ``predicate(N)`` True, assumindo predicado
    monotônico (True...True False...False). O(log(hi-lo)) avaliações.

    Retorna None se nem ``lo`` é viável.
    """
    if not predicate(lo):
        return None
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def find_best_n(batch_size: int, edge_budget: float, scheme: str = "partition",
                graphs_per_step: int = 1, n_min: int = 2,
                n_max: Optional[int] = None,
                feasible_fn: Optional[Callable[[int], bool]] = None) -> Optional[int]:
    """Busca binária pelo melhor (= maior viável) N para ``batch_size``.

    Use ``feasible_fn`` para injetar uma medição REAL (ex.: roda 1 passo de
    treino com N nós e devolve memória < limite / sem OOM / tempo < limite).
    Sem ela, usa o modelo de custo (``edge_budget``).
    """
    n_max = n_max or batch_size
    if feasible_fn is None:
        def feasible_fn(N):
            return step_edge_cost(batch_size, N, scheme, graphs_per_step) <= edge_budget
    return largest_feasible_n(feasible_fn, n_min, n_max)


def find_knee_n(quality_fn: Callable[[int], float], lo: int = 2, hi: int = 128,
                rel_tol: float = 0.01) -> int:
    """Variante para quando há RETORNOS DECRESCENTES (não "∝ N" puro).

    Busca binária pelo maior N cujo ganho relativo de qualidade ao DOBRAR N
    (de N/2 para N) ainda supera ``rel_tol``. ``quality_fn(N)`` deve ser uma
    medição real (ex.: recall/top-1 do student destilado com grafos de N nós).
    """
    def worth_it(N):
        half = max(lo, N // 2)
        if half == N:
            return True
        q_half, q_n = quality_fn(half), quality_fn(N)
        if q_half <= 0:
            return True
        return (q_n - q_half) / q_half >= rel_tol
    best = largest_feasible_n(worth_it, lo, hi)
    return best if best is not None else lo


# --------------------------------------------------------------------------- #
# Regra de escala empírica                                                    #
# --------------------------------------------------------------------------- #
def derive_scaling_rule(batch_sizes, edge_budget: float, scheme: str = "partition",
                        graphs_per_step: int = 1) -> dict:
    """Roda ``find_best_n`` para vários batches e ajusta a forma analítica.

    O custo da partição é E = B·(N-1)/2, logo o melhor N viável é AFIM em 1/B:

        N*(B) ≈ a·(1/B) + b ,   com   a = 2E ,  b = 1.

    Ajustamos N* = a·(1/B) + b por mínimos quadrados (não lei de potência — o
    termo "+1" da partição enviesaria um ajuste log-log). Para ``sample`` o N*
    independe de B (N* ≈ sqrt(2E/G)).
    """
    import numpy as np

    best = [find_best_n(B, edge_budget, scheme, graphs_per_step) for B in batch_sizes]
    Barr = np.array(batch_sizes, dtype=float)
    Narr = np.array([n if n else float("nan") for n in best], dtype=float)
    # Só o regime limitado pelo ORÇAMENTO é informativo: descarta pontos
    # "clampados" pelo teto N<=B (batches pequenos), que enviesam o ajuste.
    clamped = np.array([(n is not None) and (n >= B) for n, B in zip(best, batch_sizes)])
    mask = np.isfinite(Narr) & (Narr > 0) & (~clamped)
    if mask.sum() < 2:
        mask = np.isfinite(Narr) & (Narr > 0)

    out = {"batch_sizes": list(batch_sizes), "best_N": best,
           "edge_budget": edge_budget, "scheme": scheme}

    if scheme == "sample":
        out["analytic_rule"] = "N* ≈ sqrt(2·E/G) = %.3g (independe de B)" % (
            math.sqrt(2 * edge_budget / max(1, graphs_per_step)))
        out["fitted_rule"] = "N* ≈ %.3g (mediana, ~constante)" % float(
            np.median(Narr[mask]))
        return out

    # afim em 1/B
    inv = 1.0 / Barr[mask]
    a, b = np.polyfit(inv, Narr[mask], 1)            # N ≈ a*(1/B) + b
    pred = a * (1.0 / Barr[mask]) + b
    ss_res = float(np.sum((Narr[mask] - pred) ** 2))
    ss_tot = float(np.sum((Narr[mask] - Narr[mask].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    out.update({
        "fit_a": float(a), "fit_b": float(b), "fit_r2": r2,
        "fitted_rule": "N* ≈ %.0f·(1/B) + %.2f" % (a, b),
        "analytic_rule": "N* ≈ 2E/B + 1 = %.0f/B + 1" % (2 * edge_budget),
        "max_abs_residual": float(np.max(np.abs(Narr[mask] - pred))),
    })
    return out


# --------------------------------------------------------------------------- #
# Gráfico                                                                     #
# --------------------------------------------------------------------------- #
def plot_unique_graphs(B: int = 128, path: str = "unique_graphs_128.png") -> str:
    """Plota o nº de grafos únicos C(B, N) vs N (e P(B, N) p/ comparação)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = np.arange(0, B + 1)
    sets = np.array([unique_graphs(B, int(n)) for n in Ns], dtype=float)
    tups = np.array([ordered_tuples(B, int(n)) for n in Ns], dtype=float)
    peak = int(np.argmax(sets))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.semilogy(Ns, np.where(sets > 0, sets, np.nan), color="#2b6cb0", lw=2,
                label=r"grafos únicos = $\binom{%d}{N}$  (sets — método novo)" % B)
    ax.semilogy(Ns, np.where(tups > 0, tups, np.nan), "--", color="#c05621", lw=1.6,
                label=r"$N$-tuplas ordenadas = $P(%d,N)$  (RKD tradicional)" % B)
    ax.axvline(peak, color="gray", ls=":", lw=1)
    ax.annotate(r"pico em $N=%d$" % peak + "\n" + r"$\binom{%d}{%d}\approx%.2e$" % (B, peak, sets[peak]),
                xy=(peak, sets[peak]), xytext=(peak + 4, sets[peak] / 1e6),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("N (nós por grafo)")
    ax.set_ylabel("nº de grafos / tuplas  (escala log)")
    ax.set_title("Grafos únicos vs N-tuplas ordenadas — batch de %d amostras" % B)
    ax.legend(loc="lower center")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    B = 128
    print("=== Contagem para batch de %d (alguns N) ===" % B)
    print(" N | grafos únicos C(B,N) | tuplas ordenadas P(B,N) | reducao N!")
    for n in (2, 3, 4, 8, 16, 32, 64, 128):
        print("%3d | %22d | %24d | %d!" % (
            n, unique_graphs(B, n), ordered_tuples(B, n), n))

    print("\n=== Busca binária do melhor N (scheme=partition) ===")
    for budget in (256, 1024, 4096):
        n = find_best_n(B, edge_budget=budget, scheme="partition")
        print("batch=%d  orcamento=%5d arestas/passo -> melhor N = %s" % (B, budget, n))

    print("\n=== Regra de escala (varios batches, orcamento fixo) ===")
    rule = derive_scaling_rule([32, 64, 128, 256, 512, 1024, 2048],
                               edge_budget=1024, scheme="partition")
    print("N* por batch:", dict(zip(rule["batch_sizes"], rule["best_N"])))
    print("ajuste empirico:", rule["fitted_rule"], "(R^2=%.4f, resid max=%.2g)"
          % (rule["fit_r2"], rule["max_abs_residual"]))
    print("esperado analitico:", rule["analytic_rule"])

    out = plot_unique_graphs(B, path="graph_rkd/unique_graphs_128.png")
    print("\ngrafico salvo em:", out)
