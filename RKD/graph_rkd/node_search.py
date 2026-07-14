"""Graph-RKD: choosing the number of nodes N via binary search + combinatorial analysis.

Method context (generalized RKD): each node is an object (a batch sample) and
each edge is the distance between the embeddings of two objects. The graph is
complete and undirected, and its embedding is **permutation-invariant** — if the
graph is the same (same set of nodes), the embedding is the same. Hence a graph
is defined by a *set* of N nodes, not by an ordered tuple.

This module covers two requests:

1. ``find_best_n`` — binary search for the LARGEST feasible N for a given batch
   size, under a monotonic constraint (compute/memory budget, or an injected
   real measurement). Quality is assumed to grow with N (so, within the
   budget, "larger is better"); the binary search finds the feasibility
   frontier in O(log B) evaluations.
2. ``unique_graphs`` / ``plot_unique_graphs`` — count of unique graphs
   C(B, N) and comparison with the ordered N-tuples P(B, N) of traditional RKD.

Honest notes:
* C(B, N) is NOT "non-exponential": it is combinatorial and peaks at N=B/2. The
  advantage of permutation invariance is the N! factor (P(B,N) = N! · C(B,N)) and
  the fact that the PER-STEP cost is controllable when a fixed number of graphs is sampled.
* Binary search requires a monotonic predicate; that is why it searches for the largest N under
  a constraint, and we use a smooth cost (strictly increasing in N).
"""

import math
from typing import Callable, Optional

__all__ = [
    "unique_graphs", "ordered_tuples", "permutation_reduction",
    "step_edge_cost", "largest_feasible_n", "find_best_n",
    "find_knee_n", "derive_scaling_rule", "plot_unique_graphs",
    "adaptive_num_graphs", "log_spaced_orders", "select_order",
]


# --------------------------------------------------------------------------- #
# Combinatorics                                                                #
# --------------------------------------------------------------------------- #
def unique_graphs(B: int, N: int) -> int:
    """Number of DISTINCT complete graphs of N nodes from a batch of B
    samples = C(B, N) (node/edge order does not matter)."""
    if N < 0 or N > B:
        return 0
    return math.comb(B, N)


def ordered_tuples(B: int, N: int) -> int:
    """Number of ORDERED N-tuples (N-ary relation of traditional RKD)
    = B!/(B-N)! = P(B, N). It holds that: P(B,N) == N! * C(B,N)."""
    if N < 0 or N > B:
        return 0
    return math.perm(B, N)


def permutation_reduction(N: int) -> int:
    """Set-vs-tuple reduction factor = N! (how many orderings collapse into 1 graph)."""
    return math.factorial(N)


def adaptive_num_graphs(batch_size: int, n_nodes: int, alpha: float = 0.5,
                        g_min: Optional[int] = None,
                        g_max: Optional[int] = None) -> int:
    """Number of graphs to sample that GROWS with the Pascal's triangle row.

    The row value C(K, N) is huge (up to ~1e37), so we scale with
    ``log2(C(K,N))`` — the *description length* (bits to specify one graph) of the
    graph space. It is bounded, grows with the row and peaks at N = K/2:

        G(N) = clamp( round(alpha · log2 C(K,N)) , g_min , g_max )

    g_min default = ⌊K/N⌋ (at least one full partition of the batch), ensuring
    coverage ≥ ``partition``. ``g_max`` bounds the cost (None = no cap).
    """
    if g_min is None:
        g_min = max(1, batch_size // n_nodes)
    bits = math.log2(unique_graphs(batch_size, n_nodes) + 1)
    g = max(g_min, round(alpha * bits))
    if g_max is not None:
        g = min(g, g_max)
    return max(1, g)


# --------------------------------------------------------------------------- #
# Per-step cost model                                                          #
# --------------------------------------------------------------------------- #
def step_edge_cost(B: int, N: int, scheme: str = "partition",
                   graphs_per_step: int = 1) -> float:
    """Cost of one step in number of edges processed (proxy for compute/memory).

    * ``partition``: each batch sample is used ONCE -> ~B/N disjoint
      graphs of N nodes. Cost = (B/N)·C(N,2) = B·(N-1)/2 (exact when N|B;
      strictly increasing in N -> monotonic predicate, safe for binary
      search). Gives the elegant rule N*·B ≈ const.
    * ``sample``: ``graphs_per_step`` graphs of N nodes -> G·C(N,2) = G·N(N-1)/2
      (independent of B).
    """
    if N < 2 or N > B:
        return float("inf")
    if scheme == "partition":
        return B * (N - 1) / 2.0           # = (B/N) * C(N,2), exact for N|B
    if scheme == "sample":
        return graphs_per_step * N * (N - 1) / 2.0
    raise ValueError("scheme must be 'partition' or 'sample'")


# --------------------------------------------------------------------------- #
# Binary search                                                               #
# --------------------------------------------------------------------------- #
def largest_feasible_n(predicate: Callable[[int], bool],
                       lo: int = 2, hi: int = 128) -> Optional[int]:
    """Largest N in [lo, hi] with ``predicate(N)`` True, assuming a monotonic
    predicate (True...True False...False). O(log(hi-lo)) evaluations.

    Returns None if not even ``lo`` is feasible.
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
    """Binary search for the best (= largest feasible) N for ``batch_size``.

    Use ``feasible_fn`` to inject a REAL measurement (e.g.: runs 1 training
    step with N nodes and returns memory < limit / no OOM / time < limit).
    Without it, uses the cost model (``edge_budget``).
    """
    n_max = n_max or batch_size
    if feasible_fn is None:
        def feasible_fn(N):
            return step_edge_cost(batch_size, N, scheme, graphs_per_step) <= edge_budget
    return largest_feasible_n(feasible_fn, n_min, n_max)


def log_spaced_orders(n_min: int, n_max: int, base: int = 2):
    """Log-spaced (geometric) N candidates in [n_min, n_max].

    E.g.: n_min=2, n_max=17 -> [2, 4, 8, 16, 17]. They are ~log_base(n_max) points
    (same budget as binary search), but reveal the SHAPE of the quality-N curve
    and do not assume monotonicity.
    """
    if n_max < n_min:
        return []
    orders, n = [], n_min
    while n < n_max:
        orders.append(int(n))
        n *= base
    orders.append(int(n_max))
    return sorted({o for o in orders if n_min <= o <= n_max})


def select_order(orders, means, sems=None, rule: str = "argmax") -> int:
    """Chooses N from the measured qualities (validation top-1).

    ``argmax``: N with the highest mean.
    ``1se``   : smallest N whose mean is within 1 standard error of the best
                (one-standard-error rule: parsimony — smaller N is cheaper —
                with no statistically significant loss). Requires ``sems`` and
                ``orders`` in increasing order.
    """
    best_i = max(range(len(orders)), key=lambda i: means[i])
    if rule == "argmax" or sems is None:
        return int(orders[best_i])
    threshold = means[best_i] - sems[best_i]
    for i in range(len(orders)):            # orders increasing -> smallest eligible N
        if means[i] >= threshold:
            return int(orders[i])
    return int(orders[best_i])


def find_knee_n(quality_fn: Callable[[int], float], lo: int = 2, hi: int = 128,
                rel_tol: float = 0.01) -> int:
    """Variant for when there are DIMINISHING RETURNS (not pure "∝ N").

    Binary search for the largest N whose relative quality gain when DOUBLING N
    (from N/2 to N) still exceeds ``rel_tol``. ``quality_fn(N)`` must be a
    real measurement (e.g.: recall/top-1 of the student distilled with graphs of N nodes).
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
# Empirical scaling rule                                                       #
# --------------------------------------------------------------------------- #
def derive_scaling_rule(batch_sizes, edge_budget: float, scheme: str = "partition",
                        graphs_per_step: int = 1) -> dict:
    """Runs ``find_best_n`` for several batches and fits the analytic form.

    The partition cost is E = B·(N-1)/2, so the best feasible N is AFFINE in 1/B:

        N*(B) ≈ a·(1/B) + b ,   with   a = 2E ,  b = 1.

    We fit N* = a·(1/B) + b by least squares (not a power law — the
    "+1" partition term would bias a log-log fit). For ``sample`` the N*
    is independent of B (N* ≈ sqrt(2E/G)).
    """
    import numpy as np

    best = [find_best_n(B, edge_budget, scheme, graphs_per_step) for B in batch_sizes]
    Barr = np.array(batch_sizes, dtype=float)
    Narr = np.array([n if n else float("nan") for n in best], dtype=float)
    # Only the BUDGET-limited regime is informative: discard points
    # "clamped" by the ceiling N<=B (small batches), which bias the fit.
    clamped = np.array([(n is not None) and (n >= B) for n, B in zip(best, batch_sizes)])
    mask = np.isfinite(Narr) & (Narr > 0) & (~clamped)
    if mask.sum() < 2:
        mask = np.isfinite(Narr) & (Narr > 0)

    out = {"batch_sizes": list(batch_sizes), "best_N": best,
           "edge_budget": edge_budget, "scheme": scheme}

    if scheme == "sample":
        out["analytic_rule"] = "N* ≈ sqrt(2·E/G) = %.3g (independent of B)" % (
            math.sqrt(2 * edge_budget / max(1, graphs_per_step)))
        out["fitted_rule"] = "N* ≈ %.3g (median, ~constant)" % float(
            np.median(Narr[mask]))
        return out

    # affine in 1/B
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
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def plot_unique_graphs(B: int = 128, path: str = "unique_graphs_128.png") -> str:
    """Plots the number of unique graphs C(B, N) vs N (and P(B, N) for comparison)."""
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
                label=r"unique graphs = $\binom{%d}{N}$  (sets — new method)" % B)
    ax.semilogy(Ns, np.where(tups > 0, tups, np.nan), "--", color="#c05621", lw=1.6,
                label=r"ordered $N$-tuples = $P(%d,N)$  (traditional RKD)" % B)
    ax.axvline(peak, color="gray", ls=":", lw=1)
    ax.annotate(r"peak at $N=%d$" % peak + "\n" + r"$\binom{%d}{%d}\approx%.2e$" % (B, peak, sets[peak]),
                xy=(peak, sets[peak]), xytext=(peak + 4, sets[peak] / 1e6),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("N (nodes per graph)")
    ax.set_ylabel("number of graphs / tuples  (log scale)")
    ax.set_title("Unique graphs vs ordered N-tuples — batch of %d samples" % B)
    ax.legend(loc="lower center")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    B = 128
    print("=== Count for batch of %d (some N) ===" % B)
    print(" N | unique graphs C(B,N) | ordered tuples P(B,N) | reduction N!")
    for n in (2, 3, 4, 8, 16, 32, 64, 128):
        print("%3d | %22d | %24d | %d!" % (
            n, unique_graphs(B, n), ordered_tuples(B, n), n))

    print("\n=== Binary search for the best N (scheme=partition) ===")
    for budget in (256, 1024, 4096):
        n = find_best_n(B, edge_budget=budget, scheme="partition")
        print("batch=%d  budget=%5d edges/step -> best N = %s" % (B, budget, n))

    print("\n=== Scaling rule (several batches, fixed budget) ===")
    rule = derive_scaling_rule([32, 64, 128, 256, 512, 1024, 2048],
                               edge_budget=1024, scheme="partition")
    print("N* per batch:", dict(zip(rule["batch_sizes"], rule["best_N"])))
    print("empirical fit:", rule["fitted_rule"], "(R^2=%.4f, max resid=%.2g)"
          % (rule["fit_r2"], rule["max_abs_residual"]))
    print("analytic expected:", rule["analytic_rule"])

    out = plot_unique_graphs(B, path="graph_rkd/unique_graphs_128.png")
    print("\nplot saved to:", out)
