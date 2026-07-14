"""Graph-RKD distillation loss.

Generalizes relational RKD: instead of matching distances between pairs (RKD-D) or
angles between triples (RKD-A), it matches the **whole-graph embedding** of N nodes between
teacher and student. For each graph (set of N batch indices):

    D_s = dist(student[idx]) ;  g_s = embed(D_s)
    D_t = dist(teacher[idx]) ;  g_t = embed(D_t)        (teacher without gradient)
    loss += || g_s - g_t ||_p

The same set of indices is used on both sides, so the graphs
correspond; permutation invariance guarantees that the embedding represents the
*set*, not the order (which is what allows using sets and cutting the N! factor of the
tuples). N comes from the binary search (see node_search.find_best_n).
"""

import torch
import torch.nn as nn

from .embeddings import (NORM_SCHEMES, batch_distance_mean, embed_graphs,
                         zscore_descriptor)
from .node_search import adaptive_num_graphs

__all__ = ["GraphRKDLoss", "sample_graphs", "norm_flags"]


def norm_flags(norm):
    """Normalization scheme -> (per_graph_normalize, uses_μ_batch, zscore).

    Maps the ``norm`` axis (EXPERIMENTS_EN §5 / H2) to the concrete flags that
    ``embed_graphs`` and the descriptor z-score consume. Shared between the
    regression loss and the contrastive one.
    """
    if norm not in NORM_SCHEMES:
        raise ValueError("norm must be one of %s" % (NORM_SCHEMES,))
    return {"per_graph": (True, False, False),
            "none": (False, False, False),
            "minibatch": (False, True, False),
            "hybrid": (False, True, True)}[norm]


def sample_graphs(batch_size, n_nodes, sampling="partition", graphs_per_step=None,
                  alpha=0.5, g_min=None, g_max=None, generator=None, device="cpu"):
    """Generates graph indices (sets of ``n_nodes`` nodes) from the batch.

    * ``partition``: shuffles the batch and slices into ⌊B/N⌋ disjoint graphs
      (each sample used once). Number of graphs = ⌊B/N⌋.
    * ``random``: ``graphs_per_step`` random subsets of N nodes.
    * ``log``: ADAPTIVE number of graphs, growing with the Pascal's triangle
      row — ``adaptive_num_graphs(B, N, alpha, g_min, g_max)`` random
      subsets (recommended; see node_search.adaptive_num_graphs).

    Returns a LongTensor ``(num_graphs, n_nodes)``.
    """
    if n_nodes < 2 or n_nodes > batch_size:
        raise ValueError("n_nodes must be in [2, batch_size]")
    if sampling == "partition":
        perm = torch.randperm(batch_size, generator=generator, device=device)
        g = batch_size // n_nodes
        return perm[: g * n_nodes].reshape(g, n_nodes)
    if sampling in ("random", "log"):
        if sampling == "log":
            g = adaptive_num_graphs(batch_size, n_nodes, alpha, g_min, g_max)
        else:
            g = graphs_per_step or max(1, batch_size // n_nodes)
        idx = [torch.randperm(batch_size, generator=generator, device=device)[:n_nodes]
               for _ in range(g)]
        return torch.stack(idx, dim=0)
    raise ValueError("sampling must be 'partition', 'random' or 'log'")


class GraphRKDLoss(nn.Module):
    """Distillation via graph embedding (Method 1 'profile' or Method 2 'mds').

    Args:
        method: ``"profile"`` (ordered node profile) or ``"mds"`` (eigenvalues).
        n_nodes: N — number of nodes per graph (use node_search.find_best_n).
        sampling: ``"partition"``, ``"random"`` or ``"log"`` (adaptive).
        graphs_per_step: number of graphs when ``sampling="random"``.
        alpha/g_min/g_max: hyperparameters of ``sampling="log"`` (see
            node_search.adaptive_num_graphs).
        p: order of the Minkowski norm used in the per-graph loss.
        norm: scale normalization scheme — ``per_graph`` (per-graph off-diag mean,
            default), ``minibatch`` (μ_batch), ``none`` (raw) or ``hybrid``
            (μ_batch + descriptor z-score). See ``embeddings.NORM_SCHEMES`` / H2.
        squared: uses squared distances in the matrix.
        sort_key: sorting key of Method 1 (``"lex"`` by default).
    """

    def __init__(self, method="profile", n_nodes=8, sampling="partition",
                 graphs_per_step=None, alpha=0.5, g_min=None, g_max=None,
                 p=2, norm="per_graph", squared=False, sort_key="lex"):
        super().__init__()
        if method not in ("profile", "mds"):
            raise ValueError("method must be 'profile' or 'mds'")
        self.method = method
        self.n_nodes = n_nodes
        self.sampling = sampling
        self.graphs_per_step = graphs_per_step
        self.alpha = alpha
        self.g_min = g_min
        self.g_max = g_max
        self.p = p
        self.norm = norm
        self._normalize, self._use_scale, self._zscore = norm_flags(norm)
        self.squared = squared
        self.sort_key = sort_key

    def _embed(self, node_emb_graphs, batch_scale=None):
        g = embed_graphs(node_emb_graphs, method=self.method,
                         normalize=self._normalize, squared=self.squared,
                         sort_key=self.sort_key, batch_scale=batch_scale)
        return zscore_descriptor(g) if self._zscore else g

    def forward(self, student_emb, teacher_emb, graphs=None, generator=None):
        """student_emb/teacher_emb: ``(B, d)``. ``graphs`` optional ``(G, N)``;
        if None, samples with ``sample_graphs``."""
        B = student_emb.shape[0]
        if graphs is None:
            graphs = sample_graphs(B, self.n_nodes, self.sampling,
                                   self.graphs_per_step, self.alpha, self.g_min,
                                   self.g_max, generator=generator,
                                   device=student_emb.device)

        # μ_batch (cross scale) computed from the whole K×K matrix, per side.
        s_scale = batch_distance_mean(student_emb) if self._use_scale else None
        s_nodes = student_emb[graphs]                 # (G, N, d_s)
        with torch.no_grad():
            t_scale = batch_distance_mean(teacher_emb) if self._use_scale else None
            t_nodes = teacher_emb[graphs]             # (G, N, d_t)
            g_t = self._embed(t_nodes, batch_scale=t_scale)
        g_s = self._embed(s_nodes, batch_scale=s_scale)

        # per-graph Minkowski loss, mean over graphs
        return torch.norm(g_s - g_t, p=self.p, dim=-1).mean()
