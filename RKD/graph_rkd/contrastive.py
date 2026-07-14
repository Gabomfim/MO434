"""Sampling-based contrastive (InfoNCE) loss for Graph-RKD.

Instead of direct regression between the student's and teacher's graph embeddings
(``GraphRKDLoss``), it uses InfoNCE-style contrast with **stochastic negative
sampling** (GraphSAGE/PinSage/CRD): the cost stops depending on the graph
space and becomes O(G · M) (G graphs, M negatives).

DISTILLATION framing (cross-model, Contrastive Representation
Distillation style):
    anchor   = STUDENT's embedding of graph i
    positive = TEACHER's embedding of the SAME graph i
    negatives = M TEACHER embeddings of graphs j != i (sampled)
Since the graph embedding size depends only on N, anchor and positive live in the
same space — the cosine similarity is direct.

All vectorized (no per-sample loop) and via ``cross_entropy(logits, 0)``, which is
numerically stable InfoNCE and equivalent to -log(pos/(pos+Σneg)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import batch_distance_mean, embed_graphs, zscore_descriptor
from .loss import norm_flags, sample_graphs

__all__ = ["SampledGraphContrastiveLoss", "GraphContrastiveDistillLoss"]


class SampledGraphContrastiveLoss(nn.Module):
    """InfoNCE with sampled negatives (vectorized).

    Same idea as the requested protocol, but: (a) without the per-sample ``for``
    loop; (b) option to exclude indices (avoids drawing the positive itself as a
    negative); (c) stable ``cross_entropy`` instead of raw exp/log.
    """

    def __init__(self, temperature=0.07, num_negative_samples=10):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negative_samples

    def forward(self, anchor_embeddings, positive_embeddings,
                all_available_embeddings, exclude_index=None, generator=None):
        """anchor/positive: ``(A, E)`` aligned; pool: ``(P, E)``.
        ``exclude_index``: ``(A,)`` index in the pool NOT to sample as a negative
        (typically the positive's index, when pool == positives)."""
        A = anchor_embeddings.size(0)
        P = all_available_embeddings.size(0)
        device = anchor_embeddings.device

        a = F.normalize(anchor_embeddings, dim=-1)
        p = F.normalize(positive_embeddings, dim=-1)
        pool = F.normalize(all_available_embeddings, dim=-1)

        pos_sim = (a * p).sum(-1, keepdim=True)                  # (A, 1)

        neg_idx = torch.randint(0, P, (A, self.num_negatives),
                                device=device, generator=generator)
        if exclude_index is not None:
            # if it drew the index itself, shift by +1 (mod P)
            collide = neg_idx == exclude_index.view(-1, 1)
            neg_idx = torch.where(collide, (neg_idx + 1) % P, neg_idx)

        neg = pool[neg_idx]                                      # (A, M, E)
        neg_sim = (a.unsqueeze(1) * neg).sum(-1)                 # (A, M)

        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
        labels = torch.zeros(A, dtype=torch.long, device=device)  # positive = col 0
        return F.cross_entropy(logits, labels)


class GraphContrastiveDistillLoss(nn.Module):
    """Per-graph contrastive distillation (cross-model InfoNCE + sampling).

    Samples G graphs from the batch; embeds student (anchor) and teacher (positive) with
    the SAME set of indices; negatives are other teacher graphs.
    """

    def __init__(self, method="profile", n_nodes=8, sampling="partition",
                 graphs_per_step=None, alpha=0.5, g_min=None, g_max=None,
                 num_negatives=10, temperature=0.07,
                 norm="per_graph", squared=False, sort_key="lex"):
        super().__init__()
        self.method = method
        self.n_nodes = n_nodes
        self.sampling = sampling
        self.graphs_per_step = graphs_per_step
        self.alpha = alpha
        self.g_min = g_min
        self.g_max = g_max
        self.norm = norm
        self._normalize, self._use_scale, self._zscore = norm_flags(norm)
        self.squared = squared
        self.sort_key = sort_key
        self.nce = SampledGraphContrastiveLoss(temperature, num_negatives)

    def _embed(self, node_emb_graphs, batch_scale=None):
        g = embed_graphs(node_emb_graphs, method=self.method,
                         normalize=self._normalize, squared=self.squared,
                         sort_key=self.sort_key, batch_scale=batch_scale)
        return zscore_descriptor(g) if self._zscore else g

    def forward(self, student_emb, teacher_emb, graphs=None, generator=None):
        """student_emb/teacher_emb: ``(B, d)``."""
        B = student_emb.shape[0]
        if graphs is None:
            graphs = sample_graphs(B, self.n_nodes, self.sampling,
                                   self.graphs_per_step, self.alpha, self.g_min,
                                   self.g_max, generator=generator,
                                   device=student_emb.device)
        G = graphs.shape[0]

        s_scale = batch_distance_mean(student_emb) if self._use_scale else None
        g_s = self._embed(student_emb[graphs], batch_scale=s_scale)  # anchor, grad
        with torch.no_grad():
            t_scale = batch_distance_mean(teacher_emb) if self._use_scale else None
            g_t = self._embed(teacher_emb[graphs], batch_scale=t_scale)  # pos/pool

        idx = torch.arange(G, device=student_emb.device)     # positive of graph i = i
        return self.nce(g_s, g_t, g_t, exclude_index=idx, generator=generator)
