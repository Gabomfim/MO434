"""Perda contrastiva (InfoNCE) por amostragem para Graph-RKD.

Em vez de regressão direta entre o embedding do grafo do student e do teacher
(``GraphRKDLoss``), usa contraste estilo InfoNCE com **amostragem negativa
estocástica** (GraphSAGE/PinSage/CRD): o custo deixa de depender do espaço de
grafos e passa a ser O(G · M) (G grafos, M negativos).

Enquadramento de DESTILAÇÃO (cross-model, estilo Contrastive Representation
Distillation):
    âncora   = embedding do grafo i do STUDENT
    positivo = embedding do MESMO grafo i do TEACHER
    negativos = M embeddings de grafos j != i do TEACHER (amostrados)
Como o tamanho do embedding do grafo depende só de N, âncora e positivo vivem no
mesmo espaço — a similaridade de cosseno é direta.

Tudo vetorizado (sem loop por amostra) e via ``cross_entropy(logits, 0)``, que é
a InfoNCE numericamente estável e equivale a -log(pos/(pos+Σneg)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import embed_graphs
from .loss import sample_graphs

__all__ = ["SampledGraphContrastiveLoss", "GraphContrastiveDistillLoss"]


class SampledGraphContrastiveLoss(nn.Module):
    """InfoNCE com negativos amostrados (vetorizado).

    Mesma ideia do protocolo pedido, mas: (a) sem o loop ``for`` por amostra;
    (b) opção de excluir índices (evita sortear o próprio positivo como
    negativo); (c) ``cross_entropy`` estável no lugar de exp/log crus.
    """

    def __init__(self, temperature=0.07, num_negative_samples=10):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negative_samples

    def forward(self, anchor_embeddings, positive_embeddings,
                all_available_embeddings, exclude_index=None, generator=None):
        """anchor/positive: ``(A, E)`` alinhados; pool: ``(P, E)``.
        ``exclude_index``: ``(A,)`` índice no pool a NÃO amostrar como negativo
        (tipicamente o índice do positivo, quando pool == positivos)."""
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
            # se sorteou o próprio índice, desloca em +1 (mod P)
            collide = neg_idx == exclude_index.view(-1, 1)
            neg_idx = torch.where(collide, (neg_idx + 1) % P, neg_idx)

        neg = pool[neg_idx]                                      # (A, M, E)
        neg_sim = (a.unsqueeze(1) * neg).sum(-1)                 # (A, M)

        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
        labels = torch.zeros(A, dtype=torch.long, device=device)  # positivo = col 0
        return F.cross_entropy(logits, labels)


class GraphContrastiveDistillLoss(nn.Module):
    """Destilação contrastiva por grafo (InfoNCE cross-model + amostragem).

    Amostra G grafos do batch; embute student (âncora) e teacher (positivo) com
    o MESMO conjunto de índices; negativos são outros grafos do teacher.
    """

    def __init__(self, method="profile", n_nodes=8, sampling="partition",
                 graphs_per_step=None, alpha=0.5, g_min=None, g_max=None,
                 num_negatives=10, temperature=0.07,
                 normalize=True, squared=False, sort_key="lex"):
        super().__init__()
        self.method = method
        self.n_nodes = n_nodes
        self.sampling = sampling
        self.graphs_per_step = graphs_per_step
        self.alpha = alpha
        self.g_min = g_min
        self.g_max = g_max
        self.normalize = normalize
        self.squared = squared
        self.sort_key = sort_key
        self.nce = SampledGraphContrastiveLoss(temperature, num_negatives)

    def _embed(self, node_emb_graphs):
        return embed_graphs(node_emb_graphs, method=self.method,
                            normalize=self.normalize, squared=self.squared,
                            sort_key=self.sort_key)

    def forward(self, student_emb, teacher_emb, graphs=None, generator=None):
        """student_emb/teacher_emb: ``(B, d)``."""
        B = student_emb.shape[0]
        if graphs is None:
            graphs = sample_graphs(B, self.n_nodes, self.sampling,
                                   self.graphs_per_step, self.alpha, self.g_min,
                                   self.g_max, generator=generator,
                                   device=student_emb.device)
        G = graphs.shape[0]

        g_s = self._embed(student_emb[graphs])               # âncora (student), grad
        with torch.no_grad():
            g_t = self._embed(teacher_emb[graphs])           # positivo/pool (teacher)

        idx = torch.arange(G, device=student_emb.device)     # positivo do grafo i = i
        return self.nce(g_s, g_t, g_t, exclude_index=idx, generator=generator)
