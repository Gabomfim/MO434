"""Perda de destilação Graph-RKD.

Generaliza o RKD relacional: em vez de casar distâncias entre pares (RKD-D) ou
ângulos entre trios (RKD-A), casa o **embedding do grafo inteiro** de N nós entre
teacher e student. Para cada grafo (conjunto de N índices do batch):

    D_s = dist(student[idx]) ;  g_s = embed(D_s)
    D_t = dist(teacher[idx]) ;  g_t = embed(D_t)        (teacher sem gradiente)
    perda += || g_s - g_t ||_p

O mesmo conjunto de índices é usado nos dois lados, então os grafos
correspondem; a invariância a permutação garante que o embedding represente o
*conjunto*, não a ordem (é o que permite usar sets e cortar o fator N! das
tuplas). N vem da busca binária (ver node_search.find_best_n).
"""

import torch
import torch.nn as nn

from .embeddings import embed_graphs
from .node_search import adaptive_num_graphs

__all__ = ["GraphRKDLoss", "sample_graphs"]


def sample_graphs(batch_size, n_nodes, sampling="partition", graphs_per_step=None,
                  alpha=0.5, g_min=None, g_max=None, generator=None, device="cpu"):
    """Gera índices de grafos (conjuntos de ``n_nodes`` nós) a partir do batch.

    * ``partition``: embaralha o batch e fatia em ⌊B/N⌋ grafos disjuntos
      (cada amostra usada uma vez). Nº de grafos = ⌊B/N⌋.
    * ``random``: ``graphs_per_step`` subconjuntos aleatórios de N nós.
    * ``log``: nº de grafos ADAPTATIVO, crescendo com a linha do triângulo de
      Pascal — ``adaptive_num_graphs(B, N, alpha, g_min, g_max)`` subconjuntos
      aleatórios (recomendado; ver node_search.adaptive_num_graphs).

    Retorna LongTensor ``(num_graphs, n_nodes)``.
    """
    if n_nodes < 2 or n_nodes > batch_size:
        raise ValueError("n_nodes deve estar em [2, batch_size]")
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
    raise ValueError("sampling deve ser 'partition', 'random' ou 'log'")


class GraphRKDLoss(nn.Module):
    """Destilação pelo embedding de grafo (Método 1 'profile' ou Método 2 'mds').

    Args:
        method: ``"profile"`` (perfil de nós ordenado) ou ``"mds"`` (autovalores).
        n_nodes: N — nº de nós por grafo (use node_search.find_best_n).
        sampling: ``"partition"``, ``"random"`` ou ``"log"`` (adaptativo).
        graphs_per_step: nº de grafos quando ``sampling="random"``.
        alpha/g_min/g_max: hiperparâmetros do ``sampling="log"`` (ver
            node_search.adaptive_num_graphs).
        p: ordem da norma de Minkowski usada na perda por grafo.
        normalize: normaliza cada matriz de distância pela média (escala-invar.).
        squared: usa distâncias ao quadrado na matriz.
        sort_key: chave de ordenação do Método 1 (``"lex"`` por padrão).
    """

    def __init__(self, method="profile", n_nodes=8, sampling="partition",
                 graphs_per_step=None, alpha=0.5, g_min=None, g_max=None,
                 p=2, normalize=True, squared=False, sort_key="lex"):
        super().__init__()
        if method not in ("profile", "mds"):
            raise ValueError("method deve ser 'profile' ou 'mds'")
        self.method = method
        self.n_nodes = n_nodes
        self.sampling = sampling
        self.graphs_per_step = graphs_per_step
        self.alpha = alpha
        self.g_min = g_min
        self.g_max = g_max
        self.p = p
        self.normalize = normalize
        self.squared = squared
        self.sort_key = sort_key

    def _embed(self, node_emb_graphs):
        return embed_graphs(node_emb_graphs, method=self.method,
                            normalize=self.normalize, squared=self.squared,
                            sort_key=self.sort_key)

    def forward(self, student_emb, teacher_emb, graphs=None, generator=None):
        """student_emb/teacher_emb: ``(B, d)``. ``graphs`` opcional ``(G, N)``;
        se None, amostra com ``sample_graphs``."""
        B = student_emb.shape[0]
        if graphs is None:
            graphs = sample_graphs(B, self.n_nodes, self.sampling,
                                   self.graphs_per_step, self.alpha, self.g_min,
                                   self.g_max, generator=generator,
                                   device=student_emb.device)

        s_nodes = student_emb[graphs]                 # (G, N, d_s)
        with torch.no_grad():
            t_nodes = teacher_emb[graphs]             # (G, N, d_t)
            g_t = self._embed(t_nodes)
        g_s = self._embed(s_nodes)

        # perda de Minkowski por grafo, média sobre grafos
        return torch.norm(g_s - g_t, p=self.p, dim=-1).mean()
