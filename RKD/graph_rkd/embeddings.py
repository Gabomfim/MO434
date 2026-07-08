"""Embeddings de grafo invariantes a permutação para Graph-RKD.

O grafo é completo e não-direcionado; a matriz de adjacência é a **matriz de
dissimilaridade** (distâncias entre embeddings de objetos). Implementamos dois
embeddings determinísticos do grafo INTEIRO, ambos invariantes ao embaralhamento
físico de linhas/colunas da matriz, mas sensíveis a mudanças estruturais (mexer
quem-está-perto-de-quem):

  1. ``node_profile_embedding``  — Perfil Multidimensional de Nós Ordenado.
  2. ``mds_spectral_embedding``  — autovalores do MDS clássico (espectro da Gram).

Tudo em PyTorch e **diferenciável** (o lado do student precisa de gradiente):
- o ``torch.sort`` propaga gradiente pelas posições ordenadas;
- a reordenação lexicográfica dos perfis é um *gather* (a seleção de ordem é
  constante por partes — ok — mas os valores reordenados carregam gradiente);
- ``torch.linalg.eigvalsh`` é diferenciável (pode ser instável com autovalores
  degenerados; use ``jitter`` se necessário).

Aceitam matriz única ``(N, N)`` ou batch ``(G, N, N)``.
"""

import numpy as np
import torch

__all__ = [
    "pairwise_distance_matrix", "normalize_distance_matrix",
    "batch_distance_mean", "zscore_descriptor", "NORM_SCHEMES",
    "node_profile_embedding", "mds_spectral_embedding",
    "node_profile_embedding_np", "embed_graphs",
]

# Esquemas de normalização (eixo do EXPERIMENTS_EN §5 / hipótese H2):
#   per_graph  — divide cada grafo pela sua própria média off-diagonal (escala
#                por grafo; perde escala CRUZADA entre grafos).
#   minibatch  — divide TODOS os grafos por μ_batch (média off-diagonal da matriz
#                K×K do minibatch inteiro); restaura a escala cruzada. Espelha a
#                μ-normalização do RkdDistance (batch-mean).
#   none       — sem normalização (mantém a escala bruta das distâncias).
#   hybrid     — μ_batch (escala cruzada) + descritor scale-invariante (z-score
#                do descritor entre os grafos amostrados), recuperando também a
#                invariância teacher/student.
NORM_SCHEMES = ("per_graph", "minibatch", "none", "hybrid")


def pairwise_distance_matrix(node_emb, squared=False, eps=1e-12):
    """Embeddings de nós ``(..., N, d)`` -> matriz de distâncias ``(..., N, N)``."""
    d = torch.cdist(node_emb, node_emb, p=2)
    if squared:
        d = d.clamp_min(0).pow(2)
    return d


def normalize_distance_matrix(D, eps=1e-12):
    """Divide cada matriz pela média das distâncias fora da diagonal (per-graph).

    Torna o embedding invariante à ESCALA das distâncias (essencial para comparar
    teacher e student, que vivem em espaços de dimensões diferentes), preservando
    a geometria relativa. Espelha a normalização do ``RkdDistance``.
    """
    N = D.shape[-1]
    off_sum = D.sum(dim=(-2, -1))
    mean = off_sum / (N * (N - 1))            # média só das (N*(N-1)) off-diagonais
    mean = mean.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    return D / mean


def batch_distance_mean(node_emb, eps=1e-12):
    """μ_batch — média das distâncias off-diagonal da matriz K×K do minibatch.

    Escalar único (por lado teacher/student) usado pela normalização ``minibatch``
    e ``hybrid``: restaura a escala CRUZADA entre grafos que a per-graph descarta.
    """
    B = node_emb.shape[0]
    if B < 2:
        return node_emb.new_tensor(1.0)
    D = torch.cdist(node_emb, node_emb, p=2)
    mean = D.sum() / (B * (B - 1))
    return mean.clamp_min(eps)


def zscore_descriptor(g, eps=1e-6):
    """Padroniza o descritor de grafo entre os grafos amostrados (dim 0).

    ``g`` ``(G, E)`` -> z-score por dimensão sobre os G grafos. Torna o descritor
    invariante a escala/deslocamento (componente scale-invariante da norm hybrid).
    """
    if g.dim() != 2 or g.shape[0] < 2:
        return g
    mean = g.mean(dim=0, keepdim=True)
    std = g.std(dim=0, keepdim=True).clamp_min(eps)
    return (g - mean) / std


def _remove_diagonal(D):
    """``(..., N, N)`` -> ``(..., N, N-1)`` removendo a diagonal de cada linha."""
    N = D.shape[-1]
    mask = ~torch.eye(N, dtype=torch.bool, device=D.device)
    return D[..., mask].reshape(*D.shape[:-2], N, N - 1)


def _apply_norm(D, normalize, batch_scale, eps=1e-12):
    """Aplica a normalização de escala à matriz de distâncias.

    * ``batch_scale`` (escalar) fornecido -> normalização ``minibatch``/``hybrid``.
    * senão ``normalize=True`` -> per-graph; ``False`` -> none.
    """
    if batch_scale is not None:
        return D / batch_scale.clamp_min(eps)
    if normalize:
        return normalize_distance_matrix(D, eps=eps)
    return D


def node_profile_embedding(D, sort_key="lex", normalize=False, batch_scale=None):
    """Método 1 — Perfil Multidimensional de Nós Ordenado.

    Para cada nó: remove a diagonal e ordena suas distâncias (perfil de
    vizinhança). Depois ordena os PERFIS entre si (lexicograficamente por padrão)
    e achata. Saída de tamanho fixo ``N*(N-1)``.

    Args:
        D: ``(N, N)`` ou ``(G, N, N)`` matriz(es) de distância.
        sort_key: ``"lex"`` (canônica, robusta a empates) ou ``"mean"``
            (paridade com a referência; pode quebrar invariância em empates).
        normalize: normaliza pela distância média antes (escala-invariante).
    """
    single = D.dim() == 2
    if single:
        D = D.unsqueeze(0)
    D = _apply_norm(D, normalize, batch_scale)

    off = _remove_diagonal(D)                       # (G, N, N-1)
    profiles = torch.sort(off, dim=-1).values       # ordena dentro de cada nó

    out = []
    for g in range(profiles.shape[0]):
        prof = profiles[g]                          # (N, N-1)
        if sort_key == "mean":
            order = torch.argsort(prof.mean(dim=1))
        elif sort_key == "lex":
            # ordem lexicográfica canônica (col 0 primária). A ordem é decidida
            # a partir dos valores destacados; os valores reordenados mantêm grad.
            key = prof.detach().cpu().numpy()
            order_np = np.lexsort(key[:, ::-1].T)   # último argumento = chave primária
            order = torch.as_tensor(order_np, device=prof.device, dtype=torch.long)
        else:
            raise ValueError("sort_key deve ser 'lex' ou 'mean'")
        out.append(prof[order].reshape(-1))         # gather (diferenciável) + flatten

    emb = torch.stack(out, dim=0)                   # (G, N*(N-1))
    return emb.squeeze(0) if single else emb


def mds_spectral_embedding(D, normalize=False, jitter=0.0, batch_scale=None):
    """Método 2 — autovalores do MDS clássico (espectro da matriz de Gram).

    Dupla-centralização de D² -> Gram ``B = -1/2 · J D² J`` (J = I - 11ᵀ/N);
    autovalores de B em ordem decrescente. Invariante a permutação (autovalores
    são invariantes por similaridade de permutação). Saída de tamanho fixo ``N``.
    """
    single = D.dim() == 2
    if single:
        D = D.unsqueeze(0)
    D = _apply_norm(D, normalize, batch_scale)

    G, N, _ = D.shape
    D2 = D.pow(2)
    J = torch.eye(N, device=D.device, dtype=D.dtype) - (1.0 / N)
    B = -0.5 * (J @ D2 @ J)
    B = 0.5 * (B + B.transpose(-2, -1))             # força simetria (estabilidade)
    if jitter > 0:
        B = B + jitter * torch.eye(N, device=D.device, dtype=D.dtype)
    # eigvalsh não tem kernel CUDA p/ Half (AMP); calcula em float32 e converte
    # de volta. A decomposição espectral precisa de precisão plena de qualquer forma.
    eig = torch.linalg.eigvalsh(B.float()).to(B.dtype)   # crescente
    eig = eig.flip(-1)                              # decrescente
    return eig.squeeze(0) if single else eig


def embed_graphs(node_emb_graphs, method="profile", normalize=True, squared=False,
                 sort_key="lex", batch_scale=None):
    """Embeddings de nós por grafo ``(G, N, d)`` -> embedding do grafo ``(G, *)``.

    method ``"profile"`` -> tamanho N·(N-1); ``"mds"`` -> tamanho N. O tamanho
    depende SÓ de N (não da dimensão dos features), então embeddings de
    teacher e student do mesmo grafo são diretamente comparáveis.

    ``batch_scale`` (escalar) força a normalização ``minibatch``/``hybrid``
    (divide todas as matrizes por esse μ_batch); quando None, usa ``normalize``
    (per-graph se True, none se False).
    """
    D = pairwise_distance_matrix(node_emb_graphs, squared=squared)
    if method == "profile":
        return node_profile_embedding(D, sort_key=sort_key, normalize=normalize,
                                      batch_scale=batch_scale)
    if method == "mds":
        return mds_spectral_embedding(D, normalize=normalize, batch_scale=batch_scale)
    raise ValueError("method deve ser 'profile' ou 'mds'")


# --------------------------------------------------------------------------- #
# Referência numpy (idêntica à fornecida; para paridade/validação)            #
# --------------------------------------------------------------------------- #
def node_profile_embedding_np(distance_matrix):
    """Referência numpy do Método 1 com ordenação por média (como no enunciado)."""
    n = distance_matrix.shape[0]
    node_profiles = []
    for i in range(n):
        row = np.delete(distance_matrix[i], i)
        node_profiles.append(np.sort(row))
    node_profiles = np.array(node_profiles)
    row_means = np.mean(node_profiles, axis=1)
    sorted_node_profiles = node_profiles[np.argsort(row_means)]
    return sorted_node_profiles.flatten()
