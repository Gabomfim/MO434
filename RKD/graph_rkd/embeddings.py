"""Permutation-invariant graph embeddings for Graph-RKD.

The graph is complete and undirected; the adjacency matrix is the **dissimilarity
matrix** (distances between object embeddings). We implement two
deterministic embeddings of the WHOLE graph, both invariant to the physical
shuffling of matrix rows/columns, but sensitive to structural changes (changing
who-is-close-to-whom):

  1. ``node_profile_embedding``  — Ordered Multidimensional Node Profile.
  2. ``mds_spectral_embedding``  — classical MDS eigenvalues (Gram spectrum).

All in PyTorch and **differentiable** (the student side needs gradient):
- ``torch.sort`` propagates gradient through the sorted positions;
- the lexicographic reordering of the profiles is a *gather* (the order
  selection is piecewise constant — ok — but the reordered values carry gradient);
- ``torch.linalg.eigvalsh`` is differentiable (it can be unstable with degenerate
  eigenvalues; use ``jitter`` if necessary).

They accept a single matrix ``(N, N)`` or a batch ``(G, N, N)``.
"""

import numpy as np
import torch

__all__ = [
    "pairwise_distance_matrix", "normalize_distance_matrix",
    "batch_distance_mean", "zscore_descriptor", "NORM_SCHEMES",
    "node_profile_embedding", "mds_spectral_embedding",
    "node_profile_embedding_np", "embed_graphs",
]

# Normalization schemes (axis of EXPERIMENTS_EN §5 / hypothesis H2):
#   per_graph  — divides each graph by its own off-diagonal mean (per-graph
#                scale; loses the CROSS scale between graphs).
#   minibatch  — divides ALL graphs by μ_batch (off-diagonal mean of the
#                K×K matrix of the whole minibatch); restores the cross scale. Mirrors the
#                μ-normalization of RkdDistance (batch-mean).
#   none       — no normalization (keeps the raw scale of the distances).
#   hybrid     — μ_batch (cross scale) + scale-invariant descriptor (z-score
#                of the descriptor across the sampled graphs), also recovering the
#                teacher/student invariance.
NORM_SCHEMES = ("per_graph", "minibatch", "none", "hybrid")


def pairwise_distance_matrix(node_emb, squared=False, eps=1e-12):
    """Node embeddings ``(..., N, d)`` -> distance matrix ``(..., N, N)``."""
    d = torch.cdist(node_emb, node_emb, p=2)
    if squared:
        d = d.clamp_min(0).pow(2)
    return d


def normalize_distance_matrix(D, eps=1e-12):
    """Divides each matrix by the mean of the off-diagonal distances (per-graph).

    Makes the embedding invariant to the SCALE of the distances (essential to compare
    teacher and student, which live in spaces of different dimensions), preserving
    the relative geometry. Mirrors the normalization of ``RkdDistance``.
    """
    N = D.shape[-1]
    off_sum = D.sum(dim=(-2, -1))
    mean = off_sum / (N * (N - 1))            # mean of only the (N*(N-1)) off-diagonals
    mean = mean.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    return D / mean


def batch_distance_mean(node_emb, eps=1e-12):
    """μ_batch — mean of the off-diagonal distances of the K×K minibatch matrix.

    Single scalar (per teacher/student side) used by the ``minibatch``
    and ``hybrid`` normalization: restores the CROSS scale between graphs that per-graph discards.
    """
    B = node_emb.shape[0]
    if B < 2:
        return node_emb.new_tensor(1.0)
    D = torch.cdist(node_emb, node_emb, p=2)
    mean = D.sum() / (B * (B - 1))
    return mean.clamp_min(eps)


def zscore_descriptor(g, eps=1e-6):
    """Standardizes the graph descriptor across the sampled graphs (dim 0).

    ``g`` ``(G, E)`` -> per-dimension z-score over the G graphs. Makes the descriptor
    invariant to scale/shift (scale-invariant component of the hybrid norm).
    """
    if g.dim() != 2 or g.shape[0] < 2:
        return g
    mean = g.mean(dim=0, keepdim=True)
    std = g.std(dim=0, keepdim=True).clamp_min(eps)
    return (g - mean) / std


def _remove_diagonal(D):
    """``(..., N, N)`` -> ``(..., N, N-1)`` removing the diagonal of each row."""
    N = D.shape[-1]
    mask = ~torch.eye(N, dtype=torch.bool, device=D.device)
    return D[..., mask].reshape(*D.shape[:-2], N, N - 1)


def _apply_norm(D, normalize, batch_scale, eps=1e-12):
    """Applies the scale normalization to the distance matrix.

    * ``batch_scale`` (scalar) provided -> ``minibatch``/``hybrid`` normalization.
    * otherwise ``normalize=True`` -> per-graph; ``False`` -> none.
    """
    if batch_scale is not None:
        return D / batch_scale.clamp_min(eps)
    if normalize:
        return normalize_distance_matrix(D, eps=eps)
    return D


def node_profile_embedding(D, sort_key="lex", normalize=False, batch_scale=None):
    """Method 1 — Ordered Multidimensional Node Profile.

    For each node: removes the diagonal and sorts its distances (neighborhood
    profile). Then sorts the PROFILES among themselves (lexicographically by default)
    and flattens. Fixed-size output ``N*(N-1)``.

    Args:
        D: ``(N, N)`` or ``(G, N, N)`` distance matrix(es).
        sort_key: ``"lex"`` (canonical, robust to ties) or ``"mean"``
            (parity with the reference; may break invariance on ties).
        normalize: normalizes by the mean distance beforehand (scale-invariant).
    """
    single = D.dim() == 2
    if single:
        D = D.unsqueeze(0)
    D = _apply_norm(D, normalize, batch_scale)

    off = _remove_diagonal(D)                       # (G, N, N-1)
    profiles = torch.sort(off, dim=-1).values       # sorts within each node

    out = []
    for g in range(profiles.shape[0]):
        prof = profiles[g]                          # (N, N-1)
        if sort_key == "mean":
            order = torch.argsort(prof.mean(dim=1))
        elif sort_key == "lex":
            # canonical lexicographic order (col 0 primary). The order is decided
            # from the detached values; the reordered values keep grad.
            key = prof.detach().cpu().numpy()
            order_np = np.lexsort(key[:, ::-1].T)   # last argument = primary key
            order = torch.as_tensor(order_np, device=prof.device, dtype=torch.long)
        else:
            raise ValueError("sort_key must be 'lex' or 'mean'")
        out.append(prof[order].reshape(-1))         # gather (differentiable) + flatten

    emb = torch.stack(out, dim=0)                   # (G, N*(N-1))
    return emb.squeeze(0) if single else emb


def mds_spectral_embedding(D, normalize=False, jitter=0.0, batch_scale=None):
    """Method 2 — classical MDS eigenvalues (Gram matrix spectrum).

    Double-centering of D² -> Gram ``B = -1/2 · J D² J`` (J = I - 11ᵀ/N);
    eigenvalues of B in decreasing order. Permutation-invariant (eigenvalues
    are invariant under permutation similarity). Fixed-size output ``N``.
    """
    single = D.dim() == 2
    if single:
        D = D.unsqueeze(0)
    D = _apply_norm(D, normalize, batch_scale)

    G, N, _ = D.shape
    D2 = D.pow(2)
    J = torch.eye(N, device=D.device, dtype=D.dtype) - (1.0 / N)
    B = -0.5 * (J @ D2 @ J)
    B = 0.5 * (B + B.transpose(-2, -1))             # forces symmetry (stability)
    if jitter > 0:
        B = B + jitter * torch.eye(N, device=D.device, dtype=D.dtype)
    # eigvalsh has no CUDA kernel for Half (AMP); computes in float32 and converts
    # back. The spectral decomposition needs full precision anyway.
    eig = torch.linalg.eigvalsh(B.float()).to(B.dtype)   # increasing
    eig = eig.flip(-1)                              # decreasing
    return eig.squeeze(0) if single else eig


def embed_graphs(node_emb_graphs, method="profile", normalize=True, squared=False,
                 sort_key="lex", batch_scale=None):
    """Per-graph node embeddings ``(G, N, d)`` -> graph embedding ``(G, *)``.

    method ``"profile"`` -> size N·(N-1); ``"mds"`` -> size N. The size
    depends ONLY on N (not on the feature dimension), so teacher and student
    embeddings of the same graph are directly comparable.

    ``batch_scale`` (scalar) forces the ``minibatch``/``hybrid`` normalization
    (divides all matrices by that μ_batch); when None, uses ``normalize``
    (per-graph if True, none if False).
    """
    D = pairwise_distance_matrix(node_emb_graphs, squared=squared)
    if method == "profile":
        return node_profile_embedding(D, sort_key=sort_key, normalize=normalize,
                                      batch_scale=batch_scale)
    if method == "mds":
        return mds_spectral_embedding(D, normalize=normalize, batch_scale=batch_scale)
    raise ValueError("method must be 'profile' or 'mds'")


# --------------------------------------------------------------------------- #
# numpy reference (identical to the one provided; for parity/validation)       #
# --------------------------------------------------------------------------- #
def node_profile_embedding_np(distance_matrix):
    """numpy reference of Method 1 with mean-based ordering (as in the prompt)."""
    n = distance_matrix.shape[0]
    node_profiles = []
    for i in range(n):
        row = np.delete(distance_matrix[i], i)
        node_profiles.append(np.sort(row))
    node_profiles = np.array(node_profiles)
    row_means = np.mean(node_profiles, axis=1)
    sorted_node_profiles = node_profiles[np.argsort(row_means)]
    return sorted_node_profiles.flatten()
