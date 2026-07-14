"""OFFLINE probe of graph descriptor fidelity/stability (§6/§7.3).

Needs neither a trained model nor data: generates random graphs (point
configurations -> distance matrices), computes the ``profile`` and ``mds`` descriptors
under each normalization scheme and measures, per order N:

  * COLLISION (fidelity): fraction of pairs of STRUCTURALLY DISTINCT graphs whose
    descriptors are ε-close. High = descriptor loses information (profile loses
    node correspondence; mds suffers from cospectrality). Evidence for H3.
  * MDS near-degenerate: fraction of spectra with minimum gap between consecutive
    eigenvalues < ε·(range) — numerical instability of mds.
  * Profile tie rate: fraction of adjacent edges (within a node's profile)
    tied within ε — ordering churn of the profile.

Structural and descriptor distances are normalized by the median to make the
ε thresholds relative and comparable across N and schemes. Writes a CSV.

Usage: python analysis/descriptor_probe.py [--n-graphs 2000] [--out descriptor_probe.csv]
"""

import argparse
import csv
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_rkd.embeddings import (batch_distance_mean, embed_graphs,  # noqa: E402
                                  mds_spectral_embedding, pairwise_distance_matrix)
from graph_rkd.loss import norm_flags  # noqa: E402
from graph_rkd.embeddings import zscore_descriptor  # noqa: E402

NORMS = ("per_graph", "minibatch", "none", "hybrid")
METHODS = ("profile", "mds")


def random_node_embeddings(g, n, dim=32, seed=0):
    """(g, n, dim) random point configurations -> distinct complete graphs."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(g, n, dim, generator=gen)


def _descriptor(node_emb, method, norm):
    normalize, use_scale, zscore = norm_flags(norm)
    scale = batch_distance_mean(node_emb.reshape(-1, node_emb.shape[-1])) if use_scale else None
    g = embed_graphs(node_emb, method=method, normalize=normalize, batch_scale=scale)
    return zscore_descriptor(g) if zscore else g


def _pdist_flat(x, max_pairs=40000, seed=0):
    """Pairwise distances (subsampled) of a tensor (G, D)."""
    G = x.shape[0]
    pairs = list(itertools.combinations(range(G), 2))
    gen = torch.Generator().manual_seed(seed)
    if len(pairs) > max_pairs:
        idx = torch.randperm(len(pairs), generator=gen)[:max_pairs].tolist()
        pairs = [pairs[i] for i in idx]
    ij = torch.tensor(pairs)
    d = (x[ij[:, 0]] - x[ij[:, 1]]).norm(dim=-1)
    return d, ij


def structural_distance(node_emb, ij):
    """Permutation-invariant structural distance between graphs: ||sorted profiles||.

    Uses the SORTED multiset of edges (all off-diagonals sorted) —
    permutation-invariant and independent of the descriptor tested, so it serves as a
    reference to tell whether two graphs are in fact distinct."""
    D = pairwise_distance_matrix(node_emb)                    # (G,N,N)
    N = D.shape[-1]
    mask = ~torch.eye(N, dtype=torch.bool)
    edges = D[:, mask].reshape(D.shape[0], -1)                # (G, N*(N-1))
    edges = torch.sort(edges, dim=-1).values
    return (edges[ij[:, 0]] - edges[ij[:, 1]]).norm(dim=-1)


def collision_rate(desc, node_emb, ij, eps_desc=0.05, delta_struct=0.20):
    """Fraction of structurally distinct pairs (struct > delta·median) whose
    descriptors collide (desc < eps·median)."""
    # descriptor distances on the SAME ij pairs used in struct
    dd = (desc[ij[:, 0]] - desc[ij[:, 1]]).norm(dim=-1)
    sd = structural_distance(node_emb, ij)
    dd_med = dd.median().clamp_min(1e-9)
    sd_med = sd.median().clamp_min(1e-9)
    distinct = sd > delta_struct * sd_med
    collide = dd < eps_desc * dd_med
    n_distinct = int(distinct.sum())
    if n_distinct == 0:
        return 0.0, 0
    return float((distinct & collide).sum()) / n_distinct, n_distinct


def mds_degeneracy(node_emb, norm, eps=0.02):
    normalize, use_scale, _ = norm_flags(norm)
    scale = batch_distance_mean(node_emb.reshape(-1, node_emb.shape[-1])) if use_scale else None
    D = pairwise_distance_matrix(node_emb)
    eig = mds_spectral_embedding(D, normalize=normalize, batch_scale=scale)  # (G,N) desc
    eig, _ = torch.sort(eig, dim=-1, descending=True)
    gaps = (eig[:, :-1] - eig[:, 1:]).abs()
    rng = (eig[:, 0] - eig[:, -1]).abs().clamp_min(1e-9).unsqueeze(-1)
    min_gap = (gaps / rng).min(dim=-1).values
    return float((min_gap < eps).float().mean())


def profile_tie_rate(node_emb, eps=0.01):
    D = pairwise_distance_matrix(node_emb)
    N = D.shape[-1]
    mask = ~torch.eye(N, dtype=torch.bool)
    rows = D[:, mask].reshape(D.shape[0], N, N - 1)
    prof = torch.sort(rows, dim=-1).values
    diffs = (prof[..., 1:] - prof[..., :-1]).abs()
    scale = D.mean(dim=(-2, -1)).clamp_min(1e-9).view(-1, 1, 1)
    return float((diffs < eps * scale).float().mean())


def run(n_graphs, out, n_list=(3, 4, 8, 16, 17), dim=32, seed=0):
    rows = []
    for N in n_list:
        node_emb = random_node_embeddings(n_graphs, N, dim=dim, seed=seed + N)
        _, ij = _pdist_flat(torch.zeros(n_graphs, 1), seed=seed)  # fixed pairs per N
        prof_tie = profile_tie_rate(node_emb)
        for norm in NORMS:
            mds_deg = mds_degeneracy(node_emb, norm)
            for method in METHODS:
                desc = _descriptor(node_emb, method, norm)
                coll, ndist = collision_rate(desc, node_emb, ij)
                rows.append({"N": N, "norm": norm, "method": method,
                             "descriptor_dim": int(desc.shape[-1]),
                             "collision_rate": round(coll, 5),
                             "n_distinct_pairs": ndist,
                             "mds_degenerate_rate": round(mds_deg, 5) if method == "mds" else "",
                             "profile_tie_rate": round(prof_tie, 5) if method == "profile" else ""})
                print(f"N={N:2d} norm={norm:9s} {method:7s} dim={desc.shape[-1]:3d} "
                      f"collision={coll*100:5.2f}% "
                      + (f"mds_degen={mds_deg*100:5.2f}%" if method == "mds"
                         else f"profile_tie={prof_tie*100:5.2f}%"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {len(rows)} rows written to {out}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-graphs", type=int, default=2000)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "descriptor_probe.csv"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.n_graphs, a.out, seed=a.seed)
