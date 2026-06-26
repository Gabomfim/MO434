"""Testes dos embeddings de grafo: paridade c/ numpy, invariância, sensibilidade,
diferenciabilidade e a perda Graph-RKD. Rode: python graph_rkd/selftest.py"""

import numpy as np
import torch

import torch.nn.functional as F

from graph_rkd import (
    GraphContrastiveDistillLoss,
    GraphRKDLoss,
    SampledGraphContrastiveLoss,
    mds_spectral_embedding,
    node_profile_embedding,
    node_profile_embedding_np,
    pairwise_distance_matrix,
    sample_graphs,
)


def _matrices():
    n = 4
    rng = np.random.RandomState(100)
    A = rng.uniform(0.1, 2.0, size=(n, n))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    p = np.array([2, 0, 3, 1])
    A_shuf = A[p][:, p]                 # mesmo grafo, índices embaralhados
    B = A.copy()
    B[0, 1] = B[1, 0] = 9.9            # grafo estruturalmente diferente
    return A, A_shuf, B


def test_numpy_reference_validation():
    A, A_shuf, B = _matrices()
    eA = node_profile_embedding_np(A)
    eAs = node_profile_embedding_np(A_shuf)
    eB = node_profile_embedding_np(B)
    d_inv = np.linalg.norm(eA - eAs)
    d_sen = np.linalg.norm(eA - eB)
    print(f"[numpy ref] A vs A_shuffled = {d_inv:.6f} (=0)  | A vs B = {d_sen:.6f} (>0)")
    assert d_inv < 1e-9 and d_sen > 1e-6


def test_torch_parity_with_numpy():
    A, _, _ = _matrices()
    D = torch.tensor(A, dtype=torch.float64)
    torch_mean = node_profile_embedding(D, sort_key="mean").numpy()
    np_ref = node_profile_embedding_np(A)
    err = np.max(np.abs(torch_mean - np_ref))
    print(f"[parity] torch(mean) vs numpy ref: max abs err = {err:.2e}")
    assert err < 1e-9


def test_invariance_and_sensitivity_torch():
    A, A_shuf, B = _matrices()
    tA, tAs, tB = (torch.tensor(M, dtype=torch.float64) for M in (A, A_shuf, B))
    for name, fn in [("profile/lex", lambda D: node_profile_embedding(D, "lex")),
                     ("mds", mds_spectral_embedding)]:
        inv = torch.norm(fn(tA) - fn(tAs)).item()
        sen = torch.norm(fn(tA) - fn(tB)).item()
        print(f"[{name}] A vs A_shuffled = {inv:.2e} (=0) | A vs B = {sen:.4f} (>0)")
        assert inv < 1e-9 and sen > 1e-6


def test_mean_sort_can_break_invariance():
    # Duas linhas com a MESMA média mas perfis diferentes -> 'mean' pode quebrar
    # sob permutação (argsort desempata por posição); 'lex' não.
    D = torch.tensor([
        [0., 1., 2., 3.],
        [1., 0., 4., 1.],   # off-diag {1,4,1} média 2.0
        [2., 4., 0., 0.],   # off-diag {2,4,0} média 2.0  (empata com a linha 1)
        [3., 1., 0., 0.],
    ], dtype=torch.float64)
    D = (D + D.T) / 2
    D.fill_diagonal_(0)
    p = [1, 0, 2, 3]
    Dp = D[p][:, p]
    for key in ("mean", "lex"):
        diff = torch.norm(node_profile_embedding(D, key)
                          - node_profile_embedding(Dp, key)).item()
        print(f"[tie test] sort_key={key}: A vs permutado = {diff:.2e}")
    # lex deve ser exatamente invariante
    assert torch.norm(node_profile_embedding(D, "lex")
                      - node_profile_embedding(Dp, "lex")).item() < 1e-9


def test_differentiability():
    torch.manual_seed(0)
    x = torch.randn(6, 8, requires_grad=True)   # 6 nós, dim 8
    D = pairwise_distance_matrix(x)
    for name, emb in [("profile", node_profile_embedding(D, "lex")),
                      ("mds", mds_spectral_embedding(D))]:
        g, = torch.autograd.grad(emb.sum(), x, retain_graph=True)
        ok = torch.isfinite(g).all().item() and g.abs().sum().item() > 0
        print(f"[grad] {name}: grad finito e não-nulo = {ok}")
        assert ok


def test_graph_rkd_loss():
    torch.manual_seed(0)
    B, ds, dt = 32, 16, 24
    student = torch.randn(B, ds, requires_grad=True)
    teacher = torch.randn(B, dt)
    gen = torch.Generator().manual_seed(0)
    graphs = sample_graphs(B, n_nodes=8, sampling="partition", generator=gen)
    for method in ("profile", "mds"):
        loss_fn = GraphRKDLoss(method=method, n_nodes=8)
        loss = loss_fn(student, teacher, graphs=graphs)
        g, = torch.autograd.grad(loss, student, retain_graph=True)
        print(f"[loss/{method}] loss={loss.item():.4f} grad_norm={g.norm().item():.4f}")
        assert loss.item() > 0 and torch.isfinite(g).all()
    # perda ~0 quando student == teacher (mesma dim)
    z = torch.randn(B, dt)
    loss_same = GraphRKDLoss(method="profile", n_nodes=8)(z, z.detach(), graphs=graphs)
    print(f"[loss/profile] student==teacher -> loss={loss_same.item():.2e} (~0)")
    assert loss_same.item() < 1e-6


def test_infonce_matches_reference_formula():
    # Vetorizado (cross_entropy) deve igualar -log(pos/(pos+Σneg)) sobre OS MESMOS
    # negativos. Fixamos os negativos passando um pool e gerador idêntico.
    torch.manual_seed(0)
    A, E, P, M = 5, 16, 40, 8
    anchor = torch.randn(A, E)
    positive = anchor + 0.1 * torch.randn(A, E)
    pool = torch.randn(P, E)
    tau = 0.1

    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    pooln = F.normalize(pool, dim=-1)
    gen = torch.Generator().manual_seed(123)
    neg_idx = torch.randint(0, P, (A, M), generator=gen)
    pos_sim = torch.exp((a * p).sum(-1) / tau)
    neg = pooln[neg_idx]
    neg_sim = torch.exp((a.unsqueeze(1) * neg).sum(-1) / tau).sum(1)
    ref = (-torch.log(pos_sim / (pos_sim + neg_sim))).mean()

    gen2 = torch.Generator().manual_seed(123)
    mine = SampledGraphContrastiveLoss(temperature=tau, num_negative_samples=M)(
        anchor, positive, pool, generator=gen2)
    print(f"[infonce] vetorizado={mine.item():.6f} vs formula ref={ref.item():.6f}")
    assert abs(mine.item() - ref.item()) < 1e-5


def test_contrastive_distill():
    torch.manual_seed(0)
    B, ds, dt = 48, 16, 24
    student = torch.randn(B, ds, requires_grad=True)
    teacher = torch.randn(B, dt)
    gen = torch.Generator().manual_seed(0)
    graphs = sample_graphs(B, n_nodes=8, sampling="partition", generator=gen)
    for method in ("profile", "mds"):
        loss_fn = GraphContrastiveDistillLoss(method=method, n_nodes=8,
                                              num_negatives=5)
        g0 = torch.Generator().manual_seed(7)
        loss = loss_fn(student, teacher, graphs=graphs, generator=g0)
        grad, = torch.autograd.grad(loss, student, retain_graph=True)
        print(f"[contrastive/{method}] loss={loss.item():.4f} "
              f"grad_norm={grad.norm().item():.4f}")
        assert loss.item() > 0 and torch.isfinite(grad).all() and grad.norm() > 0


if __name__ == "__main__":
    test_numpy_reference_validation()
    test_torch_parity_with_numpy()
    test_invariance_and_sensitivity_torch()
    test_mean_sort_can_break_invariance()
    test_differentiability()
    test_graph_rkd_loss()
    test_infonce_matches_reference_formula()
    test_contrastive_distill()
    print("\nTODOS OS TESTES PASSARAM.")
