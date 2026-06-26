# Graph-RKD — escolha do número de nós (N) e combinatória

Generalização do Relational Knowledge Distillation: cada **nó** é um objeto
(amostra do batch), cada **aresta** é a distância entre os embeddings de dois
objetos, o grafo é completo e não-direcionado, e seu embedding é **invariante a
permutação** (mesmo conjunto de nós ⇒ mesmo embedding). Logo um grafo é um
*conjunto* de N nós, não uma tupla ordenada.

Os métodos de *embedding* desses grafos entram depois; este pacote cobre a
**escolha de N** (busca binária) e a **análise combinatória**.

## Combinatória (e uma correção importante)

Para um batch de B amostras e grafos de N nós:

| | fórmula | B=128, N=8 |
|---|---|---|
| grafos únicos (sets — método novo) | `C(B, N)` | 1.43×10¹² |
| N-tuplas ordenadas (RKD tradicional) | `P(B, N) = N!·C(B, N)` | 5.76×10¹⁶ |

- O nº de grafos únicos é **`C(B, N)`** — combinatório, com **pico em N = B/2**
  (`C(128,64) ≈ 2.4×10³⁷`). Ou seja, o *espaço* de grafos possíveis **não** é
  pequeno nem "não-exponencial".
- A vantagem real da invariância a permutação é o **fator N!**: cada grafo
  substitui N! tuplas ordenadas. No RKD tradicional, relações N-árias ordenadas
  custam O(Bᴺ) (exponencial em N); com sets você processa cada grafo uma vez.
- E o **custo por passo** só não explode se você **amostra um nº fixo de grafos**
  (ou particiona o batch) em vez de enumerar `C(B,N)`. Veja o modelo de custo.

Gráfico `C(128,N)` vs `P(128,N)`: ![unique graphs](unique_graphs_128.png)
(gerado por `plot_unique_graphs(128)`).

## Busca binária pelo melhor N

Busca binária acha uma **fronteira monotônica**, não o máximo de uma função
monotônica. Como assumimos que a qualidade cresce com N, "melhor N" = **maior N
viável sob uma restrição** (orçamento de compute/memória). `find_best_n` faz
isso em O(log B) avaliações:

```python
from graph_rkd import find_best_n

# modelo de custo (default): orçamento em arestas/passo
find_best_n(batch_size=128, edge_budget=1024)          # -> 17

# medição REAL (quando tiver GPU + os métodos de embedding):
#   feasible_fn(N) deve ser monotônico: True enquanto couber (sem OOM / tempo ok)
find_best_n(128, edge_budget=0, feasible_fn=lambda N: roda_um_passo_ok(N))
```

Se houver **retornos decrescentes** (qualidade não é "∝ N" puro), use
`find_knee_n(quality_fn, ...)`: busca binária pelo maior N cujo ganho relativo
de qualidade ao dobrar N ainda supera `rel_tol` (o "joelho" da curva).
`quality_fn(N)` é uma medição real (ex.: recall/top-1 do student).

## Regra matemática derivada

Esquema **partition** (cada amostra usada 1×): o batch é dividido em ⌊B/N⌋
grafos completos disjuntos `K_N`. Arestas por grafo = `C(N,2) = N(N-1)/2`, logo:

```
E (arestas/passo) = (B/N) · C(N,2) = B·(N-1)/2          (exato quando N | B)
```

Note que **E cresce só linearmente em N** (dobrar N ≈ dobra o custo, pois o nº de
grafos cai pela metade) — é nesse sentido preciso que "não cresce
exponencialmente" vale. Resolvendo para o maior N sob orçamento E:

```
B·(N-1)/2 ≤ E   ⟹   N* = ⌊ 2E/B + 1 ⌋     ⟹     (N*-1)·B = 2E = const
```

Ou seja **N\* é afim em 1/B** (`N* ≈ 2E/B + 1`). O ajuste empírico de
`derive_scaling_rule` sobre B ∈ {128…2048} reproduz isso **exatamente**
(`N* ≈ 2048·(1/B) + 1.00`, R² = 1.0000, resíduo ~10⁻¹⁴ para E=1024).

Esquema **sample** (G grafos fixos de N nós): `E = G·C(N,2)` ⟹
`N* ≈ sqrt(2E/G)`, **independente de B**.

> Estas regras vêm do *modelo de custo* (proxy = nº de arestas). Quando os
> métodos de embedding e a GPU estiverem disponíveis, troque o predicado por uma
> medição real (`feasible_fn`/`quality_fn`) e re-derive — a forma funcional
> (afim em 1/B para partition; constante para sample) deve se manter se o custo
> do embedding for ~linear no nº de arestas.

## API

- `unique_graphs(B, N)` / `ordered_tuples(B, N)` / `permutation_reduction(N)`
- `step_edge_cost(B, N, scheme, graphs_per_step)`
- `largest_feasible_n(predicate, lo, hi)` — busca binária monotônica genérica
- `find_best_n(batch_size, edge_budget, scheme, ..., feasible_fn=None)`
- `find_knee_n(quality_fn, lo, hi, rel_tol)`
- `derive_scaling_rule(batch_sizes, edge_budget, scheme, graphs_per_step)`
- `plot_unique_graphs(B, path)`

Demo: `python graph_rkd/node_search.py`.
