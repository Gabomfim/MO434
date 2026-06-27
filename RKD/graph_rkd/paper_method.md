# Graph Relational Knowledge Distillation

## Related Work

**Knowledge distillation.** Knowledge distillation was introduced to compress a
large teacher into a smaller student by matching softened output distributions
(Hinton et al., 2015). Subsequent work transferred *intermediate* knowledge
rather than only logits: FitNets matched hidden feature maps (Romero et al.,
2015), and Attention Transfer matched spatial attention maps derived from
activations (Zagoruyko & Komodakis, 2017). These methods align individual
representations point by point.

**Relational knowledge distillation.** Relational Knowledge Distillation (RKD)
shifted the target from individual representations to the *relations* between
examples, transferring pairwise distances and triplet angles (Park et al.,
2019). Our method generalizes this idea to relations of arbitrary order $N$ by
modeling each minibatch as a complete weighted graph; rather than enumerating the
super-exponentially many ordered tuples that a naive higher-order RKD would
require, we exploit permutation invariance to work with unordered node sets and
sample graphs to keep the loss tractable.

**Contrastive distillation and InfoNCE.** Contrastive learning maximizes
agreement between related views while separating unrelated ones, formalized by
the InfoNCE objective and its mutual-information bound (van den Oord et al.,
2018) and scaled up by SimCLR (Chen et al., 2020) and MoCo (He et al., 2020).
Contrastive Representation Distillation (CRD) applied this cross-model, pulling
the student's representation toward the teacher's for the same input and pushing
it away from others (Tian et al., 2020). Our contrastive objective adopts this
view at the level of *graph* embeddings, with stochastic negative sampling in the
tradition of noise-contrastive estimation (Gutmann & Hyvärinen, 2010; Mikolov et
al., 2013).

**Graph learning and permutation invariance.** Sampling-based training over
large relational structures is standard in graph machine learning: GraphSAGE
samples fixed-size neighborhoods (Hamilton et al., 2017) and PinSage scales this
to web-scale graphs (Ying et al., 2018). The requirement that a set-level
representation be invariant to input ordering is the defining property of set
functions (Zaheer et al., 2017). Our graph descriptors follow this principle:
the ordered node-profile embedding builds a permutation-invariant summary by
sorting, while the spectral embedding uses the eigenvalues of the classical
multidimensional-scaling Gram matrix (Torgerson, 1952; Cox & Cox, 2000), which
are invariant under permutation similarity.

## From relational tuples to relational graphs

Relational Knowledge Distillation (RKD) transfers the *mutual relations* between
data examples from a teacher to a student, rather than their individual
representations. In its original form, these relations are defined over
**ordered tuples** of examples: pairwise distances are computed over ordered
pairs and angular relations over ordered triples. This formulation has two
limitations. First, it fixes the relational arity to small constants (2 and 3).
Second, generalizing it to higher-order relations of size $N$ is combinatorially
prohibitive, because the number of ordered $N$-tuples drawn without repetition
from a minibatch of size $K$ is the falling factorial

$$P(K, N) \;=\; \frac{K!}{(K-N)!} \;=\; K (K-1)\cdots(K-N+1),$$

which grows super-exponentially in the relational order $N$. Crucially, much of
this count is redundant: the relational structure induced by a set of examples
does not depend on the order in which they are listed.

## Modeling the structure as graphs

We instead model the relational structure of a minibatch as a **complete,
undirected, weighted graph**, in which each node is an example and each edge
weight is the dissimilarity (distance) between the embeddings of its two
endpoints. Because the graph is complete and undirected, it is fully determined
by the *set* of its nodes, and the ordering of nodes and edges carries no
information. The number of distinct $N$-node graphs that can be formed from a
minibatch of size $K$ is therefore the binomial coefficient

$$\binom{K}{N} \;=\; \frac{P(K,N)}{N!},$$

i.e., a single row of Pascal's triangle. Adopting the set view rather than the
tuple view thus removes a factor of $N!$ — exactly the orderings that the tuple
formulation counts redundantly.

## Tractability through graph sampling

The number of distinct graphs is nonetheless very large: $\binom{K}{N}$ is
maximized at $N=K/2$ and reaches astronomical magnitudes (for $K=128$,
$\binom{128}{64}\approx 2.4\times10^{37}$), so enumerating all graphs to
evaluate the loss is infeasible. We make the per-minibatch loss computable by
**sampling** a small set of graphs at each step rather than enumerating them.
Each step either partitions the shuffled minibatch into $\lfloor K/N \rfloor$
node-disjoint graphs or draws a set of random $N$-node subsets, and the sampled
set is re-randomized every iteration. Provided the sampling is representative,
the gradient computed over the sample is an unbiased estimator of the gradient
over the full graph space, so the student converges toward the same optimum at a
per-step cost of $O(G\cdot N^2)$ (with $G$ the number of sampled graphs),
independent of $\binom{K}{N}$.

Because the size of the graph space varies strongly with the relational order —
$\binom{K}{N}$ spans many orders of magnitude across a Pascal row — we let the
number of sampled graphs grow with that size rather than holding it fixed. As the
raw count is unusable directly, we scale $G$ with its *description length*
(the number of bits needed to identify one graph),

$$G(N) \;=\; \mathrm{clamp}\!\big(\,\alpha\,\log_2\!\tbinom{K}{N},\; g_{\min},\; g_{\max}\big),$$

so the sampler draws more graphs where the space is exponentially larger (peaking
at $N=K/2$) and fewer at the extremes, while remaining bounded. We use
$\alpha=0.5$ and $g_{\min}=\lfloor K/N\rfloor$ (at least one full-batch partition,
ensuring coverage no worse than node-disjoint sampling), with an optional cap
$g_{\max}$ to bound compute. This is a heuristic for matching sample diversity to
the space rather than a statistical requirement: the variance of a Monte-Carlo
estimator does not depend on population size, but sampling more graphs at larger
$N$ covers a more diverse and higher-variance space.

## Permutation-invariant graph embeddings

Each sampled graph is summarized by a fixed-length descriptor computed from its
dissimilarity matrix $D\in\mathbb{R}^{N\times N}$. The descriptor must be
invariant to permutations of the node indexing (relabeling rows/columns of $D$
leaves the graph unchanged) while remaining sensitive to its structure. We use
two such embeddings.

1. **Ordered node-profile embedding.** The off-diagonal entries of each row are
   sorted to form a node's neighborhood profile, the profiles are then ordered
   lexicographically across nodes, and the result is flattened into a vector of
   size $N(N-1)$.
2. **Spectral (classical-MDS) embedding.** The squared dissimilarity matrix is
   double-centered into a Gram matrix
   $B=-\tfrac{1}{2}\,J D^{(2)} J$ with $J=I-\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top$,
   whose eigenvalues, sorted in decreasing order, give a descriptor of size $N$;
   eigenvalues are invariant under permutation similarity by construction.

Both embeddings are differentiable and are computed on a distance matrix
normalized by its mean edge weight, which makes them scale-invariant and
therefore directly comparable between teacher and student. Notably, the
descriptor dimensionality depends only on $N$, so teacher and student graph
embeddings inhabit the same space regardless of their (different) representation
dimensions.

## Distillation objectives

Given the same sampled node set on both sides, we transfer the teacher's
relational geometry to the student through two complementary objectives.

- **Regression.** Minimize the Minkowski distance between the student's and the
  teacher's graph embeddings, $\lVert g_s - g_t \rVert_p$, directly matching the
  teacher's geometry graph by graph. Simple and parameter-free.
- **Contrastive.** An InfoNCE formulation in the spirit of contrastive
  representation distillation: the student's embedding of graph $i$ is the
  anchor, the teacher's embedding of the same graph $i$ is the positive, and the
  teacher's embeddings of other graphs $j\neq i$ provide the negatives, of which
  $M$ are sampled per anchor. The loss is the cross-entropy of the resulting
  $(1{+}M)$-way classification with the positive as the target, scaled by a
  temperature $\tau$. This bounds the number of inter-graph comparisons to
  $G\cdot M$ and tends to transfer relational structure more effectively.

## Choosing the relational order $N$

The relational order $N$ is the central hyperparameter of the method. Its
selection has two distinct parts, and we treat them differently.

**Compute-budget ceiling (binary search).** The per-step cost under the
partition scheme is $E = K(N-1)/2$ edges, so the largest order that fits a
budget $E$ has the closed form

$$N_{\max} \;\le\; \frac{2E}{K} + 1,$$

i.e., the feasible order is inversely proportional to the batch size. Feasibility
is a *monotonic, deterministic* predicate (if $N$ is infeasible, so is every
larger $N$), so the ceiling $N_{\max}$ is found exactly by **binary search** in
$O(\log K)$ evaluations and without any training.

**Quality-driven selection (budget-bounded log-spaced sweep).** We deliberately
do *not* use binary search on the validation quality $q(N)$. Binary search makes
an irreversible left/right cut at each midpoint and is only valid when the search
signal is monotonic and noiseless — neither holds here: $q(N)$ may rise and then
fall (larger $N$ yields richer relations but fewer graphs per batch and a harder
optimization), and each $q(N)$ is a single, stochastic training estimate. Instead
we evaluate a set of **log-spaced candidate orders**

$$\mathcal{C} \;=\; \{\,n_{\min},\; b\,n_{\min},\; b^2 n_{\min},\;\dots,\; N_{\max}\,\}, \qquad b=2,$$

which contains $\approx\log_b N_{\max}$ points — the *same* training budget a
binary search would spend — but reveals the shape of the quality–$N$ curve
without assuming monotonicity. Each candidate is trained (short schedule) with
$R$ seeds; its score is the mean best validation recall@1 (test is never used for
selection). We then pick $N^\*$ either by $\arg\max$ or, for parsimony, by the
**one-standard-error rule** — the smallest $N$ whose mean score is within one
standard error of the best, preferring cheaper (smaller-$N$) configurations when
the difference is not statistically significant. A final long run is trained at
$N^\*$. This costs $O(R\log_b N_{\max})$ trainings, is robust to non-monotonic and
noisy quality curves, and exposes the curve for inspection.

## Algorithms

**Algorithm 1 — Permutation-invariant graph embedding.** Maps an $N\times N$
dissimilarity matrix to a fixed-length, order-invariant descriptor.

```
Input : D in R^{N x N} (distances); method in {profile, mds}; normalize
if normalize:                       # scale-invariance (teacher vs student)
    D <- D / mean_{i != j} D_ij
if method = profile:
    for each row i:
        p_i <- sort_ascending({ D_ij : j != i })      # neighborhood profile
    order <- lexicographic_argsort(p_1, ..., p_N)      # canonical node order
    return flatten([ p_order(1) ; ... ; p_order(N) ])  # in R^{N(N-1)}
else:                               # mds (classical MDS spectrum)
    J <- I - (1/N) * 1 1^T
    B <- -1/2 * J (D ⊙ D) J                            # double-centered Gram
    return eigenvalues(B) sorted descending            # in R^{N}
```

**Algorithm 2 — Adaptive graph sampling.** Number of graphs grows with the
Pascal-row size $\binom{K}{N}$ via its description length.

```
Input : batch size K, order N, alpha, g_min (default floor(K/N)), g_max
G <- round( alpha * log2( C(K, N) ) )
G <- clamp(G, g_min, g_max)
return G random N-subsets of {1, ..., K}   # (or floor(K/N) disjoint, if partition)
```

**Algorithm 3 — Graph-RKD distillation step.** Standard cross-entropy plus the
relational graph loss; teacher is frozen.

```
Input : student f_s, teacher f_t, minibatch (X, y), order N, weight lambda_g,
        objective in {regression, contrastive}, p, M, tau
Z_s <- f_s.features(X)                       # node embeddings (B x d_s)
Z_t <- f_t.features(X)   (no grad)           # node embeddings (B x d_t)
{ S_1, ..., S_G } <- sample graphs (Algorithm 2)
for g = 1 .. G:                              # same node set on both sides
    e_s[g] <- GraphEmbedding( pdist_euclid(Z_s[S_g]) )      # Algorithm 1
    e_t[g] <- GraphEmbedding( pdist_euclid(Z_t[S_g]) )      # (no grad)
if objective = regression:
    L_rel <- mean_g  || e_s[g] - e_t[g] ||_p
else:                                        # contrastive (InfoNCE, CRD-style)
    for g: logits <- [ sim(e_s[g], e_t[g]) ,  { sim(e_s[g], e_t[h]) : h in negs(g, M) } ] / tau
    L_rel <- mean_g  CrossEntropy(logits, target = 0)       # positive at index 0
L <- CrossEntropy(f_s(X), y) + lambda_g * L_rel
backprop and update f_s
```

**Algorithm 4 — Selecting $N$ (budget-bounded log-spaced sweep).** Binary search
fixes only the compute ceiling (exact); the quality-driven choice is a log-spaced
sweep with seed averaging, which is robust to non-monotonic, noisy quality curves.

```
Input : batch K, edge budget E, n_min, base b, seeds R, rule in {argmax, 1se}
# 1) compute ceiling: binary search on the monotone feasibility predicate
N_max <- largest N with K (N-1)/2 <= E         # = floor(2E/K + 1), no training
# 2) log-spaced candidates (same budget as binary search, but shape-revealing)
C <- { n_min, b*n_min, b^2*n_min, ..., N_max }  (sorted, unique)
# 3) seed-averaged validation quality of each candidate
for N in C:
    vals <- [ best_val_recall@1( distill(order=N, short schedule, seed=r) ) : r in 1..R ]
    mean[N] <- mean(vals);  sem[N] <- std(vals) / sqrt(R)
# 4) selection
if rule = argmax:
    N* <- argmax_N mean[N]
else:                                           # one-standard-error rule
    best <- argmax_N mean[N]
    N* <- smallest N in C with mean[N] >= mean[best] - sem[best]
# 5) final long run at N*
return N*
```

## Datasets and splits

We evaluate on two standard fine-grained **deep-metric-learning / retrieval**
benchmarks. **Stanford Cars-196** (Krause et al., 2013) contains $16{,}185$
images of $196$ car models; **Caltech-UCSD Birds-200-2011 (CUB-200)** (Wah et
al., 2011) contains $11{,}788$ images of $200$ bird species.

We use the standard **metric-learning split**, in which the training and test
**classes are disjoint** (e.g., for Cars-196 the first $98$ classes for training
and the remaining $98$ for testing; analogously the first/last $100$ for
CUB-200). The model is evaluated by **recall@K** ($K\in\{1,2,4,8\}$): for each
test image, whether a same-class neighbor appears among its $K$ nearest
neighbors in embedding space. Because the disjoint split provides no validation
set, we hold out a **disjoint subset of the training classes** ($20\%$, fixed
seed) as validation — so validation classes are unseen during training. All
hyperparameter selection — the checkpoint kept for final evaluation and the
relational order $N$ — relies **only on validation recall@1**; the test set is
read once, with the validation-selected checkpoint, and recall@K is reported on
train, validation, and test.

## Preprocessing and data augmentation

All images are processed as $224\times224$ RGB tensors. During **training** we
apply `RandomResizedCrop(224)` followed by a random horizontal flip; at
**evaluation** (validation and test) we resize the shorter side to $256$ and take
a central $224$ crop. In both cases pixels are scaled to $[0,1]$ and normalized
with ImageNet statistics (mean $(0.485, 0.456, 0.406)$, std
$(0.229, 0.224, 0.225)$), matching the ImageNet-pretrained teachers. We
deliberately **do not** use mixup or cutmix in the distillation runs: mixing
samples would destroy the per-example correspondence that the graph loss needs to
pair student and teacher on identical node sets. Pairwise edge weights are
Euclidean distances between embeddings, and each graph's distance matrix is
normalized by its mean edge weight before embedding.

## Training configuration

**Teacher fine-tuning.** Teachers are ImageNet-1k pretrained backbones —
ResNet-18 or ConvNeXt-Tiny — fine-tuned as **embedding networks** with a
**triplet loss** (distance-weighted sampling, margin $0.2$) on the disjoint
training classes, with L2-normalized embeddings and NPairs class-balanced
batches; they are evaluated by recall@K. AdamW (lr $10^{-4}$, weight decay
$10^{-5}$), $60$ epochs, batch $128$, cosine schedule with $3$-epoch warmup,
mixed precision.

**Student distillation.** The student is **ConvNextMicro** (dims
$(24,48,96,192)$, depths $(1,1,3,1)$, $\approx 0.67$M parameters), trained as an
embedding network with **triplet + the graph loss only** (no Hinton KD — metric
learning has no shared logits). The relational graph term is balanced against the
triplet by a **warm-up** that ramps its weight $0\!\to\!1$ over the first
$10\%$ of epochs (early student embeddings are noise). Optimization: AdamW
(lr $10^{-3}$, weight decay $0.05$), cosine schedule with $5$-epoch warmup,
$120$ epochs, batch $K=128$, NPairs sampling, stochastic depth $0.1$, mixed
precision. The **classic baselines** replace the graph term with a single
relational loss — **RKD-distance alone** or **RKD-angle alone** — under the same
triplet + warm-up. A from-scratch ConvNextMicro trained with **triplet only**
(identical hyperparameters) is the reference that isolates the distillation gain.

**Graph loss.** Default values:

| Hyperparameter | Value |
|---|---|
| edge metric | Euclidean (distance matrix mean-normalized) |
| embedding | `profile` ($\in\mathbb{R}^{N(N-1)}$) or `mds` ($\in\mathbb{R}^{N}$) |
| objective | `regression` (Minkowski, $p=2$) or `contrastive` (InfoNCE) |
| sampling | adaptive `log`: $\alpha=0.5$, $g_{\min}=\lfloor K/N\rfloor$, $g_{\max}=64$ |
| contrastive negatives $M$ | $10$ |
| contrastive temperature $\tau$ | $0.07$ (constant; schedule is an optional ablation) |
| loss weight $\lambda_g$ | $\approx 1000$ (regression) / $\approx 1$ (contrastive), to match scales |
| relational order $N$ | log-spaced sweep up to the budget ceiling; $E=1024 \Rightarrow N_{\max}=17$ at $K=128$ |

Each (objective $\times$ embedding) combination is run separately, and within
each the order $N$ is selected by the budget-bounded log-spaced sweep of
Algorithm 4 on the **validation recall@1**.
