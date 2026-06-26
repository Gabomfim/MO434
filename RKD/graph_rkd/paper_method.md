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

The relational order $N$ is the central hyperparameter of the method. Under the
assumption that distillation quality increases monotonically with $N$, the
optimal $N$ is the largest value that remains within the available compute
budget; because the per-step cost under the partition scheme is
$E = K(N-1)/2$ edges, this yields the closed-form ceiling

$$N \;\le\; \frac{2E}{K} + 1,$$

i.e., the best feasible order is inversely proportional to the batch size. We
tune $N$ with a **binary search**: feasibility under a resource budget is a
monotonic predicate, so the largest feasible $N$ is found in $O(\log K)$
evaluations; when quality exhibits diminishing returns rather than strict
monotonicity, the same binary search locates the "knee" of the validation-quality
curve, using the student's validation accuracy at each candidate $N$ as the
search signal.
