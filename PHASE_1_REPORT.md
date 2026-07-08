# PHASE 1 REPORT — λg viability gate (H0)

**Slice:** Cars-196 · ResNet-18 teacher · ConvNextMicro student · descriptor
`profile` · objective `regression` · normalization `minibatch` · order N=4 ·
30 epochs · 1 seed. Run on Modal. Figure: `RKD/analysis/figures/fig_h0_lambda.pdf`.

## Result (validation mAP@R)

| config | val mAP@R | test mAP@R |
|---|---|---|
| triplet-only **floor** | 0.0232 | 0.0070 |
| Graph-RKD λg=0.01 | **0.0280** | 0.0075 |
| Graph-RKD λg=0.1 | 0.0170 | 0.0052 |
| Graph-RKD λg=1 | 0.0112 | 0.0023 |
| Graph-RKD λg=10 | 0.0086 | 0.0016 |
| Graph-RKD λg=100 | 0.0094 | (below floor) |
| Graph-RKD λg=1000 | 0.0081 | 0.0015 |
| teacher (reference) | — | 0.188 |

## Verdict: H0 — marginally ACCEPT
- A viable band exists at **λg = 0.01** (val 0.028 ≥ floor 0.023); it does **not**
  collapse to the historical ~0.002 floor.
- But every λg ≥ 0.1 falls **below** the triplet-only floor and decreases
  monotonically — the graph term **hurts** as it grows.
- So Graph-RKD-regression here only *survives* at tiny λg and provides **no clear
  gain** over triplet-only.

## Caveats (why this is a weak signal, not a conclusion)
- **Short schedule:** 30 epochs; the student is far from the 0.188 teacher
  (floor test mAP@R ≈ 0.7% ≈ near-floor). Absolute numbers are tiny and noisy.
- **Single seed:** no sem (I5 not met); differences are within run-to-run noise.
- This is one slice (profile/regression/minibatch/N=4). Other descriptors/
  objectives/norms are untested.

## Implication
Before committing to a large campaign, the method/config should be **evolved**
(descriptor, objective, normalization, λg range, longer teacher/student) using
cheap Modal tests, then the improved configuration run at full budget locally.
See the "cheap Modal test" presets and the local full-GPU plan.

## Dev iteration (Modal, cheap, reuses teacher) — design-space probe
Slice cars196/r18/N4, λg=0.01, 30 ep, 1 seed. 14/16 configs; floor val=0.0232.
**11/14 configs beat the floor.** Best: `mds/reg/per_graph` (val 0.0322, test 0.0080),
`mds/reg/none` (0.0308/0.0074), `mds/con/per_graph` (0.0307/0.0072),
`prof/reg/*` (~0.029/0.0075). **`hybrid` normalization is consistently worst** → drop it.

**Candidates to carry to the LOCAL full run (multi-seed, longer schedule):**
- objective: **regression** (more consistent than contrastive here)
- descriptor: **profile and mds** (both viable at N=4)
- normalization: **per_graph / minibatch / none** (NOT hybrid)
- λg: small (~0.01–0.1); larger λg hurts (see gate curve above)
Caveat: gains are small and single-seed/short — confirm with ≥3 seeds + full epochs locally.

## Convergence test (conv) — go/no-go for the full run
Floor + top-2 dev configs at **80 epochs** (Cars-196/R18/N4/λg=0.01, 1 seed, reuse teacher):

| config | val mAP@R | test mAP@R |
|---|---|---|
| triplet-only floor | 0.0942 | 0.0214 |
| mds / reg / per_graph | 0.1025 | 0.0252 (+18% rel) |
| prof / reg / minibatch | 0.1107 | 0.0243 (+14% rel) |

**PASS.** Longer training lifts everyone far off the near-floor 30-ep regime
(floor test 0.007→0.021), and **both Graph-RKD configs beat the triplet-only floor**
by ~14–18% (test). The edge survives/grows at convergence → the trimmed campaign
is worth running (confirm multi-seed in phase 5). Single-seed, so treat as a strong
signal, not a final number.
