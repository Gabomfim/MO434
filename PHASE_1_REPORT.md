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
