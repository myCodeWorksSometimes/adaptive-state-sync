# Methodology Changes — Phase 2 Design → Phase 3 Implementation

This file records every place where the Phase 3 implementation differs from the
methodology proposed in Phase 2, and why. The same list appears in Section 2 of
`Phase3_Implementation.ipynb`.

| Component  | Phase 2 design                                   | Phase 3 implementation                                                        |
|------------|--------------------------------------------------|-------------------------------------------------------------------------------|
| Predictor  | RBF Network + BOCPD                              | RBF Network (K-means centres, least-squares output weights)                   |
| Drift      | Full Bayesian Online Changepoint Detection       | Sliding-window error z-score monitor (CUSUM-style approximation)              |
| Controller | Monotonic threshold policy on predicted severity | Implemented as specified (lookup table, `src/adaptive_controller.py`)          |
| Baselines  | A Static, B ALC, C CLAAP gate, D static placement | Implemented as documented approximations (`src/baselines.py`)                 |
| Dataset    | Real-world multi-path, multi-genre traffic       | Fully synthetic, parameterised stochastic simulation (`generate_dataset.py`)   |
| Labels     | Majority-vote severity labelling                 | Per-tick threshold rule on RTT and window loss rate                           |

## Change 1 — BOCPD approximated
Full BOCPD (Adams & MacKay, 2007) keeps a posterior over run lengths, which is
too heavy for this project. `BOCPDApproximation` instead compares the mean of a
sliding window of prediction errors against the training-error baseline and
flags a changepoint when the z-score exceeds 2.5. **Online retraining after a
detected changepoint is not implemented**; the monitor only reports detections.
In the reported validation run it detected no changepoints.

## Change 2 — Synthetic dataset
No public dataset combines latency, jitter, loss, bandwidth and game-state
signals across several genres and paths, and live collection was out of scope.
The dataset is generated with AR(1) RTT/bandwidth processes, random congestion
spikes and a burst-loss model. All results are simulation results.

## Change 3 — Threshold severity labels
Severity is assigned per tick by thresholds on RTT and the 20-tick mean loss rate
(see `severity_label()` in `generate_dataset.py`), instead of majority voting.
This is simpler but creates label noise at class boundaries, and random
congestion spikes make the future label hard to predict 5 ticks ahead.

## Change 4 — Future prediction targets
The predictor is trained on targets 5 ticks ahead (`future_rtt_ms_h5`,
`future_bw_mbps_h5`, `future_severity_h5`, ≈83 ms at 60 Hz). These columns are
never used as inputs; a leakage assertion in the notebook and in
`src/evaluation.py` enforces this.

## Change 5 — Session-based splitting
Sessions (path × genre × client count × repetition) are assigned whole to
train / validation / test (70/15/15) with `GroupShuffleSplit`, so no tick from a
test session is seen during training.

## Change 6 — Traffic is estimated, not capacity-subtracted
Synchronisation traffic is estimated as
`(payload × IR × CF + 28 B) × 8 × clients × 60 Hz × UR`, and reduction is
reported relative to Baseline A. An earlier formula
(`available bandwidth − traffic`) was dimensionally wrong and was removed.

## Change 7 — Gameplay metrics are proxies
FPS, reconciliation error and fairness cannot be measured without a game engine.
The only quality measure kept is a clearly labelled *simulated* synchronisation
quality proxy (lower is better).

## Change 8 — Predictor is genuinely trained
An early prototype script used a placeholder predictor that perturbed the
ground-truth label. It has been removed from the codebase. All reported
predictor numbers come from the trained RBF model on held-out test sessions.

## Change 9 — Additional evaluation added in Phase 3
- **Naive forecasting baselines** (persistence / window mean) to test whether the
  RBF predictor adds value over using the current measurement.
- **Predictor-source ablations**: the controller driven by the current severity
  (`ASS_Reactive`, predictor removed) and by the true future severity
  (`ASS_Oracle`, perfect-prediction upper bound).
