# Phase 3 — Adaptive State Synchronization (ASS) for Multiplayer Games

CS3001 Computer Networks research project, Phase 3: implementation and experimental evaluation.
Group: Afaq Ahsan (24i-6516) · Qanitah Ali (24i-3125)

> **Synthetic data notice:** all results come from a simulated dataset, not from real multiplayer traffic.

## Contents

| Path | What it is |
|------|------------|
| `Phase3_Implementation.ipynb` | Main deliverable: preprocessing, predictor, baselines, experiments, ablations, statistics, discussion (executed, with outputs) |
| `Technical_Report.docx` | Written report |
| `METHODOLOGY_CHANGES.md` | Every change from the Phase 2 design, with justification |
| `generate_dataset.py` | Deterministic synthetic dataset generator (seed 42) |
| `src/predictor.py` | RBF Network, BOCPD approximation (drift monitor), naive persistence forecaster |
| `src/adaptive_controller.py` | ASS controller policy table and traffic model |
| `src/baselines.py` | Baselines A–D (documented approximations of the cited work) |
| `src/evaluation.py` | Vectorised evaluation of all methods and ablations, session-level aggregation |
| `dataset/network_dataset.csv` | 552,000 rows = 960 sessions × 575 ticks (66 MB, not stored in the repo; run `python generate_dataset.py` to recreate it) |
| `dataset/simulation_parameters.yaml` | Simulation settings |
| `results/figures/` | Figures 1–13 |
| `results/tables/` | Summary tables (CSV) and gzipped per-tick raw results |
| `results/logs/split_sessions.csv` | Which session went to train / validation / test |

## Reproducing the results

```bash
pip install -r requirements.txt
python generate_dataset.py          # optional: re-creates dataset/network_dataset.csv byte-for-byte
jupyter nbconvert --to notebook --execute --inplace Phase3_Implementation.ipynb
```

Run both commands from this folder. The notebook takes about 2–3 minutes (RBF training ≈ 1–1.5 min).
All randomness is seeded (seed 42); only the measured inference time depends on the machine.

## Methods evaluated

- **Baselines:** A static full sync · B fixed-rule throttle (ALC-inspired) · C prediction with binary gate
  (CLAAP-inspired) · D static placement (García Fernández de Castro et al.-inspired)
- **Proposed:** ASS — RBF-predicted future severity drives update rate, compression and interest radius
- **Ablations:** no update-rate scaling · no compression · no interest management ·
  Reactive (current severity, no predictor) · Oracle (true future severity, upper bound)
- **Predictor baselines:** persistence (last value), persistence (window mean), majority class
