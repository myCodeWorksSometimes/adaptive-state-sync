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

## Extra: validation on real network traces

`real_traces/` reruns the pipeline on real cloud-gaming telemetry (RTT, jitter, loss, link throughput)
from the LORIA cloud-gaming study (Marchal et al., *JNSM* 2023, <https://cloud-gaming-traces.lhs.loria.fr/>).
The game-side context (entities, payload, client count) is still synthetic. The raw traces are not
stored here because the source states no licence; the script downloads them.

```bash
pip install py7zr                                  # needed to open the .7z archive
python real_traces/prepare_loria.py                # download + convert -> real_traces/loria_dataset.csv
python real_traces/real_trace_validation.py        # writes results/real_traces/*.csv
```

Two experiments: RBF trained on the synthetic split and tested on all 1,647 real sessions, and RBF
trained and tested on real data split by recording (8 of 25 files held out). On real telemetry,
last-value persistence predicts the severity class 5 ticks ahead with 98.8% accuracy, while the RBF
does no better than always predicting class 0.
