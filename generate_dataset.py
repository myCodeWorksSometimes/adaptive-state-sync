"""
generate_dataset.py
====================
Generates the synthetic-but-realistic multi-genre, multi-path network dataset
described in Phase 2 Deliverable 3 (Steps 1-4).

IMPORTANT — SYNTHETIC DATASET NOTICE
-------------------------------------
This dataset is entirely synthetic. It was generated using a parameterized
stochastic simulation (AR(1) processes, burst loss model, correlated bandwidth)
calibrated to approximate realistic multiplayer game network conditions.
It is NOT collected real-world multiplayer traffic.

Network paths:  LAN, Home_Broadband, Mobile_4G, CrossRegion_Cloud
Game genres:    Shooter (FPS-like), Racing (CLAAP-style)
Clients:        4, 8, 16, 32 concurrent
Repetitions:    30 per configuration cell
Window size:    20 ticks (rolling window for statistical features)
Tick rate:      60 Hz
Ticks/session:  600 (10 simulated seconds)
Warm-up:        First 20 ticks discarded (insufficient window history)
Horizon:        Last 5 ticks discarded (no future target available)
Rows/session:   575  (ticks 20..594)

Total rows: 4 paths × 2 genres × 4 client_counts × 30 reps × 575 ticks
          = 552,000 rows

Each row = one tick observation with:
  - raw network measurements (rtt_ms, jitter_ms, loss_rate, bandwidth_mbps)
  - rolling-window statistical features over the previous 20 ticks
  - gameplay context (entity_count, payload_bytes, genre, client count)
  - current severity label (threshold-based, see severity_label() below)
  - future prediction targets at horizon H=5 ticks (~83ms at 60Hz)
    - future_rtt_ms_h5    : RTT 5 ticks ahead (for regression)
    - future_bw_mbps_h5   : bandwidth 5 ticks ahead (for regression)
    - future_severity_h5  : severity class 5 ticks ahead (for classification)

SEVERITY LABELLING (limitation documented):
  Labels use threshold rules on RTT and loss rate:
    0 (None):     RTT < 50ms  AND loss < 0.5%
    1 (Mild):     RTT < 100ms AND loss < 1.5%
    2 (Moderate): RTT < 180ms AND loss < 3.0%
    3 (Severe):   otherwise
  This differs from Phase 2's majority-vote labelling scheme
  because per-tick thresholds are simpler and have no smoothing. The limitation
  is documented in METHODOLOGY_CHANGES.md and Section 2 of Phase3_Implementation.ipynb.

SEEDING NOTE:
  Session seeds use deterministic arithmetic (not Python's hash()) to ensure
  reproducibility across separate Python processes (PYTHONHASHSEED-independent).
  Formula: local_seed = SEED + path_idx*10000 + genre_idx*2000 + client_idx*400 + rep
"""

import numpy as np
import pandas as pd
import os
import yaml

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

os.makedirs("dataset", exist_ok=True)

# ── Path profiles (mean_rtt_ms, rtt_std_ms, base_loss_rate, mean_bw_mbps) ─────
PATH_PROFILES = {
    "LAN":               dict(base_rtt=5,   rtt_std=1,   loss=0.001, bw=100),
    "Home_Broadband":    dict(base_rtt=35,  rtt_std=8,   loss=0.005, bw=50),
    "Mobile_4G":         dict(base_rtt=80,  rtt_std=25,  loss=0.015, bw=20),
    "CrossRegion_Cloud": dict(base_rtt=130, rtt_std=20,  loss=0.010, bw=30),
}

# ── Genre profiles (entity_count range, payload_bytes range) ──────────────────
GENRE_PROFILES = {
    "Shooter": dict(entity_min=10, entity_max=32,  payload_min=200, payload_max=800),
    "Racing":  dict(entity_min=4,  entity_max=12,  payload_min=100, payload_max=400),
}

# ── Severity thresholds ────────────────────────────────────────────────────────
# Limitation: These are per-tick threshold rules, not majority-vote labels.
# They produce some label noise near boundaries (ticks just above/below threshold).
def severity_label(rtt, loss_rate_mean):
    """
    Classify network severity into 4 classes using threshold rules.

    Parameters
    ----------
    rtt : float
        Current RTT in milliseconds.
    loss_rate_mean : float
        Mean packet loss rate over the rolling window (0.0–1.0).

    Returns
    -------
    int : 0=None, 1=Mild, 2=Moderate, 3=Severe
    """
    if rtt < 50  and loss_rate_mean < 0.005: return 0  # None
    if rtt < 100 and loss_rate_mean < 0.015: return 1  # Mild
    if rtt < 180 and loss_rate_mean < 0.030: return 2  # Moderate
    return 3                                             # Severe


# ── Simulation parameters ──────────────────────────────────────────────────────
TICKS_PER_SESSION = 600   # 10 seconds at 60 Hz per repetition
WINDOW            = 20    # rolling-window for feature computation
HORIZON           = 5     # future prediction horizon (5 ticks ≈ 83ms at 60Hz)
REPETITIONS       = 30
CLIENT_COUNTS     = [4, 8, 16, 32]

# Update simulation_parameters.yaml to match actual generator settings
sim_params = {
    "latency_budget_ms": 50,
    "tick_rate_hz": 60,
    "ticks_per_session": TICKS_PER_SESSION,
    "rolling_window_size": WINDOW,
    "rows_per_session": TICKS_PER_SESSION - WINDOW - HORIZON,  # 575
    "prediction_horizon_ticks": HORIZON,
    "random_seed": SEED,
    "paths": list(PATH_PROFILES.keys()),
    "genres": list(GENRE_PROFILES.keys()),
    "client_counts": CLIENT_COUNTS,
    "repetitions": REPETITIONS,
    "note": (
        "SYNTHETIC DATASET — not real-world collected traffic. "
        f"Total rows: 4x2x4x{REPETITIONS}x{TICKS_PER_SESSION-WINDOW-HORIZON} = "
        f"{4*2*4*REPETITIONS*(TICKS_PER_SESSION-WINDOW-HORIZON):,}"
    )
}
with open("dataset/simulation_parameters.yaml", "w") as f:
    yaml.dump(sim_params, f, default_flow_style=False, sort_keys=False)


# ── Main generation loop ───────────────────────────────────────────────────────
rows = []

PATH_NAMES   = list(PATH_PROFILES.keys())
GENRE_NAMES  = list(GENRE_PROFILES.keys())

for path_idx, path_name in enumerate(PATH_NAMES):
    path = PATH_PROFILES[path_name]
    for genre_idx, genre_name in enumerate(GENRE_NAMES):
        genre = GENRE_PROFILES[genre_name]
        for client_idx, n_clients in enumerate(CLIENT_COUNTS):
            for rep in range(REPETITIONS):

                # Deterministic seed — no Python hash(), reproducible across processes
                local_seed = (
                    SEED
                    + path_idx  * 10000
                    + genre_idx * 2000
                    + client_idx * 400
                    + rep
                )
                rng_local = np.random.default_rng(local_seed)

                # ── Simulate a correlated RTT trace (AR(1) + spikes) ──────────
                rtt_trace = []
                rtt = path["base_rtt"]
                for t in range(TICKS_PER_SESSION):
                    rtt = 0.85 * rtt + 0.15 * path["base_rtt"] + rng_local.normal(0, path["rtt_std"])
                    if rng_local.random() < 0.02:           # congestion spike (~2% chance)
                        rtt += rng_local.uniform(50, 200)
                    rtt = max(1.0, rtt)
                    rtt_trace.append(rtt)

                # ── Simulate jitter (absolute difference of consecutive RTTs) ──
                jitter_trace = [0.0] + [
                    abs(rtt_trace[i] - rtt_trace[i - 1])
                    for i in range(1, TICKS_PER_SESSION)
                ]

                # ── Simulate packet loss (Bernoulli + burst model) ────────────
                loss_trace = []
                in_burst = False
                for t in range(TICKS_PER_SESSION):
                    if rng_local.random() < 0.01:       # burst start probability
                        in_burst = True
                    if in_burst and rng_local.random() < 0.3:   # burst end
                        in_burst = False
                    p = path["loss"] * (5 if in_burst else 1)
                    loss_trace.append(1 if rng_local.random() < p else 0)

                # ── Simulate bandwidth (AR(1), anti-correlated with RTT) ───────
                bw_trace = []
                bw = path["bw"]
                for t in range(TICKS_PER_SESSION):
                    bw = 0.90 * bw + 0.10 * path["bw"] + rng_local.normal(0, path["bw"] * 0.05)
                    bw -= 0.01 * (rtt_trace[t] - path["base_rtt"])
                    bw = max(1.0, bw)
                    bw_trace.append(bw)

                # ── Simulate entity count ─────────────────────────────────────
                entity_trace = rng_local.integers(
                    genre["entity_min"],
                    min(genre["entity_max"], genre["entity_min"] + n_clients * 2),
                    size=TICKS_PER_SESSION,
                )

                # ── Build rows (skip first WINDOW ticks — no rolling stats yet) ─
                # Also need HORIZON future ticks, so stop HORIZON before end.
                # Rows where the future horizon falls outside session are excluded.
                for t in range(WINDOW, TICKS_PER_SESSION - HORIZON):
                    w_rtt  = rtt_trace[t - WINDOW:t]
                    w_jit  = jitter_trace[t - WINDOW:t]
                    w_bw   = bw_trace[t - WINDOW:t]
                    w_loss = loss_trace[t - WINDOW:t]

                    rtt_now       = rtt_trace[t]
                    bw_now        = bw_trace[t]
                    loss_now_mean = float(np.mean(w_loss))
                    sev_now       = severity_label(rtt_now, loss_now_mean)
                    payload       = rng_local.integers(genre["payload_min"], genre["payload_max"])

                    # Future targets at horizon H=5
                    t_future     = t + HORIZON
                    rtt_future   = rtt_trace[t_future]
                    bw_future    = bw_trace[t_future]
                    # future loss_mean: rolling window ending at t_future
                    loss_future_mean = float(np.mean(loss_trace[t_future - WINDOW:t_future]))
                    sev_future   = severity_label(rtt_future, loss_future_mean)

                    rows.append({
                        # ── Identifiers ───────────────────────────────────────
                        "path":            path_name,
                        "genre":           genre_name,
                        "n_clients":       n_clients,
                        "repetition":      rep,
                        "tick":            t,
                        # ── Raw measurements at tick t ────────────────────────
                        "rtt_ms":          round(rtt_now,  2),
                        "jitter_ms":       round(jitter_trace[t], 2),
                        "loss_rate":       round(loss_now_mean, 4),
                        "bandwidth_mbps":  round(bw_now,  2),
                        "entity_count":    int(entity_trace[t]),
                        "payload_bytes":   int(payload),
                        # ── Rolling-window features (20 ticks ending at t-1) ──
                        "rtt_mean":        round(float(np.mean(w_rtt)), 2),
                        "rtt_std":         round(float(np.std(w_rtt)),  2),
                        "rtt_min":         round(float(np.min(w_rtt)),  2),
                        "rtt_max":         round(float(np.max(w_rtt)),  2),
                        "jitter_mean":     round(float(np.mean(w_jit)), 2),
                        "jitter_std":      round(float(np.std(w_jit)),  2),
                        "bw_mean":         round(float(np.mean(w_bw)),  2),
                        "bw_std":          round(float(np.std(w_bw)),   2),
                        "loss_rate_mean":  round(loss_now_mean, 4),
                        # ── Gameplay context ──────────────────────────────────
                        "genre_encoded":   0 if genre_name == "Shooter" else 1,
                        # ── Current severity label ────────────────────────────
                        "severity":        sev_now,
                        # ── Future prediction targets (H=5 ticks ≈ 83ms) ─────
                        # IMPORTANT: These columns must NEVER be used as predictor inputs.
                        # They are targets for the supervised training task.
                        "future_rtt_ms_h5":    round(rtt_future,          2),
                        "future_bw_mbps_h5":   round(bw_future,           2),
                        "future_severity_h5":  int(sev_future),
                    })

df = pd.DataFrame(rows)

# ── Save raw dataset — do NOT modify after this point ─────────────────────────
RAW_PATH = "dataset/network_dataset.csv"
df.to_csv(RAW_PATH, index=False, lineterminator="\n")  # LF on every OS -> byte-identical output
print(f"Synthetic raw dataset saved: {len(df):,} rows × {len(df.columns)} columns")
print(f"Path: {RAW_PATH}")
print("\nRow counts by path:")
print(df["path"].value_counts().to_string())
print("\nRow counts by genre:")
print(df["genre"].value_counts().to_string())
print("\nSeverity distribution:")
print(df["severity"].value_counts().sort_index().to_string())
print("\nFuture severity distribution (H=5):")
print(df["future_severity_h5"].value_counts().sort_index().to_string())

# ── Verify no missing values ───────────────────────────────────────────────────
missing = df.isnull().sum().sum()
assert missing == 0, f"Unexpected nulls: {missing}"
print(f"\nMissing values: {missing} (OK)")

# ── Verify unique sessions ─────────────────────────────────────────────────────
n_sessions = df[["path", "genre", "n_clients", "repetition"]].drop_duplicates().shape[0]
expected   = len(PATH_NAMES) * len(GENRE_NAMES) * len(CLIENT_COUNTS) * REPETITIONS
assert n_sessions == expected, f"Expected {expected} sessions, got {n_sessions}"
print(f"Sessions: {n_sessions} (expected {expected}) — OK")
print("\nDataset generation complete.")
print("NOTE: Splitting and normalization are performed inside Phase3_Implementation.ipynb")
print("      using session-based GroupShuffleSplit to prevent session leakage.")
