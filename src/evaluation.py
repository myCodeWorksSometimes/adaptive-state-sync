"""
src/evaluation.py
=================
Evaluation utilities for the Phase 3 experimental framework.

Metric definitions (all clearly labelled):
  - traffic_mbps       : ESTIMATED total server→clients sync traffic in Mbps.
                         Formula: (payload×IR×CF + overhead) × 8 × n_clients × tick_rate × UR / 1e6
                         Assumptions documented in adaptive_controller.py.
  - traffic_reduction_pct : % reduction vs Baseline A. NOT vs available capacity.
  - update_rate_hz     : achieved tick rate = UR × 60 Hz.
  - sync_quality_proxy : SIMULATED state divergence proxy. NOT a game measurement.
  - inference_us       : wall-clock time of ONE batched predictor.predict() call on
                         all test rows (scaling + 3 RBF sub-models), divided by the
                         number of rows. Controller lookup and traffic estimation
                         are NOT included. The same value is attached to every
                         method because the prediction is computed once and reused.

Performance notes:
  - run_experiment() is fully vectorised (no iterrows).
  - RBF activation uses batched computation (see predictor.py) to cap memory.
  - Projections for 82,800 test rows × 8 methods: typically <30s.
"""

import numpy as np
import pandas as pd
import time

from src.adaptive_controller import (
    TICK_RATE_HZ, PROTOCOL_OVERHEAD, COMPRESSION_FACTORS,
    compute_bandwidth_reduction
)

# Feature columns used as predictor inputs — must NOT include future targets
FEATURE_COLS = [
    "rtt_mean", "rtt_std", "rtt_min", "rtt_max",
    "jitter_mean", "jitter_std",
    "bw_mean", "bw_std",
    "loss_rate_mean",
    "entity_count", "payload_bytes",
    "genre_encoded", "n_clients",
]

# Forbidden: must never appear as predictor inputs
_FORBIDDEN_FEATURES = {
    "future_rtt_ms_h5", "future_bw_mbps_h5", "future_severity_h5", "severity"
}
for _f in _FORBIDDEN_FEATURES:
    if _f in FEATURE_COLS:
        raise RuntimeError(f"CRITICAL: forbidden feature '{_f}' in FEATURE_COLS")

# Policy lookup arrays (indexed by severity 0-3)
_UR_TABLE = np.array([1.00, 0.75, 0.50, 0.25])
_CL_TABLE = np.array([0,    1,    2,    3   ], dtype=int)
_IR_TABLE = np.array([1.00, 0.80, 0.60, 0.40])
_CF_TABLE = np.array([1.00, 0.80, 0.55, 0.35])  # compression factors


def _traffic_vec(payload, ur, cl_int, ir, n_cl):
    """
    Vectorised traffic estimate in Mbps.
    cl_int: integer array indexing _CF_TABLE.
    """
    comp  = _CF_TABLE[cl_int]
    bpp   = payload * ir * comp + PROTOCOL_OVERHEAD      # bytes per packet
    return bpp * 8.0 * n_cl * TICK_RATE_HZ * ur / 1e6


def _sq_vec(rtt, ur, ir, jitter, rng=None):
    """
    Vectorised SIMULATED sync quality proxy.
    Label: SIMULATED — not a game-engine measurement.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    tick_ms   = 1000.0 / (TICK_RATE_HZ * np.maximum(ur, 0.01))
    state_age = rtt + tick_ms / 2.0
    cov_pen   = 1.0 / np.maximum(ir, 0.01)
    noise     = np.abs(rng.normal(0, jitter * 0.05 + 0.1))
    return np.maximum(0.0, state_age * cov_pen * 0.01 + noise)


def run_experiment(df, predictor, methods_config, seed=42):
    """
    Run the full experimental evaluation across all methods.
    Fully vectorised — no iterrows().

    Parameters
    ----------
    df : pd.DataFrame
        Test dataset. Must contain FEATURE_COLS and standard columns.
    predictor : AdaptivePredictor
        Fitted predictor. Called once; predictions reused for all methods.
    methods_config : dict
        {method_name: {'type': <type_string>}}
        Valid types: 'baseline_a', 'baseline_b', 'baseline_c', 'baseline_d',
                     'proposed', 'ablation_no_ur', 'ablation_no_pl', 'ablation_no_im',
                     'ablation_reactive'  (controller driven by CURRENT severity,
                                           i.e. the predictor is removed),
                     'ablation_oracle'    (controller driven by the TRUE future
                                           severity — upper bound for a perfect
                                           predictor; uses future labels, so it
                                           is an analysis tool, not deployable)
    seed : int
        Seed for sync quality proxy noise.

    Returns
    -------
    pd.DataFrame : one row per (method, test row). Shape = N_methods × N_rows.
    """
    rng = np.random.default_rng(seed)
    n   = len(df)

    # === Pre-compute predictions ONCE ========================================
    X_arr = df[FEATURE_COLS].values.astype(np.float64)

    t0   = time.perf_counter()
    preds = predictor.predict(X_arr)
    t_pred_us = (time.perf_counter() - t0) * 1e6
    per_row_us = t_pred_us / n

    pred_sev   = preds['severity'].astype(int)   # (N,)

    # === Extract column arrays ===============================================
    payload_arr  = df['payload_bytes'].values.astype(np.float64)
    n_cl_arr     = df['n_clients'].values.astype(np.float64)  # float for broadcasting
    rtt_arr      = df['rtt_ms'].values.astype(np.float64)
    jitter_arr   = df['jitter_ms'].values.astype(np.float64)
    bw_arr       = df['bandwidth_mbps'].values.astype(np.float64)
    sev_true_arr = df['severity'].values.astype(int)

    all_frames = []

    for method_name, cfg in methods_config.items():
        mtype = cfg['type']

        # === Determine control arrays =========================================
        if mtype == 'baseline_a':
            ur_arr  = np.ones(n)
            cl_arr  = np.zeros(n, dtype=int)
            ir_arr  = np.ones(n)
            sev_pred = np.full(n, -1, dtype=int)

        elif mtype == 'baseline_b':
            ur_arr  = np.full(n, 0.75)
            cl_arr  = np.ones(n, dtype=int)
            ir_arr  = np.full(n, 0.90)
            sev_pred = np.full(n, -1, dtype=int)

        elif mtype == 'baseline_c':
            sev_pred = pred_sev
            ur_arr  = np.where(sev_pred == 0, 1.00, 0.50)
            cl_arr  = np.where(sev_pred == 0, 0,    1   ).astype(int)
            ir_arr  = np.where(sev_pred == 0, 1.00, 0.80)

        elif mtype == 'baseline_d':
            ur_arr  = np.full(n, 0.85)
            cl_arr  = np.zeros(n, dtype=int)
            ir_arr  = np.full(n, 0.95)
            sev_pred = np.full(n, -1, dtype=int)

        else:
            # Proposed / ablation — policy table on a severity signal
            if mtype == 'ablation_reactive':
                sev_pred = sev_true_arr                       # no prediction
            elif mtype == 'ablation_oracle':
                sev_pred = df['future_severity_h5'].values.astype(int)
            else:
                sev_pred = pred_sev                           # RBF prediction
            sev_clamped  = np.clip(sev_pred, 0, 3)
            no_ur = 'no_ur' in mtype
            no_pl = 'no_pl' in mtype
            no_im = 'no_im' in mtype

            ur_arr = np.ones(n)                      if no_ur else _UR_TABLE[sev_clamped]
            cl_arr = np.zeros(n, dtype=int)          if no_pl else _CL_TABLE[sev_clamped]
            ir_arr = np.ones(n)                      if no_im else _IR_TABLE[sev_clamped]

        # === Compute metrics ==================================================
        traffic_arr = _traffic_vec(payload_arr, ur_arr, cl_arr, ir_arr, n_cl_arr)
        sq_arr      = _sq_vec(rtt_arr, ur_arr, ir_arr, jitter_arr, rng)

        frame = pd.DataFrame({
            'method':             method_name,
            'path':               df['path'].values,
            'genre':              df['genre'].values,
            'n_clients':          df['n_clients'].values,
            'repetition':         df['repetition'].values,
            'tick':               df['tick'].values,
            'rtt_ms':             rtt_arr,
            'jitter_ms':          jitter_arr,
            'loss_rate':          df['loss_rate'].values,
            'bandwidth_mbps':     bw_arr,
            'severity_true':      sev_true_arr,
            'severity_pred':      sev_pred,
            'ur_applied':         ur_arr,
            'cl_applied':         cl_arr,
            'ir_applied':         ir_arr,
            'update_rate_hz':     (TICK_RATE_HZ * ur_arr).round(1),
            'traffic_mbps':       traffic_arr.round(5),
            'sync_quality_proxy': sq_arr.round(4),
            'inference_us':       per_row_us,
        })
        all_frames.append(frame)

    return pd.concat(all_frames, ignore_index=True)


def aggregate_by_session(results_df):
    """
    Aggregate per-tick results to session level.

    Session key: (method, path, genre, n_clients, repetition).
    Returns one row per session — the correct level for statistical comparison.
    Individual ticks are temporally correlated; sessions are approximately
    independent experimental trials.
    """
    group_cols = ['method', 'path', 'genre', 'n_clients', 'repetition']
    agg = results_df.groupby(group_cols, observed=True).agg(
        mean_traffic_mbps    = ('traffic_mbps',       'mean'),
        mean_sync_quality    = ('sync_quality_proxy', 'mean'),
        mean_update_rate_hz  = ('update_rate_hz',     'mean'),
        mean_ur              = ('ur_applied',         'mean'),
        mean_cl              = ('cl_applied',         'mean'),
        mean_ir              = ('ir_applied',         'mean'),
        mean_rtt_ms          = ('rtt_ms',             'mean'),
        p95_rtt_ms           = ('rtt_ms',             lambda x: np.percentile(x, 95)),
        mean_jitter_ms       = ('jitter_ms',          'mean'),
        mean_loss_rate       = ('loss_rate',          'mean'),
        mean_inference_us    = ('inference_us',       'mean'),
        n_ticks              = ('tick',               'count'),
    ).reset_index()
    return agg


def compute_traffic_reduction(session_df,
                              baseline_name='Baseline_A_StaticFullSync'):
    """
    Add traffic_reduction_pct column to session-level DataFrame.

    For each session (path, genre, n_clients, repetition), compute the
    percentage reduction relative to Baseline A's traffic for that same session.
    Positive = less traffic than baseline; negative = more traffic.
    """
    base_traffic = (
        session_df[session_df['method'] == baseline_name]
        .set_index(['path', 'genre', 'n_clients', 'repetition'])
        ['mean_traffic_mbps']
    )

    def _get_reduction(row):
        key = (row['path'], row['genre'], row['n_clients'], row['repetition'])
        try:
            bt = base_traffic.loc[key]
            return compute_bandwidth_reduction(row['mean_traffic_mbps'], bt)
        except KeyError:
            return np.nan

    df_out = session_df.copy()
    df_out['traffic_reduction_pct'] = df_out.apply(_get_reduction, axis=1)
    return df_out
