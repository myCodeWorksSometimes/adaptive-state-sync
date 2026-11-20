"""
Real-trace validation of the Phase 3 pipeline on the LORIA cloud-gaming traces.

Experiment 1 (transfer): RBF trained on the synthetic training split exactly as in the
  notebook (GroupShuffleSplit 70/15/15, seed 42), tested on all real sessions.
Experiment 2 (in-domain): RBF trained and tested on real data, split by source trace
  file (GroupShuffleSplit 70/30 over the 25 files, seed 42) so no recording is shared.

Run from the project root after prepare_loria.py:
  python real_traces/real_trace_validation.py
Writes CSV tables to results/real_traces/.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.getcwd())
from src.predictor import AdaptivePredictor, PersistenceForecaster  # noqa: E402
from src.evaluation import (run_experiment, aggregate_by_session,  # noqa: E402
                            compute_traffic_reduction, FEATURE_COLS)

SEED = 42
OUT_DIR = os.path.join("results", "real_traces")
TARGETS = ("future_rtt_ms_h5", "future_bw_mbps_h5", "future_severity_h5")
METHODS_CONFIG = {
    'Baseline_A_StaticFullSync':  {'type': 'baseline_a'},
    'Baseline_B_FixedRuleALC':    {'type': 'baseline_b'},
    'Baseline_C_PredictNoAct':    {'type': 'baseline_c'},
    'Baseline_D_StaticPlacement': {'type': 'baseline_d'},
    'ASS_Proposed':               {'type': 'proposed'},
    'ASS_no_UR':                  {'type': 'ablation_no_ur'},
    'ASS_no_PL':                  {'type': 'ablation_no_pl'},
    'ASS_no_IM':                  {'type': 'ablation_no_im'},
    'ASS_Reactive':               {'type': 'ablation_reactive'},
    'ASS_Oracle':                 {'type': 'ablation_oracle'},
}
COMPARE = ['Baseline_A_StaticFullSync', 'Baseline_B_FixedRuleALC', 'Baseline_C_PredictNoAct',
           'Baseline_D_StaticPlacement', 'ASS_Reactive']


def path_group(p):
    cond = p.split('__')[1]
    return {'ftth': 'FTTH', 'highway': '4G highway'}.get(cond, '4G static')


def fit_rbf(train):
    pred = AdaptivePredictor(n_centres=32, seed=SEED)
    pred.fit(train[FEATURE_COLS].values.astype(float),
             *(train[c].values for c in TARGETS))
    return pred


def forecast_table(pred, test):
    y_rtt, y_bw, y_sev = (test[c].values for c in TARGETS)
    rows = []
    m = pred.evaluate(test[FEATURE_COLS].values.astype(float), y_rtt, y_bw, y_sev)
    rows.append({'model': 'RBF predictor', 'rtt_mae_ms': m['rtt']['mae'], 'rtt_rmse_ms': m['rtt']['rmse'],
                 'bw_mae_mbps': m['bw']['mae'], 'bw_rmse_mbps': m['bw']['rmse'],
                 'sev_accuracy': m['severity']['accuracy'], 'sev_f1_macro': m['severity']['f1_macro']})
    for mode, name in (('last', 'Persistence: last value'), ('window', 'Persistence: window mean')):
        p = PersistenceForecaster(mode).evaluate(test, y_rtt, y_bw, y_sev)
        rows.append({'model': name, **{k: p[k] for k in ('rtt_mae_ms', 'rtt_rmse_ms', 'bw_mae_mbps',
                                                         'bw_rmse_mbps', 'sev_accuracy', 'sev_f1_macro')}})
    from sklearn.metrics import accuracy_score, f1_score
    maj = np.zeros_like(y_sev)
    rows.append({'model': 'Majority class (always 0)', 'sev_accuracy': accuracy_score(y_sev, maj),
                 'sev_f1_macro': f1_score(y_sev, maj, average='macro', zero_division=0)})
    return pd.DataFrame(rows)


def controller_tables(pred, test, tag):
    res = run_experiment(test, pred, METHODS_CONFIG, seed=SEED)
    sess = compute_traffic_reduction(aggregate_by_session(res))
    sess['path_group'] = sess['path'].map(path_group)
    overall = sess.groupby('method').agg(
        mean_traffic_mbps=('mean_traffic_mbps', 'mean'),
        mean_reduction_pct=('traffic_reduction_pct', 'mean'),
        std_reduction_pct=('traffic_reduction_pct', 'std'),
        mean_sync_quality=('mean_sync_quality', 'mean'),
        mean_update_rate_hz=('mean_update_rate_hz', 'mean'),
        n_sessions=('n_ticks', 'count')).reindex(list(METHODS_CONFIG)).reset_index()
    by_group = (sess[sess.method.isin(['ASS_Proposed', 'Baseline_B_FixedRuleALC', 'ASS_Reactive'])]
                .pivot_table(index='path_group', columns='method', values='traffic_reduction_pct', aggfunc='mean'))
    by_group['n_sessions'] = sess[sess.method == 'ASS_Proposed'].groupby('path_group').size()

    key = ['path', 'genre', 'n_clients', 'repetition']
    ass = sess[sess.method == 'ASS_Proposed'].set_index(key)['mean_traffic_mbps']
    rng = np.random.default_rng(SEED)
    stats = []
    for b in COMPARE:
        base = sess[sess.method == b].set_index(key)['mean_traffic_mbps'].loc[ass.index]
        diff = (base - ass).values
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        nz = not np.allclose(diff, 0)
        stats.append({'comparison': f'ASS vs {b}', 'n_sessions': len(diff), 'mean_diff_mbps': diff.mean(),
                      'ci95_lo': np.percentile(boot, 2.5), 'ci95_hi': np.percentile(boot, 97.5),
                      'wilcoxon_p_two_sided': wilcoxon(ass.values, base.values).pvalue if nz else np.nan,
                      'cohens_dz': diff.mean() / (diff.std() + 1e-12)})
    for name, t in (('controllers', overall), ('by_path_group', by_group.reset_index()),
                    ('stats', pd.DataFrame(stats))):
        p = t.pop('wilcoxon_p_two_sided') if 'wilcoxon_p_two_sided' in t else None
        t = t.round(5)
        if p is not None:
            t['wilcoxon_p_two_sided'] = p.map(lambda v: f'{v:.3e}')   # keep tiny p-values
        t.to_csv(os.path.join(OUT_DIR, f'{tag}_{name}.csv'), index=False)
    return overall, by_group, pd.DataFrame(stats)


def show(title, df):
    print(f'\n=== {title} ===')
    print(df.round(4).to_string(index=False if 'model' in df.columns or 'method' in df.columns else True))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    real = pd.read_csv(os.path.join('real_traces', 'loria_dataset.csv'))
    syn = pd.read_csv(os.path.join('dataset', 'network_dataset.csv'))

    # --- synthetic training split, identical to the notebook (cell 12) ---
    sid = (syn['path'] + '__' + syn['genre'] + '__' + syn['n_clients'].astype(str) + '__'
           + syn['repetition'].astype(str)).values
    tr_idx, _ = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(syn, groups=sid))
    syn_train = syn.iloc[tr_idx]
    print(f'Synthetic train rows: {len(syn_train):,} | real rows: {len(real):,}')
    print('Real future-severity share (%):',
          (real['future_severity_h5'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict())

    # --- Experiment 1: train synthetic, test real ---
    pred1 = fit_rbf(syn_train)
    f1 = forecast_table(pred1, real)
    f1.round(5).to_csv(os.path.join(OUT_DIR, 'exp1_transfer_forecasting.csv'), index=False)
    show('Exp 1 (train synthetic, test real): forecasting', f1)
    o1, g1, s1 = controller_tables(pred1, real, 'exp1_transfer')
    show('Exp 1: controllers', o1); show('Exp 1: reduction by path group', g1); show('Exp 1: stats', s1)

    # --- Experiment 2: train and test on real, split by trace file ---
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(real, groups=real['path']))
    real_tr, real_te = real.iloc[tr], real.iloc[te]
    print('\nExp 2 test files:', sorted(real_te['path'].unique()))
    pred2 = fit_rbf(real_tr)
    f2 = forecast_table(pred2, real_te)
    f2.round(5).to_csv(os.path.join(OUT_DIR, 'exp2_indomain_forecasting.csv'), index=False)
    show('Exp 2 (train real, test real by file): forecasting', f2)
    o2, g2, s2 = controller_tables(pred2, real_te, 'exp2_indomain')
    show('Exp 2: controllers', o2); show('Exp 2: reduction by path group', g2); show('Exp 2: stats', s2)


if __name__ == '__main__':
    main()
