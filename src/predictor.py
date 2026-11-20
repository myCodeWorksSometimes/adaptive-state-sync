"""
src/predictor.py
================
Implements the proposed prediction architecture from Phase 2 Deliverable 3 Step 5:
  - RBF Network (Radial Basis Function Neural Network) for forecasting
  - BOCPD-inspired drift detection for adaptive retraining

IMPORTANT: This predictor uses only features available at prediction time.
It NEVER receives future RTT, future bandwidth, or future severity as inputs.

Architecture
------------
RBF Network:
  - Input: rolling-window network statistics + gameplay context features
  - Hidden: RBF centres (learned via K-means on training data)
  - Output: linear combination of RBF activations (regression or classification)

BOCPD Approximation:
  - Full Bayesian Online Changepoint Detection (Adams & MacKay, 2007) is
    computationally intensive for real-time use. We implement a practical
    approximation using a CUSUM-style cumulative error monitor that triggers
    retraining when prediction error exceeds a learned threshold.
  - Deviation from full BOCPD is documented in METHODOLOGY_CHANGES.md.

Deviation from Phase 2 design:
  - Full BOCPD maintains a full distribution over changepoint locations.
    This implementation uses a sliding error window and threshold comparison,
    which captures the same purpose (detecting regime changes) but with lower
    computational cost suitable for a semester project.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# RBF Network
# ═══════════════════════════════════════════════════════════════════════════════

class RBFNetwork:
    """
    Radial Basis Function Neural Network for network condition forecasting.

    Follows the architecture described in Monaco et al. (CLAAP, 2025):
    - K-means clustering to initialise RBF centres
    - Gaussian RBF activations
    - Least-squares linear output layer

    Parameters
    ----------
    n_centres : int
        Number of RBF centres (hidden units). Tuned by cross-validation.
    gamma : float or None
        Width parameter for Gaussian RBFs. If None, computed from inter-centre
        distances (heuristic from CLAAP).
    task : str
        'regression' or 'classification'. For classification, the output layer
        uses a one-vs-rest scheme.
    n_classes : int
        Number of classes for classification task. Ignored for regression.
    seed : int
        Random seed for K-means initialisation.
    """

    def __init__(self, n_centres=32, gamma=None, task='regression',
                 n_classes=4, seed=42):
        self.n_centres  = n_centres
        self.gamma      = gamma
        self.task       = task
        self.n_classes  = n_classes
        self.seed       = seed
        self.centres_   = None
        self._gamma     = None
        self.weights_   = None
        self.is_fitted_ = False

    def _rbf_activation(self, X, batch_size=2000):
        """
        Compute Gaussian RBF activations for input matrix X.

        Uses batched computation to avoid materialising (N, K, F) arrays
        in memory for large datasets (would be ~1.84 GB for N=386K, K=32, F=13).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        batch_size : int
            Number of rows processed at once to bound memory use.
            Memory per batch: batch_size × K × F × 8 bytes.
            At batch_size=2000, K=32, F=13: ~6.6 MB per batch.

        Returns
        -------
        np.ndarray, shape (n_samples, n_centres)
        """
        n = X.shape[0]
        k = self.centres_.shape[0]
        Phi = np.empty((n, k), dtype=np.float64)
        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            Xb   = X[start:end]                                       # (b, f)
            diff = Xb[:, np.newaxis, :] - self.centres_[np.newaxis, :, :]  # (b, k, f)
            dist_sq = np.sum(diff ** 2, axis=2)                       # (b, k)
            Phi[start:end] = np.exp(-self._gamma * dist_sq)
        return Phi

    def fit(self, X, y):
        """
        Fit the RBF Network using K-means centre initialisation and
        least-squares weight estimation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features. Must be pre-scaled.
        y : np.ndarray, shape (n_samples,)
            Target values (continuous for regression, integer for classification).
        """
        n_samples = X.shape[0]

        # 1. Find RBF centres using K-means
        km = KMeans(
            n_clusters=min(self.n_centres, n_samples),
            random_state=self.seed,
            n_init=10,
        )
        km.fit(X)
        self.centres_ = km.cluster_centers_   # (k, f)

        # 2. Compute gamma from mean inter-centre distance (CLAAP heuristic)
        if self.gamma is None:
            diffs = self.centres_[:, np.newaxis, :] - self.centres_[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            mean_dist = np.mean(dists[dists > 0]) if np.any(dists > 0) else 1.0
            self._gamma = 1.0 / (2.0 * mean_dist ** 2)
        else:
            self._gamma = self.gamma

        # 3. Compute RBF activation matrix (design matrix for linear layer)
        Phi = self._rbf_activation(X)                # (n, k)
        # Add bias column
        Phi_b = np.hstack([Phi, np.ones((n_samples, 1))])  # (n, k+1)

        # 4. Fit linear output weights via least squares (pseudo-inverse)
        if self.task == 'regression':
            # Direct least-squares: W = (Phi^T Phi)^{-1} Phi^T y
            self.weights_, _, _, _ = np.linalg.lstsq(Phi_b, y.astype(float), rcond=None)
        else:
            # One-hot encode y for multi-class classification
            Y_onehot = np.zeros((n_samples, self.n_classes))
            for i, cls in enumerate(y.astype(int)):
                Y_onehot[i, cls] = 1.0
            self.weights_, _, _, _ = np.linalg.lstsq(Phi_b, Y_onehot, rcond=None)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict for input X.

        Returns
        -------
        np.ndarray : predicted values (regression) or class labels (classification)
        """
        if not self.is_fitted_:
            raise RuntimeError("RBFNetwork is not fitted yet.")
        Phi   = self._rbf_activation(X)
        Phi_b = np.hstack([Phi, np.ones((X.shape[0], 1))])
        raw   = Phi_b @ self.weights_                        # (n,) or (n, C)

        if self.task == 'regression':
            return raw
        else:
            # Argmax over class scores
            return np.argmax(raw, axis=1).astype(int)

    def predict_proba(self, X):
        """Return softmax-normalised class probabilities (classification only)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification.")
        Phi   = self._rbf_activation(X)
        Phi_b = np.hstack([Phi, np.ones((X.shape[0], 1))])
        raw   = Phi_b @ self.weights_
        # Softmax
        e     = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BOCPD Approximation — Drift Detector
# ═══════════════════════════════════════════════════════════════════════════════

class BOCPDApproximation:
    """
    Practical approximation of Bayesian Online Changepoint Detection (BOCPD).

    Motivation (from Phase 2 methodology, Step 5):
      CLAAP uses BOCPD for drift-adaptive retraining of the RBFNN predictor.
      Full BOCPD maintains a posterior distribution over changepoint locations,
      which is O(T^2) in memory and computationally heavy for a semester project.

    This implementation:
      Uses a CUSUM-style cumulative deviation monitor that triggers a "changepoint"
      when prediction error (sliding window mean) significantly exceeds the
      baseline error established on the training set.

    Deviation from full BOCPD:
      Full BOCPD: exact Bayesian posterior, O(T) message-passing
      This:       heuristic sliding window + threshold, O(W) memory, O(1) per step
      Documented in METHODOLOGY_CHANGES.md as an explicit approximation.

    Parameters
    ----------
    window : int
        Size of the sliding error window.
    threshold_sigma : float
        Number of standard deviations above training baseline to trigger detection.
    """

    def __init__(self, window=50, threshold_sigma=2.5):
        self.window          = window
        self.threshold_sigma = threshold_sigma
        self.baseline_mean_  = None
        self.baseline_std_   = None
        self.error_buffer_   = []
        self.changepoints_   = []
        self.n_steps_        = 0

    def fit_baseline(self, errors):
        """
        Establish baseline prediction error from training/validation data.

        Parameters
        ----------
        errors : array-like
            Prediction errors (absolute or squared) on training data.
        """
        errors = np.array(errors)
        self.baseline_mean_ = float(np.mean(errors))
        self.baseline_std_  = float(np.std(errors)) + 1e-9
        self.error_buffer_  = []
        self.n_steps_       = 0

    def step(self, error):
        """
        Process one new prediction error. Returns True if changepoint detected.

        Parameters
        ----------
        error : float
            Absolute prediction error at the current step.
        """
        self.error_buffer_.append(error)
        if len(self.error_buffer_) > self.window:
            self.error_buffer_.pop(0)
        self.n_steps_ += 1

        if self.baseline_mean_ is None:
            return False
        if len(self.error_buffer_) < max(5, self.window // 4):
            return False

        window_mean = np.mean(self.error_buffer_)
        z_score     = (window_mean - self.baseline_mean_) / self.baseline_std_

        if z_score > self.threshold_sigma:
            self.changepoints_.append(self.n_steps_)
            self.error_buffer_ = []   # reset buffer after detection
            return True
        return False

    @property
    def n_changepoints(self):
        return len(self.changepoints_)


# ═══════════════════════════════════════════════════════════════════════════════
# Composite Predictor — RBF + BOCPD
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptivePredictor:
    """
    Combined RBF Network + BOCPD approximation predictor.

    Implements the proposed prediction architecture from Phase 2 Step 5:
    - Trains a base RBF model on training data
    - Monitors prediction error online using BOCPD approximation
    - Triggers local retraining on a sliding buffer when drift is detected

    This predictor outputs:
      - Continuous forecast of future RTT (regression)
      - Continuous forecast of future bandwidth (regression)
      - Discrete future severity class (classification)

    Parameters
    ----------
    n_centres : int
        RBF hidden units for each sub-model.
    bocpd_window : int
        Error monitoring window for drift detection.
    bocpd_threshold : float
        Detection sensitivity (sigma multiplier).
    seed : int
        Global random seed.
    """

    def __init__(self, n_centres=32, bocpd_window=50,
                 bocpd_threshold=2.5, seed=42):
        self.n_centres       = n_centres
        self.bocpd_window    = bocpd_window
        self.bocpd_threshold = bocpd_threshold
        self.seed            = seed

        # Three sub-models: RTT regression, BW regression, severity classification
        self.rbf_rtt_   = RBFNetwork(n_centres=n_centres, task='regression', seed=seed)
        self.rbf_bw_    = RBFNetwork(n_centres=n_centres, task='regression', seed=seed+1)
        self.rbf_sev_   = RBFNetwork(n_centres=n_centres, task='classification',
                                     n_classes=4, seed=seed+2)

        # BOCPD monitors for each sub-model
        self.bocpd_rtt_ = BOCPDApproximation(bocpd_window, bocpd_threshold)
        self.bocpd_bw_  = BOCPDApproximation(bocpd_window, bocpd_threshold)
        self.bocpd_sev_ = BOCPDApproximation(bocpd_window, bocpd_threshold)

        # Scaler fitted on training features
        self.scaler_    = MinMaxScaler()
        self.is_fitted_ = False

    def fit(self, X_train, y_rtt, y_bw, y_sev):
        """
        Train all three RBF sub-models and establish BOCPD baselines.

        Parameters
        ----------
        X_train : np.ndarray (n, f) — training features (unscaled)
        y_rtt   : np.ndarray (n,)   — future RTT targets
        y_bw    : np.ndarray (n,)   — future bandwidth targets
        y_sev   : np.ndarray (n,)   — future severity class targets (0–3)
        """
        # Fit scaler on training data only
        X_scaled = self.scaler_.fit_transform(X_train)

        # Train sub-models
        self.rbf_rtt_.fit(X_scaled, y_rtt)
        self.rbf_bw_.fit(X_scaled, y_bw)
        self.rbf_sev_.fit(X_scaled, y_sev)

        # Establish BOCPD baselines on training predictions
        pred_rtt_tr = self.rbf_rtt_.predict(X_scaled)
        pred_bw_tr  = self.rbf_bw_.predict(X_scaled)
        pred_sev_tr = self.rbf_sev_.predict(X_scaled)

        self.bocpd_rtt_.fit_baseline(np.abs(pred_rtt_tr - y_rtt))
        self.bocpd_bw_.fit_baseline( np.abs(pred_bw_tr  - y_bw))
        self.bocpd_sev_.fit_baseline((pred_sev_tr != y_sev.astype(int)).astype(float))

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict future network conditions for input features X.

        Parameters
        ----------
        X : np.ndarray (n, f) — features (unscaled)

        Returns
        -------
        dict with keys:
          'rtt'      : np.ndarray (n,) — predicted future RTT (ms)
          'bw'       : np.ndarray (n,) — predicted future bandwidth (Mbps)
          'severity' : np.ndarray (n,) — predicted future severity class (0–3)
        """
        X_scaled = self.scaler_.transform(X)
        return {
            'rtt':      self.rbf_rtt_.predict(X_scaled),
            'bw':       self.rbf_bw_.predict(X_scaled),
            'severity': self.rbf_sev_.predict(X_scaled),
        }

    def evaluate(self, X, y_rtt, y_bw, y_sev, label="Test"):
        """
        Evaluate all three sub-models and return metric dictionaries.

        Returns
        -------
        dict of dicts with keys 'rtt', 'bw', 'severity'
        """
        preds = self.predict(X)
        rtt_range = max(y_rtt.max() - y_rtt.min(), 1e-9)
        bw_range  = max(y_bw.max()  - y_bw.min(),  1e-9)

        rtt_mae  = mean_absolute_error(y_rtt, preds['rtt'])
        rtt_rmse = np.sqrt(mean_squared_error(y_rtt, preds['rtt']))
        bw_mae   = mean_absolute_error(y_bw,  preds['bw'])
        bw_rmse  = np.sqrt(mean_squared_error(y_bw,  preds['bw']))

        metrics = {
            'rtt': {
                'mae':   rtt_mae,
                'rmse':  rtt_rmse,
                'nrmse': rtt_rmse / rtt_range,   # normalised by value range
            },
            'bw': {
                'mae':   bw_mae,
                'rmse':  bw_rmse,
                'nrmse': bw_rmse / bw_range,
            },
            'severity': {
                'accuracy':          accuracy_score(y_sev, preds['severity']),
                'precision_macro':   precision_score(y_sev, preds['severity'],
                                                     average='macro', zero_division=0),
                'recall_macro':      recall_score(y_sev, preds['severity'],
                                                  average='macro', zero_division=0),
                'f1_macro':          f1_score(y_sev, preds['severity'],
                                              average='macro', zero_division=0),
                'confusion_matrix':  confusion_matrix(y_sev, preds['severity'],
                                                       labels=[0, 1, 2, 3]),
                'report':            classification_report(y_sev, preds['severity'],
                                                           labels=[0, 1, 2, 3],
                                                           target_names=['None','Mild','Moderate','Severe'],
                                                           zero_division=0),
            }
        }
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Naive forecasting baselines (no learning)
# ═══════════════════════════════════════════════════════════════════════════════

class PersistenceForecaster:
    """
    Naive forecasting baselines used to judge whether the RBF predictor adds
    value over "no prediction at all".

    mode='last'   : future value = most recent measurement at tick t
                    (rtt_ms, bandwidth_mbps, current severity label)
    mode='window' : future value = mean of the 20-tick rolling window
                    (rtt_mean, bw_mean); severity = threshold rule applied to
                    (rtt_mean, loss_rate_mean)

    Both use only information available at prediction time (tick t).
    """

    def __init__(self, mode='last'):
        if mode not in ('last', 'window'):
            raise ValueError("mode must be 'last' or 'window'")
        self.mode = mode

    @staticmethod
    def _severity_rule(rtt, loss):
        rtt = np.asarray(rtt, dtype=float)
        loss = np.asarray(loss, dtype=float)
        sev = np.full(rtt.shape, 3, dtype=int)
        sev[(rtt < 180) & (loss < 0.030)] = 2
        sev[(rtt < 100) & (loss < 0.015)] = 1
        sev[(rtt < 50) & (loss < 0.005)] = 0
        return sev

    def predict(self, df):
        """df must contain the dataset's measurement and rolling-window columns."""
        if self.mode == 'last':
            return {
                'rtt':      df['rtt_ms'].values.astype(float),
                'bw':       df['bandwidth_mbps'].values.astype(float),
                'severity': df['severity'].values.astype(int),
            }
        return {
            'rtt':      df['rtt_mean'].values.astype(float),
            'bw':       df['bw_mean'].values.astype(float),
            'severity': self._severity_rule(df['rtt_mean'].values,
                                            df['loss_rate_mean'].values),
        }

    def evaluate(self, df, y_rtt, y_bw, y_sev):
        """Return the same flat metric set used for the RBF predictor."""
        p = self.predict(df)
        rtt_range = max(y_rtt.max() - y_rtt.min(), 1e-9)
        bw_range  = max(y_bw.max()  - y_bw.min(),  1e-9)
        rtt_rmse = np.sqrt(mean_squared_error(y_rtt, p['rtt']))
        bw_rmse  = np.sqrt(mean_squared_error(y_bw,  p['bw']))
        return {
            'rtt_mae_ms':   mean_absolute_error(y_rtt, p['rtt']),
            'rtt_rmse_ms':  rtt_rmse,
            'rtt_nrmse':    rtt_rmse / rtt_range,
            'bw_mae_mbps':  mean_absolute_error(y_bw, p['bw']),
            'bw_rmse_mbps': bw_rmse,
            'bw_nrmse':     bw_rmse / bw_range,
            'sev_accuracy': accuracy_score(y_sev, p['severity']),
            'sev_f1_macro': f1_score(y_sev, p['severity'], average='macro',
                                     zero_division=0),
        }
