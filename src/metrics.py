"""
Log-normal and legacy comparison metrics for density time series.

Following Picone (2002) and Fitzpatrick (2025):
  - beta  = mean( ln(model / truth) )      — log-mean bias
  - sigma = std ( ln(model / truth) )       — log-std
  - SD%   = 100 * (exp(sigma) - 1)          — percentage scatter (Eq. 10)
  - r     = Pearson correlation
  - r²    = Pearson r-squared
"""

import numpy as np
import pandas as pd


def log_metrics(truth, model):
    """
    Compute log-normal and legacy metrics between truth and model arrays.

    Parameters
    ----------
    truth, model : array-like of float (same length)

    Returns
    -------
    dict with keys:
      beta, sigma, SD_pct, RMSE_log, r, r_squared, RMS_pct, MAPE, bias_pct, N
    """
    truth = np.asarray(truth, dtype=float)
    model = np.asarray(model, dtype=float)

    # Mask invalid
    valid = (np.isfinite(truth) & np.isfinite(model) &
             (truth > 0) & (model > 0))
    t = truth[valid]
    m = model[valid]
    N = len(t)

    if N < 2:
        return {k: np.nan for k in
                ['beta', 'sigma', 'SD_pct', 'RMSE_log', 'r', 'r_squared',
                 'RMS_pct', 'MAPE', 'bias_pct', 'N']}

    # Log-normal metrics (Picone 2002, Fitzpatrick 2025)
    ln_ratio = np.log(m / t)
    beta = np.mean(ln_ratio)
    sigma = np.std(ln_ratio, ddof=1)
    RMSE_log = np.sqrt(np.mean(ln_ratio**2))
    SD_pct = 100.0 * (np.exp(sigma) - 1.0)

    # Pearson correlation
    r = np.corrcoef(t, m)[0, 1] if N > 1 else np.nan
    r_squared = r**2 if np.isfinite(r) else np.nan

    # Legacy metrics
    pct_err = (m - t) / t * 100.0
    RMS_pct = np.sqrt(np.mean(pct_err**2))
    MAPE = np.mean(np.abs(pct_err))
    bias_pct = np.mean(pct_err)

    return {
        'beta': beta,
        'sigma': sigma,
        'SD_pct': SD_pct,
        'RMSE_log': RMSE_log,
        'r': r,
        'r_squared': r_squared,
        'RMS_pct': RMS_pct,
        'MAPE': MAPE,
        'bias_pct': bias_pct,
        'N': N,
    }


def compute_all_metrics(truth, models_dict):
    """
    Compute metrics for multiple models against a single truth.

    Parameters
    ----------
    truth       : array-like
    models_dict : dict  {name: array-like}

    Returns
    -------
    pd.DataFrame — one row per model, columns = metric names
    """
    rows = {}
    for name, model in models_dict.items():
        rows[name] = log_metrics(truth, model)
    return pd.DataFrame(rows).T
