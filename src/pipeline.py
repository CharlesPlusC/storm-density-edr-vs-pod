"""
End-to-end analysis pipeline for DensityInversion2.

Loads raw ephemeris + TU Delft files (or a pre-merged CSV), runs EDR and POD
density inversion, computes effective densities, evaluates log-normal metrics,
and produces comparison / diagnostic plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tqdm.auto import tqdm

from .config import get_satellite, ballistic_coeff, SATELLITES
from .data_loaders import (
    load_ephemeris, load_tudelft_density, load_swarm_density,
    find_ephemeris_file, find_tudelft_file, find_accel_density_file,
)
from .edr import find_perigees, compute_edr, edr_to_density
from .pod_accelerometry import pod_density_from_positions
from .effective_density import compute_effective_density
from .metrics import log_metrics, compute_all_metrics


# ====================================================================
# Load & parse
# ====================================================================
def load_storm_csv(csv_path):
    """
    Load a storm CSV and return structured arrays.

    Returns
    -------
    df          : DataFrame with UTC index
    pos_vel_eci : (N, 6) ndarray
    times       : list of datetime
    """
    df = pd.read_csv(csv_path, parse_dates=['UTC'])
    df = df.sort_values('UTC').reset_index(drop=True)

    pos_vel_cols = ['x', 'y', 'z', 'xv', 'yv', 'zv']
    pos_vel_eci = df[pos_vel_cols].values.astype(float)
    times = df['UTC'].dt.to_pydatetime().tolist()

    return df, pos_vel_eci, times


# ====================================================================
# Main analysis
# ====================================================================
def analyze_storm(csv_path, sat_name='GRACE-FO',
                  edr_fitspan=1,
                  correct_3bp=True, correct_srp=True):
    """
    Full pipeline: load CSV → EDR + POD + ACC + models → effective density → metrics.

    Parameters
    ----------
    csv_path        : str — path to storm CSV with withEDR columns
    sat_name        : str — satellite name (must be in config.SATELLITES)
    edr_fitspan     : int — perigee arcs to span for EDR
    correct_3bp     : bool — subtract 3BP work in EDR
    correct_srp     : bool — subtract SRP work in EDR

    Returns
    -------
    results : dict with keys:
      'df', 'perigee_indices', 'orbit_times',
      'edr_density', 'edr_times',
      'pod_density_hicad', 'pod_eff_density',
      'acc_eff_density', 'jb08_eff_density', 'nrlmsise_eff_density',
      'metrics'
    """
    sat = get_satellite(sat_name)
    df, pos_vel_eci, times = load_storm_csv(csv_path)

    # ---- Perigees ----
    perigee_indices = find_perigees(pos_vel_eci, times)
    orbit_mid_times = []
    for k in range(len(perigee_indices) - 1):
        i0 = perigee_indices[k]
        i1 = perigee_indices[k + 1]
        orbit_mid_times.append(times[i0] + (times[i1] - times[i0]) / 2)

    print(f"Found {len(perigee_indices)} perigees → {len(perigee_indices)-1} orbits")

    # ---- EDR path ----
    print("Computing EDR...")
    edr_values, edr_midtimes = compute_edr(
        pos_vel_eci, times, perigee_indices,
        fitspan=edr_fitspan,
        correct_3bp=correct_3bp,
        correct_srp=correct_srp,
        Cr=sat['Cr'], area=sat['area'], mass=sat['mass'],
    )
    rho_edr, edr_times = edr_to_density(
        edr_values, pos_vel_eci, times, perigee_indices,
        sat_name=sat_name, fitspan=edr_fitspan,
    )

    # ---- POD-accel path (full, from SP3 positions) ----
    rho_pod_hicad = None
    rho_pod_eff = None
    pod_result = None

    print("\n--- POD: Full pipeline from SP3 positions ---")
    pod_result = pod_density_from_positions(
        pos_vel_eci, times,
        sat_name=sat_name,
        output_cadence=15,
    )
    rho_pod_hicad = pod_result['rho']
    pod_pv_out = pod_result['pos_vel_out']
    pod_times_out = pod_result['times_out']

    # Find perigees on the POD output grid for effective density
    pod_perigee_idx = find_perigees(pod_pv_out, pod_times_out)
    if len(pod_perigee_idx) > 1:
        print("Computing POD effective density...")
        rho_pod_eff, pod_mid_idx = compute_effective_density(
            rho_pod_hicad, pod_pv_out, pod_perigee_idx)
    else:
        print("WARNING: not enough POD perigees for effective density.")

    # ---- Accelerometer (truth) ----
    rho_acc_eff = None
    if 'AccelerometerDensity' in df.columns:
        acc_density = df['AccelerometerDensity'].values.astype(float)
        if np.isfinite(acc_density).sum() > 100:
            print("Computing ACC effective density...")
            rho_acc_eff, _ = compute_effective_density(
                acc_density, pos_vel_eci, perigee_indices)

    # ---- NRLMSISE-00 ----
    rho_nrl_eff = None
    nrl_col = None
    for c in ['NRLMSISE-00', 'NRLMSISE00', 'nrlmsise00']:
        if c in df.columns:
            nrl_col = c
            break
    if nrl_col is not None:
        nrl_density = df[nrl_col].values.astype(float)
        if np.isfinite(nrl_density).sum() > 100:
            print("Computing NRLMSISE-00 effective density...")
            rho_nrl_eff, _ = compute_effective_density(
                nrl_density, pos_vel_eci, perigee_indices)

    # ---- JB08 ----
    rho_jb08_eff = None
    if 'JB08' in df.columns:
        jb08_density = df['JB08'].values.astype(float)
        if np.isfinite(jb08_density).sum() > 100:
            print("Computing JB08 effective density...")
            rho_jb08_eff, _ = compute_effective_density(
                jb08_density, pos_vel_eci, perigee_indices)

    # ---- De-bias on log scale (Fitzpatrick 2025, Picone 2002) ----
    # Subtract mean ln-ratio from each method to remove systematic bias
    def _debias(model_arr, truth_arr):
        """De-bias model on log scale: multiply by exp(-beta)."""
        m = np.array(model_arr, dtype=float)
        t = np.array(truth_arr, dtype=float)
        valid = np.isfinite(m) & np.isfinite(t) & (m > 0) & (t > 0)
        if valid.sum() < 2:
            return m
        beta = np.mean(np.log(m[valid] / t[valid]))
        print(f"  De-bias factor: exp(-beta) = {np.exp(-beta):.4f}  (beta={beta:.4f})")
        return m * np.exp(-beta)

    # Store raw (pre-debias) for reference
    rho_edr_raw = list(rho_edr) if rho_edr is not None else None
    rho_pod_eff_raw = list(rho_pod_eff) if rho_pod_eff is not None else None

    if rho_acc_eff is not None:
        acc_arr = np.array(rho_acc_eff)
        if rho_edr is not None:
            print("De-biasing EDR...")
            rho_edr = list(_debias(rho_edr, acc_arr[:len(rho_edr)]))
        if rho_pod_eff is not None:
            print("De-biasing POD...")
            rho_pod_eff = list(_debias(rho_pod_eff, acc_arr[:len(rho_pod_eff)]))
        if rho_nrl_eff is not None:
            print("De-biasing NRLMSISE-00...")
            rho_nrl_eff = list(_debias(rho_nrl_eff, acc_arr[:len(rho_nrl_eff)]))
        if rho_jb08_eff is not None:
            print("De-biasing JB08...")
            rho_jb08_eff = list(_debias(rho_jb08_eff, acc_arr[:len(rho_jb08_eff)]))

    # ---- Metrics (post-debias) ----
    metrics_raw = None
    metrics = None
    if rho_acc_eff is not None:
        truth = np.array(rho_acc_eff)

        # Raw metrics (pre-debias)
        models_raw = {}
        if rho_edr_raw is not None:
            models_raw['EDR'] = np.array(rho_edr_raw)
        if rho_pod_eff_raw is not None:
            models_raw['POD'] = np.array(rho_pod_eff_raw)

        # De-biased metrics
        models = {}
        if rho_edr is not None:
            models['EDR'] = np.array(rho_edr)
        if rho_pod_eff is not None:
            models['POD'] = np.array(rho_pod_eff)
        if rho_nrl_eff is not None:
            models['NRLMSISE-00'] = np.array(rho_nrl_eff)
        if rho_jb08_eff is not None:
            models['JB08'] = np.array(rho_jb08_eff)

        if models_raw:
            min_len = min(len(truth), *(len(v) for v in models_raw.values()))
            truth_trim = truth[:min_len]
            mr = {k: np.array(v[:min_len]) for k, v in models_raw.items()}
            metrics_raw = compute_all_metrics(truth_trim, mr)
            print("\n=== Raw Metrics (pre-debias, vs Accelerometer) ===")
            print(metrics_raw.to_string())

        if models:
            min_len = min(len(truth), *(len(v) for v in models.values()))
            truth_trim = truth[:min_len]
            models_trim = {k: np.array(v[:min_len]) for k, v in models.items()}
            metrics = compute_all_metrics(truth_trim, models_trim)
            print("\n=== De-biased Metrics (vs Accelerometer) ===")
            print(metrics.to_string())

    results = {
        'df': df,
        'pos_vel_eci': pos_vel_eci,
        'times': times,
        'perigee_indices': perigee_indices,
        'orbit_mid_times': orbit_mid_times,
        'edr_values': edr_values,
        'edr_density': rho_edr,
        'edr_density_raw': rho_edr_raw,
        'edr_times': edr_times,
        'pod_result': pod_result,
        'pod_density_hicad': rho_pod_hicad,
        'pod_times_out': pod_times_out,
        'pod_eff_density': rho_pod_eff,
        'pod_eff_density_raw': rho_pod_eff_raw,
        'acc_eff_density': rho_acc_eff,
        'jb08_eff_density': rho_jb08_eff,
        'nrlmsise_eff_density': rho_nrl_eff,
        'metrics_raw': metrics_raw,
        'metrics': metrics,
    }
    return results


# ====================================================================
# Raw-file entry point
# ====================================================================
def analyze_storm_from_files(ephem_path, tudelft_path=None, sat_name='GRACE-FO-A',
                              storm_start=None, storm_end=None,
                              diagnostics=True, output_dir='output',
                              accel_path=None, accel_fmt=None):
    """
    Full pipeline from raw files: load ephemeris + truth density → POD → effective density → metrics.

    Parameters
    ----------
    ephem_path   : str — path to ephemeris text file
    tudelft_path : str or list of str — path(s) to TU Delft density file(s) (legacy)
    sat_name     : str — satellite name (must be in config.SATELLITES)
    storm_start  : datetime or None — trim data to this start
    storm_end    : datetime or None — trim data to this end
    diagnostics  : bool — produce intermediate diagnostic plots
    output_dir   : str — directory for saving plots
    accel_path   : str or None — path to accelerometer/truth density file (unified)
    accel_fmt    : str or None — 'tudelft' or 'swarm' (used with accel_path)

    Returns
    -------
    results : dict
    """
    sat = get_satellite(sat_name)

    # ---- Load ephemeris ----
    print(f"Loading ephemeris: {ephem_path}")
    pos_vel_eci, times = load_ephemeris(ephem_path)
    print(f"  {len(times)} points, {times[0]} → {times[-1]}")

    # ---- Trim to storm window if specified ----
    if storm_start or storm_end:
        mask = np.ones(len(times), dtype=bool)
        if storm_start:
            mask &= np.array([t >= storm_start for t in times])
        if storm_end:
            mask &= np.array([t <= storm_end for t in times])
        pos_vel_eci = pos_vel_eci[mask]
        times = [t for t, m in zip(times, mask) if m]
        print(f"  Trimmed to {len(times)} points, {times[0]} → {times[-1]}")

    # ---- Load truth density (unified: TU Delft or Swarm) ----
    # Prefer multi-month tudelft_path (covers full storm window); fall back to accel_path
    if tudelft_path:
        truth_path = tudelft_path
        truth_fmt = 'tudelft'
    elif accel_path:
        truth_path = accel_path
        truth_fmt = accel_fmt
    else:
        truth_path = None
        truth_fmt = None
    nn_threshold = 15.0  # nearest-neighbor threshold in seconds

    if truth_path and truth_fmt == 'swarm':
        print(f"Loading Swarm density: {truth_path}")
        td_times, td_density = load_swarm_density(
            truth_path,
            start_date=times[0] - timedelta(hours=1),
            end_date=times[-1] + timedelta(hours=1),
        )
        nn_threshold = 30.0  # Swarm POD is 30s cadence
        print(f"  {len(td_times)} density points loaded (Swarm, NN threshold={nn_threshold}s)")
    elif truth_path and truth_fmt == 'tudelft':
        print(f"Loading TU Delft density: {truth_path}")
        td_times, td_density = load_tudelft_density(
            truth_path,
            start_date=times[0] - timedelta(hours=1),
            end_date=times[-1] + timedelta(hours=1),
        )
        print(f"  {len(td_times)} density points loaded")
    else:
        print("No accelerometer density data available — skipping truth comparison")
        td_times, td_density = [], np.array([])

    # ---- Perigees on the SP3 grid ----
    perigee_indices = find_perigees(pos_vel_eci, times)
    orbit_mid_times = []
    for k in range(len(perigee_indices) - 1):
        i0 = perigee_indices[k]
        i1 = perigee_indices[k + 1]
        orbit_mid_times.append(times[i0] + (times[i1] - times[i0]) / 2)
    print(f"Found {len(perigee_indices)} perigees → {len(perigee_indices)-1} orbits")

    # ---- POD-accelerometry ----
    print("\n=== POD-Accelerometry ===")
    pod_result = pod_density_from_positions(
        pos_vel_eci, times,
        sat_name=sat_name,
        output_cadence=15,
    )
    rho_pod_hicad = pod_result['rho']
    pod_pv_out = pod_result['pos_vel_out']
    pod_times_out = pod_result['times_out']

    # POD effective density
    pod_perigee_idx = find_perigees(pod_pv_out, pod_times_out)
    rho_pod_eff = None
    if len(pod_perigee_idx) > 1:
        print("Computing POD effective density...")
        rho_pod_eff, pod_mid_idx = compute_effective_density(
            rho_pod_hicad, pod_pv_out, pod_perigee_idx)
        print(f"  {len(rho_pod_eff)} orbit-effective values")
    else:
        print("WARNING: not enough POD perigees for effective density.")

    # ---- Truth effective density ----
    # Interpolate truth density to the SP3 timestamps (nearest-neighbor via searchsorted)
    print("Interpolating truth density to SP3 grid...")
    acc_density_on_sp3 = np.full(len(times), np.nan)
    if len(td_times) > 0:
        # Convert to seconds from a common epoch for fast searchsorted
        t0_ref = td_times[0]
        td_sec = np.array([(t - t0_ref).total_seconds() for t in td_times])
        sp3_sec = np.array([(t - t0_ref).total_seconds() for t in times])

        # searchsorted gives insertion point; check both neighbours
        idx = np.searchsorted(td_sec, sp3_sec, side='left')
        for i in tqdm(range(len(sp3_sec)), desc='Truth density interpolation'):
            best_diff = 999.0
            best_j = -1
            for j in [idx[i] - 1, idx[i]]:
                if 0 <= j < len(td_sec):
                    diff = abs(sp3_sec[i] - td_sec[j])
                    if diff < best_diff:
                        best_diff = diff
                        best_j = j
            if best_diff <= nn_threshold:
                acc_density_on_sp3[i] = td_density[best_j]

    n_matched = np.isfinite(acc_density_on_sp3).sum()
    print(f"  Matched {n_matched}/{len(times)} SP3 points to truth density")

    rho_acc_eff = None
    if n_matched > 100:
        print("Computing truth effective density...")
        rho_acc_eff, _ = compute_effective_density(
            acc_density_on_sp3, pos_vel_eci, perigee_indices)
        print(f"  {len(rho_acc_eff)} orbit-effective values")

    # ---- EDR (Energy Dissipation Rate) ----
    print("\n=== EDR (Energy Dissipation Rate) ===")
    edr_values, edr_midtimes = compute_edr(
        pos_vel_eci, times, perigee_indices,
        fitspan=1,
        correct_3bp=True, correct_srp=True,
        Cr=sat['Cr'], area=sat['area'], mass=sat['mass'],
    )
    rho_edr, edr_times = edr_to_density(
        edr_values, pos_vel_eci, times, perigee_indices,
        sat_name=sat_name, fitspan=1,
    )
    print(f"  {len(edr_values)} EDR values, {len(rho_edr)} density values")

    # ---- Diagnostic plots ----
    if diagnostics:
        os.makedirs(output_dir, exist_ok=True)
        plot_pod_diagnostics(pod_result, save_dir=output_dir)
        plot_edr_diagnostics(edr_values, edr_midtimes, rho_edr, edr_times,
                             save_dir=output_dir)

    # ---- De-bias helper ----
    def _debias(model_list, truth_arr):
        m = np.array(model_list, dtype=float)
        n = min(len(m), len(truth_arr))
        valid = (np.isfinite(m[:n]) & np.isfinite(truth_arr[:n])
                 & (m[:n] > 0) & (truth_arr[:n] > 0))
        if valid.sum() < 2:
            return model_list, None
        beta = np.mean(np.log(m[:n][valid] / truth_arr[:n][valid]))
        factor = np.exp(-beta)
        return [v * factor for v in model_list], factor

    # ---- Raw metrics (pre-debias) ----
    metrics_raw = None
    metrics = None
    rho_pod_eff_raw = list(rho_pod_eff) if rho_pod_eff is not None else None
    rho_edr_raw = list(rho_edr)

    if rho_acc_eff is not None:
        truth = np.array(rho_acc_eff)
        models_raw = {}
        n_common = len(truth)
        if rho_pod_eff is not None:
            n_common = min(n_common, len(rho_pod_eff))
            models_raw['POD'] = np.array(rho_pod_eff[:n_common])
        if rho_edr is not None:
            n_common = min(n_common, len(rho_edr))
            models_raw['EDR'] = np.array(rho_edr[:n_common])
        if models_raw:
            truth_trim = truth[:n_common]
            models_raw = {k: v[:n_common] for k, v in models_raw.items()}
            metrics_raw = compute_all_metrics(truth_trim, models_raw)
            print("\n=== Raw Metrics (pre-debias, vs Accelerometer) ===")
            print(metrics_raw.to_string())

    # ---- De-bias all methods on log scale ----
    pod_debias_factor = None
    edr_debias_factor = None
    if rho_acc_eff is not None:
        acc_arr = np.array(rho_acc_eff)
        if rho_pod_eff is not None:
            rho_pod_eff, pod_debias_factor = _debias(rho_pod_eff, acc_arr)
            if pod_debias_factor is not None:
                print(f"De-biasing POD: factor={pod_debias_factor:.4f}")
        if rho_edr is not None:
            rho_edr, edr_debias_factor = _debias(rho_edr, acc_arr)
            if edr_debias_factor is not None:
                print(f"De-biasing EDR: factor={edr_debias_factor:.4f}")
        # De-biased metrics
        models_db = {}
        n_common = len(acc_arr)
        if rho_pod_eff is not None:
            n_common = min(n_common, len(rho_pod_eff))
            models_db['POD'] = np.array(rho_pod_eff[:n_common])
        if rho_edr is not None:
            n_common = min(n_common, len(rho_edr))
            models_db['EDR'] = np.array(rho_edr[:n_common])
        if models_db:
            truth_trim = acc_arr[:n_common]
            models_db = {k: v[:n_common] for k, v in models_db.items()}
            metrics = compute_all_metrics(truth_trim, models_db)
            print("\n=== De-biased Metrics (vs Accelerometer) ===")
            print(metrics.to_string())

    # ---- Comparison plot: time series + log-ratio residuals ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    # Panel 1: Time series
    ax = axes[0]
    if rho_acc_eff is not None:
        n = min(len(orbit_mid_times), len(rho_acc_eff))
        ax.plot(orbit_mid_times[:n],
                rho_acc_eff[:n], 'k-', lw=1.5, label='Accelerometer (truth)', zorder=5)
    if rho_pod_eff is not None:
        n = min(len(orbit_mid_times), len(rho_pod_eff))
        ax.plot(orbit_mid_times[:n],
                rho_pod_eff[:n], 'b-', lw=1.0, label='POD (de-biased)', alpha=0.8)
    if rho_edr is not None:
        n = min(len(edr_times), len(rho_edr))
        ax.plot(edr_times[:n],
                rho_edr[:n], 'r-', lw=1.0, label='EDR (de-biased)', alpha=0.8)
    ax.set_ylabel('Effective Density [kg/m$^3$]')
    ax.set_title(f'{sat_name} — Density Inversion Comparison (orbit-effective)')
    ax.legend(loc='upper right')
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.3)

    # Panel 2: Log-ratio residuals
    ax = axes[1]
    if rho_acc_eff is not None:
        acc_arr = np.array(rho_acc_eff)

        def _plot_logratio(model_list, model_times, label, color):
            m = np.array(model_list, dtype=float)
            n = min(len(m), len(acc_arr), len(model_times))
            m = m[:n]
            t = np.array(model_times[:n])
            a = acc_arr[:n]
            valid = np.isfinite(m) & np.isfinite(a) & (m > 0) & (a > 0)
            ratio = np.full_like(m, np.nan)
            ratio[valid] = np.log(m[valid] / a[valid])
            ax.plot(t, ratio, color=color, lw=1.0, alpha=0.8, label=label)

        if rho_pod_eff is not None:
            _plot_logratio(rho_pod_eff, orbit_mid_times, 'POD', 'blue')
        if rho_edr is not None:
            _plot_logratio(rho_edr, edr_times, 'EDR', 'red')
        ax.axhline(0, color='k', lw=0.5, ls='--')

    ax.set_ylabel('ln(model / truth)')
    ax.set_xlabel('UTC')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        a.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'density_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    results = {
        'pos_vel_eci': pos_vel_eci,
        'times': times,
        'perigee_indices': perigee_indices,
        'orbit_mid_times': orbit_mid_times,
        'pod_result': pod_result,
        'pod_density_hicad': rho_pod_hicad,
        'pod_times_out': pod_times_out,
        'pod_eff_density': rho_pod_eff,
        'pod_eff_density_raw': rho_pod_eff_raw,
        'pod_debias_factor': pod_debias_factor,
        'edr_values': edr_values,
        'edr_density': rho_edr,
        'edr_density_raw': rho_edr_raw,
        'edr_times': edr_times,
        'edr_midtimes': edr_midtimes,
        'edr_debias_factor': edr_debias_factor,
        'acc_density_on_sp3': acc_density_on_sp3,
        'acc_eff_density': rho_acc_eff,
        'td_times': td_times,
        'td_density': td_density,
        'metrics_raw': metrics_raw,
        'metrics': metrics,
    }

    # ---- Save derived density data ----
    os.makedirs(output_dir, exist_ok=True)

    # Orbit-effective densities (POD + ACC share orbit_mid_times)
    n_pod = len(rho_pod_eff) if rho_pod_eff is not None else 0
    n_acc = len(rho_acc_eff) if rho_acc_eff is not None else 0
    n_orb = len(orbit_mid_times)
    def _pad(lst, n):
        if lst is None:
            return [np.nan] * n
        arr = list(lst)
        return (arr + [np.nan] * n)[:n]
    eff_df = pd.DataFrame({
        'time': orbit_mid_times[:n_orb],
        'acc_effective': _pad(rho_acc_eff, n_orb),
        'pod_raw': _pad(rho_pod_eff_raw, n_orb),
        'pod_debiased': _pad(rho_pod_eff, n_orb),
    })
    eff_path = os.path.join(output_dir, 'pod_acc_effective.csv')
    eff_df.to_csv(eff_path, index=False)
    print(f"Saved: {eff_path}")

    # EDR orbit-effective densities (own time axis from arc midpoints)
    n_edr = len(rho_edr) if rho_edr is not None else 0
    edr_df = pd.DataFrame({
        'time': edr_times[:n_edr],
        'edr_raw': _pad(rho_edr_raw, n_edr),
        'edr_debiased': _pad(rho_edr, n_edr),
    })
    edr_path = os.path.join(output_dir, 'edr_effective.csv')
    edr_df.to_csv(edr_path, index=False)
    print(f"Saved: {edr_path}")

    # High-cadence POD density (15s cadence)
    hicad_df = pd.DataFrame({
        'time': pod_times_out,
        'pod_density': rho_pod_hicad,
        'pod_density_raw': pod_result['rho_raw'],
    })
    hicad_path = os.path.join(output_dir, 'pod_density_hicad.csv')
    hicad_df.to_csv(hicad_path, index=False)
    print(f"Saved: {hicad_path}")

    return results


# ====================================================================
# POD diagnostic plots
# ====================================================================
def plot_pod_diagnostics(pod_result, save_dir='output'):
    """
    Four diagnostic plots for the POD-accelerometry pipeline.

    1. Raw vs smoothed velocity (one component)
    2. Non-conservative acceleration in HCL frame
    3. Force model magnitudes
    4. High-cadence density time series
    """
    times = pod_result['times_out']
    t_hours = np.array([(t - times[0]).total_seconds() / 3600.0 for t in times])

    os.makedirs(save_dir, exist_ok=True)

    # --- 1. SG-smoothed velocity ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    labels = ['Vx', 'Vy', 'Vz']
    for j in range(3):
        ax = axes[j]
        ax.plot(t_hours, pod_result['vel_smooth'][:, j], 'b-', lw=0.8,
                label='SG-smoothed velocity')
        ax.set_ylabel(f'{labels[j]} [m/s]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [hours from start]')
    axes[0].set_title('POD Diagnostic: SG-Smoothed Velocity')
    plt.tight_layout()
    path = os.path.join(save_dir, 'diag_1_velocity_smoothing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # --- 2. Non-conservative acceleration in HCL ---
    h_nc, c_nc, l_nc = pod_result['nc_hcl']
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, data, label in zip(axes, [h_nc, c_nc, l_nc], ['H (radial)', 'C (cross-track)', 'L (along-track)']):
        ax.plot(t_hours, data, 'k-', lw=0.5)
        ax.set_ylabel(f'a_{label.split()[0]} [m/s²]')
        ax.set_title(f'{label}')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-8, -4))
    axes[-1].set_xlabel('Time [hours from start]')
    fig.suptitle('POD Diagnostic: Non-Conservative Acceleration (HCL)', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'diag_2_nc_acceleration_hcl.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # --- 3. Force model magnitudes ---
    a_obs_mag = np.linalg.norm(pod_result['a_obs'], axis=1)
    a_grav_mag = np.linalg.norm(pod_result['a_grav'], axis=1)
    a_3bp_mag = np.linalg.norm(pod_result['a_3bp'], axis=1)
    a_srp_mag = np.linalg.norm(pod_result['a_srp'], axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.semilogy(t_hours, a_obs_mag, 'k-', lw=0.8, label='|a_observed|')
    ax.semilogy(t_hours, a_grav_mag, 'b-', lw=0.8, label='|a_gravity|', alpha=0.7)
    ax.semilogy(t_hours, a_3bp_mag, 'g-', lw=0.8, label='|a_3BP|', alpha=0.7)
    ax.semilogy(t_hours, a_srp_mag, 'r-', lw=0.8, label='|a_SRP|', alpha=0.7)
    ax.set_xlabel('Time [hours from start]')
    ax.set_ylabel('Acceleration magnitude [m/s²]')
    ax.set_title('POD Diagnostic: Force Model Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'diag_3_force_magnitudes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # --- 4. High-cadence density ---
    rho = pod_result['rho']
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_hours, rho, 'b-', lw=0.5)
    ax.set_xlabel('Time [hours from start]')
    ax.set_ylabel('Density [kg/m³]')
    ax.set_title('POD Diagnostic: High-Cadence Density (before orbit-averaging)')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-13, -10))
    ax.grid(True, alpha=0.3)
    # Mark negative densities
    neg = rho < 0
    if neg.any():
        ax.axhline(0, color='r', ls='--', lw=0.5)
        pct_neg = 100.0 * neg.sum() / len(rho)
        ax.set_title(f'POD Diagnostic: High-Cadence Density ({pct_neg:.1f}% negative)')
    plt.tight_layout()
    path = os.path.join(save_dir, 'diag_4_hicad_density.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ====================================================================
# EDR diagnostic plots
# ====================================================================
def plot_edr_diagnostics(edr_values, edr_midtimes, rho_edr, edr_times,
                         save_dir='output'):
    """
    Two diagnostic plots for the EDR pipeline.

    1. EDR values (W/kg) vs time — should be smooth, no wavy oscillation
    2. EDR density vs time
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. EDR values (W/kg) vs time ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(edr_midtimes, edr_values, 'ro-', ms=3, lw=0.8)
    ax.set_xlabel('UTC')
    ax.set_ylabel('EDR [W/kg = m²/s³]')
    ax.set_title('EDR Diagnostic: Energy Dissipation Rate')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-5, -2))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    path = os.path.join(save_dir, 'diag_edr_1_values.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # --- 2. EDR density vs time ---
    rho_arr = np.array(rho_edr, dtype=float)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(edr_times[:len(rho_arr)], rho_arr, 'ro-', ms=3, lw=0.8)
    ax.set_xlabel('UTC')
    ax.set_ylabel('Density [kg/m³]')
    ax.set_title('EDR Diagnostic: Orbit-Effective Density')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-13, -10))
    ax.grid(True, alpha=0.3)
    neg = rho_arr < 0
    if np.any(neg):
        ax.axhline(0, color='r', ls='--', lw=0.5)
        pct_neg = 100.0 * np.nansum(neg) / len(rho_arr)
        ax.set_title(f'EDR Diagnostic: Orbit-Effective Density ({pct_neg:.1f}% negative)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    path = os.path.join(save_dir, 'diag_edr_2_density.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ====================================================================
# Plotting
# ====================================================================
def plot_storm_comparison(results, title=None, save_path=None):
    """
    Three-panel comparison plot:
      1. Time series: ACC, EDR, POD effective densities
      2. Time series: ACC vs empirical models (NRLMSISE-00, JB08)
      3. Residual ratios on log scale
    """
    orbit_times = results['orbit_mid_times']
    n_orbits = len(orbit_times)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ---- Panel 1: Inversion methods ----
    ax = axes[0]
    if results['acc_eff_density'] is not None:
        acc = np.array(results['acc_eff_density'][:n_orbits])
        ax.plot(orbit_times, acc, 'k-', lw=1.5, label='Accelerometer', zorder=5)

    if results['edr_density'] is not None:
        edr = np.array(results['edr_density'][:n_orbits])
        edr_t = results['edr_times'][:n_orbits]
        ax.plot(edr_t, edr, 'ro-', ms=3, lw=1.0, label='EDR', alpha=0.8)

    if results['pod_eff_density'] is not None:
        pod = np.array(results['pod_eff_density'][:n_orbits])
        ax.plot(orbit_times, pod, 'b-', lw=1.0, label='POD-accel', alpha=0.8)

    ax.set_ylabel('Density [kg/m$^3$]')
    ax.legend(loc='upper right')
    ax.set_title(title or 'Density Inversion Comparison')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-13, -10))
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Models ----
    ax = axes[1]
    if results['acc_eff_density'] is not None:
        ax.plot(orbit_times, acc, 'k-', lw=1.5, label='Accelerometer', zorder=5)

    if results['nrlmsise_eff_density'] is not None:
        nrl = np.array(results['nrlmsise_eff_density'][:n_orbits])
        ax.plot(orbit_times, nrl, 'g-', lw=1.0, label='NRLMSISE-00', alpha=0.8)

    if results['jb08_eff_density'] is not None:
        jb08 = np.array(results['jb08_eff_density'][:n_orbits])
        ax.plot(orbit_times, jb08, 'm-', lw=1.0, label='JB08', alpha=0.8)

    ax.set_ylabel('Density [kg/m$^3$]')
    ax.legend(loc='upper right')
    ax.set_title('Empirical Models')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-13, -10))
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Log ratios ----
    ax = axes[2]
    if results['acc_eff_density'] is not None:
        acc_arr = np.array(results['acc_eff_density'][:n_orbits])

        def _plot_ratio(data, times, label, color, marker='o'):
            data = np.array(data[:len(times)])
            valid = (np.isfinite(data) & np.isfinite(acc_arr[:len(data)]) &
                     (data > 0) & (acc_arr[:len(data)] > 0))
            if valid.sum() > 0:
                ratio = np.full_like(data, np.nan)
                ratio[valid] = np.log(data[valid] / acc_arr[:len(data)][valid])
                t = np.array(times)
                ax.plot(t, ratio, marker=marker, ms=2, lw=0.8,
                        color=color, label=label, alpha=0.8)

        if results['edr_density'] is not None:
            _plot_ratio(results['edr_density'], results['edr_times'][:n_orbits],
                        'EDR', 'red')
        if results['pod_eff_density'] is not None:
            _plot_ratio(results['pod_eff_density'], orbit_times,
                        'POD-accel', 'blue')
        if results.get('nrlmsise_eff_density') is not None:
            _plot_ratio(results['nrlmsise_eff_density'], orbit_times,
                        'NRLMSISE-00', 'olive')
        if results.get('jb08_eff_density') is not None:
            _plot_ratio(results['jb08_eff_density'], orbit_times,
                        'JB08', 'purple')

    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('ln(model / ACC)')
    ax.set_xlabel('UTC')
    ax.legend(loc='upper right')
    ax.set_title('Log-Ratio Residuals')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        a.tick_params(axis='x', rotation=30)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_all_methods_timeseries(results, title=None, save_path=None):
    """
    Single-panel overlay of ALL density time series (ACC, EDR, POD, NRLMSISE-00, JB08).
    """
    orbit_times = results['orbit_mid_times']
    n_orbits = len(orbit_times)

    fig, ax = plt.subplots(figsize=(14, 6))

    if results['acc_eff_density'] is not None:
        acc = np.array(results['acc_eff_density'][:n_orbits])
        ax.plot(orbit_times, acc, 'k-', lw=2.0, label='Accelerometer (truth)', zorder=10)

    if results['edr_density'] is not None:
        edr = np.array(results['edr_density'][:n_orbits])
        edr_t = results['edr_times'][:n_orbits]
        ax.plot(edr_t, edr, 'r-', lw=1.2, label='EDR', alpha=0.85)

    if results['pod_eff_density'] is not None:
        pod = np.array(results['pod_eff_density'][:n_orbits])
        ax.plot(orbit_times, pod, 'b-', lw=1.2, label='POD-accel', alpha=0.85)

    if results.get('nrlmsise_eff_density') is not None:
        nrl = np.array(results['nrlmsise_eff_density'][:n_orbits])
        ax.plot(orbit_times, nrl, 'g--', lw=1.0, label='NRLMSISE-00', alpha=0.7)

    if results.get('jb08_eff_density') is not None:
        jb08 = np.array(results['jb08_eff_density'][:n_orbits])
        ax.plot(orbit_times, jb08, 'm--', lw=1.0, label='JB08', alpha=0.7)

    ax.set_ylabel('Effective Density [kg/m$^3$]')
    ax.set_xlabel('UTC')
    ax.set_title(title or 'Storm-Time Density: All Methods')
    ax.legend(loc='upper right', fontsize=9)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-13, -10))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


# ====================================================================
# CLI entry point
# ====================================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <density_csv> [satellite_name]")
        sys.exit(1)
    csv = sys.argv[1]
    sat = sys.argv[2] if len(sys.argv) > 2 else 'GRACE-FO'

    results = analyze_storm(csv, sat_name=sat)
    plot_storm_comparison(results, title=f'{sat} Storm Density Comparison')
    plot_all_methods_timeseries(results, title=f'{sat} Storm: All Methods')
