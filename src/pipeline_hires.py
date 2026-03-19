"""
Sub-orbital resolution pipeline.

Runs both POD window sweep and EDR arc-length sweep, compares to TU Delft
at each resolution, finds optimal windows, and produces trade-off plots
and comparison time series.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from tqdm.auto import tqdm

from .config import get_satellite
from .data_loaders import load_ephemeris, load_tudelft_density
from .pod_accelerometry import pod_density_from_positions
from .pod_accelerometry_hires import (
    interpolate_tudelft_to_grid,
    sweep_pod_windows,
    sweep_pod_effective,
    find_optimal_window,
    compute_mean_drag_acceleration,
    debias_density,
)
from .edr_hires import (
    precompute_forces,
    sweep_edr_arcs,
    find_optimal_arc,
    suborbit_edr_density,
)
from .metrics import log_metrics


def analyze_storm_hires(ephem_path, tudelft_path, sat_name='GRACE-FO-A',
                        storm_start=None, storm_end=None,
                        pod_windows=None, edr_arcs=None,
                        output_dir='output_hires'):
    """
    Full sub-orbital resolution analysis for one storm.

    Parameters
    ----------
    ephem_path   : str
    tudelft_path : str
    sat_name     : str
    storm_start  : datetime or None
    storm_end    : datetime or None
    pod_windows  : list of float — POD SG windows in minutes (None = default)
    edr_arcs     : list of float — EDR arc lengths in minutes (None = default)
    output_dir   : str

    Returns
    -------
    results : dict with sweep results, optimal windows, and metadata
    """
    sat = get_satellite(sat_name)

    # ---- Load ephemeris ----
    print(f"Loading ephemeris: {ephem_path}")
    pos_vel_eci, times = load_ephemeris(ephem_path)
    print(f"  {len(times)} points, {times[0]} → {times[-1]}")

    if storm_start or storm_end:
        mask = np.ones(len(times), dtype=bool)
        if storm_start:
            mask &= np.array([t >= storm_start for t in times])
        if storm_end:
            mask &= np.array([t <= storm_end for t in times])
        pos_vel_eci = pos_vel_eci[mask]
        times = [t for t, m in zip(times, mask) if m]
        print(f"  Trimmed to {len(times)} points, {times[0]} → {times[-1]}")

    # ---- Load TU Delft ----
    print(f"Loading TU Delft density: {tudelft_path}")
    td_times, td_density = load_tudelft_density(
        tudelft_path,
        start_date=times[0] - timedelta(hours=1),
        end_date=times[-1] + timedelta(hours=1),
    )
    print(f"  {len(td_times)} density points loaded")

    # ---- POD pipeline (run once with no smoothing to get raw density) ----
    print("\n=== POD-Accelerometry (raw, no smoothing) ===")
    pod_result = pod_density_from_positions(
        pos_vel_eci, times, sat_name=sat_name,
        output_cadence=15, density_smooth_min=0,
    )
    rho_raw = pod_result['rho_raw']
    pod_times = pod_result['times_out']
    pod_pv = pod_result['pos_vel_out']

    # NaN out raw density edges (spline + differentiation boundary artifacts)
    edge_buf = 8  # ~2 min at 15s cadence
    rho_raw[:edge_buf] = np.nan
    rho_raw[-edge_buf:] = np.nan

    # Interpolate TU Delft onto POD 15s grid
    print("Interpolating TU Delft to POD grid...")
    td_on_pod = interpolate_tudelft_to_grid(td_times, td_density, pod_times)
    n_matched = np.isfinite(td_on_pod).sum()
    print(f"  Matched {n_matched}/{len(pod_times)} POD points to TU Delft")

    # ---- POD window sweep ----
    print("\n=== POD Window Sweep ===")
    pod_sweep = sweep_pod_windows(
        rho_raw, pod_times, td_on_pod, pod_pv,
        output_cadence=15, windows_min=pod_windows,
    )
    for res in pod_sweep:
        print(f"  Window={res['window_min']:5.0f} min: "
              f"r²_matched={res['r']**2:.3f}, r²_native={res['r_native']**2:.3f}, "
              f"EV={res['expl_var_pct']:.1f}%, neg={res['neg_frac']:.1f}%")

    pod_optimal = find_optimal_window(pod_sweep)
    if pod_optimal:
        print(f"  → Optimal POD SG window: {pod_optimal['window_min']} min "
              f"(r²_native={pod_optimal['r_native']**2:.3f}, "
              f"EV={pod_optimal['expl_var_pct']:.1f}%)")
    else:
        print("  → No POD SG window meets thresholds")

    # ---- POD effective density sweep (Picone-weighted integration) ----
    print("\n=== POD Effective Density Sweep ===")
    _, _, l_drag = pod_result['drag_hcl']
    pod_eff_sweep = sweep_pod_effective(
        l_drag, pod_pv, pod_times, td_on_pod, sat_name,
        output_cadence=15, windows_min=pod_windows,
    )
    for res in pod_eff_sweep:
        print(f"  Window={res['window_min']:5.0f} min: "
              f"r²_matched={res['r']**2:.3f}, r²_native={res['r_native']**2:.3f}, "
              f"EV={res['expl_var_pct']:.1f}%, neg={res['neg_frac']:.1f}%")

    pod_eff_optimal = find_optimal_window(pod_eff_sweep)
    if pod_eff_optimal:
        print(f"  → Optimal POD effective window: {pod_eff_optimal['window_min']} min "
              f"(r²_native={pod_eff_optimal['r_native']**2:.3f}, "
              f"EV={pod_eff_optimal['expl_var_pct']:.1f}%)")
    else:
        print("  → No POD effective window meets thresholds")

    # ---- EDR force precomputation (on SP3 30s grid) ----
    print("\n=== EDR Force Pre-computation ===")
    forces = precompute_forces(pos_vel_eci, times, sat_name)

    # Interpolate TU Delft onto SP3 30s grid
    print("Interpolating TU Delft to SP3 grid...")
    td_on_sp3 = interpolate_tudelft_to_grid(td_times, td_density, times)
    n_matched_sp3 = np.isfinite(td_on_sp3).sum()
    print(f"  Matched {n_matched_sp3}/{len(times)} SP3 points to TU Delft")

    # ---- EDR arc sweep ----
    print("\n=== EDR Arc-Length Sweep ===")
    edr_sweep = sweep_edr_arcs(
        pos_vel_eci, times, forces, sat_name,
        td_on_sp3, arc_minutes_list=edr_arcs,
    )
    for res in edr_sweep:
        print(f"  Arc={res['arc_minutes']:5.0f} min: "
              f"r²_matched={res['r']**2:.3f}, r²_native={res['r_native']**2:.3f}, "
              f"EV={res['expl_var_pct']:.1f}%, neg={res['neg_frac']:.1f}%")

    edr_optimal = find_optimal_arc(edr_sweep)
    if edr_optimal:
        print(f"  → Optimal EDR arc: {edr_optimal['arc_minutes']} min "
              f"(r²_native={edr_optimal['r_native']**2:.3f}, "
              f"EV={edr_optimal['expl_var_pct']:.1f}%)")
    else:
        print("  → No EDR arc meets thresholds")

    # ---- Mean drag acceleration ----
    mean_drag = compute_mean_drag_acceleration(pod_result)
    print(f"\nMean |a_drag| = {mean_drag:.2e} m/s²")

    # ---- Save plots ----
    os.makedirs(output_dir, exist_ok=True)

    plot_tradeoff_curves(pod_eff_sweep, edr_sweep, pod_eff_optimal, edr_optimal,
                         sat_name, output_dir)

    plot_best_resolution_comparison(
        pod_eff_sweep, pod_eff_optimal, pod_times,
        edr_sweep, edr_optimal, times,
        td_on_pod, td_on_sp3,
        sat_name, output_dir,
    )

    # ---- Save CSV summary ----
    save_sweep_csv(pod_eff_sweep, edr_sweep, sat_name, mean_drag, output_dir)

    results = {
        'sat_name': sat_name,
        'pod_result': pod_result,
        'pod_sweep': pod_sweep,
        'pod_optimal': pod_optimal,
        'pod_eff_sweep': pod_eff_sweep,
        'pod_eff_optimal': pod_eff_optimal,
        'edr_sweep': edr_sweep,
        'edr_optimal': edr_optimal,
        'forces': forces,
        'mean_drag_acc': mean_drag,
        'pod_times': pod_times,
        'sp3_times': times,
        'td_on_pod': td_on_pod,
        'td_on_sp3': td_on_sp3,
    }
    return results


# ====================================================================
# Plotting
# ====================================================================
def plot_tradeoff_curves(pod_sweep, edr_sweep, pod_opt, edr_opt,
                         sat_name, output_dir):
    """
    Plot resolution trade-off with both matched and native-truth metrics.

    Left panel: r against native (unaveraged) TU Delft — reveals the genuine
    optimal resolution where information content peaks.
    Right panel: explained variance against native truth.

    Also shows r_matched as faded lines for reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pod_w = [r['window_min'] for r in pod_sweep]
    edr_w = [r['arc_minutes'] for r in edr_sweep]

    # ---- r²_native vs window (primary metric) ----
    ax = axes[0]
    pod_r2_nat = [r['r_native']**2 for r in pod_sweep]
    edr_r2_nat = [r['r_native']**2 for r in edr_sweep]
    pod_r2_mat = [r['r']**2 for r in pod_sweep]
    edr_r2_mat = [r['r']**2 for r in edr_sweep]

    # Faded matched-truth lines for reference
    ax.plot(pod_w, pod_r2_mat, 'b--', lw=0.8, ms=3, alpha=0.3)
    ax.plot(edr_w, edr_r2_mat, 'r--', lw=0.8, ms=3, alpha=0.3)

    # Native-truth lines (primary)
    ax.plot(pod_w, pod_r2_nat, 'bo-', lw=1.5, ms=6, label='POD')
    ax.plot(edr_w, edr_r2_nat, 'rs-', lw=1.5, ms=6, label='EDR')

    if pod_opt is not None:
        r2_pod = pod_opt['r_native']**2
        ax.plot(pod_opt['window_min'], r2_pod, 'bo', ms=14,
                mfc='none', mew=2.5, zorder=5)
        ax.annotate(f"{pod_opt['window_min']:.0f} min\nr\u00b2={r2_pod:.3f}",
                    xy=(pod_opt['window_min'], r2_pod),
                    xytext=(15, -20), textcoords='offset points',
                    fontsize=8, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=0.8))
    if edr_opt is not None:
        r2_edr = edr_opt['r_native']**2
        ax.plot(edr_opt['arc_minutes'], r2_edr, 'rs', ms=14,
                mfc='none', mew=2.5, zorder=5)
        ax.annotate(f"{edr_opt['arc_minutes']:.0f} min\nr\u00b2={r2_edr:.3f}",
                    xy=(edr_opt['arc_minutes'], r2_edr),
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=8, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    ax.set_xlabel('Window / Arc length [min]')
    ax.set_ylabel('Pearson r\u00b2 vs native TU Delft')
    ax.set_xscale('log')
    ax.set_title('Correlation vs Native Truth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- Explained variance vs window ----
    ax = axes[1]
    pod_ev = [r['expl_var_pct'] for r in pod_sweep]
    edr_ev = [r['expl_var_pct'] for r in edr_sweep]

    ax.plot(pod_w, pod_ev, 'bo-', lw=1.5, ms=6, label='POD')
    ax.plot(edr_w, edr_ev, 'rs-', lw=1.5, ms=6, label='EDR')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)

    if pod_opt is not None:
        ax.plot(pod_opt['window_min'], pod_opt['expl_var_pct'], 'bo', ms=14,
                mfc='none', mew=2.5, zorder=5)
    if edr_opt is not None:
        ax.plot(edr_opt['arc_minutes'], edr_opt['expl_var_pct'], 'rs', ms=14,
                mfc='none', mew=2.5, zorder=5)

    ax.set_xlabel('Window / Arc length [min]')
    ax.set_ylabel('Explained variance [%]')
    ax.set_xscale('log')
    ax.set_title('Information Content vs Resolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{sat_name} — Resolution Trade-off (vs native truth)',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'hires_tradeoff.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_best_resolution_comparison(pod_sweep, pod_opt, pod_times,
                                    edr_sweep, edr_opt, sp3_times,
                                    td_on_pod, td_on_sp3,
                                    sat_name, output_dir):
    """
    Time series at optimal resolution vs TU Delft.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    t_pod_dt = pod_times
    t_sp3_dt = sp3_times

    # Panel 1: density time series
    ax = axes[0]

    # TU Delft at native cadence (on POD grid)
    ax.plot(t_pod_dt, td_on_pod, color='gray', lw=0.3, alpha=0.5,
            label='TU Delft (10s)')

    if pod_opt is not None:
        ax.plot(t_pod_dt, pod_opt['td_avg'], 'k-', lw=1.0,
                label=f'TU Delft ({pod_opt["window_min"]:.0f}-min avg)')
        ax.plot(t_pod_dt, pod_opt['rho'], 'b-', lw=0.8, alpha=0.8,
                label=f'POD ({pod_opt["window_min"]:.0f} min, '
                      f'r\u00b2={pod_opt["r_native"]**2:.3f})')

    if edr_opt is not None:
        ax.plot(t_sp3_dt, edr_opt['rho'], 'r-', lw=0.8, alpha=0.8,
                label=f'EDR ({edr_opt["arc_minutes"]:.0f} min, '
                      f'r\u00b2={edr_opt["r_native"]**2:.3f})')

    # Also show orbit-effective for reference (90-min POD)
    orb_eff = next((r for r in pod_sweep if r['window_min'] == 90), None)
    if orb_eff is not None:
        ax.plot(t_pod_dt, orb_eff['rho'], 'b--', lw=0.6, alpha=0.4,
                label='POD (90 min)')

    ax.set_ylabel('Density [kg/m$^3$]')
    ax.set_title(f'{sat_name} — Best-Resolution Density Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-13, -10))
    ax.grid(True, alpha=0.3)

    # Panel 2: log-ratio residuals at optimal resolution
    ax = axes[1]

    if pod_opt is not None:
        rho = pod_opt['rho']
        td = pod_opt['td_avg']
        valid = np.isfinite(rho) & np.isfinite(td) & (rho > 0) & (td > 0)
        ratio = np.full(len(rho), np.nan)
        ratio[valid] = np.log(rho[valid] / td[valid])
        ax.plot(t_pod_dt, ratio, 'b-', lw=0.5, alpha=0.8,
                label=f'POD ({pod_opt["window_min"]:.0f} min)')

    if edr_opt is not None:
        rho = edr_opt['rho']
        td = edr_opt['td_avg']
        valid = np.isfinite(rho) & np.isfinite(td) & (rho > 0) & (td > 0)
        ratio = np.full(len(rho), np.nan)
        ratio[valid] = np.log(rho[valid] / td[valid])
        ax.plot(t_sp3_dt, ratio, 'r-', lw=0.5, alpha=0.8,
                label=f'EDR ({edr_opt["arc_minutes"]:.0f} min)')

    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('ln(method / truth)')
    ax.set_xlabel('UTC')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        a.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = os.path.join(output_dir, 'hires_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def save_sweep_csv(pod_sweep, edr_sweep, sat_name, mean_drag, output_dir):
    """Save sweep results to CSV."""
    rows = []
    for r in pod_sweep:
        rows.append({
            'method': 'POD',
            'window_min': r['window_min'],
            'r_matched': r['r'],
            'r_squared_matched': r.get('r_squared', r['r']**2 if np.isfinite(r['r']) else np.nan),
            'SD_pct_matched': r['SD_pct'],
            'r_native': r['r_native'],
            'r_squared_native': r.get('r_squared_native', r['r_native']**2 if np.isfinite(r['r_native']) else np.nan),
            'SD_pct_native': r['SD_pct_native'],
            'expl_var_pct': r['expl_var_pct'],
            'sigma': r['sigma'],
            'beta': r['beta'],
            'beta_debias': r.get('beta_debias', np.nan),
            'neg_frac': r['neg_frac'],
            'n_valid': r['n_valid'],
        })
    for r in edr_sweep:
        rows.append({
            'method': 'EDR',
            'window_min': r['arc_minutes'],
            'r_matched': r['r'],
            'r_squared_matched': r.get('r_squared', r['r']**2 if np.isfinite(r['r']) else np.nan),
            'SD_pct_matched': r['SD_pct'],
            'r_native': r['r_native'],
            'r_squared_native': r.get('r_squared_native', r['r_native']**2 if np.isfinite(r['r_native']) else np.nan),
            'SD_pct_native': r['SD_pct_native'],
            'expl_var_pct': r['expl_var_pct'],
            'sigma': r['sigma'],
            'beta': r['beta'],
            'beta_debias': r.get('beta_debias', np.nan),
            'neg_frac': r['neg_frac'],
            'n_valid': r['n_valid'],
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'sweep_results.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
