"""
High-resolution POD-accelerometry density with variable smoothing.

Reuses the existing POD pipeline (which produces raw 15s density) and applies
variable-width SG filters to explore the resolution/accuracy trade-off.

The 90-min SG smooth in the standard pipeline kills sub-orbital structure.
Here we sweep shorter windows to find the minimum viable smoothing that still
gives acceptable agreement with TU Delft accelerometer truth.
"""

import numpy as np
from scipy.signal import savgol_filter
from .pod_accelerometry import pod_density_from_positions, flag_outliers
from .config import OMEGA_VEC, ballistic_coeff


def debias_density(rho, truth):
    """
    Median log-ratio de-biasing (robust to outliers).

    Parameters
    ----------
    rho   : (N,) ndarray — model density
    truth : (N,) ndarray — reference density (same grid)

    Returns
    -------
    rho_db : (N,) ndarray — de-biased density
    beta   : float — median log bias that was removed
    """
    valid = (np.isfinite(rho) & np.isfinite(truth)
             & (rho > 0) & (truth > 0))
    if valid.sum() < 10:
        return rho.copy(), 0.0

    beta = np.median(np.log(rho[valid] / truth[valid]))
    return rho * np.exp(-beta), beta


def smooth_density(rho_raw, window_min, output_cadence=15, sg_order=3):
    """
    Apply SG smoothing to raw POD density at a given window width.

    NaN-fills half-window at both edges to suppress SG boundary artifacts.

    Parameters
    ----------
    rho_raw        : (N,) ndarray — raw density at output_cadence [kg/m³]
    window_min     : float — SG window width in minutes (0 = no smoothing)
    output_cadence : float — cadence in seconds
    sg_order       : int — SG polynomial order

    Returns
    -------
    rho : (N,) ndarray — smoothed density (NaN at edges)
    """
    if window_min <= 0:
        return rho_raw.copy()

    win = int(window_min * 60 / output_cadence)
    if win % 2 == 0:
        win += 1
    win = max(win, sg_order + 2)
    if win % 2 == 0:
        win += 1

    rho = savgol_filter(rho_raw, win, min(sg_order, win - 1))

    # NaN-fill edges where SG polynomial extrapolation is unreliable
    edge = win // 2
    rho[:edge] = np.nan
    rho[-edge:] = np.nan
    return rho


def moving_average_tudelft(td_density_on_grid, window_min, output_cadence=15):
    """
    Apply a uniform moving average to TU Delft density for fair comparison
    at a given resolution.

    Parameters
    ----------
    td_density_on_grid : (N,) ndarray — TU Delft density interpolated to POD grid
    window_min         : float — averaging window in minutes
    output_cadence     : float — cadence in seconds

    Returns
    -------
    td_avg : (N,) ndarray — moving-averaged density (NaN where insufficient data)
    """
    if window_min <= 0:
        return td_density_on_grid.copy()

    half_win = int(window_min * 60 / output_cadence / 2)
    N = len(td_density_on_grid)
    td_avg = np.full(N, np.nan)

    for i in range(N):
        i0 = max(0, i - half_win)
        i1 = min(N, i + half_win + 1)
        chunk = td_density_on_grid[i0:i1]
        valid = np.isfinite(chunk) & (chunk > 0)
        if valid.sum() >= 0.5 * len(chunk):
            td_avg[i] = np.nanmean(chunk[valid])

    return td_avg


def interpolate_tudelft_to_grid(td_times, td_density, grid_times, tolerance_s=15.0):
    """
    Nearest-neighbor interpolation of TU Delft density to an arbitrary time grid.

    Parameters
    ----------
    td_times    : list of datetime — TU Delft timestamps
    td_density  : (M,) ndarray — TU Delft density
    grid_times  : list of datetime — target timestamps
    tolerance_s : float — max allowed time difference in seconds

    Returns
    -------
    density_on_grid : (N,) ndarray — interpolated density (NaN where no match)
    """
    N = len(grid_times)
    result = np.full(N, np.nan)

    if len(td_times) == 0:
        return result

    t0_ref = td_times[0]
    td_sec = np.array([(t - t0_ref).total_seconds() for t in td_times])
    grid_sec = np.array([(t - t0_ref).total_seconds() for t in grid_times])

    idx = np.searchsorted(td_sec, grid_sec, side='left')
    for i in range(N):
        best_diff = 999.0
        best_j = -1
        for j in [idx[i] - 1, idx[i]]:
            if 0 <= j < len(td_sec):
                diff = abs(grid_sec[i] - td_sec[j])
                if diff < best_diff:
                    best_diff = diff
                    best_j = j
        if best_diff <= tolerance_s:
            result[i] = td_density[best_j]

    return result


def sweep_pod_windows(rho_raw, pod_times, td_on_pod_grid, pos_vel_out,
                      output_cadence=15,
                      windows_min=None):
    """
    Sweep SG smoothing windows and compute metrics at each resolution.

    Computes two sets of metrics:
      - r, SD%: against resolution-matched (averaged) TU Delft
      - r_native, EV: against native (unaveraged) TU Delft

    Parameters
    ----------
    rho_raw        : (N,) ndarray — raw POD density at 15s
    pod_times      : list of datetime — POD output timestamps
    td_on_pod_grid : (N,) ndarray — TU Delft density on POD grid (native)
    pos_vel_out    : (N, 6) ndarray — ECI states on POD grid
    output_cadence : float — cadence in seconds
    windows_min    : list of float — windows to sweep (minutes)

    Returns
    -------
    results : list of dict, one per window
    """
    from .metrics import log_metrics

    if windows_min is None:
        windows_min = [2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 75, 90, 100, 120]

    results = []

    for wm in windows_min:
        rho = smooth_density(rho_raw, wm, output_cadence)
        neg_frac = 100.0 * np.nansum(rho < 0) / np.sum(np.isfinite(rho))

        # De-bias against native truth using median log-ratio
        rho, beta_db = debias_density(rho, td_on_pod_grid)

        # Matched-truth metrics
        td_avg = moving_average_tudelft(td_on_pod_grid, wm, output_cadence)
        valid = (np.isfinite(rho) & np.isfinite(td_avg)
                 & (rho > 0) & (td_avg > 0))
        n_valid = valid.sum()
        if n_valid >= 10:
            m = log_metrics(td_avg[valid], rho[valid])
        else:
            m = {'r': np.nan, 'r_squared': np.nan, 'SD_pct': np.nan, 'sigma': np.nan, 'beta': np.nan}

        # Native-truth metrics
        v_nat = (np.isfinite(rho) & np.isfinite(td_on_pod_grid)
                 & (rho > 0) & (td_on_pod_grid > 0))
        if v_nat.sum() >= 10:
            m_nat = log_metrics(td_on_pod_grid[v_nat], rho[v_nat])
            resid = rho[v_nat] - td_on_pod_grid[v_nat]
            expl_var = 100.0 * (1.0 - np.var(resid) / np.var(td_on_pod_grid[v_nat]))
        else:
            m_nat = {'r': np.nan, 'r_squared': np.nan, 'SD_pct': np.nan}
            expl_var = np.nan

        results.append({
            'window_min': wm,
            'rho': rho,
            'td_avg': td_avg,
            'r': m['r'],
            'r_squared': m['r_squared'],
            'SD_pct': m['SD_pct'],
            'sigma': m['sigma'],
            'beta': m.get('beta', np.nan),
            'beta_debias': beta_db,
            'neg_frac': neg_frac,
            'n_valid': n_valid,
            'r_native': m_nat['r'],
            'r_squared_native': m_nat['r_squared'],
            'SD_pct_native': m_nat['SD_pct'],
            'expl_var_pct': expl_var,
        })

    return results


def pod_effective_density(l_drag, pos_vel_out, times_out, sat_name, window_min,
                          output_cadence=15):
    """
    Compute POD effective density using Picone velocity-weighted integration.

    Uses the same weighting as EDR:
        rho_eff = (2/B) * trapz(l_drag * |v|, dt) / trapz(v_rel^2 * |v|, dt)

    This makes POD and EDR directly comparable — both produce the same
    physical quantity (Picone orbit-average density) from different drag
    estimates (differentiated acceleration vs energy balance).

    Parameters
    ----------
    l_drag         : (N,) ndarray — along-track drag acceleration [m/s²]
    pos_vel_out    : (N, 6) ndarray — ECI states at output cadence [m, m/s]
    times_out      : list of datetime (length N)
    sat_name       : str
    window_min     : float — integration window in minutes
    output_cadence : float — cadence in seconds

    Returns
    -------
    rho : (N,) ndarray — effective density [kg/m³] (NaN at edges)
    """
    B = ballistic_coeff(sat_name)
    N = len(l_drag)

    # Relative velocity (co-rotating atmosphere, no winds)
    v_rel_vec = pos_vel_out[:, 3:] - np.cross(OMEGA_VEC, pos_vel_out[:, :3])
    v_rel_mag = np.linalg.norm(v_rel_vec, axis=1)
    speeds = np.linalg.norm(pos_vel_out[:, 3:], axis=1)

    # Integrands
    numer_integrand = l_drag * speeds            # drag power per unit mass
    denom_integrand = v_rel_mag**2 * speeds      # Picone weighting

    # Sliding window (same structure as suborbit_edr_density)
    half_win_idx = int(round(window_min * 60 / (2 * output_cadence)))

    # Time array in seconds from start
    t0 = times_out[0]
    dt_sec = np.array([(t - t0).total_seconds() for t in times_out])

    rho = np.full(N, np.nan)

    for i in range(N):
        i0 = i - half_win_idx
        i1 = i + half_win_idx
        if i0 < 0 or i1 >= N:
            continue

        t_arc = dt_sec[i0:i1 + 1] - dt_sec[i0]
        if t_arc[-1] <= 0:
            continue

        numer = np.trapz(numer_integrand[i0:i1 + 1], t_arc)
        denom = np.trapz(denom_integrand[i0:i1 + 1], t_arc)

        if denom == 0:
            continue

        rho[i] = 2.0 * numer / (B * denom)

    return rho


def multiorbit_pod_density(l_drag, pos_vel_out, times_out, sat_name,
                           perigee_indices, n_orbits=1):
    """
    Compute POD effective density for consecutive N-orbit spans.

    Integrates the Picone-weighted drag power from perigee_i to
    perigee_{i+n_orbits}:

        rho_eff = (2/B) * trapz(l_drag * |v|, dt) / trapz(v_rel^2 * |v|, dt)

    This is the genuine multi-orbit POD density — the drag acceleration is
    integrated over the full N-orbit span before density inversion, not
    single-orbit densities averaged post-hoc.

    Parameters
    ----------
    l_drag          : (N,) ndarray — along-track drag acceleration [m/s²]
    pos_vel_out     : (N, 6) ndarray — ECI states
    times_out       : list of datetime (length N)
    sat_name        : str
    perigee_indices : 1-d int array — perigee passage indices
    n_orbits        : int — number of orbits per span (1, 2, or 3)

    Returns
    -------
    rho_eff : ndarray of float — one density per N-orbit span
    """
    B = ballistic_coeff(sat_name)

    # Relative velocity (co-rotating atmosphere)
    v_rel_vec = pos_vel_out[:, 3:] - np.cross(OMEGA_VEC, pos_vel_out[:, :3])
    v_rel_mag = np.linalg.norm(v_rel_vec, axis=1)
    speeds = np.linalg.norm(pos_vel_out[:, 3:], axis=1)

    # Integrands
    numer_integrand = l_drag * speeds            # drag power per unit mass
    denom_integrand = v_rel_mag**2 * speeds      # Picone weighting

    # Time array in seconds
    t0 = times_out[0]
    dt_sec = np.array([(t - t0).total_seconds() for t in times_out])

    # Non-overlapping N-orbit spans
    peri_N = perigee_indices[::n_orbits]

    rho_eff = np.full(len(peri_N) - 1, np.nan)
    for k in range(len(peri_N) - 1):
        i0 = peri_N[k]
        i1 = peri_N[k + 1]

        t_arc = dt_sec[i0:i1 + 1] - dt_sec[i0]
        if t_arc[-1] <= 0:
            continue

        numer = np.trapz(numer_integrand[i0:i1 + 1], t_arc)
        denom = np.trapz(denom_integrand[i0:i1 + 1], t_arc)
        if denom == 0:
            continue

        rho_eff[k] = 2.0 * numer / (B * denom)

    return rho_eff


def sweep_pod_effective(l_drag, pos_vel_out, times_out, td_on_pod_grid,
                        sat_name, output_cadence=15, windows_min=None):
    """
    Sweep integration windows for POD effective density and compute metrics.

    Drop-in replacement for sweep_pod_windows() in sweep CSVs —
    same output dict keys, same metric computation.

    Parameters
    ----------
    l_drag         : (N,) ndarray — along-track drag acceleration [m/s²]
    pos_vel_out    : (N, 6) ndarray — ECI states at output cadence [m, m/s]
    times_out      : list of datetime (length N)
    td_on_pod_grid : (N,) ndarray — TU Delft density on POD grid (native)
    sat_name       : str
    output_cadence : float — cadence in seconds
    windows_min    : list of float — windows to sweep (minutes)

    Returns
    -------
    results : list of dict, one per window (same format as sweep_pod_windows)
    """
    from .metrics import log_metrics

    if windows_min is None:
        windows_min = [2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 75, 90, 100, 120]

    results = []

    for wm in windows_min:
        print(f"  POD effective window = {wm} min...")
        rho = pod_effective_density(l_drag, pos_vel_out, times_out, sat_name,
                                    wm, output_cadence)

        finite_mask = np.isfinite(rho)
        if finite_mask.sum() == 0:
            neg_frac = 0.0
        else:
            neg_frac = 100.0 * np.nansum(rho < 0) / finite_mask.sum()

        # De-bias against native truth using median log-ratio
        rho, beta_db = debias_density(rho, td_on_pod_grid)

        # Matched-truth metrics (resolution-matched averaging)
        td_avg = moving_average_tudelft(td_on_pod_grid, wm, output_cadence)
        valid = (np.isfinite(rho) & np.isfinite(td_avg)
                 & (rho > 0) & (td_avg > 0))
        n_valid = valid.sum()
        if n_valid >= 10:
            m = log_metrics(td_avg[valid], rho[valid])
        else:
            m = {'r': np.nan, 'r_squared': np.nan, 'SD_pct': np.nan, 'sigma': np.nan, 'beta': np.nan}

        # Native-truth metrics (against unaveraged TU Delft)
        v_nat = (np.isfinite(rho) & np.isfinite(td_on_pod_grid)
                 & (rho > 0) & (td_on_pod_grid > 0))
        if v_nat.sum() >= 10:
            m_nat = log_metrics(td_on_pod_grid[v_nat], rho[v_nat])
            resid = rho[v_nat] - td_on_pod_grid[v_nat]
            expl_var = 100.0 * (1.0 - np.var(resid) / np.var(td_on_pod_grid[v_nat]))
        else:
            m_nat = {'r': np.nan, 'r_squared': np.nan, 'SD_pct': np.nan}
            expl_var = np.nan

        results.append({
            'window_min': wm,
            'rho': rho,
            'td_avg': td_avg,
            'r': m['r'],
            'r_squared': m['r_squared'],
            'SD_pct': m['SD_pct'],
            'sigma': m['sigma'],
            'beta': m.get('beta', np.nan),
            'beta_debias': beta_db,
            'neg_frac': neg_frac,
            'n_valid': n_valid,
            'r_native': m_nat['r'],
            'r_squared_native': m_nat['r_squared'],
            'SD_pct_native': m_nat['SD_pct'],
            'expl_var_pct': expl_var,
        })

    return results


def find_optimal_window(sweep_results, neg_threshold=10.0):
    """
    Find the window that maximizes r against native (unaveraged) truth.

    This finds the genuine optimal resolution — the window that captures
    the most information about the real density field.

    Parameters
    ----------
    sweep_results : list of dict from sweep_pod_windows()
    neg_threshold : maximum % negative densities

    Returns
    -------
    optimal : dict or None
    """
    valid = [r for r in sweep_results
             if np.isfinite(r.get('r_native', np.nan))
             and r['neg_frac'] <= neg_threshold]
    if not valid:
        return None

    return max(valid, key=lambda r: r['r_native'])


def compute_mean_drag_acceleration(pod_result):
    """
    Compute mean along-track drag acceleration magnitude from POD result.

    Parameters
    ----------
    pod_result : dict from pod_density_from_positions()

    Returns
    -------
    float — mean |a_drag_L| in m/s²
    """
    _, _, l_drag = pod_result['drag_hcl']
    return np.nanmean(np.abs(l_drag))


