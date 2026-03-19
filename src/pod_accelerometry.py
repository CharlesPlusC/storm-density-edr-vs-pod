"""
POD-accelerometry density inversion from SP3 positions/velocities.

Steps:
  1. Cubic-spline interpolate SP3 velocities to 0.01s cadence
  2. Savitzky-Golay smooth velocities (window=21, order=7 at 0.01s = 0.21s)
  3. Differentiate via np.gradient at 0.01s → acceleration
  4. Downsample to 15s output cadence
  5. Subtract conservative forces (100x100 gravity + 3BP) and SRP
  6. Residual drag → HCL along-track → invert for density
  7. SG-smooth density at ~90 min to suppress orbit-frequency noise
  8. Outlier flagging → orbit-effective density via Picone weighting

The 0.01s step size is critical: finite-difference truncation error scales as h².
At 0.01s the error is ~1.5e-7 m/s² (comparable to drag); at 0.1s it's 100x larger.
"""

import gc
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

from .config import OMEGA_VEC, get_satellite
from .orekit_utils import (
    compute_gravity_acceleration,
    compute_3bp_acceleration,
    compute_srp_acceleration,
)


# ====================================================================
# Outlier detection — rolling median + MAD
# ====================================================================
def flag_outliers(rho, window=361, k=10.0):
    if window % 2 == 0:
        window += 1

    med = median_filter(rho, size=window, mode='reflect')
    abs_dev = np.abs(rho - med)
    mad = median_filter(abs_dev, size=window, mode='reflect')

    mad_floor = np.nanmedian(abs_dev[abs_dev > 0]) * 0.01 if np.any(abs_dev > 0) else 1e-20
    mad = np.maximum(mad, mad_floor)

    outlier = abs_dev > k * mad
    n_flagged = int(outlier.sum())
    rho[outlier] = np.nan
    return n_flagged


# ====================================================================
# HCL projection (vectorised for arrays)
# ====================================================================
def project_acc_into_HCL(acc, pos, vel):
    if acc.ndim == 1:
        r_hat = pos / np.linalg.norm(pos)
        rxv = np.cross(pos, vel)
        c_hat = rxv / np.linalg.norm(rxv)
        l_hat = np.cross(r_hat, c_hat)
        return np.dot(acc, r_hat), np.dot(acc, c_hat), np.dot(acc, l_hat)

    r_norm = np.linalg.norm(pos, axis=1, keepdims=True)
    r_hat = pos / r_norm
    rxv = np.cross(pos, vel)
    c_hat = rxv / np.linalg.norm(rxv, axis=1, keepdims=True)
    l_hat = np.cross(r_hat, c_hat)
    h = np.sum(acc * r_hat, axis=1)
    c = np.sum(acc * c_hat, axis=1)
    l = np.sum(acc * l_hat, axis=1)
    return h, c, l


# ====================================================================
# Interpolation + SG smoothing + differentiation
# ====================================================================
def interpolate_and_differentiate(pos_vel_eci, times,
                                  interp_dt=0.01,
                                  sg_vel_window=21,
                                  sg_vel_order=7,
                                  output_cadence=15):
    """
    Interpolate SP3 velocities to fine cadence, SG-smooth, differentiate.

    Memory-efficient: processes one component at a time and immediately
    downsamples to output cadence, avoiding multi-GB intermediate arrays.

    Parameters
    ----------
    pos_vel_eci    : (N, 6) ndarray — ECI [m, m/s] at SP3 cadence
    times          : list of datetime (length N)
    interp_dt      : float — interpolation timestep in seconds (default 0.01)
    sg_vel_window  : int — SG window for velocity smoothing in points (default 21)
    sg_vel_order   : int — SG polynomial order for velocity (default 7)
    output_cadence : float — output timestep in seconds (default 15)

    Returns
    -------
    t_out     : 1-d array — seconds from epoch at output cadence
    pos_out   : (M, 3) — interpolated positions [m]
    vel_out   : (M, 3) — smoothed velocities at output cadence [m/s]
    acc_out   : (M, 3) — accelerations at output cadence [m/s²]
    t0        : datetime — reference epoch
    """
    t0 = times[0]
    t_sec = np.array([(t - t0).total_seconds() for t in times])

    t_fine = np.arange(t_sec[0], t_sec[-1], interp_dt)
    M_fine = len(t_fine)
    print(f"  Fine grid: {M_fine} points at {interp_dt}s "
          f"({M_fine * interp_dt / 3600:.1f} hrs)")

    # Output indices (downsample)
    step = max(1, int(output_cadence / interp_dt))
    idx_out = np.arange(0, M_fine, step)
    N_out = len(idx_out)
    t_out = t_fine[idx_out]

    # Process each component: spline → SG smooth → differentiate → downsample
    vel_out = np.empty((N_out, 3))
    acc_out = np.empty((N_out, 3))

    print(f"  SG smoothing velocity: window={sg_vel_window} pts "
          f"({sg_vel_window * interp_dt:.2f}s), order={sg_vel_order}")

    for j in range(3):
        v = CubicSpline(t_sec, pos_vel_eci[:, 3 + j])(t_fine)
        v = savgol_filter(v, sg_vel_window, sg_vel_order)
        vel_out[:, j] = v[idx_out]
        a = np.gradient(v, interp_dt)
        acc_out[:, j] = a[idx_out]
        del v, a

    gc.collect()

    # Positions: interpolate directly to output cadence (no need for 0.01s)
    pos_out = np.empty((N_out, 3))
    for j in range(3):
        pos_out[:, j] = CubicSpline(t_sec, pos_vel_eci[:, j])(t_out)

    print(f"  Output grid: {N_out} points at {output_cadence}s")
    return t_out, pos_out, vel_out, acc_out, t0


# ====================================================================
# Force model computation at output cadence
# ====================================================================
def compute_model_accelerations(pos_vel_out, times_out, sat_name):
    sat = get_satellite(sat_name)
    mass = sat['mass']
    Cr, area = sat['Cr'], sat['area']
    N = len(pos_vel_out)

    a_grav = np.empty((N, 3))
    a_3bp  = np.empty((N, 3))
    a_srp  = np.empty((N, 3))

    for i in tqdm(range(N), desc='Force models (grav+3BP+SRP)'):
        sv = pos_vel_out[i]
        t  = times_out[i]
        a_grav[i] = compute_gravity_acceleration(sv, t, mass)
        a_3bp[i]  = compute_3bp_acceleration(sv, t, mass)
        a_srp[i]  = compute_srp_acceleration(sv, t, Cr, area, mass)

    a_conservative = a_grav + a_3bp
    return a_grav, a_3bp, a_srp, a_conservative


# ====================================================================
# Full POD pipeline
# ====================================================================
def pod_density_from_positions(pos_vel_eci, times, sat_name='GRACE-FO',
                               output_cadence=15,
                               interp_dt=0.01,
                               sg_vel_window=21,
                               sg_vel_order=7,
                               density_smooth_min=90):
    """
    Full POD-accelerometry density inversion from SP3 positions/velocities.

    Parameters
    ----------
    pos_vel_eci       : (N, 6) ndarray — ECI state from SP3 [m, m/s]
    times             : list of datetime (length N)
    sat_name          : str
    output_cadence    : int — output density cadence in seconds
    interp_dt         : float — interpolation timestep (default 0.01s)
    sg_vel_window     : int — SG window for velocity smoothing (default 21)
    sg_vel_order      : int — SG polynomial order for velocity (default 7)
    density_smooth_min : float — density SG smoothing window in minutes (0 = off)

    Returns
    -------
    result : dict with keys:
      'rho', 'rho_raw', 'times_out', 'pos_vel_out',
      'vel_smooth', 'a_obs', 'a_grav', 'a_3bp', 'a_srp', 'a_nc', 'a_drag',
      'nc_hcl', 'drag_hcl'
    """
    from datetime import timedelta

    sat = get_satellite(sat_name)
    Cd, A, M = sat['Cd'], sat['area'], sat['mass']

    # Step 1-2: Interpolate + SG smooth velocity + differentiate
    print("Step 1-2: Interpolate, SG-smooth velocity, differentiate...")
    t_out, pos_out, vel_out, acc_obs_out, t0 = \
        interpolate_and_differentiate(pos_vel_eci, times,
                                      interp_dt=interp_dt,
                                      sg_vel_window=sg_vel_window,
                                      sg_vel_order=sg_vel_order,
                                      output_cadence=output_cadence)

    times_out = [t0 + timedelta(seconds=float(t)) for t in t_out]
    pos_vel_out = np.hstack([pos_out, vel_out])
    N_out = len(times_out)

    # Step 3-4: Compute force models at output cadence
    print("Step 3-4: Computing force models...")
    a_grav, a_3bp, a_srp, a_conservative = \
        compute_model_accelerations(pos_vel_out, times_out, sat_name)

    # Non-conservative residual = observed - conservative
    a_nc = acc_obs_out - a_conservative
    a_drag = a_nc - a_srp

    # Step 5: Project into HCL and invert for density
    print("Step 5: HCL projection and density inversion...")
    h_nc, c_nc, l_nc = project_acc_into_HCL(a_nc, pos_out, vel_out)
    h_drag, c_drag, l_drag = project_acc_into_HCL(a_drag, pos_out, vel_out)

    rho_raw = np.empty(N_out)
    for i in range(N_out):
        v_rel = vel_out[i] - np.cross(OMEGA_VEC, pos_out[i])
        rho_raw[i] = 2.0 * l_drag[i] * M / (Cd * A * np.linalg.norm(v_rel)**2)

    neg_frac = 100.0 * np.sum(rho_raw < 0) / N_out
    print(f"  Raw density: {neg_frac:.1f}% negative")

    # Step 6: Smooth density with SG filter
    if density_smooth_min > 0:
        win = int(density_smooth_min * 60 / output_cadence)
        if win % 2 == 0:
            win += 1
        rho = savgol_filter(rho_raw, win, min(3, win - 1))
        neg_after = 100.0 * np.sum(rho < 0) / N_out
        print(f"Step 6: Density smoothing: SG window={win} pts "
              f"({density_smooth_min:.0f} min), {neg_after:.1f}% negative after")
    else:
        rho = rho_raw.copy()

    # Step 7: Outlier flagging
    n_flagged = flag_outliers(rho, window=361, k=10.0)
    print(f"Step 7: Flagged {n_flagged}/{N_out} outliers "
          f"({100*n_flagged/N_out:.1f}%)")

    return {
        'rho': rho,
        'rho_raw': rho_raw,
        'times_out': times_out,
        'pos_vel_out': pos_vel_out,
        'vel_smooth': vel_out,
        'a_obs': acc_obs_out,
        'a_grav': a_grav,
        'a_3bp': a_3bp,
        'a_srp': a_srp,
        'a_nc': a_nc,
        'a_drag': a_drag,
        'nc_hcl': (h_nc, c_nc, l_nc),
        'drag_hcl': (h_drag, c_drag, l_drag),
    }
