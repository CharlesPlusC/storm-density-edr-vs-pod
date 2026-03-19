"""
Sub-orbital EDR density inversion (Ray et al. 2024 formulation).

Uses the ECEF Jacobi energy approach (same as our working orbit-effective EDR)
generalized to arbitrary arc endpoints instead of perigee-to-perigee:

    1. Evaluate Jacobi energy at arc start/end (endpoint potential, no integration)
    2. Correct for 3BP and SRP via work integrals using v_rot
    3. Invert for density via Picone weighting

The Jacobi approach evaluates the gravitational potential exactly at the
endpoints, avoiding numerical integration errors from undersampled high-degree
gravity harmonics. Only the much smaller 3BP and SRP work integrals require
numerical integration.

Density is produced at every SP3 timestep (30s) as a moving-average over the
chosen arc length, centered on each output epoch.
"""

import numpy as np
from tqdm.auto import tqdm

from .config import OMEGA_EARTH, OMEGA_VEC, MU_EARTH, get_satellite, ballistic_coeff
from .orekit_utils import (
    eci_to_ecef,
    compute_nonspherical_potential,
    compute_3bp_acceleration,
    compute_srp_acceleration,
)


def precompute_forces(pos_vel_eci, times, sat_name):
    """
    Pre-compute 3BP/SRP accelerations and Jacobi energy on the full time grid.

    The Jacobi energy is evaluated at every point (endpoint evaluation), while
    3BP and SRP are stored for work integral computation.

    Parameters
    ----------
    pos_vel_eci : (N, 6) ndarray — ECI states [m, m/s]
    times       : list of datetime (length N)
    sat_name    : str

    Returns
    -------
    forces : dict
    """
    sat = get_satellite(sat_name)
    mass = sat['mass']
    Cr, area = sat['Cr'], sat['area']
    N = len(times)

    # Pre-compute Jacobi energy at every point
    energy = np.empty(N)
    a_3bp = np.empty((N, 3))
    a_srp = np.empty((N, 3))

    for i in tqdm(range(N), desc='Pre-computing energy + forces'):
        sv = pos_vel_eci[i]
        t = times[i]

        # Jacobi energy in ECEF rotating frame
        pv_ecef = eci_to_ecef(sv, t)
        pos_ecef = pv_ecef[:3]
        vel_ecef = pv_ecef[3:]
        v2 = np.dot(vel_ecef, vel_ecef)
        r = np.linalg.norm(pos_ecef)
        centrifugal = OMEGA_EARTH**2 * (pos_ecef[0]**2 + pos_ecef[1]**2)
        U_ns = compute_nonspherical_potential(pos_ecef, t)
        energy[i] = v2 / 2.0 - centrifugal / 2.0 - MU_EARTH / r - U_ns

        # 3BP and SRP accelerations in ECI (ERP dropped — ~5e-9 m/s², negligible)
        a_3bp[i] = compute_3bp_acceleration(sv, t, mass)
        a_srp[i] = compute_srp_acceleration(sv, t, Cr, area, mass)

    # Pre-compute rotating-frame velocity in ECI coords: v_rot = v_ECI - omega x r_ECI
    v_rot = pos_vel_eci[:, 3:] - np.cross(OMEGA_VEC, pos_vel_eci[:, :3])

    # Pre-compute work integrands: a_pert . v_rot
    bp3_dot_vrot = np.sum(a_3bp * v_rot, axis=1)
    srp_dot_vrot = np.sum(a_srp * v_rot, axis=1)
    erp_dot_vrot = np.zeros(N)  # ERP dropped

    # Relative velocity (co-rotating atmosphere, no winds)
    v_rel = np.linalg.norm(v_rot, axis=1)
    speeds = np.linalg.norm(pos_vel_eci[:, 3:], axis=1)

    # Picone denominator integrand: v_rel^2 * |v|
    picone_integrand = v_rel**2 * speeds

    t0 = times[0]
    dt_sec = np.array([(t - t0).total_seconds() for t in times])

    return {
        'energy': energy,
        'bp3_dot_vrot': bp3_dot_vrot,
        'srp_dot_vrot': srp_dot_vrot,
        'erp_dot_vrot': erp_dot_vrot,
        'picone_integrand': picone_integrand,
        'v_rel': v_rel,
        'speeds': speeds,
        'dt_sec': dt_sec,
    }


def suborbit_edr_density(pos_vel_eci, times, forces, arc_minutes, sat_name):
    """
    Compute sub-orbital EDR density at every SP3 timestep.

    Uses a moving-average window of arc_minutes centered on each epoch.
    Energy balance via Jacobi integral (endpoint evaluation) corrected for
    3BP and SRP work integrals.

    Parameters
    ----------
    pos_vel_eci : (N, 6) ndarray
    times       : list of datetime (length N)
    forces      : dict from precompute_forces()
    arc_minutes : float — arc length in minutes
    sat_name    : str

    Returns
    -------
    rho       : (N,) ndarray — density [kg/m³] (NaN at edges)
    mid_times : list of datetime — same as input times
    """
    sat = get_satellite(sat_name)
    Cd, A, mass = sat['Cd'], sat['area'], sat['mass']
    N = len(times)

    dt_sec = forces['dt_sec']
    energy = forces['energy']
    bp3_dot_vrot = forces['bp3_dot_vrot']
    srp_dot_vrot = forces['srp_dot_vrot']
    erp_dot_vrot = forces['erp_dot_vrot']
    picone_integrand = forces['picone_integrand']

    # Determine half-window in index space
    cadence_s = dt_sec[1] - dt_sec[0] if N > 1 else 30.0
    half_win_idx = int(round(arc_minutes * 60 / (2 * cadence_s)))

    rho = np.full(N, np.nan)

    for i in range(N):
        i0 = i - half_win_idx
        i1 = i + half_win_idx
        if i0 < 0 or i1 >= N:
            continue

        # Time array for this arc (relative, for trapz)
        t_arc = dt_sec[i0:i1 + 1] - dt_sec[i0]
        dt_total = t_arc[-1]
        if dt_total <= 0:
            continue

        # Energy change via Jacobi integral (endpoint evaluation — exact)
        # dE = -(E2 - E1) = E1 - E2  (positive for drag-dominated arcs)
        dE = -(energy[i1] - energy[i0])

        # Add back 3BP work (perturbation does work that is NOT drag)
        W_3bp = np.trapz(bp3_dot_vrot[i0:i1 + 1], t_arc)
        dE += W_3bp

        # Add back SRP work
        W_srp = np.trapz(srp_dot_vrot[i0:i1 + 1], t_arc)
        dE += W_srp

        # Add back ERP work
        W_erp = np.trapz(erp_dot_vrot[i0:i1 + 1], t_arc)
        dE += W_erp

        # EDR = dE / dt
        edr = dE / dt_total

        # Picone denominator: integral(v_rel^2 * |v| dt)
        denom = np.trapz(picone_integrand[i0:i1 + 1], t_arc)
        if denom == 0:
            continue

        # Density: rho = 2 * EDR * dt / (B * denom)
        B = Cd * A / mass
        rho[i] = 2.0 * edr * dt_total / (B * denom)

    return rho, times


def multiorbit_edr_density(forces, sat_name, perigee_indices, n_orbits=1):
    """
    Compute EDR effective density for consecutive N-orbit spans.

    Integrates the Jacobi energy balance from perigee_i to perigee_{i+n_orbits}
    for each non-overlapping N-orbit group.  This is the genuine multi-orbit
    EDR — the energy balance spans the full N orbits, not an average of
    single-orbit EDR values.

    Parameters
    ----------
    forces          : dict from precompute_forces()
    sat_name        : str
    perigee_indices : 1-d int array — perigee passage indices
    n_orbits        : int — number of orbits per span (1, 2, or 3)

    Returns
    -------
    rho_eff : ndarray of float — one density per N-orbit span
    """
    B = ballistic_coeff(sat_name)

    dt_sec = forces['dt_sec']
    energy = forces['energy']
    bp3_dot_vrot = forces['bp3_dot_vrot']
    srp_dot_vrot = forces['srp_dot_vrot']
    erp_dot_vrot = forces['erp_dot_vrot']
    picone_integrand = forces['picone_integrand']

    # Non-overlapping N-orbit spans
    peri_N = perigee_indices[::n_orbits]

    rho_eff = np.full(len(peri_N) - 1, np.nan)
    for k in range(len(peri_N) - 1):
        i0 = peri_N[k]
        i1 = peri_N[k + 1]

        t_arc = dt_sec[i0:i1 + 1] - dt_sec[i0]
        dt_total = t_arc[-1]
        if dt_total <= 0:
            continue

        # Energy change via Jacobi integral (endpoint evaluation)
        dE = -(energy[i1] - energy[i0])

        # Correct for 3BP, SRP, and ERP work
        dE += np.trapz(bp3_dot_vrot[i0:i1 + 1], t_arc)
        dE += np.trapz(srp_dot_vrot[i0:i1 + 1], t_arc)
        dE += np.trapz(erp_dot_vrot[i0:i1 + 1], t_arc)

        edr = dE / dt_total

        # Picone denominator
        denom = np.trapz(picone_integrand[i0:i1 + 1], t_arc)
        if denom == 0:
            continue

        rho_eff[k] = 2.0 * edr * dt_total / (B * denom)

    return rho_eff


def sweep_edr_arcs(pos_vel_eci, times, forces, sat_name,
                   td_on_sp3_grid,
                   arc_minutes_list=None):
    """
    Sweep arc lengths and compute metrics for each.

    Computes two sets of metrics:
      - r, SD%: against resolution-matched (averaged) TU Delft
      - r_native, EV: against native (unaveraged) TU Delft

    The native-truth metrics reveal the genuine optimal resolution —
    the arc length that maximizes information content about the real
    density field. Matched-truth metrics monotonically improve with
    arc length and are misleading for optimal selection.

    Parameters
    ----------
    pos_vel_eci    : (N, 6) ndarray
    times          : list of datetime (length N)
    forces         : dict from precompute_forces()
    sat_name       : str
    td_on_sp3_grid : (N,) ndarray — TU Delft density on SP3 grid (native)
    arc_minutes_list : list of float

    Returns
    -------
    results : list of dict, one per arc length
    """
    from .metrics import log_metrics
    from .pod_accelerometry_hires import moving_average_tudelft, debias_density

    if arc_minutes_list is None:
        arc_minutes_list = [2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 75, 90, 100, 120]

    # SP3 cadence
    dt_sec = forces['dt_sec']
    cadence_s = dt_sec[1] - dt_sec[0] if len(dt_sec) > 1 else 30.0

    results = []

    for am in arc_minutes_list:
        print(f"  EDR arc = {am} min...")
        rho, _ = suborbit_edr_density(pos_vel_eci, times, forces, am, sat_name)

        neg_frac = 100.0 * np.nansum(rho < 0) / np.sum(np.isfinite(rho))

        # De-bias against native truth using median log-ratio
        rho, beta_db = debias_density(rho, td_on_sp3_grid)

        # Matched-truth metrics (resolution-matched averaging)
        td_avg = moving_average_tudelft(td_on_sp3_grid, am, cadence_s)
        valid = (np.isfinite(rho) & np.isfinite(td_avg)
                 & (rho > 0) & (td_avg > 0))
        n_valid = valid.sum()
        if n_valid >= 10:
            m = log_metrics(td_avg[valid], rho[valid])
        else:
            m = {'r': np.nan, 'r_squared': np.nan, 'SD_pct': np.nan, 'sigma': np.nan, 'beta': np.nan}

        # Native-truth metrics (against unaveraged TU Delft)
        v_nat = (np.isfinite(rho) & np.isfinite(td_on_sp3_grid)
                 & (rho > 0) & (td_on_sp3_grid > 0))
        if v_nat.sum() >= 10:
            m_nat = log_metrics(td_on_sp3_grid[v_nat], rho[v_nat])
            resid = rho[v_nat] - td_on_sp3_grid[v_nat]
            expl_var = 100.0 * (1.0 - np.var(resid) / np.var(td_on_sp3_grid[v_nat]))
        else:
            m_nat = {'r': np.nan, 'r_squared': np.nan, 'SD_pct': np.nan}
            expl_var = np.nan

        results.append({
            'arc_minutes': am,
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


def find_optimal_arc(sweep_results, neg_threshold=10.0):
    """
    Find the arc length that maximizes r against native (unaveraged) truth.

    This finds the genuine optimal resolution — the arc length that captures
    the most information about the real density field. Too short = noise
    dominates, too long = real structure is destroyed.

    Parameters
    ----------
    sweep_results : list of dict from sweep_edr_arcs()
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
