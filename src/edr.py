"""
Energy Dissipation Rate (EDR) density inversion.

Implements the ECEF rotating-frame energy approach of Sutton (2021) and
Fitzpatrick (2025), with explicit 3BP and SRP work corrections.

Energy:  xi = v_ecef^2/2  -  omega^2*(x^2+y^2)/2  -  mu/r  -  U_ns

EDR:     -(E2 - E1) / dt   corrected for  int(a_3bp . v_ecef) dt
                                        +  int(a_srp . v_ecef) dt

Density: rho_eff = 2 * EDR * dt / [B * int(v_rel^2 * |v| dt)]
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

# Pre-import for work integral — rotating-frame velocity in ECI coords
# v_rot = v_ECI - omega x r_ECI


# ====================================================================
# Perigee detection
# ====================================================================
def find_perigees(pos_vel_eci, times):
    """
    Find perigee passages via radial-velocity sign change.

    Parameters
    ----------
    pos_vel_eci : (N, 6) ndarray
    times       : list/array of datetime  (length N)

    Returns
    -------
    perigee_indices : 1-d int array — indices into *pos_vel_eci*
    """
    times = np.asarray(times)
    r_dot = np.sum(pos_vel_eci[:, :3] * pos_vel_eci[:, 3:], axis=1) / \
            np.linalg.norm(pos_vel_eci[:, :3], axis=1)

    # Sign change negative→positive = perigee
    candidates = np.where(np.diff(np.sign(r_dot)) > 0)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)

    # Filter: keep one candidate per ~90 min window (80–110 min)
    accepted = [candidates[0]]
    i = 1
    while i < len(candidates):
        gap_min = (times[candidates[i]] - times[accepted[-1]]).total_seconds() / 60.0
        if gap_min < 80:
            i += 1
            continue
        # Collect all candidates within the 80–110 min window
        window = []
        while i < len(candidates):
            gap_min = (times[candidates[i]] - times[accepted[-1]]).total_seconds() / 60.0
            if gap_min > 110:
                break
            if gap_min >= 80:
                window.append(candidates[i])
            i += 1
        if window:
            best = min(window, key=lambda idx: abs(
                (times[idx] - times[accepted[-1]]).total_seconds() / 60.0 - 90))
            accepted.append(best)
        elif i < len(candidates):
            accepted.append(candidates[i])
            i += 1

    return np.array(accepted, dtype=int)


# ====================================================================
# Orbital energy in rotating (ECEF) frame
# ====================================================================
def compute_orbital_energy(pos_vel_eci, times, indices):
    """
    Jacobi-like orbital energy at selected indices.

    xi = v_ecef^2/2 - omega^2*(x_ecef^2 + y_ecef^2)/2 - mu/r - U_ns

    Returns (len(indices),) array.
    """
    energy = np.empty(len(indices))
    for k, idx in enumerate(tqdm(indices, desc='Orbital energy')):
        pv_ecef = eci_to_ecef(pos_vel_eci[idx], times[idx])
        pos_ecef = pv_ecef[:3]
        vel_ecef = pv_ecef[3:]

        v2 = np.dot(vel_ecef, vel_ecef)
        r  = np.linalg.norm(pos_ecef)
        centrifugal = OMEGA_EARTH**2 * (pos_ecef[0]**2 + pos_ecef[1]**2)
        U_ns = compute_nonspherical_potential(pos_ecef, times[idx])

        energy[k] = v2 / 2.0 - centrifugal / 2.0 - MU_EARTH / r - U_ns

    return energy


# ====================================================================
# 3BP / SRP work integrals over an arc
# ====================================================================
def _integrate_perturbation_work(pos_vel_eci, times, i_start, i_end,
                                 accel_func, **kwargs):
    """
    Trapezoidal integration of  int( a_pert . v_rot ) dt  over [i_start, i_end].

    Both a_pert and v_rot are expressed in ECI coordinates.
    v_rot = v_ECI - omega x r_ECI  (rotating-frame velocity in ECI coords).
    """
    integrand = np.empty(i_end - i_start + 1)
    for j, idx in enumerate(tqdm(range(i_start, i_end + 1),
                                  desc='Perturbation work', leave=False)):
        a_eci = accel_func(pos_vel_eci[idx], times[idx], **kwargs)
        r_eci = pos_vel_eci[idx, :3]
        v_eci = pos_vel_eci[idx, 3:]
        v_rot = v_eci - np.cross(OMEGA_VEC, r_eci)
        integrand[j] = np.dot(a_eci, v_rot)

    # Time grid in seconds
    t_sec = np.array([(times[idx] - times[i_start]).total_seconds()
                      for idx in range(i_start, i_end + 1)])
    return np.trapz(integrand, t_sec)


# ====================================================================
# EDR computation
# ====================================================================
def compute_edr(pos_vel_eci, times, perigee_indices,
                fitspan=1,
                correct_3bp=True, correct_srp=True,
                Cr=1.5, area=1.0, mass=600.2):
    """
    Compute Energy Dissipation Rate on a perigee-to-perigee basis.

    Parameters
    ----------
    pos_vel_eci     : (N, 6) ndarray
    times           : list of datetime (length N)
    perigee_indices : 1-d int array from find_perigees()
    fitspan         : int — number of perigee-to-perigee arcs to span
    correct_3bp     : subtract third-body work integral
    correct_srp     : subtract SRP work integral
    Cr, area, mass  : SRP spacecraft parameters

    Returns
    -------
    edr_values : list of float  (W/kg = m^2/s^3)
    arc_midtimes : list of datetime
    """
    # Compute energy at all perigee points
    energy = compute_orbital_energy(pos_vel_eci, times, perigee_indices)

    n_arcs = len(perigee_indices) - fitspan
    edr_values = []
    arc_midtimes = []

    for i in tqdm(range(n_arcs), desc='EDR arcs'):
        i0 = perigee_indices[i]
        i1 = perigee_indices[i + fitspan]
        dt = (times[i1] - times[i0]).total_seconds()
        if dt <= 0:
            edr_values.append(np.nan)
            arc_midtimes.append(times[i0])
            continue

        # Energy change: dE = -(E2 - E1) = E1 - E2
        # From the energy balance: E2 - E1 = W_drag + W_3bp + W_srp
        # So: W_drag = (E2 - E1) - W_3bp - W_srp
        # And: EDR = -W_drag/dt = (E1 - E2 + W_3bp + W_srp) / dt
        #                       = (dE + W_3bp + W_srp) / dt
        dE = -(energy[i + fitspan] - energy[i])

        # ADD perturbation work back (3BP/SRP do work that is NOT drag)
        if correct_3bp:
            work_3bp = _integrate_perturbation_work(
                pos_vel_eci, times, i0, i1,
                compute_3bp_acceleration, mass=mass)
            dE += work_3bp

        if correct_srp:
            work_srp = _integrate_perturbation_work(
                pos_vel_eci, times, i0, i1,
                compute_srp_acceleration, Cr=Cr, area=area, mass=mass)
            dE += work_srp

        edr_values.append(dE / dt)
        mid = times[i0] + (times[i1] - times[i0]) / 2
        arc_midtimes.append(mid)

    return edr_values, arc_midtimes


# ====================================================================
# EDR → density
# ====================================================================
def _relative_velocity_eci(pos_vel_eci):
    """
    |v_rel| = |v - omega x r|  (co-rotating atmosphere, no winds).

    pos_vel_eci : (M, 6) or (6,) — ECI state(s)
    Returns      : (M,) or scalar — relative speed [m/s]
    """
    if pos_vel_eci.ndim == 1:
        v_rel = pos_vel_eci[3:] - np.cross(OMEGA_VEC, pos_vel_eci[:3])
        return np.linalg.norm(v_rel)
    v_rel = pos_vel_eci[:, 3:] - np.cross(OMEGA_VEC, pos_vel_eci[:, :3])
    return np.linalg.norm(v_rel, axis=1)


def edr_to_density(edr_values, pos_vel_eci, times, perigee_indices,
                   sat_name='GRACE-FO', fitspan=1):
    """
    Convert EDR values to effective density via Picone (2005)/Sutton (2021):

        rho_eff = 2 * EDR * dt / [B * int(v_rel^2 * |v| dt)]

    Returns
    -------
    rho_edr  : list of float  (kg/m^3)
    mid_times: list of datetime
    """
    B = ballistic_coeff(sat_name)
    n_arcs = len(perigee_indices) - fitspan
    rho_edr = []
    mid_times = []

    for i in range(n_arcs):
        i0 = perigee_indices[i]
        i1 = perigee_indices[i + fitspan]
        dt_total = (times[i1] - times[i0]).total_seconds()
        if dt_total <= 0 or np.isnan(edr_values[i]):
            rho_edr.append(np.nan)
            mid_times.append(times[i0])
            continue

        arc = pos_vel_eci[i0:i1 + 1]
        v_rel = _relative_velocity_eci(arc)
        speeds = np.linalg.norm(arc[:, 3:], axis=1)

        # Integrand: v_rel^2 * |v|
        integrand = v_rel**2 * speeds
        dt_step = dt_total / (len(arc) - 1) if len(arc) > 1 else 1.0
        integral = np.trapz(integrand, dx=dt_step)

        if integral == 0 or np.isnan(integral):
            rho_edr.append(np.nan)
        else:
            rho_edr.append(2.0 * edr_values[i] * dt_total / (B * integral))

        mid = times[i0] + (times[i1] - times[i0]) / 2
        mid_times.append(mid)

    return rho_edr, mid_times
