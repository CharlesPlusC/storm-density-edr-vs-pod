"""
Picone (2005) effective (orbit-average) density.

For any high-cadence density time series (accelerometer, POD, model), compute
one effective density per orbit using the velocity-weighted formula:

    rho_eff = int(rho * v_rel^2 * |v| dt) / int(v_rel^2 * |v| dt)

This is equivalent to Picone (2005) Eq.13 for a co-rotating atmosphere (no
winds), where F ≈ (v_rel/v)^2 * (v_rel_hat . v_hat) simplifies so that
v^3 * F ≈ v_rel^2 * |v|.

EDR already returns effective density by construction (Sutton 2021 Eq.4),
so this module is only needed for ACC, POD, JB08, MSIS, etc.
"""

import numpy as np
from .config import OMEGA_VEC


def compute_effective_density(rho, pos_vel_eci, perigee_indices):
    """
    Orbit-average a high-cadence density series using Picone weighting.

    Parameters
    ----------
    rho             : (N,) ndarray — density at native cadence [kg/m^3]
    pos_vel_eci     : (N, 6) ndarray — ECI states [m, m/s]
    perigee_indices : 1-d int array — perigee passage indices

    Returns
    -------
    rho_eff   : list of float — one effective density per orbit
    mid_times_idx : list of int — index of orbit midpoint (for time mapping)
    """
    rho_eff = []
    mid_indices = []

    for k in range(len(perigee_indices) - 1):
        i0 = perigee_indices[k]
        i1 = perigee_indices[k + 1]

        arc_rho = rho[i0:i1]
        arc_pv  = pos_vel_eci[i0:i1]

        # Relative velocity (co-rotating atmosphere)
        v_rel = arc_pv[:, 3:] - np.cross(OMEGA_VEC, arc_pv[:, :3])
        v_rel_mag = np.linalg.norm(v_rel, axis=1)
        speeds = np.linalg.norm(arc_pv[:, 3:], axis=1)

        weight = v_rel_mag**2 * speeds   # Picone weighting

        # Mask NaN / negative densities
        valid = np.isfinite(arc_rho) & (arc_rho > 0) & np.isfinite(weight)
        if valid.sum() < 2:
            rho_eff.append(np.nan)
        else:
            rho_eff.append(
                np.sum(arc_rho[valid] * weight[valid]) / np.sum(weight[valid]))

        mid_indices.append((i0 + i1) // 2)

    return rho_eff, mid_indices
