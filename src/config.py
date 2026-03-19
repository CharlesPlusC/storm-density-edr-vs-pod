"""
Satellite parameters and physical constants for density inversion.

All methods (EDR, POD-accelerometry) use the same parameters from this file
to ensure consistency (Reviewer 2, point 1).
"""

import numpy as np

# ---------- Satellite catalogue ----------
# Cd estimated from EDR median log-bias vs accelerometer truth (see scripts/estimate_cd.py)
SATELLITES = {
    'GRACE-FO': {
        'mass': 600.2,       # kg (Mehta 2013)
        'area': 1.04,        # m^2 cross-sectional (Mehta 2013)
        'Cd':   3.2,         # drag coefficient (de-biased away; Table 2)
        'Cr':   1.5,         # radiation pressure coefficient
        'norad': 43476,
    },
    'GRACE-FO-A': {          # alias
        'mass': 600.2,
        'area': 1.04,
        'Cd':   3.2,
        'Cr':   1.5,
        'norad': 43476,
    },
    'CHAMP': {
        'mass': 522.0,
        'area': 1.0,         # m^2 cross-sectional (Mehta 2017)
        'Cd':   2.2,         # drag coefficient (de-biased away; Table 2)
        'Cr':   1.0,
        'norad': 26405,
    },
    'Swarm-A': {
        'mass': 473.0,
        'area': 0.9,
        'Cd':   4.92,        # EDR median, 7 storms vs Swarm POD density
        'Cr':   1.5,
        'norad': 39452,
        'ephem_dir': 'Swarm-A',
    },
    'Swarm-B': {
        'mass': 473.0,
        'area': 0.9,
        'Cd':   4.59,        # EDR median, 7 storms vs Swarm POD density
        'Cr':   1.5,
        'norad': 39451,
        'ephem_dir': 'Swarm-B',
    },
    'Swarm-C': {
        'mass': 473.0,
        'area': 0.9,
        'Cd':   4.92,        # EDR median, 6 storms vs Swarm ACC density
        'Cr':   1.5,
        'norad': 39453,
        'ephem_dir': 'Swarm-C',
    },
}

# ---------- Gravity field ----------
# Ray (2024): degree 80-100 for POD-accel, 60-70 for EDR above 400 km.
# Use 100x100 for both methods (consistent).
GRAVITY_DEGREE = 100
GRAVITY_ORDER  = 100

# ---------- Physical constants ----------
OMEGA_EARTH = 7.2921159e-5   # rad/s  — Earth's mean rotation rate
OMEGA_VEC   = np.array([0.0, 0.0, OMEGA_EARTH])
MU_EARTH    = 3.986004418e14  # m^3/s^2  — WGS84


def get_satellite(name):
    """Return satellite dict; raise KeyError if unknown."""
    if name not in SATELLITES:
        raise KeyError(f"Unknown satellite '{name}'. Known: {list(SATELLITES.keys())}")
    return SATELLITES[name]


def ballistic_coeff(name):
    """B = Cd * A / M  [m^2/kg]."""
    s = get_satellite(name)
    return s['Cd'] * s['area'] / s['mass']
