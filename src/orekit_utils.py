"""
Orekit helpers with caching for DensityInversion2.

Force model matches Paper Table 2:
  - Gravity field: EIGEN-6S4 100x100 (loaded from orekit-data.zip)
  - Third-body:    Sun & Moon point masses, DE421 ephemerides
  - SRP:           Cannonball (Cr), cone shadow (Orekit penumbra model)
  - ERP:           Knocke 1x1 deg rediffused Earth radiation model
  - IERS:          IERS 2010 conventions (paper says "2014"; Orekit enum
                   caps at IERS_2010 — "2014" refers to EOP series C04-14)
"""

import numpy as np
import functools

import orekit
from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions, PVCoordinates, Constants
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    NewtonianAttraction,
    ThirdBodyAttraction,
)
from org.orekit.forces.radiation import (
    SolarRadiationPressure,
    IsotropicRadiationSingleCoefficient,
    KnockeRediffusedForceModel,
)
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.forces import ForceModel

from .config import GRAVITY_DEGREE, GRAVITY_ORDER, OMEGA_EARTH

# Cache force models at module level (built once per sat config)
_srp_model_cache = {}
_erp_model_cache = {}


# ------------------------------------------------------------------ helpers
def _to_vec3d(arr):
    return Vector3D(float(arr[0]), float(arr[1]), float(arr[2]))


def _pv_to_numpy(pv):
    p = pv.getPosition()
    v = pv.getVelocity()
    return np.array([p.getX(), p.getY(), p.getZ(),
                     v.getX(), v.getY(), v.getZ()])


# ------------------------------------------------------------------ frames
@functools.lru_cache(maxsize=1)
def get_eci_frame():
    return FramesFactory.getEME2000()


@functools.lru_cache(maxsize=1)
def get_ecef_frame():
    return FramesFactory.getITRF(IERSConventions.IERS_2010, True)


# ------------------------------------------------------------------ gravity
@functools.lru_cache(maxsize=4)
def _get_gravity_provider(degree, order):
    return GravityFieldFactory.getNormalizedProvider(int(degree), int(order))


@functools.lru_cache(maxsize=4)
def _get_gravity_field(degree, order):
    provider = _get_gravity_provider(degree, order)
    return HolmesFeatherstoneAttractionModel(get_ecef_frame(), provider)


@functools.lru_cache(maxsize=1)
def _get_monopole():
    return NewtonianAttraction(Constants.WGS84_EARTH_MU)


def compute_gravity_acceleration(state_vector_eci, dt, mass):
    """
    Full gravitational acceleration (monopole + non-spherical) at a single ECI state.

    Uses cached 100x100 Holmes-Featherstone + Newtonian monopole.

    Returns (3,) ndarray  [m/s^2] in ECI.
    """
    date = datetime_to_absolutedate(dt)
    pv = PVCoordinates(_to_vec3d(state_vector_eci[:3]),
                       _to_vec3d(state_vector_eci[3:]))
    orbit = CartesianOrbit(pv, get_eci_frame(), date, Constants.WGS84_EARTH_MU)
    state = SpacecraftState(orbit, float(mass))

    acc = np.zeros(3)
    # Monopole
    monopole = _get_monopole()
    params = ForceModel.cast_(monopole).getParameters()
    a = monopole.acceleration(state, params)
    acc += np.array([a.getX(), a.getY(), a.getZ()])

    # Non-spherical (100x100)
    grav = _get_gravity_field(GRAVITY_DEGREE, GRAVITY_ORDER)
    params = ForceModel.cast_(grav).getParameters()
    a = grav.acceleration(state, params)
    acc += np.array([a.getX(), a.getY(), a.getZ()])

    return acc


def compute_nonspherical_potential(pos_ecef, dt):
    """
    Non-central gravitational potential U_ns at *pos_ecef* (ECEF metres).

    Parameters
    ----------
    pos_ecef : (3,) array — ECEF position [m]
    dt       : datetime   — epoch

    Returns
    -------
    float — U_ns  [m^2/s^2]
    """
    grav = _get_gravity_field(GRAVITY_DEGREE, GRAVITY_ORDER)
    date = datetime_to_absolutedate(dt)
    return grav.nonCentralPart(date, _to_vec3d(pos_ecef), Constants.WGS84_EARTH_MU)


# ------------------------------------------------------------------ ECI ↔ ECEF
def eci_to_ecef(pos_vel_eci, dt):
    """
    Convert a single ECI state to ECEF.

    Parameters
    ----------
    pos_vel_eci : (6,) array — [x,y,z,vx,vy,vz] in ECI [m, m/s]
    dt          : datetime

    Returns
    -------
    (6,) ndarray — ECEF [m, m/s]
    """
    date = datetime_to_absolutedate(dt)
    pv_eci = PVCoordinates(_to_vec3d(pos_vel_eci[:3]),
                           _to_vec3d(pos_vel_eci[3:]))
    transform = get_eci_frame().getTransformTo(get_ecef_frame(), date)
    pv_ecef = transform.transformPVCoordinates(pv_eci)
    return _pv_to_numpy(pv_ecef)


# ------------------------------------------------------------------ 3BP
@functools.lru_cache(maxsize=1)
def _get_3bp_models():
    moon = CelestialBodyFactory.getMoon()
    sun  = CelestialBodyFactory.getSun()
    return ThirdBodyAttraction(moon), ThirdBodyAttraction(sun)


def compute_3bp_acceleration(state_vector_eci, dt, mass):
    """
    Sun + Moon third-body acceleration at a single ECI state.

    Returns (3,) ndarray  [m/s^2] in ECI.
    """
    date = datetime_to_absolutedate(dt)
    moon_model, sun_model = _get_3bp_models()

    pv = PVCoordinates(_to_vec3d(state_vector_eci[:3]),
                       _to_vec3d(state_vector_eci[3:]))
    orbit = CartesianOrbit(pv, get_eci_frame(), date, Constants.WGS84_EARTH_MU)
    state = SpacecraftState(orbit, float(mass))

    acc = np.zeros(3)
    for model in (moon_model, sun_model):
        params = ForceModel.cast_(model).getParameters()
        a = model.acceleration(state, params)
        acc += np.array([a.getX(), a.getY(), a.getZ()])
    return acc


# ------------------------------------------------------------------ SRP
def _get_srp_model(Cr, area):
    """Build an isotropic SRP model, cached by (Cr, area)."""
    key = (float(Cr), float(area))
    if key not in _srp_model_cache:
        radiation = IsotropicRadiationSingleCoefficient(float(area), float(Cr))
        earth = OneAxisEllipsoid(
            Constants.IERS2010_EARTH_EQUATORIAL_RADIUS,
            Constants.IERS2010_EARTH_FLATTENING,
            get_ecef_frame(),
        )
        sun = CelestialBodyFactory.getSun()
        srp = SolarRadiationPressure(sun, earth, radiation)
        srp.addOccultingBody(CelestialBodyFactory.getMoon(),
                             Constants.MOON_EQUATORIAL_RADIUS)
        _srp_model_cache[key] = srp
    return _srp_model_cache[key]


def compute_srp_acceleration(state_vector_eci, dt, Cr, area, mass):
    """
    Solar radiation pressure acceleration at a single ECI state.

    Returns (3,) ndarray  [m/s^2] in ECI.
    """
    date = datetime_to_absolutedate(dt)
    srp = _get_srp_model(Cr, area)

    pv = PVCoordinates(_to_vec3d(state_vector_eci[:3]),
                       _to_vec3d(state_vector_eci[3:]))
    orbit = CartesianOrbit(pv, get_eci_frame(), date, Constants.WGS84_EARTH_MU)
    state = SpacecraftState(orbit, float(mass))

    params = ForceModel.cast_(srp).getParameters()
    a = srp.acceleration(state, params)
    return np.array([a.getX(), a.getY(), a.getZ()])


# ------------------------------------------------------------------ ERP
def _get_erp_model(Cr, area):
    """Build a Knocke 1x1 deg Earth radiation pressure model, cached by (Cr, area)."""
    key = (float(Cr), float(area))
    if key not in _erp_model_cache:
        radiation = IsotropicRadiationSingleCoefficient(float(area), float(Cr))
        sun = CelestialBodyFactory.getSun()
        erp = KnockeRediffusedForceModel(
            sun, radiation,
            Constants.IERS2010_EARTH_EQUATORIAL_RADIUS,
            Constants.IERS2010_EARTH_FLATTENING,
        )
        _erp_model_cache[key] = erp
    return _erp_model_cache[key]


def compute_erp_acceleration(state_vector_eci, dt, Cr, area, mass):
    """
    Earth radiation pressure (Knocke 1x1 deg) acceleration at a single ECI state.

    Returns (3,) ndarray  [m/s^2] in ECI.
    """
    date = datetime_to_absolutedate(dt)
    erp = _get_erp_model(Cr, area)

    pv = PVCoordinates(_to_vec3d(state_vector_eci[:3]),
                       _to_vec3d(state_vector_eci[3:]))
    orbit = CartesianOrbit(pv, get_eci_frame(), date, Constants.WGS84_EARTH_MU)
    state = SpacecraftState(orbit, float(mass))

    params = ForceModel.cast_(erp).getParameters()
    a = erp.acceleration(state, params)
    return np.array([a.getX(), a.getY(), a.getZ()])
