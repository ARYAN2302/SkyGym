"""ENU world frame helpers and geometry transforms.

Conventions
-----------
- World frame: ENU (East, North, Up), metres.
- Sensor site sits at the origin by default; drones live in the +z half space.
- Azimuth: degrees, 0 = North, 90 = East, range [0, 360).
- Elevation: degrees above local horizon, range [-90, +90].
- Range: metres, > 0.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def cartesian_to_spherical(d_enu: np.ndarray) -> tuple[float, float, float]:
    """ENU offset vector -> (az_deg, el_deg, range_m)."""
    d = np.asarray(d_enu, dtype=float)
    east, north, up = float(d[0]), float(d[1]), float(d[2])
    rng = float(np.sqrt(east * east + north * north + up * up))
    az = float(np.degrees(np.arctan2(east, north))) % 360.0
    el = float(np.degrees(np.arctan2(up, np.hypot(east, north) + EPS)))
    return az, el, rng


def spherical_to_cartesian(az_deg: float, el_deg: float, range_m: float) -> np.ndarray:
    """(az_deg, el_deg, range_m) -> ENU offset vector."""
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    e = range_m * np.cos(el) * np.sin(az)
    n = range_m * np.cos(el) * np.cos(az)
    u = range_m * np.sin(el)
    return np.array([e, n, u])


def wrap_az_deg(az: float) -> float:
    """Wrap azimuth to [0, 360)."""
    return float(az) % 360.0


def ang_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference a-b in degrees, wrapped to [-180, 180]."""
    return float((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def measurement_jacobian(az_deg: float, el_deg: float, range_m: float) -> np.ndarray:
    """Jacobian d(cartesian)/d(az, el, range) at a spherical point.

    Used by the EKF to convert measurement covariance (spherical) to
    cartesian innovation covariance.
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    ce, se = np.cos(el), np.sin(el)
    ca, sa = np.cos(az), np.sin(az)
    J = np.zeros((3, 3))
    # d/d(range)
    J[:, 2] = [ce * sa, ce * ca, se]
    # d/d(az)
    J[:, 0] = [range_m * ce * ca, -range_m * ce * sa, 0.0]
    # d/d(el)
    J[:, 1] = [-range_m * se * sa, -range_m * se * ca, range_m * ce]
    return J


def spherical_cov_to_cartesian(
    az_deg: float, el_deg: float, range_m: float,
    sig_az_deg: float, sig_el_deg: float, sig_r_m: float,
) -> np.ndarray:
    """Map diagonal spherical noise sigma to a 3x3 cartesian covariance."""
    J = measurement_jacobian(az_deg, el_deg, range_m)
    R = np.diag([
        np.radians(sig_az_deg) ** 2,
        np.radians(sig_el_deg) ** 2,
        sig_r_m ** 2,
    ])
    return J @ R @ J.T


def ground_range_m(range_m: float, el_deg: float) -> float:
    """Slant range to horizontal distance."""
    return range_m * np.cos(np.radians(el_deg))
