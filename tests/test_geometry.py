"""Geometry validation: roundtrip transforms, Jacobian consistency."""
import numpy as np
import pytest

from skygym import world


def test_spherical_cartesian_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        az, el = rng.uniform(0, 360), rng.uniform(-85, 85)
        r = rng.uniform(10, 10000)
        p = world.spherical_to_cartesian(az, el, r)
        az2, el2, r2 = world.cartesian_to_spherical(p)
        assert abs(world.ang_diff_deg(az, az2 % 360)) < 1e-6
        assert abs(el - el2) < 1e-6
        assert abs(r - r2) < 1e-6


def test_known_vectors():
    # due North, level
    az, el, r = world.cartesian_to_spherical(np.array([0.0, 100.0, 0.0]))
    assert abs(az) < 1e-9 and abs(el) < 1e-9 and abs(r - 100) < 1e-9
    # due East, 45 deg up
    az, el, r = world.cartesian_to_spherical(np.array([100.0, 0.0, 100.0]))
    assert abs(az - 90) < 1e-9 and abs(el - 45) < 1e-6
    # straight up
    az, el, r = world.cartesian_to_spherical(np.array([0.0, 0.0, 500.0]))
    assert abs(el - 90) < 1e-6


def test_jacobian_numerical():
    az, el, r = 37.0, 12.0, 2500.0
    J = world.measurement_jacobian(az, el, r)
    h = 1e-6
    # d/drange
    num = (world.spherical_to_cartesian(az, el, r + h)
           - world.spherical_to_cartesian(az, el, r - h)) / (2 * h)
    assert np.allclose(num, J[:, 2], atol=1e-4)
    # d/daz
    num = (world.spherical_to_cartesian(az + np.degrees(h), el, r)
           - world.spherical_to_cartesian(az - np.degrees(h), el, r)) / (2 * h)
    assert np.allclose(num, J[:, 0], atol=1e-4)


def test_cov_mapping_psd():
    R = world.spherical_cov_to_cartesian(120.0, 5.0, 3000.0, 1.0, 1.0, 20.0)
    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0)
