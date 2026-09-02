"""Sensor models: noise spec match, Pd monotonicity, RF silence, ID confusion."""
import numpy as np
import pytest

from skygym.config import RadarCfg, EOCfg, RFCfg
from skygym.sensors.radar import RadarSensor
from skygym.sensors.eo import EOSensor
from skygym.sensors.rf import RFSensor
from skygym.sensors.base import class_posterior, CLASSES
from skygym.world import spherical_to_cartesian

META = {"site_enu": (0.0, 0.0, 0.0), "true_class": "quad", "tx_on": True,
        "noise_scale": 1.0}


def test_radar_noise_matches_spec():
    """Monte-Carlo: empirical sigma ~= configured sigma at fixed geometry."""
    cfg = RadarCfg(rate_hz=1e9)  # fire as fast as we poll
    rng = np.random.default_rng(1)
    s = RadarSensor(cfg, rng)
    pos = spherical_to_cartesian(30.0, 10.0, 2000.0)
    az_true, el_true, r_true = 30.0, 10.0, 2000.0
    n = 4000
    azs, els, rs = [], [], []
    s._next_t = 0.0
    for i in range(n):
        t = i * 1.0
        d = s._observe(t, pos, META)
        if d is None:
            continue
        azs.append(d.az_deg); els.append(d.el_deg); rs.append(d.range_m)
    azs = np.array(azs); els = np.array(els); rs = np.array(rs)
    # detection rate at 2 km, quad RCS should be healthy but < 1
    det_rate = len(azs) / n
    assert 0.4 < det_rate <= 1.0, f"Pd at 2km = {det_rate}"
    sig_az_cfg = s.ang_sigma(2000.0)
    # empirical angular sigma (deg) within 20% of spec
    assert abs(np.std(azs) - sig_az_cfg) / sig_az_cfg < 0.25
    assert abs(np.std(els) - sig_az_cfg) / sig_az_cfg < 0.25
    sig_r_cfg = s.range_sigma(2000.0)
    assert abs(np.std(rs) - sig_r_cfg) / sig_r_cfg < 0.25
    # unbiased (no systematic drift)
    assert abs(np.mean(rs) - r_true) < 0.2 * sig_r_cfg


def test_pd_monotonic_decreasing_with_range():
    s = RadarSensor(RadarCfg(), np.random.default_rng(2))
    snrs = [s.snr_db(r, 0.05) for r in (500.0, 2000.0, 5000.0, 8000.0)]
    pds = [s.pd(v) for v in snrs]
    assert all(pds[i] > pds[i + 1] for i in range(len(pds) - 1)), pds
    assert pds[0] > 0.95
    assert pds[-1] < 0.5


def test_rcs_matters():
    s = RadarSensor(RadarCfg(), np.random.default_rng(3))
    assert s.snr_db(2000.0, 0.5) - s.snr_db(2000.0, 0.05) == pytest.approx(10.0)


def test_clutter_present_and_flagged():
    cfg = RadarCfg(clutter_rate_per_scan=5.0)
    s = RadarSensor(cfg, np.random.default_rng(4))
    far_pos = spherical_to_cartesian(10.0, 5.0, 20000.0)  # out of range
    n_flag = 0
    for i in range(50):
        dets = s._clutter(i * 1.0)
        for d in dets:
            assert d.from_clutter
            n_flag += 1
    assert n_flag > 50  # ~5/scan * 50


def test_rf_silent_when_tx_off():
    s = RFSensor(RFCfg(rate_hz=1e9), np.random.default_rng(5))
    s._next_t = 0.0
    pos = spherical_to_cartesian(45.0, 15.0, 1500.0)
    meta = dict(META, tx_on=False)
    for i in range(20):
        assert s._observe(i * 1.0, pos, meta) is None
    meta_on = dict(META, tx_on=True)
    hits = [s._observe(i * 1.0, pos, meta_on) for i in range(20)]
    assert sum(h is not None for h in hits) > 15


def test_rf_has_no_el_no_range():
    s = RFSensor(RFCfg(rate_hz=1e9), np.random.default_rng(6))
    s._next_t = 0.0
    pos = spherical_to_cartesian(200.0, 30.0, 900.0)
    d = s._observe(0.0, pos, META)
    assert d is not None
    assert np.isnan(d.el_deg) and np.isnan(d.range_m)
    assert abs(((d.az_deg - 200.0) + 180) % 360 - 180) < 12.0  # within ~4deg*3


def test_class_posterior_sums_and_concentrates():
    rng = np.random.default_rng(7)
    p_hi = class_posterior(rng, "quad", 0.95)
    p_lo = class_posterior(rng, "quad", 0.10)
    assert abs(sum(p_hi.values()) - 1.0) < 1e-6
    assert p_hi["quad"] > 0.85
    assert p_lo["quad"] < p_hi["quad"]
    assert set(p_lo.keys()) == set(CLASSES)


def test_eo_pixel_size_drives_pd():
    cfg = EOCfg(rate_hz=1e9, slaved=True, p50_pixel=4.0, pix_slope=1.0)
    s = EOSensor(cfg, np.random.default_rng(8))
    s._next_t = 0.0
    close = spherical_to_cartesian(10.0, 10.0, 150.0)   # big pixels
    far = spherical_to_cartesian(10.0, 10.0, 4000.0)    # sub-pixel blob
    hits_close = sum(s._observe(i * 0.1, close, META) is not None for i in range(60))
    hits_far = sum(s._observe(i * 0.1 + 100.0, far, META) is not None for i in range(60))
    assert hits_close > 45
    assert hits_far < 25
