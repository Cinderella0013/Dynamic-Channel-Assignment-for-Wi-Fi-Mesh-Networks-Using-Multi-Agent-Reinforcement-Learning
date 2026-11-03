import numpy as np
from mesh.sim.radio import rx_power_dbm, snr_db, rate_mbps, NOISE_DBM


def test_rx_power_ns3_defaults():
    p = rx_power_dbm(np.array([0.5, 1.0, 10.0]))
    assert np.allclose(p, [20 - 46.6777, 20 - 46.6777, 20 - 46.6777 - 30])


def test_snr_adds_noise_linearly():
    assert np.isclose(snr_db(np.array([-64.0]))[0], 30.0)
    # equal extra noise doubles noise power -> -3.01 dB
    assert np.isclose(snr_db(np.array([-64.0]), NOISE_DBM)[0], 30.0 - 3.0103, atol=1e-3)


def test_rate_thresholds():
    assert np.allclose(rate_mbps(np.array([1.9, 2.0, 10.0, 40.0])), [0.0, 8.6, 25.8, 143.4])
