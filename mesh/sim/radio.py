"""PHY model: log-distance path loss (ns-3 defaults), SNR, 802.11ax MCS rate."""
import numpy as np

TX_POWER_DBM = 20.0
NOISE_DBM = -94.0
PL_EXPONENT = 3.0
PL_REF_DB = 46.6777  # ns3::LogDistancePropagationLossModel ReferenceLoss at 1 m
LINK_THRESHOLD_DBM = -76.0
CS_THRESHOLD_DBM = -82.0
CHANNELS = (1, 6, 11, 36, 40, 44, 48)
N_CHANNELS = len(CHANNELS)

# 802.11ax, 20 MHz, 1 SS, GI 0.8 us
MCS_SNR_DB = (2, 5, 9, 11, 15, 18, 20, 25, 29, 31, 34, 37)
MCS_RATE_MBPS = (8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.4)


def rx_power_dbm(d_m):
    d = np.maximum(np.asarray(d_m, dtype=float), 1.0)
    return TX_POWER_DBM - PL_REF_DB - 10 * PL_EXPONENT * np.log10(d)


def snr_db(rx_dbm, extra_noise_dbm=None):
    noise_mw = 10 ** (NOISE_DBM / 10)
    if extra_noise_dbm is not None:
        noise_mw = noise_mw + 10 ** (np.asarray(extra_noise_dbm, dtype=float) / 10)
    return np.asarray(rx_dbm, dtype=float) - 10 * np.log10(noise_mw)


def rate_mbps(snr):
    idx = np.searchsorted(MCS_SNR_DB, np.asarray(snr, dtype=float), side="right") - 1
    table = np.array((0.0,) + MCS_RATE_MBPS)
    return table[idx + 1]
