"""Static radio graph + per-step channel-aware link rate and airtime share."""
from dataclasses import dataclass

import numpy as np

from mesh.sim.radio import CS_THRESHOLD_DBM, N_CHANNELS, rate_mbps, rx_power_dbm, snr_db
from mesh.sim.topology import distances, link_adjacency


@dataclass(frozen=True)
class RadioGraph:
    rx_dbm: np.ndarray    # (n, n)
    link: np.ndarray      # (n, n) bool
    conflict: np.ndarray  # (n, n) bool, rx >= CS threshold, diag False


def build_radio_graph(top):
    rx = rx_power_dbm(distances(top.pos))
    conflict = rx >= CS_THRESHOLD_DBM
    np.fill_diagonal(conflict, False)
    return RadioGraph(rx_dbm=rx, link=link_adjacency(top.pos), conflict=conflict)


def common_channel(channels):
    """Largest shared channel index per node pair, -1 if none.

    Largest, not smallest: a common control channel on radio 0 (index 0) must
    not swallow every link when radio 1 offers a dedicated channel.
    """
    n = len(channels)
    cc = np.full((n, n), -1, dtype=int)
    for c in range(N_CHANNELS):
        has = (channels == c).any(axis=1)
        cc[has[:, None] & has[None, :]] = c
    return cc


def link_rate(rg, cc, ext_noise_dbm):
    usable = rg.link & (cc >= 0)
    extra = None
    if ext_noise_dbm is not None:
        extra = np.where(usable, np.asarray(ext_noise_dbm, dtype=float)[np.maximum(cc, 0)], -np.inf)
    rate = rate_mbps(snr_db(rg.rx_dbm, extra))
    return np.where(usable, rate, 0.0)


def link_capacity(rg, cc, rate, active):
    # ponytail: count-based airtime share (1/k). Upper bound on real CSMA loss;
    # swap for clique-based share if Plan 2 calibration error > 15%.
    cap = rate.copy()
    sym = active | active.T
    us, vs = np.nonzero(np.triu(sym, 1))
    near = rg.conflict | np.eye(len(rate), dtype=bool)
    for i, (u, v) in enumerate(zip(us, vs)):
        k = 1
        for j, (x, y) in enumerate(zip(us, vs)):
            if i != j and cc[x, y] == cc[u, v] and near[np.ix_((u, v), (x, y))].any():
                k += 1
        cap[u, v] = cap[v, u] = rate[u, v] / k
    return cap
