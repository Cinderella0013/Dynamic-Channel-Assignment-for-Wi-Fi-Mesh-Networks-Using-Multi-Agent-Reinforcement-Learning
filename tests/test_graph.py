import numpy as np
from mesh.sim.topology import Topology
from mesh.sim.graph import build_radio_graph, common_channel, link_rate, link_capacity

LINE = Topology(pos=np.array([[0, 0], [30, 0], [60, 0], [90, 0]], float), gateways=(0,))


def _two_active():
    active = np.zeros((4, 4), bool)
    active[0, 1] = active[1, 0] = active[2, 3] = active[3, 2] = True
    return active


def test_common_channel():
    cc = common_channel(np.array([[0, 1], [1, 2], [3, 4], [0, 4]]))
    assert cc[0, 1] == 1 and cc[1, 2] == -1 and cc[2, 3] == 4 and cc[0, 3] == 0


def test_two_conflicting_links_share_airtime():
    rg = build_radio_graph(LINE)
    cc = common_channel(np.array([[0, 1]] * 4))
    rate = link_rate(rg, cc, None)
    cap = link_capacity(rg, cc, rate, _two_active())
    assert rate[0, 1] > 0 and np.isclose(cap[0, 1], rate[0, 1] / 2)


def test_different_channels_no_sharing():
    rg = build_radio_graph(LINE)
    cc = common_channel(np.array([[0, 1], [0, 1], [2, 3], [2, 3]]))
    rate = link_rate(rg, cc, None)
    assert np.isclose(link_capacity(rg, cc, rate, _two_active())[0, 1], rate[0, 1])


def test_external_noise_lowers_rate():
    rg = build_radio_graph(LINE)
    cc = common_channel(np.array([[0, 1]] * 4))
    noise = np.full(7, -np.inf)
    noise[1] = -70.0   # links use channel 1 (largest shared)
    assert link_rate(rg, cc, noise)[0, 1] < link_rate(rg, cc, None)[0, 1]


def test_contention_defaults_unchanged():
    rg = build_radio_graph(LINE)
    cc = common_channel(np.array([[0, 1]] * 4))
    rate = link_rate(rg, cc, None)
    assert np.allclose(link_capacity(rg, cc, rate, _two_active(), mac_eff=1.0, contention=0.0),
                       link_capacity(rg, cc, rate, _two_active()))


def test_contention_scales_by_transmitters():
    # 0->1 and 2->3 on one channel, in CS range: 2 transmitters -> eta = b / (1 + a)
    rg = build_radio_graph(LINE)
    cc = common_channel(np.array([[0, 1]] * 4))
    rate = link_rate(rg, cc, None)
    active = np.zeros((4, 4), bool)
    active[0, 1] = active[2, 3] = True
    cap = link_capacity(rg, cc, rate, active, mac_eff=0.8, contention=0.5)
    assert np.isclose(cap[0, 1], rate[0, 1] / 2 * 0.8 / 1.5)
    single = np.zeros((4, 4), bool)
    single[0, 1] = True
    assert np.isclose(link_capacity(rg, cc, rate, single, mac_eff=0.8, contention=0.5)[0, 1], rate[0, 1] * 0.8)
