import numpy as np
from mesh.sim.flows import hops_to_gateway, paths_from_next_hop, allocate
from mesh.sim.metrics import jain, p5, utility

ADJ = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)   # 0 - 1 - 2, gateway 0
ALIVE = np.ones(3, bool)


def test_hops():
    assert list(hops_to_gateway(ADJ, (0,), ALIVE)) == [0, 1, 2]
    dead1 = np.array([True, False, True])
    assert list(hops_to_gateway(ADJ, (0,), dead1)) == [0, -1, -1]


def test_paths_follow_next_hop():
    p = paths_from_next_hop(np.array([-1, 0, 1]), (0,), ADJ, ALIVE)
    assert p == [None, [1, 0], [2, 1, 0]]


def test_loop_is_dropped():
    p = paths_from_next_hop(np.array([-1, 2, 1]), (0,), ADJ, ALIVE)
    assert p[1] is None and p[2] is None


def test_unusable_hop_is_dropped():
    p = paths_from_next_hop(np.array([-1, 0, 0]), (0,), ADJ, ALIVE)  # 2->0 not a link
    assert p[2] is None and p[1] == [1, 0]


def test_allocate_bottleneck_shared():
    cap = np.zeros((3, 3))
    cap[1, 0] = 10.0
    cap[2, 1] = 100.0
    thr, load = allocate([None, [1, 0], [2, 1, 0]], np.array([0.0, 10.0, 10.0]), cap)
    assert np.allclose(thr, [0.0, 5.0, 5.0]) and load[1, 0] == 20.0


def test_metrics_zero_vector():
    z = np.zeros(4)
    assert jain(z) == 1.0 and p5(z) == 0.0 and np.isfinite(utility(z, 0.01))
    assert np.isclose(jain(np.array([1.0, 0.0])), 0.5)


def test_domain_allocate_work_conserving():
    # two co-channel links in CS range, each needing 30% airtime: both fit (static 1/k would cap at 50% each anyway);
    # each needing 70%: domain needs 140% -> scaled to eta / 1.4
    from mesh.sim.flows import domain_allocate
    from mesh.sim.graph import build_radio_graph, common_channel, link_rate
    from mesh.sim.topology import Topology
    top = Topology(pos=np.array([[0, 0], [30, 0], [60, 0], [90, 0]], float), gateways=(1, 3))
    rg = build_radio_graph(top)
    cc = common_channel(np.array([[0, 1]] * 4))
    rate = link_rate(rg, cc, None)
    paths = [[0, 1], None, [2, 3], None]
    r = rate[0, 1]
    thr, load, cap = domain_allocate(rg, cc, rate, paths, np.array([0.3 * r, 0, 0.3 * r, 0]), 1.0)
    assert np.allclose(thr[[0, 2]], 0.3 * r)
    thr, _, _ = domain_allocate(rg, cc, rate, paths, np.array([0.7 * r, 0, 0.7 * r, 0]), 0.8)
    assert np.allclose(thr[0], 0.7 * r * 0.8 / 1.4)
