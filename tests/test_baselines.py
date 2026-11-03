import numpy as np
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.baselines.m0 import M0
from mesh.baselines.ko import KoLocalCost
from mesh.sim.graph import common_channel


def test_m0_routes_reach_gateway():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    a = M0().act(obs)
    assert (a.channels == [0, 1]).all()
    _, _, info = env.step(a)
    assert info["n_drop"] == 0


def test_ko_parent_links_use_diverse_channels():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    a = KoLocalCost().act(obs)
    cc = common_channel(a.channels)
    kids = [u for u in range(20) if a.next_hop[u] >= 0]
    link_ch = [cc[u, a.next_hop[u]] for u in kids]
    assert min(link_ch) >= 0                 # every tree link has a shared channel
    assert len(set(link_ch)) >= 3            # and the tree uses several channels


def test_ko_routes_stable():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    ko = KoLocalCost()
    changes = []
    for _ in range(100):
        obs, _, info = env.step(ko.act(obs))
        changes.append(info["n_route_change"])
    assert np.mean(changes[10:]) < 1.0


def test_ko_routes_loop_free():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    ko = KoLocalCost()
    for _ in range(20):
        obs, _, info = env.step(ko.act(obs))
    assert info["n_drop"] == 0
