import numpy as np
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.baselines.m0 import M0
from mesh.baselines.ko import KoLocalCost


def test_m0_routes_reach_gateway():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    a = M0().act(obs)
    assert (a.channels == [0, 1]).all()
    _, _, info = env.step(a)
    assert info["n_drop"] == 0


def test_ko_diversifies_channels_and_keeps_radio0():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    a = KoLocalCost().act(obs)
    assert (a.channels[:, 0] == 0).all()
    assert len(np.unique(a.channels[:, 1])) >= 3


def test_ko_routes_loop_free():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    ko = KoLocalCost()
    for _ in range(20):
        obs, _, info = env.step(ko.act(obs))
    assert info["n_drop"] == 0
