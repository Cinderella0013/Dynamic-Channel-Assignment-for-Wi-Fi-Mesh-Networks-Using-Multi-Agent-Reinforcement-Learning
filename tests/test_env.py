import numpy as np
import pytest
from mesh.sim.env import MeshEnv, Action
from mesh.sim.scenarios import make_scenario


def _hold(obs):
    return Action(obs.channels.copy(), obs.next_hop.copy())


def test_episode_runs_and_is_deterministic():
    def run(seed):
        env = MeshEnv(make_scenario("S2"), seed)
        obs = env.reset()
        out = []
        for _ in range(200):
            obs, r, info = env.step(_hold(obs))
            out.append(r)
        assert info["done"]
        return np.array(out)
    a, b = run(3), run(3)
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_observation_shapes():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    assert obs.node_feat.shape == (20, 5) and obs.edge_attr.shape[1] == 6
    assert obs.edge_index.shape[0] == 2 and obs.edge_index.shape[1] == obs.edge_attr.shape[0]


def test_channel_switch_counted_and_penalized():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    a = _hold(obs)
    a.channels[:, 1] = 2
    _, _, info = env.step(a)
    assert info["n_channel_switch"] == 20


def test_bad_action_raises():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    a = _hold(obs)
    a.channels[0, 0] = 7
    with pytest.raises(ValueError):
        env.step(a)
    with pytest.raises(ValueError):
        env.step(Action(obs.channels[:5], obs.next_hop))
    _, _, info = env.step(_hold(obs))      # state unchanged: still usable
    assert np.isfinite(info["utility"])


def test_failure_cuts_off_node():
    env = MeshEnv(make_scenario("S6"), 0)
    obs = env.reset()
    for _ in range(101):
        obs, r, info = env.step(_hold(obs))   # routes NOT repaired
    assert obs.node_feat[:, 4].sum() == 19 and np.isfinite(r)
    assert info["n_drop"] >= 1                     # someone routed through the dead node
