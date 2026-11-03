import numpy as np
import pytest
from mesh.sim.env import MeshEnv, Action
from mesh.sim.scenarios import make_scenario


def _hold(obs):
    return Action(obs.channels.copy(), obs.next_hop.copy())


def test_reset_with_new_seed_changes_topology():
    env = MeshEnv(make_scenario("S1"), 0)
    a = env.reset().edge_attr
    b = env.reset(seed=1).edge_attr
    assert env.seed == 1 and not (a.shape == b.shape and np.array_equal(a, b))
    assert np.array_equal(env.reset(seed=0).edge_attr, a)


def test_no_interferer_lookahead():
    env = MeshEnv(make_scenario("S4"), 0)
    obs = env.reset()
    for _ in range(50):                          # steps t = 0..49 executed, interferer starts at t = 50
        obs, _, _ = env.step(_hold(obs))
    before = obs.edge_attr[:, 3].copy()
    obs, _, _ = env.step(_hold(obs))             # t = 50 executed with interferer
    assert before.sum() > obs.edge_attr[:, 3].sum()


def test_unreachable_hop_feature_is_n():
    env = MeshEnv(make_scenario("S1"), 0)
    env.reset()
    env.alive[:] = False
    env.alive[list(env.gw)] = True               # every non-gateway unreachable
    obs = env._observe(None)
    nongw = [i for i in range(20) if i not in env.gw]
    assert (obs.node_feat[nongw, 3] == 20).all() and (obs.node_feat[list(env.gw), 3] == 0).all()


def test_step_after_done_raises():
    cfg = make_scenario("S1")
    env = MeshEnv(cfg, 0)
    obs = env.reset()
    for _ in range(cfg.episode_len):
        obs, _, _ = env.step(_hold(obs))
    with pytest.raises(RuntimeError):
        env.step(_hold(obs))


def test_out_of_range_next_hop_not_a_route_change():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    u = next(i for i in range(20) if i not in obs.gateways)
    a = _hold(obs)
    a.next_hop[u] = -1                       # unroute u (one real route change)
    obs, _, _ = env.step(a)
    b = _hold(obs)
    b.next_hop[u] = -5                       # same meaning as -1: not a change
    _, _, info = env.step(b)
    assert info["n_route_change"] == 0
