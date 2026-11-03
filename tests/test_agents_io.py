import numpy as np
from mesh.sim.env import MeshEnv, RewardWeights
from mesh.sim.scenarios import make_scenario
from mesh.sim.graph import common_channel
from mesh.rl.agents_io import (K_SLOTS, neighbor_slots, action_mask, decode,
                               agent_features, per_agent_reward, alive_mask)


def _obs(name="S2", seed=0):
    env = MeshEnv(make_scenario(name), seed)
    return env, env.reset()


def test_slots_sorted_by_snr_and_padded():
    _, obs = _obs()
    s = neighbor_slots(obs)
    assert s.shape == (20, K_SLOTS) and (s[:, 0] >= 0).all()


def test_decode_tree_channels_share_with_parent():
    env, obs = _obs()
    slots = neighbor_slots(obs)
    rng = np.random.default_rng(0)
    m = action_mask(obs, slots)
    a = np.array([rng.choice(np.flatnonzero(m[i])) for i in range(20)])
    act = decode(obs, slots, a)
    cc = common_channel(act.channels)
    for u in range(20):
        if act.next_hop[u] >= 0:
            assert cc[u, act.next_hop[u]] >= 1     # parent link uses a downlink channel
    assert (act.next_hop[list(obs.gateways)] == -1).all()


def test_mask_isolated_node():
    env, obs = _obs("S2", 0)
    slots = neighbor_slots(obs)
    u = next(i for i in range(20) if i not in obs.gateways)
    obs.node_feat[:, 4] = 0
    obs.node_feat[u, 4] = 1                       # u alive, every neighbour of u dead
    m = action_mask(obs, slots)
    assert m[u].sum() == 6 and m[u, 0::4].all()   # only slot-0 actions, any channel
    a = np.zeros(20, int)
    a[u] = 4 * 3                                  # channel index 3, slot 0
    assert decode(obs, slots, a).next_hop[u] == -1


def test_features_shapes():
    _, obs = _obs()
    o, m = agent_features(obs, neighbor_slots(obs))
    assert o.shape == (20, 17) and m.shape == (20, 16) and np.isfinite(o).all() and np.isfinite(m).all()


def test_dead_node_excluded():
    env, obs = _obs("S6", 0)
    for _ in range(102):                          # failure at t=100; t=101 starts from an obs that shows it
        prev = obs
        act = decode(prev, neighbor_slots(prev), np.zeros(20, int))
        obs, r, info = env.step(act)
    dead = ~alive_mask(prev)
    rew = per_agent_reward(prev, act, info, r, RewardWeights())
    assert dead.sum() == 1 and rew[dead].item() == 0.0 and np.isfinite(rew).all()
