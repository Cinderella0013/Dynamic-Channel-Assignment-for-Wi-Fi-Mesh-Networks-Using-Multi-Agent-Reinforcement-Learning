import numpy as np
from mesh.sim.env import MeshEnv, RewardWeights
from mesh.sim.scenarios import make_scenario
from mesh.sim.graph import common_channel
from mesh.rl.agents_io import (K_SLOTS, neighbor_slots, action_mask, decode,
                               agent_features, per_agent_reward, alive_mask)


def _obs(name="S2", seed=0):
    env = MeshEnv(make_scenario(name), seed)
    return env, env.reset()


def test_slots_shape_and_padded():
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
    assert o.shape == (20, 18) and m.shape == (20, 20) and np.isfinite(o).all() and np.isfinite(m).all()


def test_dead_node_excluded():
    env, obs = _obs("S6", 0)
    for _ in range(102):                          # failure at t=100; t=101 starts from an obs that shows it
        prev = obs
        act = decode(prev, neighbor_slots(prev), np.zeros(20, int))
        obs, r, info = env.step(act)
    dead = ~alive_mask(prev)
    rew = per_agent_reward(prev, act, info, r, RewardWeights())
    assert dead.sum() == 1 and rew[dead].item() == 0.0 and np.isfinite(rew).all()


class _KoThroughInterface:
    """Ko's decisions expressed as RL actions: the interface must not cap a good policy."""

    def __init__(self):
        from mesh.baselines.ko import KoLocalCost
        self.ko = KoLocalCost()

    def act(self, obs):
        k = self.ko.act(obs)
        slots = neighbor_slots(obs)
        a = np.zeros(len(slots), int)
        for u in range(len(slots)):
            j = np.flatnonzero(slots[u] == k.next_hop[u])
            a[u] = (k.channels[u, 1] - 1) * K_SLOTS + (j[0] if len(j) else 0)
        return decode(obs, slots, a)


def test_interface_can_express_ko():
    from mesh.baselines.ko import KoLocalCost
    from mesh.eval.run import run_episode
    ko = np.mean([run_episode(MeshEnv(make_scenario("S2"), s), KoLocalCost())["reward"] for s in range(3)])
    via = np.mean([run_episode(MeshEnv(make_scenario("S2"), s), _KoThroughInterface())["reward"] for s in range(3)])
    assert via >= 0.95 * ko


def test_every_routable_node_has_a_closer_slot():
    _, obs = _obs("S2", 0)
    slots, hop = neighbor_slots(obs), obs.node_feat[:, 3]
    for u in range(20):
        if u not in obs.gateways and hop[u] < 20:
            assert any(hop[v] < hop[u] for v in slots[u] if v >= 0)


def test_parent_pays_for_children_following_its_channel():
    env, obs = _obs("S2", 0)
    slots = neighbor_slots(obs)
    m = action_mask(obs, slots)
    a = np.array([np.flatnonzero(m[i])[0] for i in range(20)])
    obs1, r, info = env.step(decode(obs, slots, a))
    act0 = decode(obs1, neighbor_slots(obs1), a)          # same choices: nothing changes
    p = next(int(v) for v in act0.next_hop if v >= 0)
    kids = np.flatnonzero(act0.next_hop == p)
    a2 = a.copy()
    a2[p] = ((a[p] // K_SLOTS + 1) % 6) * K_SLOTS + a[p] % K_SLOTS   # parent retunes its downlink only
    act = decode(obs1, neighbor_slots(obs1), a2)
    _, r2, info2 = env.step(act)
    zero = dict(info2, local_utility=np.zeros(20))
    rew = per_agent_reward(obs1, act, zero, 0.0, RewardWeights())
    assert np.isclose(rew[p], -0.1 * (1 + len(kids)))
    assert np.allclose(rew[kids], 0.0)


def test_gateway_parent_onehot_is_none():
    _, obs = _obs()
    o, _ = agent_features(obs, neighbor_slots(obs))
    for g in obs.gateways:
        assert o[g, -1] == 1.0 and o[g, -5:-1].sum() == 0


def test_sequential_channels_spread_identical_scores():
    from mesh.rl.agents_io import sequential_channels
    _, obs = _obs()
    slots = neighbor_slots(obs)
    j = np.zeros(20, int)
    c = sequential_channels(obs, slots, np.zeros((20, 6)), j)
    assert len(np.unique(c)) >= 3


def test_all_learners_greedy_channels_diverse():
    from mesh.rl.yuan_gqn import YuanAgent
    from mesh.rl.li_gat_a2c import LiAgent
    from mesh.rl.q_neutral import QNeutral
    _, obs = _obs()
    for ag in (YuanAgent(seed=0, device="cpu"), LiAgent(seed=0, device="cpu"), QNeutral(seed=0)):
        assert len(np.unique(ag.act(obs).channels[:, 1])) >= 3, type(ag).__name__
