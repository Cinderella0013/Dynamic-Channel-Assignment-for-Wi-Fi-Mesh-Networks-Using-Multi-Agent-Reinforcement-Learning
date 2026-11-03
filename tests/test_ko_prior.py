import numpy as np
from mesh.sim.env import MeshEnv, RewardWeights
from mesh.sim.scenarios import make_scenario
from mesh.baselines.ko import KoLocalCost
from mesh.rl.agents_io import neighbor_slots, decode, per_agent_reward, K_SLOTS
from mesh.model.ko_prior import ko_slot_costs, ko_channel_costs, ko_action_idx


def _obs(s="S2", seed=0):
    env = MeshEnv(make_scenario(s), seed)
    return env, env.reset()


def test_slot_cost_order_matches_ko_via():
    # Ko's hysteresis may keep an older parent, so compare cost ORDER with Ko's via matrix
    from mesh.baselines.ko import ko_route_via
    _, obs = _obs()
    slots = neighbor_slots(obs)
    alive = obs.node_feat[:, 4] > 0
    via, cost = ko_route_via(obs, alive), ko_slot_costs(obs, slots)
    for u in range(20):
        ok = np.isfinite(cost[u])
        if ok.sum() > 1:
            assert np.array_equal(np.argsort(cost[u][ok], kind="stable"),
                                  np.argsort(via[u, slots[u][ok]], kind="stable"))


def test_costs_shapes_and_invalid_inf():
    _, obs = _obs()
    slots = neighbor_slots(obs)
    sc, cc = ko_slot_costs(obs, slots), ko_channel_costs(obs)
    assert sc.shape == (20, K_SLOTS) and cc.shape == (20, 6) and np.isfinite(cc).all()
    assert np.isinf(sc[slots < 0]).all()


def test_ko_action_idx_reproduces_ko_through_interface():
    env, obs = _obs()
    slots = neighbor_slots(obs)
    a = ko_action_idx(obs, slots, KoLocalCost())
    act = decode(obs, slots, a)
    raw = KoLocalCost().act(obs)
    assert (act.channels[:, 1] == raw.channels[:, 1]).all()
    assert (act.next_hop == raw.next_hop).mean() > 0.9


def test_lam_default_matches_old_reward():
    env, obs = _obs()
    slots = neighbor_slots(obs)
    act = decode(obs, slots, np.zeros(20, int))
    _, r, info = env.step(act)
    a = per_agent_reward(obs, act, info, r, RewardWeights())
    b = per_agent_reward(obs, act, info, r, RewardWeights(), lam=0.5)
    c = per_agent_reward(obs, act, info, r, RewardWeights(), lam=0.1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
