import numpy as np
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.sim.graph import common_channel
from mesh.rl.agents_io import neighbor_slots, decode
from mesh.model.agent import EGATAgent
from mesh.model.ko_prior import ko_action_idx
from mesh.baselines.ko import KoLocalCost


def _env(s="S2", seed=0):
    env = MeshEnv(make_scenario(s), seed)
    return env, env.reset()


def test_progress_mask_routes_are_loop_free():
    env, obs = _env()
    ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, device="cpu")
    for _ in range(30):
        act = ag.act(obs, explore=True)
        for u in range(20):
            v, k = u, 0
            while act.next_hop[v] >= 0 and k <= 20:
                v, k = act.next_hop[v], k + 1
            assert k <= 20
        obs, _, _ = env.step(act)


def test_two_timescale_holds_channels_off_decision_steps():
    env, obs = _env()
    ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, device="cpu")
    obs, _, _ = env.step(ag.act(obs, explore=True))           # t=0 decision step
    for _ in range(8):                                          # t=1..8: no channel decisions
        before = obs.channels[:, 1].copy()
        obs, _, _ = env.step(ag.act(obs, explore=True))
        assert (obs.channels[:, 1] == before).all()


def test_fallback_always_equals_ko():
    env, obs = _env()
    ag = EGATAgent(seed=0, use_bc=False, fallback_entropy=-1.0, device="cpu")
    slots = neighbor_slots(obs)
    a = ag.act(obs, explore=False)
    ko = decode(obs, slots, ko_action_idx(obs, slots, KoLocalCost()))
    assert np.array_equal(a.next_hop, ko.next_hop) and np.array_equal(a.channels, ko.channels)
    assert ag.fallback_rate == 1.0


def test_fallback_merge_keeps_tree_channels():
    env, obs = _env()
    ag = EGATAgent(seed=0, use_bc=False, fallback_entropy=2.5, device="cpu")
    a = ag.act(obs, explore=False)
    cc = common_channel(a.channels)
    for u in range(20):
        if a.next_hop[u] >= 0:
            assert cc[u, a.next_hop[u]] >= 1


def test_gru_state_resets_each_episode():
    env, obs = _env()
    ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, device="cpu")
    first = ag.act(obs).next_hop.copy()
    for _ in range(5):
        obs, _, _ = env.step(ag.act(obs))
    assert np.array_equal(ag.act(env.reset()).next_hop, first)
