import numpy as np
from mesh.sim.env import MeshEnv, Action, edge_matrix
from mesh.sim.scenarios import make_scenario
from mesh.rl.agents_io import neighbor_slots, K_SLOTS
from mesh.model.v2 import raes, refine


def _obs(s="S2", seed=0, steps=3):
    env = MeshEnv(make_scenario(s), seed)
    obs = env.reset()
    for _ in range(steps):
        obs, _, _ = env.step(Action(obs.channels.copy(), obs.next_hop.copy()))
    return obs


def _mask(obs, slots):
    from mesh.model.agent import EGATAgent
    return EGATAgent(seed=0, use_bc=False, device="cpu").progress_mask(obs, slots)


def test_raes_spreads_channels_and_respects_route_mask():
    obs = _obs()
    slots = neighbor_slots(obs)
    rm = _mask(obs, slots)
    c, j, sc = raes(obs, slots, np.zeros((20, 6)), np.zeros((20, K_SLOTS)), rm)
    assert len(np.unique(c)) >= 3 and rm[np.arange(20), j].all() and np.isfinite(sc).all()


def test_raes_saturated_neighbourhood():
    obs = _obs()
    slots = neighbor_slots(obs)
    rm = _mask(obs, slots)
    q = np.zeros((20, 6))
    q[:, 0] = 10.0                                         # everyone loves channel 1
    c, j, sc = raes(obs, slots, q, np.zeros((20, K_SLOTS)), rm, a=100.0)
    assert np.isfinite(sc).all() and len(np.unique(c)) >= 2


def test_migration_cost_keeps_current_channel_when_tied():
    obs = _obs()
    slots = neighbor_slots(obs)
    rm = _mask(obs, slots)
    c, _, _ = raes(obs, slots, np.zeros((20, 6)), np.zeros((20, K_SLOTS)), rm, a=0.0, b=0.0, g=1.0)
    has_backlog = obs.node_feat[:, -1] > 0
    assert has_backlog.any()
    assert (c[has_backlog] == obs.channels[has_backlog, 1] - 1).all()


def test_refinement_can_reject_everything():
    obs = _obs()
    slots = neighbor_slots(obs)
    rm = _mask(obs, slots)
    c, j, sc = raes(obs, slots, np.random.default_rng(0).normal(size=(20, 6)), np.zeros((20, K_SLOTS)), rm)
    c2, j2, moved = refine(obs, slots, c, j, sc, np.zeros((20, K_SLOTS)), np.full(20, 3))
    assert not moved.any() and (c2 == obs.channels[:, 1] - 1).all()


def test_refinement_caps_moves_and_separates_conflicts():
    obs = _obs()
    slots = neighbor_slots(obs)
    rm = _mask(obs, slots)
    q = np.random.default_rng(1).normal(size=(20, 6)) * 5
    c, j, sc = raes(obs, slots, q, np.zeros((20, K_SLOTS)), rm, g=0.0)
    _, _, moved = refine(obs, slots, c, j, sc, np.zeros((20, K_SLOTS)), np.zeros(20, int))
    conf = edge_matrix(obs, "is_conflict") > 0
    assert 0 < moved.sum() <= 2 and not (conf & np.outer(moved, moved)).any()
