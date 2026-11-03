import numpy as np
import torch
from mesh.rl.li_gat_a2c import GATA2CNet, LiAgent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.rl.agents_io import N_ACTIONS


def test_shapes_and_mask():
    net = GATA2CNet().eval()
    x = torch.randn(1, 6, 33)
    adj = torch.eye(6, dtype=torch.bool)[None]
    mask = torch.ones(1, 6, N_ACTIONS, dtype=torch.bool)
    mask[..., 0] = False
    logits, v = net(x, adj, mask)
    assert logits.shape == (1, 6, N_ACTIONS) and v.shape == (1, 6)
    assert (torch.softmax(logits, -1)[..., 0] < 1e-6).all()


def test_adjacency_is_conflict_graph_with_self_loops():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    adj = LiAgent.adjacency(obs)
    assert adj.diagonal().all() and (adj == adj.T).all() and adj.sum() > 20


def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    ag = LiAgent(seed=0, lr=3e-3)
    for _ in range(400):
        ag.act(obs, explore=True)
        a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    ag.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (ag.last_action_idx[nongw] == 5).mean() > 0.8
