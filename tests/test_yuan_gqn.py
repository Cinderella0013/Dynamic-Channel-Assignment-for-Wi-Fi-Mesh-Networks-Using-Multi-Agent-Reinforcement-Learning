import numpy as np
import torch
from mesh.rl.yuan_gqn import GQNNet, YuanAgent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.rl.agents_io import N_ACTIONS


def _batch(n=5):
    o = torch.randn(2, n, 17)
    m = torch.randn(2, n, 16)
    nbr = torch.arange(n).view(1, n, 1).repeat(2, 1, 4)
    nbr[..., 1] = (nbr[..., 0] + 1) % n
    valid = torch.ones(2, n, 4, dtype=torch.bool)
    valid[..., 2:] = False
    return o, m, nbr, valid


def test_shapes_and_attention_normalised():
    q, att = GQNNet()(*_batch())
    assert q.shape == (2, 5, N_ACTIONS) and att.shape == (2, 5, 4, 4)
    assert torch.allclose(att.sum(-1), torch.ones(2, 5, 4))


def test_attention_ignores_masked():
    _, att = GQNNet()(*_batch())
    assert (att[..., 2:] == 0).all()


def test_td_target_respects_done():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    ag = YuanAgent(batch=1, kl_lambda=0.0, seed=0)
    assert np.allclose(ag.td_target(obs, np.ones(20), obs, done=True), 1.0)
    assert not np.allclose(ag.td_target(obs, np.ones(20), obs, done=False), 1.0)


def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    ag = YuanAgent(batch=4, eps_start=1.0, eps_end=1.0, seed=0, lr=3e-3, seq_decode=False)
    for _ in range(400):
        ag.act(obs, explore=True)
        a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    ag.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (ag.last_action_idx[nongw] == 5).mean() > 0.8
