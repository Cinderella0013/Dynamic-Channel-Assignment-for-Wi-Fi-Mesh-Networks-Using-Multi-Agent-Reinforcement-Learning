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
    ag = LiAgent(seed=0, lr=3e-3, seq_decode=False)
    for _ in range(400):
        ag.act(obs, explore=True)
        a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    ag.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (ag.last_action_idx[nongw] == 5).mean() > 0.8


def test_zscore_bounded_on_constant_features():
    net = GATA2CNet().train()
    x = torch.zeros(1, 6, 33)
    adj = torch.eye(6, dtype=torch.bool)[None]
    mask = torch.ones(1, 6, N_ACTIONS, dtype=torch.bool)
    for _ in range(3000):
        net(x, adj, mask)                       # constant input: EMA variance -> 0
    x2 = x.clone()
    x2[0, 0, 4] = 1.0                           # rare event (e.g. a node dies)
    z = (x2 - net.mu) / net.std()
    assert net.normalize(x2).abs().max() <= 5.0 and torch.isfinite(z).all()


def test_learn_updates_input_stats_once():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    ag = LiAgent(seed=0)
    ag.act(obs, explore=True)
    before = ag.net.mu.clone()
    s, _ = ag._prep(obs)
    batch_mean = s[0].reshape(-1, s[0].shape[-1]).mean(0)
    ag.learn(obs, ag.last_action_idx, np.zeros(20), obs, done=False)
    assert torch.allclose(ag.net.mu, 0.99 * before + 0.01 * batch_mean, atol=1e-5)


def test_policy_keeps_exploring_after_one_episode():
    from mesh.rl.agents_io import per_agent_reward
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    ag = LiAgent(seed=0, device="cpu")
    for _ in range(200):
        act = ag.act(obs, explore=True)
        a = ag.last_action_idx
        nxt, r, info = env.step(act)
        ag.learn(obs, a, per_agent_reward(obs, act, info, r, env.w), nxt, info["done"])
        obs = nxt
    s, _ = ag._prep(MeshEnv(make_scenario("S2"), 1).reset())
    ag.net.eval()
    with torch.no_grad():
        p = torch.softmax(ag.net(*s)[0][0], -1)
    assert p.max(-1).values.mean() < 0.6     # pre-fix: ~0.9+ within 60 steps
