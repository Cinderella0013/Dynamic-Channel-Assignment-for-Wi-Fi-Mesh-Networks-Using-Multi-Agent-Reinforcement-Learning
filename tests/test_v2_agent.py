import numpy as np
import torch
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.model.v2 import EGATv2Agent
from mesh.rl.train import train


def _env(s="S2", seed=0):
    env = MeshEnv(make_scenario(s), seed)
    return env, env.reset()


def test_act_sets_queue_preserve_and_respects_cap():
    env, obs = _env()
    ag = EGATv2Agent(seed=0, use_bc=False, device="cpu")
    for _ in range(5):
        a = ag.act(obs, explore=True)
        assert a.queue_preserve
        before = obs
        obs, _, info = env.step(a)
        changed = (a.channels[:, 1] != before.channels[:, 1]) | (a.next_hop != before.next_hop)
        assert changed.sum() <= 2   # k_max = ceil(0.1 * 20); children's radio 0 changes are not compared


def test_replay_stores_executed_action():
    env, obs = _env()
    ag = EGATv2Agent(seed=0, use_bc=False, batch=4, device="cpu")
    a = ag.act(obs, explore=True)
    nxt, r, info = env.step(a)
    ag.learn(obs, ag.last_action_idx, np.zeros(20), nxt, False)
    assert np.array_equal(ag.replay[-1]["a"], ag.last_action_idx)


def test_learns_constant_best_action_without_planner():
    env, obs = _env("S1")
    ag = EGATv2Agent(seed=0, use_bc=False, raes=False, migration_control=False, progress_mask=False,
                     batch=8, lr=3e-3, device="cpu")
    for _ in range(400):
        ag.act(obs, explore=True)
        a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, True)
    p = ag.prepare(obs)
    with torch.no_grad():
        out = ag._tensors([p], torch.zeros(1, 20, ag.cfg.hidden))
    best = out["chan_logits"][0].argmax(-1) * 4 + out["route_logits"][0].argmax(-1)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (best[nongw] == 5).float().mean() > 0.8


def test_v2_trains_and_roundtrips(tmp_path):
    p = str(tmp_path / "v2.pt")
    ag = train("egat_v2", steps=500, seed=0, n=10, out=p, log_every=10**9, bc_steps=200, bc_epochs=2, device="cpu")
    b = EGATv2Agent.load(p, device="cpu")
    obs = MeshEnv(make_scenario("S2", n=10), 5).reset()
    assert np.array_equal(ag.act(obs).next_hop, b.act(obs).next_hop)
