import numpy as np
import torch
from mesh.model.agent import EGATAgent, EGATv1Agent
from mesh.rl.train import train
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario


def test_v1_defaults():
    ag = EGATv1Agent(seed=0, device="cpu")
    c = ag.cfg
    assert c.use_kl_anchor and c.semi_mdp and c.lam_start == c.lam_end == 0.1 and c.lr_after_bc == 1e-4
    assert not EGATAgent(seed=0, device="cpu").cfg.use_kl_anchor


def test_kl_zero_at_bc_end():
    ag = train("egat_v1", steps=400, seed=0, n=10, log_every=10**9, bc_steps=400, bc_epochs=2, device="cpu")
    assert ag.phase == "ppo" and ag.bc_net is not None
    obs = MeshEnv(make_scenario("S2", n=10), 3).reset()
    p = ag.prepare(obs)
    h = torch.zeros(1, 10, ag.cfg.hidden)
    with torch.no_grad():
        a, b = ag._tensors([p], h), ag._bc_tensors([p], h)
    assert torch.allclose(a["route_logits"], b["route_logits"]) and torch.allclose(a["chan_logits"], b["chan_logits"])
    assert ag.opt.param_groups[0]["lr"] == 1e-4


def test_nstep_adv_cuts_at_done_and_bootstraps():
    ag = EGATv1Agent(seed=0, use_bc=False, device="cpu")
    r = np.ones((4, 1))
    v = np.zeros((4, 1))
    d = np.array([0., 0., 1., 0.])
    a = ag.nstep_adv(r, v, d, last_value=np.array([10.0]), n_steps=3)
    g = ag.cfg.gamma
    assert np.isclose(a[0, 0], 1 + g + g**2)          # 3 steps, ends at done (t=2) -> no bootstrap
    assert np.isclose(a[2, 0], 1.0)                   # done at t=2
    assert np.isclose(a[3, 0], 1 + g * 10.0)          # rollout end -> bootstrap last_value
