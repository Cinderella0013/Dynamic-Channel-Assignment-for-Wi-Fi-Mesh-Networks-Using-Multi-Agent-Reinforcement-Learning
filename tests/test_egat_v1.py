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
