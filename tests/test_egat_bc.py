from mesh.rl.train import train
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario


def test_bc_imitates_ko_on_unseen_topology():
    ag = train("egat", steps=1200, seed=0, n=10, log_every=10**9, bc_steps=1200, bc_epochs=30,
               device="cpu")
    assert ag.phase == "ppo"
    obs = MeshEnv(make_scenario("S2", n=10), 99).reset()
    assert ag.agreement(obs) > 0.7

