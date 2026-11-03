import numpy as np
from mesh.rl.q_neutral import QNeutral, discretize
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario


def test_discretize_is_hashable_and_small():
    k = discretize(np.zeros(17), np.zeros(16))
    assert hash(k) is not None and len(k) == 7


def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    q = QNeutral(eps_start=1.0, eps_end=1.0, seed=0)
    for _ in range(300):
        q.act(obs, explore=True)
        a = q.last_action_idx
        q.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    q.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (q.last_action_idx[nongw] == 5).mean() > 0.8
