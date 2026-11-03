import numpy as np
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.model.agent import EGATAgent


def test_gae_cuts_at_done():
    ag = EGATAgent(seed=0, use_bc=False, device="cpu")
    r = np.ones((3, 2))
    v = np.zeros((3, 2))
    d = np.array([0.0, 1.0, 0.0])
    adv, ret = ag.gae(r, v, d, last_value=np.full(2, 100.0))
    assert np.allclose(ret[1], 1.0)                      # episode ended at t=1: no bootstrap
    assert ret[2, 0] > 50                                # t=2 bootstraps from last_value


def test_channel_logprob_only_on_decision_steps():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, rollout=4, minibatch=4, device="cpu")
    flags = []
    for _ in range(4):
        act = ag.act(obs, explore=True)
        flags.append(ag._pending["is_chan_step"])
        nxt, r, info = env.step(act)
        ag.learn(obs, ag.last_action_idx, np.zeros(20), nxt, False)
        obs = nxt
    assert flags == [True, False, False, False]


def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, use_ko_prior=False, two_timescale=False,
                   progress_mask=False, rollout=20, minibatch=10, lr=3e-3, device="cpu")
    for t in range(600):
        ag.act(obs, explore=True)
        a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    ag.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (ag.last_action_idx[nongw] == 5).mean() > 0.8


def test_reward_lam_anneals():
    ag = EGATAgent(seed=0, device="cpu")
    assert ag.reward_lam(0, 100) == 0.5 and abs(ag.reward_lam(100, 100) - 0.1) < 1e-9
