import numpy as np
import pytest

from mesh.eval.run import run_episode
from mesh.rl.agents_io import K_SLOTS, N_ACTIONS, hold_bonus, neighbor_slots
from mesh.rl.li_gat_a2c import LiAgent
from mesh.rl.yuan_gqn import YuanAgent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario


def test_hold_bonus_marks_current_parent_and_channel():
    from mesh.baselines.ko import KoLocalCost
    env, ko = MeshEnv(make_scenario("S2"), 3), KoLocalCost()
    obs = env.reset()
    for _ in range(2):  # Ko sets real channels and parents
        obs, _, _ = env.step(ko.act(obs))
    assert (obs.next_hop >= 0).any()
    slots = neighbor_slots(obs)
    b = hold_bonus(obs, slots, 2.0)
    assert b.shape == (len(slots), N_ACTIONS)
    c = np.arange(N_ACTIONS) // K_SLOTS
    j = np.arange(N_ACTIONS) % K_SLOTS
    for u in range(len(slots)):
        keep_c = c == obs.channels[u, 1] - 1
        keep_j = (slots[u, j] == obs.next_hop[u]) & (obs.next_hop[u] >= 0)
        assert np.allclose(b[u], 2.0 * keep_c + 2.0 * keep_j)
    assert not hold_bonus(obs, slots, 0.0).any()


def _episode(cls, margin):
    a = cls(seed=0, device="cpu")
    a.switch_margin = margin
    r = run_episode(MeshEnv(make_scenario("S2"), 3), a)
    return r["n_channel_switch"] + r["n_route_change"]


@pytest.mark.parametrize("cls", [LiAgent, YuanAgent])
def test_baseline_margin_default_off_and_holds(cls):
    assert cls(seed=0, device="cpu").switch_margin == 0.0
    assert _episode(cls, 0.0) == _episode(cls, 0.0)
    assert _episode(cls, 1e4) < 0.5 * _episode(cls, 0.0)


def test_suite_agent_set_reaches_baselines(tmp_path):
    from mesh.eval.suite import _factory
    ck = tmp_path / "li.pt"
    LiAgent(seed=0, device="cpu").save(ck)
    _, make = _factory(f"li_h={ck}", {"switch_margin": 2.0})
    assert make().switch_margin == 2.0
