import numpy as np

from mesh.baselines.ko import KoLocalCost
from mesh.baselines.oracle import LookaheadOracle, rollout
from mesh.sim.env import MeshEnv
from mesh.sim.radio import N_CHANNELS
from mesh.sim.scenarios import make_scenario


def test_oracle_action_valid_and_radio_consistent():
    for scen in ("S2", "S4"):  # S4 has node failures
        env = MeshEnv(make_scenario(scen), 0)
        obs = env.reset()
        o = LookaheadOracle(env)
        for _ in range(12):
            a = o.act(obs)
            assert a.channels.min() >= 0 and a.channels.max() < N_CHANNELS
            kids = np.flatnonzero(a.next_hop >= 0)
            assert (a.channels[kids, 0] == a.channels[a.next_hop[kids], 1]).all()
            obs, _, _ = env.step(a)


def test_oracle_round_not_worse_than_ko():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    o = LookaheadOracle(env)
    assert rollout(env, o.act(obs), 10) >= rollout(env, KoLocalCost().act(obs), 10) - 1e-9


def test_rollout_queue_preserve_flag_applies_to_candidates():
    from mesh.sim.env import Action
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    for _ in range(15):
        obs, _, _ = env.step(KoLocalCost().act(obs))  # build up queues
    a = KoLocalCost().act(obs)
    moved = Action(a.channels.copy(), a.next_hop.copy())
    moved.channels[:, 1] = a.channels[:, 1] % 6 + 1  # every downlink moves to another channel
    moved.channels[:, 0] = np.where(moved.next_hop >= 0, moved.channels[np.maximum(moved.next_hop, 0), 1], 0)
    kept = Action(moved.channels.copy(), moved.next_hop.copy(), True)
    assert rollout(env, moved, 10, queue_preserve=True) == rollout(env, kept, 10)
    assert rollout(env, moved, 10) != rollout(env, kept, 10)
