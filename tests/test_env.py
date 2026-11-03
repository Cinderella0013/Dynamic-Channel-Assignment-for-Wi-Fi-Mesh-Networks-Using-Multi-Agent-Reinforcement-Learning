import numpy as np
import pytest
from mesh.sim.env import MeshEnv, Action
from mesh.sim.scenarios import make_scenario


def _hold(obs):
    return Action(obs.channels.copy(), obs.next_hop.copy())


def test_episode_runs_and_is_deterministic():
    def run(seed):
        env = MeshEnv(make_scenario("S2"), seed)
        obs = env.reset()
        out = []
        for _ in range(200):
            obs, r, info = env.step(_hold(obs))
            out.append(r)
        assert info["done"]
        return np.array(out)
    a, b = run(3), run(3)
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_observation_shapes():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    assert obs.node_feat.shape == (20, 5) and obs.edge_attr.shape[1] == 6
    assert obs.edge_index.shape[0] == 2 and obs.edge_index.shape[1] == obs.edge_attr.shape[0]


def test_channel_switch_counted_and_penalized():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    a = _hold(obs)
    a.channels[:, 1] = 2
    _, _, info = env.step(a)
    assert info["n_channel_switch"] == 20


def test_bad_action_raises():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    a = _hold(obs)
    a.channels[0, 0] = 7
    with pytest.raises(ValueError):
        env.step(a)
    with pytest.raises(ValueError):
        env.step(Action(obs.channels[:5], obs.next_hop))
    _, _, info = env.step(_hold(obs))      # state unchanged: still usable
    assert np.isfinite(info["utility"])


def test_failure_cuts_off_node():
    env = MeshEnv(make_scenario("S6"), 0)
    obs = env.reset()
    for _ in range(101):
        obs, r, info = env.step(_hold(obs))   # routes NOT repaired
    assert obs.node_feat[:, 4].sum() == 19 and np.isfinite(r)
    assert info["n_drop"] >= 1                     # someone routed through the dead node


def _m0_env(name, seed=0):
    from mesh.baselines.m0 import M0
    env = MeshEnv(make_scenario(name), seed)
    return env, env.reset(), M0()


def test_idle_routes_use_no_airtime():
    probe, obs, m0 = _m0_env("S2")
    probe.step(m0.act(obs))
    relays = np.zeros(20, bool)
    relays[probe.next_hop[probe.next_hop >= 0]] = True
    idle = (probe.demand == 0) & ~probe.is_gw & ~relays   # unroute idle leaves only
    assert idle.any()
    a_env, obs_a, _ = _m0_env("S2")
    b_env, obs_b, _ = _m0_env("S2")
    a = m0.act(obs_a)
    b = m0.act(obs_b)
    b.next_hop[idle] = -1
    _, _, ia = a_env.step(a)
    _, _, ib = b_env.step(b)
    assert np.allclose(ia["throughput"][~idle], ib["throughput"][~idle])


def test_metrics_count_only_active_flows():
    from mesh.eval.run import run_episode
    from mesh.baselines.m0 import M0
    assert run_episode(MeshEnv(make_scenario("S1"), 0), M0())["p5"] > 0


def test_s4_interference_hurts_m0():
    from dataclasses import replace
    from mesh.eval.run import run_episode
    from mesh.baselines.m0 import M0
    cfg = make_scenario("S4")
    hit = run_episode(MeshEnv(cfg, 0), M0())["aggregate"]
    clean = run_episode(MeshEnv(replace(cfg, ext_interf=None), 0), M0())["aggregate"]
    assert hit < clean


def test_dropping_own_route_never_raises_local_utility():
    probe, obs, m0 = _m0_env("S2")
    probe.step(m0.act(obs))
    a0 = m0.act(obs)
    relays = set(a0.next_hop[a0.next_hop >= 0])
    s = next(u for u in range(20) if probe.demand[u] > 0 and u not in relays and not probe.is_gw[u])
    a_env, obs_a, _ = _m0_env("S2")
    b_env, obs_b, _ = _m0_env("S2")
    b = m0.act(obs_b)
    b.next_hop[s] = -1
    _, _, ia = a_env.step(m0.act(obs_a))
    _, _, ib = b_env.step(b)
    assert ib["local_utility"][s] <= ia["local_utility"][s]
    assert ib["n_drop"] == ia["n_drop"] + 1
