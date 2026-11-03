import numpy as np
from dataclasses import replace
from mesh.sim.env import MeshEnv, Action
from mesh.sim.scenarios import make_scenario
from mesh.eval.run import run_episode
from mesh.baselines.m0 import M0
from mesh.baselines.ko import KoLocalCost


def _hold(obs, **kw):
    return Action(obs.channels.copy(), obs.next_hop.copy(), **kw)


def test_backlog_carries_unmet_demand():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    obs, _, info = env.step(_hold(obs))
    assert np.allclose(env.backlog, np.clip(env.demand - info["throughput"], 0, 5 * 10.0))   # t=0: offered = demand


def test_zero_downtime_no_interruption():
    cfg = replace(make_scenario("S2"), downtime_steps=0)
    env = MeshEnv(cfg, 0)
    obs = env.reset()
    a = _hold(obs)
    a.channels[:, 1] = 2
    _, _, info = env.step(a)
    assert info["lost"] == 0.0 and info["aggregate"] > 0


def test_switch_interrupts_and_loses_queue_unless_preserved():
    def run(preserve):
        env = MeshEnv(make_scenario("S2"), 0)
        obs = env.reset()
        for _ in range(5):
            obs, _, _ = env.step(_hold(obs))              # build up backlog
        a = _hold(obs, queue_preserve=preserve)
        a.channels[:, 1] = 3
        a.channels[:, 0] = np.where(obs.next_hop >= 0, 3, 0)
        _, _, info = env.step(a)
        return info
    assert run(False)["lost"] > run(True)["lost"]


def test_switch_loss_only_when_backlog():
    env = MeshEnv(make_scenario("S1"), 0)
    obs = env.reset()
    a = _hold(obs)
    a.channels[:, 1] = 2
    _, _, info = env.step(a)                               # t=0: backlog empty
    assert info["lost"] == 0.0


def test_dead_node_backlog_counted_once():
    env = MeshEnv(make_scenario("S6"), 0)
    obs = env.reset()
    losses = []
    for _ in range(105):
        obs, _, info = env.step(_hold(obs))
        losses.append(info["lost"])
    dead = int(np.flatnonzero(~env.alive)[0])
    assert env.backlog[dead] == 0.0 and np.isfinite(losses).all()


def test_loss_rate_reported_and_ko_pays_for_switching():
    m0 = run_episode(MeshEnv(make_scenario("S2"), 0), M0())
    ko = run_episode(MeshEnv(make_scenario("S2"), 0), KoLocalCost())
    assert 0.0 <= m0["loss_rate"] <= 1.0 and 0.0 <= ko["loss_rate"] <= 1.0


def test_utility_uses_one_second_window():
    # per node: mean throughput over the window steps in which it had something to send;
    # flow set: alive non-gateways with demand at some step of the window (method-independent)
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    thr, act, dem = [], [], []
    for _ in range(14):
        obs, _, info = env.step(_hold(obs))
        thr.append(info["throughput"]); act.append(env.offered > 0); dem.append(env.demand > 0)
    T, A = np.array(thr[-10:]), np.array(act[-10:])
    cnt = A.sum(0)
    win = np.divide((T * A).sum(0), cnt, out=np.zeros(20), where=cnt > 0)
    flows = np.array(dem[-10:]).any(0) & env.alive & ~env.is_gw
    assert np.isclose(info["utility"], np.log(win[flows] + env.w.eps).sum())
    assert info["n_flows"] == flows.sum()


def test_flow_set_is_method_independent():
    from mesh.baselines.ko import KoLocalCost
    from mesh.baselines.m0 import M0
    counts = []
    for P in (M0, KoLocalCost):
        env = MeshEnv(make_scenario("S2"), 0)
        obs = env.reset()
        pol, c = P(), []
        for _ in range(30):
            obs, _, info = env.step(pol.act(obs))
            c.append(info["n_flows"])
        counts.append(c)
    assert counts[0] == counts[1]


def test_interruptions_are_not_drops():
    from mesh.baselines.ko import KoLocalCost
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    ko, drops, intr = KoLocalCost(), 0, 0
    for _ in range(60):
        obs, _, info = env.step(ko.act(obs))
        drops += info["n_drop"]; intr += info["n_interrupted"]
    assert intr > 0 and drops == 0


def test_one_step_outage_is_not_starvation():
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    for _ in range(12):
        obs, _, before = env.step(_hold(obs))
    a = _hold(obs, queue_preserve=True)
    a.channels[:, 1] = 3
    a.channels[:, 0] = np.where(obs.next_hop >= 0, 3, 0)
    _, _, during = env.step(a)                           # every flow interrupted for this step
    assert during["utility"] > before["utility"] - 0.2 * abs(before["utility"]) - 5


def test_features_include_backlog_and_ack():
    from mesh.sim.env import NODE_FEATS, EDGE_FEATS
    from mesh.rl.agents_io import agent_features, neighbor_slots, F_O, F_M
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    for _ in range(3):
        obs, _, _ = env.step(_hold(obs))
    assert NODE_FEATS[-1] == "backlog" and EDGE_FEATS[-1] == "ack_ratio"
    assert np.allclose(obs.node_feat[:, -1], env.backlog)
    ack = obs.edge_attr[:, -1]
    assert ((ack >= 0) & (ack <= 1)).all() and (ack < 1).any()
    o, m = agent_features(obs, neighbor_slots(obs))
    assert (F_O, F_M) == (18, 20) and o.shape == (20, 18) and m.shape == (20, 20)
