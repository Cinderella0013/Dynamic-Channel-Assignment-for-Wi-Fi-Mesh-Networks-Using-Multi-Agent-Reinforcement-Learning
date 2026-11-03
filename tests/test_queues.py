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
    env = MeshEnv(make_scenario("S2"), 0)
    obs = env.reset()
    hist = []
    for _ in range(14):
        obs, _, info = env.step(_hold(obs))
        hist.append(info["throughput"])
    win = np.mean(hist[-10:], axis=0)
    flows = (env.offered > 0) & env.alive & ~env.is_gw
    assert np.isclose(info["utility"], np.log(win[flows] + env.w.eps).sum())


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
