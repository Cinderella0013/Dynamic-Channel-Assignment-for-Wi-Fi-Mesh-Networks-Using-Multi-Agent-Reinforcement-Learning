import numpy as np
from mesh.sim.scenarios import make_scenario, SUITE_A, SUITE_B, SUITE_C
from mesh.sim.env import MeshEnv, Action


def test_every_suite_scenario_runs():
    for s in SUITE_A + SUITE_B + SUITE_C:
        cfg = make_scenario(s)
        env = MeshEnv(cfg, 0)
        obs = env.reset()
        for _ in range(3):
            obs, r, _ = env.step(Action(obs.channels.copy(), obs.next_hop.copy()))
        assert np.isfinite(r) and len(obs.node_feat) == cfg.n
    assert make_scenario("C4").n == 80 and make_scenario("C5").n_gateways == 4


def test_moving_hotspot_moves():
    env = MeshEnv(make_scenario("B1"), 0)
    obs = env.reset()
    obs, _, _ = env.step(Action(obs.channels.copy(), obs.next_hop.copy()))   # t=0 sets the first hotspot
    a = env.traffic.scale.copy()
    assert (a > 1).any()
    for _ in range(40):
        obs, _, _ = env.step(Action(obs.channels.copy(), obs.next_hop.copy()))
    assert not np.array_equal(a, env.traffic.scale)


def test_static_traffic_is_constant():
    from mesh.sim.traffic import Traffic
    tr = Traffic(5, np.random.default_rng(0), 6.0, static=True)
    assert np.allclose(tr.demand(0), 6.0) and np.allclose(tr.demand(7), 6.0)


def test_suite_runs_zero_shot_sizes():
    from mesh.rl.train import train
    from mesh.eval.run import run_episode
    for name in ("yuan", "li", "egat", "q_neutral"):
        kw = {"bc_steps": 200} if name == "egat" else {}
        ag = train(name, steps=300, seed=0, n=10, log_every=10**9, **kw)
        m = run_episode(MeshEnv(make_scenario("C4"), 0), ag)
        assert np.isfinite(m["reward"]), name


def test_factory_resolves_variant_label(tmp_path):
    from mesh.eval.suite import _factory
    from mesh.model.agent import EGATv1Agent
    path = tmp_path / "m.pt"
    EGATv1Agent(seed=0).save(str(path))
    name, f = _factory(f"egat_v1_oracle={path}")
    assert name == "egat_v1_oracle" and type(f()).__name__ == "EGATv1Agent"
    import pytest
    with pytest.raises(ValueError):
        _factory(f"nope={path}")


def test_seed_start_gives_fresh_eval_seeds(tmp_path, monkeypatch):
    import json
    import sys
    from mesh.eval import suite
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["suite", "--suite", "A", "--seeds", "2", "--seed-start", "20", "--method", "m0",
                                      "--out", str(out)])
    suite.main()
    d = json.load(open(out))
    assert d["seeds"] == [20, 21]
