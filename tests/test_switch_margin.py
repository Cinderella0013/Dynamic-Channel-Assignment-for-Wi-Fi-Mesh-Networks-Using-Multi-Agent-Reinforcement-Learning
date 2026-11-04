import json

from mesh.eval.run import run_episode
from mesh.model.agent import EGATv1Agent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario


def _episode(margin, seed=3):
    a = EGATv1Agent(seed=0, device="cpu", switch_margin=margin)
    a.phase = "ppo"
    return run_episode(MeshEnv(make_scenario("S2"), seed), a)


def test_zero_margin_is_the_old_policy():
    assert EGATv1Agent(seed=0, device="cpu").cfg.switch_margin == 0.0
    a, b = _episode(0.0), _episode(0.0)
    assert a == b


def test_margin_keeps_current_channel_and_parent():
    base, held = _episode(0.0), _episode(100.0)
    assert held["n_channel_switch"] + held["n_route_change"] < 0.5 * (base["n_channel_switch"] + base["n_route_change"])


def test_suite_cli_sets_agent_config(tmp_path):
    from mesh.eval.suite import main
    import sys
    ck = tmp_path / "v1.pt"
    a = EGATv1Agent(seed=0, device="cpu")
    a.phase = "ppo"
    a.save(ck)
    out = tmp_path / "s.json"
    argv = ["suite", "--suite", "A", "--seeds", "1", "--method", f"egat_v1_h={ck}", "--agent-set", "switch_margin=100",
            "--out", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        main()
    finally:
        sys.argv = old
    d = json.load(open(out))
    assert d["agent_overrides"] == {"switch_margin": 100}
    from mesh.eval.suite import _factory
    _, make = _factory(f"egat_v1_h={ck}", {"switch_margin": 2.5})
    assert make().cfg.switch_margin == 2.5


def test_fallback_judges_the_margin_biased_choice():
    from mesh.sim.env import MeshEnv as Env
    rates = {}
    for m in (0.0, 100.0):
        a = EGATv1Agent(seed=0, device="cpu", switch_margin=m, fallback_entropy=0.5)
        a.phase = "ppo"
        env = Env(make_scenario("S2"), 3)
        obs, r = env.reset(), []
        for _ in range(30):
            obs, _, _ = env.step(a.act(obs))
            r.append(a.fallback_rate)
        rates[m] = sum(r[1:]) / len(r[1:])
    assert rates[0.0] > 0.1 and rates[100.0] == 0.0   # a held choice is a sure choice: no Ko fallback switch
