import json
from mesh.eval.composite import delta, score, report, WEIGHTS


def test_composite_edge_cases():
    assert delta(1.0, 0.0, False) == 1.0 and delta(0.0, 0.0, False) == 0.0
    assert delta(0.0, 0.0, True) == 0.0 and delta(5.0, 0.0, True) == -1.0
    assert abs(delta(110.0, 100.0, False) - 0.1) < 1e-12


def _fake(path):
    def m(agg, jain, p5, loss, sw, rc):
        return dict(aggregate=[agg], jain=[jain], p5=[p5], loss_rate=[loss], n_channel_switch=[sw],
                    n_route_change=[rc])
    res = {"ko": {"S2": m(100, .6, 1.0, .1, 100, 20)}, "egat": {"S2": m(110, .66, 1.1, .09, 50, 10)}}
    json.dump({"commit": "x", "results": res}, open(path, "w"))
    return res


def test_score_and_report(tmp_path):
    p = tmp_path / "r.json"
    res = _fake(p)
    s = score(res, "egat", "ko", "S2")
    assert s > 0.09 and abs(sum(WEIGHTS.values()) - 1.0) < 1e-12
    md = report(str(p), ours="egat")
    assert "ko" in md and "sensitivity" in md.lower()


def test_loss_rate_is_relative_and_ranks_share_ties(tmp_path):
    assert abs(delta(0.09, 0.10, True, count=False) - 0.1) < 1e-12      # 10% less loss = +0.10
    assert abs(delta(60, 120, True) - 60 / 121) < 1e-12                  # count metric keeps +1
    from mesh.eval.composite import _ranks
    assert _ranks([0.0, 0.0, 0.5], lower_better=True) == [1.5, 1.5, 3.0]
    p = tmp_path / "r.json"
    _fake(p)
    md = report(str(p), ours="egat")
    assert "per-metric" in md.lower() and "holds" in md.lower()


def test_suite_records_overrides(tmp_path):
    from mesh.eval.suite import run_suite
    from mesh.baselines.m0 import M0
    out = tmp_path / "s.json"
    res = run_suite({"m0": M0}, ["S1"], range(1), str(out), overrides={"queue_preserve": True, "downtime_steps": 2})
    d = json.load(open(out))
    assert d["overrides"] == {"queue_preserve": True, "downtime_steps": 2} and len(d["commit"]) >= 40
    assert "m0" in res
