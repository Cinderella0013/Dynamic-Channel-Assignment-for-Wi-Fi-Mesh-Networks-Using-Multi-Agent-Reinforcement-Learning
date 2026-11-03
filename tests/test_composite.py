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
