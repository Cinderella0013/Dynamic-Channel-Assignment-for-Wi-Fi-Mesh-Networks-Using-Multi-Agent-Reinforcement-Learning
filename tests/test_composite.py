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


def _seeds(path, n, ours_gain):
    rng = __import__("numpy").random.default_rng(len(str(path)))
    def m(agg, sw):
        return dict(aggregate=list(agg), jain=[.6] * n, p5=[1.0] * n, loss_rate=[.1] * n,
                    n_channel_switch=list(sw), n_route_change=[0] * n)
    base = 100 + rng.normal(size=n)
    res = {"ko": {"S2": m(base, [100] * n)}, "egat": {"S2": m(base * (1 + ours_gain), [100] * n)}}
    json.dump({"commit": "x", "results": res}, open(path, "w"))


def test_files_pool_seeds_and_significance(tmp_path):
    from mesh.eval.composite import _load, significance
    p1, p2 = tmp_path / "t0.json", tmp_path / "t1.json"
    _seeds(p1, 3, 0.2)
    _seeds(p2, 3, 0.2)
    res, _ = _load([str(p1), str(p2)])
    assert len(res["egat"]["S2"]["aggregate"]) == 6          # training seeds pooled, not overwritten
    sig = significance(res, "egat", "ko", ["S2"])
    assert sig["p"]["S2"] < 0.05 and sig["frac_sig"] == 1.0 and sig["ci"][0] > 0.04
    md = report([str(p1), str(p2)], ours="egat")
    assert "significance" in md.lower()


def test_no_effect_is_not_significant(tmp_path):
    from mesh.eval.composite import _load, significance
    p = tmp_path / "t.json"
    _seeds(p, 6, 0.0)
    res, _ = _load([str(p)])
    sig = significance(res, "egat", "ko", ["S2"])
    assert sig["frac_sig"] == 0.0 and sig["ci"][0] <= 0 <= sig["ci"][1]


def test_bootstrap_significance_tracks_the_mean_composite(tmp_path):
    from mesh.eval.composite import _load, significance
    p = tmp_path / "t.json"
    _seeds(p, 6, 0.2)
    res, _ = _load([str(p)])
    assert significance(res, "egat", "ko", ["S2"])["frac_sig_boot"] == 1.0
    q = tmp_path / "u.json"
    _seeds(q, 6, 0.0)
    res, _ = _load([str(q)])
    assert significance(res, "egat", "ko", ["S2"])["frac_sig_boot"] == 0.0


def test_holm_correction():
    from mesh.eval.composite import holm
    # m=3: thresholds .05/3, .05/2, .05/1 on sorted p; stops at the first failure
    assert holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.01, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.02, "b": 0.001, "c": 0.5}) == {"a": True, "b": True, "c": False}


def test_hierarchical_ci_counts_training_seed_spread(tmp_path):
    from mesh.eval.composite import _load, significance
    paths = []
    for i, gain in enumerate((0.0, 0.0, 0.3)):  # one good training seed out of three
        paths.append(str(tmp_path / f"t{i}.json"))
        _seeds(paths[-1], 20, gain)
    res, _ = _load(paths)
    sig = significance(res, "egat", "ko", ["S2"])
    assert sig["ci"][0] > 0.01                       # run-level CI: looks significant
    assert sig["ci_h"][0] <= 0.005 < sig["ci"][0]    # training seeds resampled: can be no gain at all
    assert "training seeds resampled" in report(paths, ours="egat")


def test_seed_matrix_ci_counts_both_methods_training_seeds(tmp_path):
    from mesh.eval.composite import _load, seed_matrix_ci
    paths = []
    for i, gain in enumerate((0.0, 0.0, 0.3)):  # one good training seed out of three
        paths.append(str(tmp_path / f"t{i}.json"))
        _seeds(paths[-1], 20, gain)
    res, _ = _load(paths)
    out = seed_matrix_ci(res, "egat", "ko", ["S2"])
    assert out["matrix"].shape == (3, 3)
    assert abs(out["mean"] - out["matrix"].mean()) < 1e-12
    lo, hi = out["ci"]
    assert lo < 0.0 < hi                               # t with few seeds: one good seed is not enough
    assert out["df"] >= 1
    md = report(paths, ours="egat")
    assert "t-interval" in md
