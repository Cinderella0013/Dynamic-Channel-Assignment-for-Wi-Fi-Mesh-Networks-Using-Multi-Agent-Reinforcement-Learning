import mesh.rl.train as T


def test_keeps_best_validation_checkpoint(monkeypatch, tmp_path):
    scores = iter([5.0, 9.0, 1.0, 2.0])                       # best is the 2nd validation
    def fake_val(agent, n):
        v = next(scores)
        agent._val_tag = v
        return v
    monkeypatch.setattr(T, "validate", fake_val)
    # validations at 200, 400, 600 (episode ends) and the final step: scores 5, 9, 1, 2 -> best is not the last
    ag = T.train("q_neutral", steps=700, seed=0, n=10, out=str(tmp_path / "q.pt"), log_every=10**9, val_every=100)
    assert getattr(ag, "_val_tag", None) == 9.0


def test_snapshot_skips_buffers_and_restores_them():
    from mesh.rl.yuan_gqn import YuanAgent
    ag = YuanAgent(seed=0, device="cpu")
    ag.mem = [1, 2, 3]
    c = T.snapshot(ag)
    assert c.mem == [] and ag.mem == [1, 2, 3]


def test_cli_agent_set_reaches_the_learner(monkeypatch):
    import sys
    seen = {}
    monkeypatch.setattr(T, "train", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(sys, "argv", ["train", "--method", "egat", "--agent-set", "use_gru=false",
                                      "--agent-set", "use_edge_feat=false"])
    T.main()
    assert seen["use_gru"] is False and seen["use_edge_feat"] is False
