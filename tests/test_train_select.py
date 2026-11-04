import torch

import mesh.rl.train as T
from mesh.baselines.ko import KoLocalCost
from mesh.baselines.m0 import M0
from mesh.model.agent import EGATv1Agent
from mesh.sim.scenarios import SUITE_A, SUITE_B


def test_validation_set_is_a_plus_b_on_held_out_topologies():
    assert T.VAL_SCENARIOS == SUITE_A + SUITE_B
    assert len(T.VAL_SEEDS) >= 4 and not set(T.VAL_SEEDS) & set(range(20))  # never an evaluation seed


def test_validate_scores_the_composite_vs_ko(monkeypatch):
    monkeypatch.setattr(T, "VAL_SCENARIOS", ("S2", "B1"))
    monkeypatch.setattr(T, "VAL_SEEDS", (1000,))
    assert T.validate(KoLocalCost(), 20) == 0.0      # Ko against itself
    assert T.validate(M0(), 20) < -0.05              # static channels lose to Ko


def _params(agent):
    return torch.cat([p.detach().flatten().cpu() for p in agent.net.parameters()])


def test_candidates_carry_ema_weights_after_bc(monkeypatch):
    seen, raw = [], []
    orig = EGATv1Agent.learn

    def spy_learn(self, *a, **k):
        if not raw:
            raw.append(self)
        return orig(self, *a, **k)

    def spy_val(cand, n):
        seen.append((cand.phase, _params(cand)))
        return float(len(seen))
    monkeypatch.setattr(EGATv1Agent, "learn", spy_learn)
    monkeypatch.setattr(T, "validate", spy_val)
    monkeypatch.setattr(T, "EMA_DECAY", 1.0)             # EMA frozen at the first post-BC weights
    T.train("egat_v1", steps=1400, n=10, val_every=200, bc_steps=200, bc_epochs=1, log_every=10**9)
    post = [p for ph, p in seen if ph != "bc"]
    assert len(post) >= 2 and all(torch.equal(post[0], p) for p in post[1:])
    assert not torch.equal(post[-1], _params(raw[0]))   # the raw net kept training; candidates did not


def test_ema_decay_zero_is_the_raw_net(monkeypatch):
    seen, raw = [], []
    orig = EGATv1Agent.learn

    def spy_learn(self, *a, **k):
        if not raw:
            raw.append(self)
        return orig(self, *a, **k)

    def spy_val(cand, n):
        seen.append(_params(cand))
        return 0.0
    monkeypatch.setattr(EGATv1Agent, "learn", spy_learn)
    monkeypatch.setattr(T, "validate", spy_val)
    monkeypatch.setattr(T, "EMA_DECAY", 0.0)
    T.train("egat_v1", steps=600, n=10, val_every=200, bc_steps=200, bc_epochs=1, log_every=10**9)
    assert torch.equal(seen[-1], _params(raw[0]))


def test_every_candidate_is_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "validate", lambda a, n: 0.0)
    T.train("q_neutral", steps=700, seed=0, n=10, out=str(tmp_path / "q.pt"), log_every=10**9, val_every=100)
    assert (tmp_path / "q.pt").exists() and len(list(tmp_path.glob("q_val*.pt"))) >= 4
