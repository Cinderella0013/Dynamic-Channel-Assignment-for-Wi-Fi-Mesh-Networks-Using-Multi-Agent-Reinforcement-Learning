import pytest

import mesh.rl.train as T
from mesh.baselines.ko import KoLocalCost
from mesh.baselines.oracle import LookaheadOracle
from mesh.model.agent import EGATv1Agent
from mesh.rl.train import train


def test_oracle_teacher_only_during_bc(monkeypatch):
    seen = []
    orig = EGATv1Agent.learn

    def spy(self, *a, **k):
        seen.append((self.phase, type(self.ko).__name__))
        return orig(self, *a, **k)
    monkeypatch.setattr(EGATv1Agent, "learn", spy)
    ag = train("egat_v1", steps=260, teacher="oracle", bc_steps=200, bc_epochs=1)
    assert {k for p, k in seen if p == "bc"} == {"LookaheadOracle"}
    assert {k for p, k in seen if p != "bc"} == {"KoLocalCost"} and isinstance(ag.ko, KoLocalCost)


def test_oracle_bound_to_current_episode_env(monkeypatch):
    envs, bound = [], []
    real_env = T.MeshEnv

    def make_env(*a, **k):
        envs.append(real_env(*a, **k))
        return envs[-1]

    class Spy(LookaheadOracle):
        def act(self, obs):
            bound.append(self.env is envs[-1])
            return super().act(obs)
    monkeypatch.setattr(T, "MeshEnv", make_env)
    monkeypatch.setattr(T, "LookaheadOracle", Spy)
    train("egat_v1", steps=420, teacher="oracle", bc_steps=400, bc_epochs=1)
    assert len(bound) >= 400 and all(bound)


def test_ko_teacher_default_unchanged():
    ag = train("egat_v2", steps=60, bc_steps=40, bc_epochs=1)
    assert isinstance(ag.ko, KoLocalCost)


def test_oracle_teacher_rejects_learners_without_bc():
    with pytest.raises(ValueError):
        train("yuan", steps=10, teacher="oracle")


def test_validation_during_bc_never_sees_the_oracle(monkeypatch):
    kinds = []
    real = T.validate

    def spy(agent, n):
        kinds.append(type(agent.ko).__name__)
        return real(agent, n)
    monkeypatch.setattr(T, "validate", spy)
    monkeypatch.setattr(T, "SCENARIOS", ("S1",))
    monkeypatch.setattr(T, "VAL_SEEDS", (1000,))
    train("egat_v1", steps=420, teacher="oracle", bc_steps=400, bc_epochs=1, val_every=200)
    assert kinds and set(kinds) == {"KoLocalCost"}
