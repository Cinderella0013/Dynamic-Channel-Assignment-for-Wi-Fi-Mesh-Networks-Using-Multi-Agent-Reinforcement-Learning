import numpy as np
import pytest
from mesh.sim.traffic import Traffic
from mesh.sim.scenarios import make_scenario


def _series(seed, **kw):
    tr = Traffic(10, np.random.default_rng(seed), 5.0, **kw)
    return np.array([tr.demand(t) for t in range(200)])


def test_deterministic_and_nonnegative():
    a, b = _series(1), _series(1)
    assert np.array_equal(a, b) and (a >= 0).all()


def test_regime_shift_raises_load():
    s = _series(2, shift_step=100)
    assert s[100:].mean() > 1.5 * s[:100].mean()


def test_scenarios():
    assert make_scenario("S4").ext_interf == (0, 50, 150, -85.0)
    assert make_scenario("S2").mean_mbps == 10.0
    with pytest.raises(ValueError):
        make_scenario("S9")
