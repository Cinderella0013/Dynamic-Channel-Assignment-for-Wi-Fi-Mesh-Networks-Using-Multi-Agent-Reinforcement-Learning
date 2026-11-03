"""Evaluation scenarios S1-S6 (spec section 5.2)."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    kind: str = "rgg"
    n: int = 20
    n_gateways: int = 2
    mean_mbps: float = 6.0
    episode_len: int = 200
    hotspot: bool = False
    shift_step: int | None = None
    ext_interf: tuple[int, int, int, float] | None = None  # (channel_idx, start, end, noise_dbm)
    failure_step: int | None = None
    downtime_steps: int = 1          # steps a switching node cannot send (802.11 switch + re-association)
    queue_preserve: bool = False     # CSA-style buffering for every action (fairness setting b)
    utility_window: int = 10         # steps (1 s): proportional fairness on windowed throughput


_SCENARIOS = {
    "S1": dict(mean_mbps=2.0),
    "S2": dict(mean_mbps=10.0),
    "S3": dict(mean_mbps=6.0, hotspot=True),
    # channel index 1: the channel M0 runs every link on (common_channel picks the largest shared)
    "S4": dict(mean_mbps=6.0, ext_interf=(1, 50, 150, -85.0)),
    "S5": dict(mean_mbps=4.0, shift_step=100),
    "S6": dict(mean_mbps=6.0, failure_step=100),
}


def make_scenario(name, n=20, kind="rgg"):
    if name not in _SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    return replace(ScenarioConfig(name=name, n=n, kind=kind), **_SCENARIOS[name])
