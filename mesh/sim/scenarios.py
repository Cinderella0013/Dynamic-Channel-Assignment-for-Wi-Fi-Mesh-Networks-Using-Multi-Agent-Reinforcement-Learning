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
    spacing: float = 30.0            # rgg area side = spacing * sqrt(n)
    hotspot_period: int | None = None
    periodic: int | None = None
    shift_period: int | None = None
    static: bool = False
    mac_eff: float = 1.0             # CSMA efficiency of a lone link (link_capacity); 1 = no MAC overhead
    airtime: str = "static"          # link_capacity 1/k share | "domain": work-conserving domain_allocate
    contention: float = 0.0          # efficiency loss per extra co-channel transmitter (link_capacity)


_SCENARIOS = {
    "S1": dict(mean_mbps=2.0),
    "S2": dict(mean_mbps=10.0),
    "S3": dict(mean_mbps=6.0, hotspot=True),
    # channel index 1: the channel M0 runs every link on (common_channel picks the largest shared)
    "S4": dict(mean_mbps=6.0, ext_interf=(1, 50, 150, -85.0)),
    "S5": dict(mean_mbps=4.0, shift_step=100),
    "S6": dict(mean_mbps=6.0, failure_step=100),
    # B: known weaknesses of the baselines (spec addendum 5.2)
    "B1": dict(mean_mbps=6.0, hotspot_period=40),     # moving hotspot: Ko is myopic, decides every t_c
    "B2": dict(mean_mbps=6.0, periodic=60),           # periodic load: Li's near-static policy
    "B3": dict(mean_mbps=6.0, spacing=37.5),          # sparse: routing loops (Yuan, Q-neutral)
    "B4": dict(mean_mbps=6.0, kind="grid"),           # unseen topology family: Q-neutral's tabular state
    "B5": dict(mean_mbps=6.0, shift_period=25),       # rapid regime shifts: slow adaptation
    # C: where EGAT may lose (reported as limitations)
    "C1": dict(mean_mbps=1.0),
    "C2": dict(mean_mbps=6.0, static=True),
    "C3": dict(mean_mbps=10.0, n=50),
    "C4": dict(mean_mbps=10.0, n=80),
    "C5": dict(mean_mbps=6.0, n_gateways=4),
}
SUITE_A = ("S1", "S2", "S3", "S4", "S5", "S6")
SUITE_B = ("B1", "B2", "B3", "B4", "B5")
SUITE_C = ("C1", "C2", "C3", "C4", "C5")


def make_scenario(name, n=20, kind="rgg"):
    if name not in _SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    return replace(ScenarioConfig(name=name, n=n, kind=kind), **_SCENARIOS[name])  # C3/C4 fix their own n
