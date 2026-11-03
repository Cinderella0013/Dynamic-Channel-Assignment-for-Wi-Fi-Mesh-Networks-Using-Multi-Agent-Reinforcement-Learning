# Plan 1: Python Mesh Simulator + M0/Ko Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the flow-level Python Wi-Fi mesh simulator (joint channel + routing actions, scenarios S1–S6) and the two non-learning baselines M0 and Ko local-cost, runnable from one CLI.

**Architecture:** Pure NumPy. Physical layer (`radio.py`) → static radio graph (`graph.py`) → per-step link capacity from current channels → flow paths from `next_hop` → proportional-share throughput allocation (`flows.py`). `MeshEnv` wires these with traffic and scenario events and returns a graph `Observation` (node features + link/conflict edges) that later plans feed to GATv2. Baselines are plain classes with `act(obs) -> Action`.

**Tech Stack:** Python ≥ 3.11, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-egat-mappo-mesh-design.md` (this plan implements spec §4.1, the M0/Ko rows of §1.2, the scenarios in §5.2, and milestones M1 + M3)

**Later plans (not here):** Plan 2 ns-3 scenario + calibration (M2), Plan 3 Q-neutral/Yuan/Li (M4), Plan 4 EGAT-MAPPO (M5), Plan 5 eval/ablation/generalization/ns-3 validation (M6–M8). The indoor topology family (spec §4.1) is deferred to Plan 5, where generalization tests need it.

## Global Constraints

- Python ≥ 3.11; runtime dependency NumPy only; dev dependency pytest. No torch in this plan.
- All randomness flows from `np.random.Generator` seeded by an int; the same seed gives identical results.
- Commands must work on Windows: use `python -m pytest`, not `pytest`.
- Propagation constants must equal ns-3 `LogDistancePropagationLossModel` defaults: exponent 3.0, reference loss 46.6777 dB at 1 m.
- Time step = 100 ms (one `step()`); episode length 200 steps.
- Channel set: `CHANNELS = (1, 6, 11, 36, 40, 44, 48)` (2.4 GHz {1,6,11} + 4 × 5 GHz); 2 radios per node.
- Reward: `Σ_f log(x_f + ε) − γ·n_channel_switch − δ·n_route_change − η·n_drop`, defaults ε=0.01, γ=0.1, δ=0.05, η=0.1.
- Ko adaptation below is this plan's concrete reading of "Ko local-cost"; if the user's own adaptation differs, Task 7 is the only place to change.

## Review Focus

1. A node with no route to any gateway (isolated, or cut off by a node failure) → its flow gets throughput 0, counts in `n_drop`, no exception, reward finite. Test: Task 6 `test_failure_cuts_off_node`.
2. `next_hop` values that form a loop → path is `None`, flow dropped, function returns (no infinite loop). Test: Task 4 `test_loop_is_dropped`.
3. `next_hop` pointing to a non-neighbor, or to a neighbor with no common channel → flow dropped. Test: Task 4 `test_unusable_hop_is_dropped`.
4. All demands zero → `jain` returns 1.0, `utility` finite, `p5` 0.0. Test: Task 4 `test_metrics_zero_vector`.
5. Malformed action (wrong shape, channel index outside `[0, 7)`) → `ValueError`, env state unchanged. Test: Task 6 `test_bad_action_raises`.

---

## File Structure

```
pyproject.toml
mesh/__init__.py
mesh/sim/__init__.py
mesh/sim/radio.py        # path loss, SNR, MCS rate, constants
mesh/sim/topology.py     # Topology dataclass + rgg/grid generators
mesh/sim/graph.py        # RadioGraph (static masks) + link_capacity (per-step)
mesh/sim/flows.py        # hops_to_gateway, paths_from_next_hop, allocate
mesh/sim/metrics.py      # jain, p5, utility
mesh/sim/traffic.py      # on/off Markov demand, hotspot, regime shift
mesh/sim/scenarios.py    # ScenarioConfig + make_scenario("S1".."S6")
mesh/sim/env.py          # Action, Observation, RewardWeights, MeshEnv
mesh/baselines/__init__.py
mesh/baselines/m0.py     # M0
mesh/baselines/ko.py     # KoLocalCost
mesh/eval/__init__.py
mesh/eval/run.py         # run_episode + CLI
tests/test_radio.py, test_topology.py, test_graph.py, test_flows.py,
tests/test_traffic.py, test_env.py, test_baselines.py, test_run.py
```

---

### Task 1: Project scaffold + radio model

**Files:**
- Create: `pyproject.toml`, `mesh/__init__.py`, `mesh/sim/__init__.py`, `mesh/sim/radio.py`
- Test: `tests/test_radio.py`

**Interfaces:**
- Produces (in `mesh/sim/radio.py`):
  - constants `TX_POWER_DBM = 20.0`, `NOISE_DBM = -94.0`, `PL_EXPONENT = 3.0`, `PL_REF_DB = 46.6777`, `LINK_THRESHOLD_DBM = -76.0`, `CS_THRESHOLD_DBM = -82.0`, `CHANNELS = (1, 6, 11, 36, 40, 44, 48)`, `N_CHANNELS = 7`
  - `MCS_SNR_DB = (2, 5, 9, 11, 15, 18, 20, 25, 29, 31, 34, 37)`, `MCS_RATE_MBPS = (8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.4)` (802.11ax, 20 MHz, 1 SS, GI 0.8 µs)
  - `rx_power_dbm(d_m: np.ndarray) -> np.ndarray` (distances below 1 m clipped to 1 m)
  - `snr_db(rx_dbm: np.ndarray, extra_noise_dbm: float | np.ndarray | None = None) -> np.ndarray` (noise powers add in linear mW)
  - `rate_mbps(snr: np.ndarray) -> np.ndarray` (highest MCS whose threshold ≤ snr; 0.0 below MCS0)

- [ ] **Step 1: Init repo and project**

```bash
git init
```
`pyproject.toml`: project name `mesh`, `requires-python = ">=3.11"`, dependencies `["numpy"]`, optional `dev = ["pytest"]`, `[tool.pytest.ini_options] testpaths = ["tests"]`. Then `python -m pip install -e .[dev]`.

- [ ] **Step 2: Write the failing tests**

```python
import numpy as np
from mesh.sim.radio import rx_power_dbm, snr_db, rate_mbps, NOISE_DBM

def test_rx_power_ns3_defaults():
    p = rx_power_dbm(np.array([0.5, 1.0, 10.0]))
    assert np.allclose(p, [20 - 46.6777, 20 - 46.6777, 20 - 46.6777 - 30])

def test_snr_adds_noise_linearly():
    assert np.isclose(snr_db(np.array([-64.0]))[0], 30.0)
    # equal extra noise doubles noise power -> -3.01 dB
    assert np.isclose(snr_db(np.array([-64.0]), NOISE_DBM)[0], 30.0 - 3.0103, atol=1e-3)

def test_rate_thresholds():
    assert np.allclose(rate_mbps(np.array([1.9, 2.0, 10.0, 40.0])), [0.0, 8.6, 25.8, 143.4])
```

- [ ] **Step 3: Run to verify fail** — `python -m pytest tests/test_radio.py -v` → FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Implement `mesh/sim/radio.py`** — `rate_mbps` via `np.searchsorted(MCS_SNR_DB, snr, side="right") - 1`, index −1 → 0.0.
- [ ] **Step 5: Run to verify pass** — same command → 3 passed.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: radio model with ns-3 log-distance defaults"`

---

### Task 2: Topology generators

**Files:**
- Create: `mesh/sim/topology.py`
- Test: `tests/test_topology.py`

**Interfaces:**
- Consumes: `rx_power_dbm`, `LINK_THRESHOLD_DBM` (Task 1)
- Produces:
  - `@dataclass(frozen=True) class Topology: pos: np.ndarray  # (n, 2) metres; gateways: tuple[int, ...]`
  - `generate_topology(kind: Literal["rgg", "grid"], n: int, n_gateways: int, rng: np.random.Generator) -> Topology`
    - `rgg`: uniform in a square of side `30 * sqrt(n)` m; resample until the link graph (rx ≥ `LINK_THRESHOLD_DBM`) is connected; give up after 1000 tries with `RuntimeError`.
    - `grid`: `ceil(sqrt(n))` columns, 35 m spacing, first `n` points.
    - gateways: `n_gateways` distinct nodes chosen by `rng` (grid too).
  - `link_adjacency(pos: np.ndarray) -> np.ndarray` bool (n, n), symmetric, diagonal False.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.topology import generate_topology, link_adjacency

def _connected(adj):
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            if v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == len(adj)

def test_rgg_connected_and_deterministic():
    a = generate_topology("rgg", 20, 2, np.random.default_rng(0))
    b = generate_topology("rgg", 20, 2, np.random.default_rng(0))
    assert a.pos.shape == (20, 2) and len(set(a.gateways)) == 2
    assert np.array_equal(a.pos, b.pos) and a.gateways == b.gateways
    assert _connected(link_adjacency(a.pos))

def test_grid_spacing_links_neighbors_only():
    t = generate_topology("grid", 9, 1, np.random.default_rng(0))
    adj = link_adjacency(t.pos)
    assert adj[0, 1] and adj[0, 3] and not adj[0, 2]   # 35 m linked, 70 m not
    assert not adj.diagonal().any() and (adj == adj.T).all()
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_topology.py -v` → FAIL, import error.
- [ ] **Step 3: Implement `mesh/sim/topology.py`.**
- [ ] **Step 4: Run to verify pass** — 2 passed.
- [ ] **Step 5: Commit** — `git commit -am "feat: rgg and grid topology generators"` (add new files first).

---

### Task 3: Radio graph + per-step link capacity

**Files:**
- Create: `mesh/sim/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `Topology`, `link_adjacency` (Task 2); radio functions/constants (Task 1)
- Produces:
  - `@dataclass(frozen=True) class RadioGraph: rx_dbm: np.ndarray (n,n); link: np.ndarray bool (n,n); conflict: np.ndarray bool (n,n)  # rx ≥ CS_THRESHOLD_DBM, diag False`
  - `build_radio_graph(top: Topology) -> RadioGraph`
  - `common_channel(channels: np.ndarray (n,2) int) -> np.ndarray (n,n) int` — **largest** shared channel index per pair, −1 if none. Largest, not smallest: Ko keeps radio 0 on channel 0 everywhere as a control channel, so "smallest" would route every link over channel 0 and make radio-1 diversity useless.
  - `link_rate(rg: RadioGraph, cc: np.ndarray, ext_noise_dbm: np.ndarray | None) -> np.ndarray (n,n) Mbps` — 0 where `~rg.link` or `cc == -1`; `ext_noise_dbm` is per-channel `(N_CHANNELS,)` with `-inf` meaning none, applied as extra noise on links using that channel.
  - `link_capacity(rg, cc, rate, active: np.ndarray bool (n,n)) -> np.ndarray (n,n)` — for each active undirected link l=(u,v): `k_l = 1 + #other active links m=(x,y) with cc[m]==cc[l] and any endpoint of l in conflict with or equal to any endpoint of m`; `cap = rate / k_l`. Inactive links get `cap = rate`.

```python
# ponytail: count-based airtime share (1/k). Upper bound on real CSMA loss;
# swap for clique-based share if Plan 2 calibration error > 15%.
```

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.topology import Topology
from mesh.sim.graph import build_radio_graph, common_channel, link_rate, link_capacity

LINE = Topology(pos=np.array([[0, 0], [30, 0], [60, 0], [90, 0]], float), gateways=(0,))

def test_common_channel():
    cc = common_channel(np.array([[0, 1], [1, 2], [3, 4], [0, 4]]))
    assert cc[0, 1] == 1 and cc[1, 2] == -1 and cc[2, 3] == 4 and cc[0, 3] == 0

def test_two_conflicting_links_share_airtime():
    rg = build_radio_graph(LINE)
    ch = np.array([[0, 1]] * 4)
    cc = common_channel(ch); rate = link_rate(rg, cc, None)
    active = np.zeros((4, 4), bool); active[0, 1] = active[1, 0] = active[2, 3] = active[3, 2] = True
    cap = link_capacity(rg, cc, rate, active)
    assert np.isclose(cap[0, 1], rate[0, 1] / 2)

def test_different_channels_no_sharing():
    rg = build_radio_graph(LINE)
    ch = np.array([[0, 1], [0, 1], [2, 3], [2, 3]])
    cc = common_channel(ch); rate = link_rate(rg, cc, None)
    active = np.zeros((4, 4), bool); active[0, 1] = active[1, 0] = active[2, 3] = active[3, 2] = True
    assert np.isclose(link_capacity(rg, cc, rate, active)[0, 1], rate[0, 1])

def test_external_noise_lowers_rate():
    rg = build_radio_graph(LINE)
    cc = common_channel(np.array([[0, 1]] * 4))
    noise = np.full(7, -np.inf); noise[1] = -70.0   # links use channel 1 (largest shared)
    assert link_rate(rg, cc, noise)[0, 1] < link_rate(rg, cc, None)[0, 1]
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_graph.py -v` → FAIL.
- [ ] **Step 3: Implement `mesh/sim/graph.py`** — O(L²) loop over active links is fine for n ≤ 80.
- [ ] **Step 4: Run to verify pass** — 4 passed.
- [ ] **Step 5: Commit** — `feat: radio graph, channel-aware rate and airtime share`.

---

### Task 4: Flows, allocation, metrics

**Files:**
- Create: `mesh/sim/flows.py`, `mesh/sim/metrics.py`
- Test: `tests/test_flows.py`

**Interfaces:**
- Produces (`flows.py`):
  - `hops_to_gateway(adj: np.ndarray bool (n,n), gateways: tuple[int, ...], alive: np.ndarray bool (n,)) -> np.ndarray int (n,)` — multi-source BFS over alive nodes; −1 if unreachable; gateways 0.
  - `paths_from_next_hop(next_hop: np.ndarray int (n,), gateways, usable: np.ndarray bool (n,n), alive) -> list[list[int] | None]` — one entry per node; gateways and dead nodes → `None` (they are not sources); a source follows `next_hop` until it reaches a gateway; `None` if a hop is not `usable`, hits a dead node, or exceeds `n` hops (loop).
  - `allocate(paths, demand: np.ndarray (n,), cap: np.ndarray (n,n)) -> tuple[np.ndarray (n,), np.ndarray (n,n)]` — returns (throughput per source, load per directed link). For each link, total demand D over flows crossing it; each flow's share = `d_f` if `D ≤ cap` else `cap * d_f / D`; flow throughput = min(d_f, shares on its path). Load = sum of demands crossing the link.

```python
# ponytail: proportional share, not max-min fair. Good enough for a fluid model;
# switch to progressive filling if fairness results look like allocation artifacts.
```

- Produces (`metrics.py`):
  - `jain(x: np.ndarray) -> float` — 1.0 when `sum(x) == 0`
  - `p5(x: np.ndarray) -> float` — `np.percentile(x, 5)`, 0.0 on empty
  - `utility(x: np.ndarray, eps: float) -> float` — `sum(log(x + eps))`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.flows import hops_to_gateway, paths_from_next_hop, allocate
from mesh.sim.metrics import jain, p5, utility

ADJ = np.array([[0,1,0],[1,0,1],[0,1,0]], bool)   # 0 - 1 - 2, gateway 0
ALIVE = np.ones(3, bool)

def test_hops():
    assert list(hops_to_gateway(ADJ, (0,), ALIVE)) == [0, 1, 2]
    dead1 = np.array([True, False, True])
    assert list(hops_to_gateway(ADJ, (0,), dead1)) == [0, -1, -1]

def test_paths_follow_next_hop():
    p = paths_from_next_hop(np.array([-1, 0, 1]), (0,), ADJ, ALIVE)
    assert p == [None, [1, 0], [2, 1, 0]]

def test_loop_is_dropped():
    p = paths_from_next_hop(np.array([-1, 2, 1]), (0,), ADJ, ALIVE)
    assert p[1] is None and p[2] is None

def test_unusable_hop_is_dropped():
    p = paths_from_next_hop(np.array([-1, 0, 0]), (0,), ADJ, ALIVE)  # 2->0 not a link
    assert p[2] is None and p[1] == [1, 0]

def test_allocate_bottleneck_shared():
    cap = np.zeros((3, 3)); cap[1, 0] = 10.0; cap[2, 1] = 100.0
    thr, load = allocate([None, [1, 0], [2, 1, 0]], np.array([0.0, 10.0, 10.0]), cap)
    assert np.allclose(thr, [0.0, 5.0, 5.0]) and load[1, 0] == 20.0

def test_metrics_zero_vector():
    z = np.zeros(4)
    assert jain(z) == 1.0 and p5(z) == 0.0 and np.isfinite(utility(z, 0.01))
    assert np.isclose(jain(np.array([1.0, 0.0])), 0.5)
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_flows.py -v` → FAIL.
- [ ] **Step 3: Implement `flows.py` and `metrics.py`.**
- [ ] **Step 4: Run to verify pass** — 6 passed.
- [ ] **Step 5: Commit** — `feat: flow paths, proportional allocation, metrics`.

---

### Task 5: Traffic + scenarios

**Files:**
- Create: `mesh/sim/traffic.py`, `mesh/sim/scenarios.py`
- Test: `tests/test_traffic.py`

**Interfaces:**
- Produces (`traffic.py`):
  - `class Traffic: __init__(self, n: int, rng: np.random.Generator, mean_mbps: float, boost_nodes: np.ndarray | None = None, boost_factor: float = 3.0, shift_step: int | None = None, shift_factor: float = 3.0)`; `demand(self, t: int) -> np.ndarray (n,)`.
    - Per node on/off Markov chain, stay-probability 0.9, start on with prob 0.7; when on, demand ~ Exponential(mean_mbps) redrawn each step.
    - `boost_nodes` (hotspot) demand × `boost_factor` always.
    - From `t >= shift_step`, a fixed random half of nodes (chosen at init) × `shift_factor`.
    - Calls must be made with consecutive `t` (0, 1, 2, …); it is stateful.
- Produces (`scenarios.py`):
  - `@dataclass(frozen=True) class ScenarioConfig: name: str; kind: str = "rgg"; n: int = 20; n_gateways: int = 2; mean_mbps: float = 6.0; episode_len: int = 200; hotspot: bool = False; shift_step: int | None = None; ext_interf: tuple[int, int, int, float] | None = None  # (channel_idx, start, end, noise_dbm); failure_step: int | None = None`
  - `make_scenario(name: str, n: int = 20, kind: str = "rgg") -> ScenarioConfig` with exact values:

| name | mean_mbps | extra |
|---|---|---|
| S1 | 2.0 | — |
| S2 | 10.0 | — |
| S3 | 6.0 | `hotspot=True` (env boosts nodes whose nearest gateway by hops is `gateways[0]`) |
| S4 | 6.0 | `ext_interf=(0, 50, 150, -85.0)` |
| S5 | 4.0 | `shift_step=100` |
| S6 | 6.0 | `failure_step=100` (env kills the non-gateway node with the highest link degree; ties → lowest id) |

  Unknown name → `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, pytest
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
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_traffic.py -v` → FAIL.
- [ ] **Step 3: Implement both files.**
- [ ] **Step 4: Run to verify pass** — 3 passed.
- [ ] **Step 5: Commit** — `feat: on/off traffic and scenario configs S1-S6`.

---

### Task 6: MeshEnv

**Files:**
- Create: `mesh/sim/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces:
  - `@dataclass class Action: channels: np.ndarray  # (n, 2) int in [0, N_CHANNELS); next_hop: np.ndarray  # (n,) int, −1 for gateways`
  - `@dataclass class Observation:`
    - `node_feat: np.ndarray (n, 5) float32` — columns `NODE_FEATS = ("offered_load", "unmet", "is_gateway", "hop_to_gw", "alive")`; `unmet` = previous-step demand − throughput.
    - `channels: np.ndarray (n, 2)`, `next_hop: np.ndarray (n,)` — currently applied action.
    - `edge_index: np.ndarray (2, E) int64` — directed pairs (u, v) where `rg.link | rg.conflict`, both directions.
    - `edge_attr: np.ndarray (E, 6) float32` — columns `EDGE_FEATS = ("is_link", "is_conflict", "snr_db", "rate_mbps", "load_frac", "same_channel")`; `rate_mbps` uses the current common channel (0 if none); `load_frac` = previous-step link load / capacity (0 if capacity 0); `same_channel` = 1 if the two nodes share any channel.
    - `gateways: tuple[int, ...]`, `t: int`
  - `@dataclass(frozen=True) class RewardWeights: eps=0.01; gamma=0.1; delta=0.05; eta=0.1`
  - `class MeshEnv: __init__(self, cfg: ScenarioConfig, seed: int, weights: RewardWeights = RewardWeights())`
    - `reset(self) -> Observation` — builds topology (`generate_topology(cfg.kind, cfg.n, cfg.n_gateways, rng)`), radio graph, traffic; initial action = all nodes `channels=[0, 1]`, `next_hop` = min-hop parent toward a gateway (lowest id on ties).
    - `step(self, action: Action) -> tuple[Observation, float, dict]` — validates action (wrong shape or channel out of range → `ValueError`, no state change); applies scenario events at this `t`; computes paths → active links → capacity → `allocate`; reward per Global Constraints.
    - `info` keys: `throughput (n,)`, `aggregate`, `jain`, `p5`, `utility`, `n_channel_switch` (radio channel changes vs previous action, alive nodes only), `n_route_change` (`next_hop` changes, alive non-gateway nodes), `n_drop` (alive sources with demand > 0 and path `None`), `local_utility (n,)` (sum of `log(x_f+eps)` over flows whose path contains node i), `done` (`t == episode_len`).
    - Metrics (`jain`, `p5`, `aggregate`, `utility`) are over alive non-gateway sources.
    - A dead node: demand 0, not a source, cannot be on a path.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, pytest
from mesh.sim.env import MeshEnv, Action
from mesh.sim.scenarios import make_scenario

def _hold(env, obs):
    return Action(obs.channels.copy(), obs.next_hop.copy())

def test_episode_runs_and_is_deterministic():
    def run(seed):
        env = MeshEnv(make_scenario("S2"), seed); obs = env.reset(); out = []
        for _ in range(200):
            obs, r, info = env.step(_hold(env, obs)); out.append(r)
        assert info["done"]
        return np.array(out)
    a, b = run(3), run(3)
    assert np.array_equal(a, b) and np.isfinite(a).all()

def test_observation_shapes():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    assert obs.node_feat.shape == (20, 5) and obs.edge_attr.shape[1] == 6
    assert obs.edge_index.shape[0] == 2 and obs.edge_index.shape[1] == obs.edge_attr.shape[0]

def test_channel_switch_counted_and_penalized():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    a = _hold(env, obs); a.channels[:, 1] = 2
    _, _, info = env.step(a)
    assert info["n_channel_switch"] == 20

def test_bad_action_raises():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    a = _hold(env, obs); a.channels[0, 0] = 7
    with pytest.raises(ValueError):
        env.step(a)
    with pytest.raises(ValueError):
        env.step(Action(obs.channels[:5], obs.next_hop))
    _, _, info = env.step(_hold(env, obs))      # state unchanged: still usable
    assert np.isfinite(info["utility"])

def test_failure_cuts_off_node():
    env = MeshEnv(make_scenario("S6"), 0); obs = env.reset()
    for _ in range(101):
        obs, r, info = env.step(_hold(env, obs))   # routes NOT repaired
    assert obs.node_feat[:, 4].sum() == 19 and np.isfinite(r)
    assert info["n_drop"] >= 1                     # someone routed through the dead node
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_env.py -v` → FAIL.
- [ ] **Step 3: Implement `mesh/sim/env.py`.** Seed three child generators from `np.random.SeedSequence(seed).spawn(3)` (topology, traffic, spare) so adding randomness later does not shift topologies.
- [ ] **Step 4: Run to verify pass** — 5 passed. If `test_failure_cuts_off_node` fails on `n_drop` because seed 0's highest-degree node carries no flow, pick the first seed in 0–9 where it does and pin it in the test with a comment.
- [ ] **Step 5: Run whole suite** — `python -m pytest -v` → all pass.
- [ ] **Step 6: Commit** — `feat: MeshEnv with graph observation and S1-S6 events`.

---

### Task 7: M0 and Ko local-cost baselines

**Files:**
- Create: `mesh/baselines/__init__.py`, `mesh/baselines/m0.py`, `mesh/baselines/ko.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `Observation`, `Action`, `EDGE_FEATS`, `NODE_FEATS` (Task 6); `hops_to_gateway` (Task 4). Baselines read **only** `Observation` (same information budget as the learned model).
- Produces:
  - `class M0: def act(self, obs: Observation) -> Action` — channels `[0, 1]` everywhere; `next_hop` = min-hop parent toward a gateway over alive nodes and `is_link` edges (lowest id on ties); unreachable → −1.
  - `class KoLocalCost: __init__(self, t_c: int = 10)`; `act(self, obs) -> Action`:
    - **Channel** (only when `obs.t % t_c == 0`, else keep `obs.channels`): radio 0 = channel 0 everywhere (common control channel, keeps connectivity). Radio 1: visit nodes in id order; each picks `c ∈ {1..6}` minimizing `Σ_{j conflict neighbor, c ∈ channels_j} (1 + offered_load_j)`, using choices already made this round; ties → lowest c.
    - **Routing** (every step): link cost `w(u,v) = (1 + load_frac(u,v)) / rate_mbps(u,v)` (∞ if rate 0); Bellman–Ford from gateways over alive nodes; `next_hop[u] = argmin_v w(u,v) + D[v]`; unreachable → −1.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.baselines.m0 import M0
from mesh.baselines.ko import KoLocalCost
from mesh.sim.flows import paths_from_next_hop

def test_m0_routes_reach_gateway():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    a = M0().act(obs)
    assert (a.channels == [0, 1]).all()
    _, _, info = env.step(a)
    assert info["n_drop"] == 0

def test_ko_diversifies_channels_and_keeps_radio0():
    env = MeshEnv(make_scenario("S2"), 0); obs = env.reset()
    a = KoLocalCost().act(obs)
    assert (a.channels[:, 0] == 0).all()
    assert len(np.unique(a.channels[:, 1])) >= 3

def test_ko_routes_loop_free():
    env = MeshEnv(make_scenario("S2"), 0); obs = env.reset(); ko = KoLocalCost()
    for _ in range(20):
        obs, _, info = env.step(ko.act(obs))
    assert info["n_drop"] == 0
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_baselines.py -v` → FAIL.
- [ ] **Step 3: Implement `m0.py` and `ko.py`.**
- [ ] **Step 4: Run to verify pass** — 3 passed.
- [ ] **Step 5: Commit** — `feat: M0 and Ko local-cost baselines`.

---

### Task 8: Episode runner, Ko-vs-M0 sanity check, CLI

**Files:**
- Create: `mesh/eval/__init__.py`, `mesh/eval/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `MeshEnv`, `make_scenario`, `M0`, `KoLocalCost`.
- Produces:
  - `POLICIES = {"m0": M0, "ko": KoLocalCost}`
  - `run_episode(env: MeshEnv, policy) -> dict[str, float]` — keys `aggregate`, `jain`, `p5`, `utility` (means over steps), `n_channel_switch`, `n_route_change`, `n_drop` (totals), `reward` (sum).
  - CLI: `python -m mesh.eval.run --scenario S2 --policy ko --seeds 5 [--n 20] [--kind rgg]` prints one line per seed and a mean row, columns in the order of the keys above.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, subprocess, sys
from mesh.eval.run import run_episode, POLICIES
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario

def test_all_scenarios_run_for_all_policies():
    for s in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        for p in POLICIES.values():
            m = run_episode(MeshEnv(make_scenario(s), 0), p())
            assert all(np.isfinite(v) for v in m.values())

def test_ko_beats_m0_under_heavy_load():
    ko = np.mean([run_episode(MeshEnv(make_scenario("S2"), s), POLICIES["ko"]())["utility"] for s in range(5)])
    m0 = np.mean([run_episode(MeshEnv(make_scenario("S2"), s), POLICIES["m0"]())["utility"] for s in range(5)])
    assert ko > m0

def test_cli():
    out = subprocess.run([sys.executable, "-m", "mesh.eval.run", "--scenario", "S1",
                          "--policy", "m0", "--seeds", "2"], capture_output=True, text=True, check=True)
    assert "mean" in out.stdout
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_run.py -v` → FAIL.
- [ ] **Step 3: Implement `mesh/eval/run.py`** with `argparse`.
- [ ] **Step 4: Run to verify pass** — 3 passed. If `test_ko_beats_m0_under_heavy_load` fails, **do not tune the test**: report Ko and M0 numbers to the user — it means either the simulator's contention model or the Ko reading is off, and that is a spec-level decision.
- [ ] **Step 5: Run whole suite** — `python -m pytest -v` → all pass.
- [ ] **Step 6: Commit** — `feat: episode runner and CLI; Ko vs M0 sanity check`.

---

## Done when

- `python -m pytest -v` all green.
- `python -m mesh.eval.run --scenario S2 --policy ko --seeds 5` and `--policy m0` print tables, with Ko utility > M0.
- Spec M1 (sim + sanity tests) and M3 (M0 + Ko on all 6 scenarios) satisfied.
