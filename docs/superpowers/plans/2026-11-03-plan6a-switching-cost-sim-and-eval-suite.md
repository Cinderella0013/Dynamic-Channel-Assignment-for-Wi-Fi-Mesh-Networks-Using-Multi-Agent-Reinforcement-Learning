# Plan 6a: Switching-Cost Simulator, Scenario Suite, Composite Score — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the simulator real switching costs (per-node queues, one-step downtime, queue loss on unprepared switches, ACK feedback). Add the pre-registered scenario suite (A: S1–S6, B1–B5, C1–C5) and the composite score. Then retrain every learned baseline and the Plan 4 EGAT on the new simulator and produce the new reference table.

**Architecture:**
- **Queues and downtime** live in `MeshEnv.step`.
- **Flows** stay proportional-share; the offered load becomes `backlog + demand`.
- **Observation:** gains one node feature, `backlog`, and one edge feature, `ack_ratio`. Both are appended at the end, so existing column indices stay valid.
- **Scenarios:** new options go into `ScenarioConfig` and `Traffic`.
- **Evaluation:** `mesh/eval/suite.py` runs methods × scenarios × seeds into a JSON file. `mesh/eval/composite.py` turns that into the composite matrix, average ranks and weight sensitivity.

**Tech Stack:** Python ≥ 3.11, NumPy, PyTorch 2.11 + CUDA 12.8, pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-egat-v1-v2-design.md`. This plan covers §2 (simulator), §5.1 (composite) and §5.2 (scenario suite), and order-of-work step 1.

## Global Constraints

- Units: rates in Mbps per 100 ms step; backlog in the same "Mbps·step" units. `Q_MAX = 5 × cfg.mean_mbps` per node, and overflow is lost.
- New `ScenarioConfig` fields (all defaults keep S1–S6 meaning):
  - `downtime_steps=1`
  - `queue_preserve=False`
  - `spacing=30.0`
  - `hotspot_period=None`
  - `periodic=None`
  - `shift_period=None`
  - `static=False`
- `Action` gains `queue_preserve: bool = False`. The env preserves queues when `cfg.queue_preserve or action.queue_preserve`.
- `NODE_FEATS` appends `"backlog"`, giving 6 columns. `EDGE_FEATS` appends `"ack_ratio"`, giving 7 columns. Existing indices are unchanged.
- `agents_io.agent_features`:
  - `o` gains `backlog` after `alive`: `F_O = 18`.
  - Each slot in `m` gains `ack_ratio`: `F_M = 20`.
  - Every learner picks up the new sizes through `F_O`/`F_M`/`EDGE_FEATS`. Tests that hard-code 17/16/33/6 are updated to 18/20/38/7.
- Composite weights are fixed now: throughput 0.25, Jain 0.20, p5 0.20, `loss_rate` 0.20, stability 0.15. Stability is `n_channel_switch + n_route_change`.
- Every run must be reproducible from (method, scenario, n, seed). The suite JSON records the git commit hash.

## Review Focus

1. **Switch with empty queue:** a node switches while its backlog is 0, so the switch loss is 0 and nothing breaks. Test: Task 1 `test_switch_loss_only_when_backlog`.
2. **`downtime_steps=0`:** the env behaves exactly like Plan 4 except for queues. Test: Task 1 `test_zero_downtime_no_interruption`.
3. **A node dies with backlog (S6):** its backlog is dropped and counted as loss once, not every step. Test: Task 1 `test_dead_node_backlog_counted_once`.
4. **Composite edge cases:**
   - Reference value 0 for a higher-is-better metric (e.g. p5 = 0 in C1): Δ is clipped to ±1 and stays finite.
   - Lower-is-better count metrics where both are 0: Δ = 0.

   Test: Task 4 `test_composite_edge_cases`.
5. **Zero-shot N = 80 (C4):** a checkpoint trained at n=20 must act on n=80 (GNN models). Q-neutral's table is size-independent. Test: Task 5 `test_suite_runs_zero_shot_sizes`.

---

## File Structure

```
mesh/sim/env.py, mesh/sim/scenarios.py, mesh/sim/traffic.py, mesh/sim/topology.py   # Tasks 1-3
mesh/rl/agents_io.py                                                                # Task 2
mesh/eval/run.py                                                                    # Task 1: loss_rate
mesh/eval/suite.py, mesh/eval/composite.py                                          # Task 4
tests/test_queues.py, tests/test_suite.py, tests/test_composite.py                  # new
tests/test_env.py, test_agents_io.py, test_li_gat_a2c.py, test_egat_net.py, ...     # shape updates
```

---

### Task 1: Queues, downtime, switch loss, `queue_preserve`, `loss_rate`

**Files:** Modify `mesh/sim/env.py`, `mesh/sim/scenarios.py`, `mesh/eval/run.py`. Test `tests/test_queues.py`.

**Interfaces:**
- **Env state:**
  - `self.backlog (n,)`
  - `self.chan_down (n,) int` and `self.route_down (n,) int`: remaining downtime steps.
- **`step`, in order:**
  1. Validate.
  2. Events.
  3. `demand`.
  4. `offered = backlog + demand`.
  5. Detect switches against the previous action:
     - `sw_ch[u] = radio1 changed` (alive nodes)
     - `sw_rt[u] = next_hop changed` (alive non-gateway)
  6. Set `chan_down[sw_ch] = route_down[sw_rt] = cfg.downtime_steps`.
  7. Compute `rate`, then zero it on interrupted links:
     - link `(u,v)` with `cc[u,v] == channels[u,1]` and `chan_down[u] > 0`
     - link `(v,u)` with `cc[v,u] == channels[u,1]` and `chan_down[u] > 0`
     - link `(u, next_hop[u])` with `route_down[u] > 0`
  8. Compute paths over the interrupted `usable`, then capacity, then `allocate(paths, offered, cap)`.
  9. Switch loss: when the action does not preserve queues, every source whose path at the start of this step crossed an interrupted link loses its `backlog`, counted in `lost_switch`.
  10. New backlog: `backlog = clip(offered − thr, 0, Q_MAX)`. The clipped excess goes to `lost_overflow`.
  11. Decrement the downtime counters.
- **Dead nodes:** demand 0, and their backlog is zeroed once at death and counted in `lost_switch`.
- **Utility and metrics:** use `flows = src & (offered > 0)`.
- **New `info` keys:**
  - `lost` = `lost_switch + lost_overflow` (this step)
  - `demand_total` = sum of `demand` (this step)
  - `backlog_total`
- **`run_episode`:** adds `loss_rate = Σ lost / max(Σ demand_total, 1e-9)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from dataclasses import replace
from mesh.sim.env import MeshEnv, Action
from mesh.sim.scenarios import make_scenario
from mesh.eval.run import run_episode
from mesh.baselines.m0 import M0
from mesh.baselines.ko import KoLocalCost

def _hold(obs, **kw):
    return Action(obs.channels.copy(), obs.next_hop.copy(), **kw)

def test_backlog_carries_unmet_demand():
    env = MeshEnv(make_scenario("S2"), 0); obs = env.reset()
    obs, _, info = env.step(_hold(obs))
    assert np.allclose(env.backlog, np.clip(env.demand - info["throughput"], 0, 5 * 10.0))   # t=0: offered = demand; Q_MAX = 5 x mean

def test_zero_downtime_no_interruption():
    cfg = replace(make_scenario("S2"), downtime_steps=0)
    env = MeshEnv(cfg, 0); obs = env.reset()
    a = _hold(obs); a.channels[:, 1] = 2
    _, _, info = env.step(a)
    assert info["lost"] == 0.0 and info["aggregate"] > 0

def test_switch_interrupts_and_loses_queue_unless_preserved():
    def run(preserve):
        env = MeshEnv(make_scenario("S2"), 0); obs = env.reset()
        for _ in range(5):
            obs, _, _ = env.step(_hold(obs))              # build up backlog
        a = _hold(obs, queue_preserve=preserve); a.channels[:, 1] = 3; a.channels[:, 0] = np.where(obs.next_hop >= 0, 3, 0)
        _, _, info = env.step(a)
        return info
    lose, keep = run(False), run(True)
    assert lose["lost"] > keep["lost"]

def test_switch_loss_only_when_backlog():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    a = _hold(obs); a.channels[:, 1] = 2
    _, _, info = env.step(a)                               # t=0: backlog empty
    assert info["lost"] == 0.0

def test_dead_node_backlog_counted_once():
    env = MeshEnv(make_scenario("S6"), 0); obs = env.reset(); losses = []
    for _ in range(105):
        obs, _, info = env.step(_hold(obs)); losses.append(info["lost"])
    dead = int(np.flatnonzero(~env.alive)[0])
    assert env.backlog[dead] == 0.0 and np.isfinite(losses).all()

def test_loss_rate_reported_and_ko_pays_for_switching():
    m0 = run_episode(MeshEnv(make_scenario("S2"), 0), M0())
    ko = run_episode(MeshEnv(make_scenario("S2"), 0), KoLocalCost())
    assert 0.0 <= m0["loss_rate"] <= 1.0 and 0.0 <= ko["loss_rate"] <= 1.0
```

- [ ] **Step 2: Run** `python -m pytest tests/test_queues.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → 6 passed; `python -m pytest -q` → all pass. Plan 1 tests that assume no queues may need updated expected values; ledger every such change as a ruling with its reason.
- [ ] **Step 5: Commit** `feat: per-node queues, switching downtime, queue loss and queue_preserve, loss_rate`.

---

### Task 2: ACK and backlog features end-to-end

**Files:** Modify `mesh/sim/env.py` (`NODE_FEATS`, `EDGE_FEATS`, `_observe`) and `mesh/rl/agents_io.py`. Update hard-coded shapes in `tests/test_env.py`, `test_agents_io.py`, `test_li_gat_a2c.py` and `test_egat_net.py`.

**Interfaces:**
- `_observe`:
  - `backlog` column = `self.backlog`.
  - `ack_ratio` per edge = `sent/offered` on that link in the last step, 1.0 where offered is 0. `sent` is the per-link sum of `thr` over flows crossing it, computed in `step` beside `load`.
- `agent_features`:
  - `o = [..., alive, backlog, onehot(radio1,7), onehot(parent,K+1)]`
  - per slot `[is_valid, snr/40, rate/143.4, load_frac, ack_ratio]`
  - `F_O = 18`, `F_M = 20`

- [ ] **Step 1: Write the failing test** (add to `tests/test_queues.py`):

```python
from mesh.sim.env import NODE_FEATS, EDGE_FEATS
from mesh.rl.agents_io import agent_features, neighbor_slots, F_O, F_M

def test_features_include_backlog_and_ack():
    env = MeshEnv(make_scenario("S2"), 0); obs = env.reset()
    for _ in range(3):
        obs, _, _ = env.step(_hold(obs))
    assert NODE_FEATS[-1] == "backlog" and EDGE_FEATS[-1] == "ack_ratio"
    assert np.allclose(obs.node_feat[:, -1], env.backlog)
    ack = obs.edge_attr[:, -1]; assert ((ack >= 0) & (ack <= 1)).all() and (ack < 1).any()
    o, m = agent_features(obs, neighbor_slots(obs))
    assert (F_O, F_M) == (18, 20) and o.shape == (20, 18) and m.shape == (20, 20)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement, then update the hard-coded shapes in existing tests** (17→18, 16→20, 33→38, 6→7 edge dims, `node_feat` 5→6).
- [ ] **Step 4: Run** `python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: backlog and ack_ratio features for every learner`.

---

### Task 3: Scenario suite B1–B5, C1–C5

**Files:** Modify `mesh/sim/scenarios.py`, `mesh/sim/traffic.py`, `mesh/sim/topology.py` and `mesh/sim/env.py` (moving-hotspot wiring). Test `tests/test_suite.py`.

**Interfaces:**
- `generate_topology(kind, n, n_gateways, rng, spacing=30.0)`: the rgg side is `spacing·√n`.
- `Traffic(..., periodic=None, shift_period=None, static=False)`:
  - `periodic=P`: demand × `(1 + 0.8·sin(2πt/P))`.
  - `shift_period=P`: the boosted half is re-drawn every P steps. It uses `shift_factor` and the generator stream already reserved for the shift.
  - `static=True`: every node always on, with demand equal to `mean_mbps`. No randomness after init.
  - `set_boost(nodes)` replaces the boost set.
- Env, when `cfg.hotspot_period`: every `hotspot_period` steps the boosted set becomes the nodes whose nearest gateway is `gateways[(t // period) % n_gw]`.
- `make_scenario(name, n=20, kind="rgg")` accepts:

  | Name | Settings |
  |---|---|
  | S1–S6 | as before |
  | B1 | mean 6, `hotspot_period=40` |
  | B2 | mean 6, `periodic=60` |
  | B3 | mean 6, `spacing=37.5` |
  | B4 | mean 6, `kind="grid"` |
  | B5 | mean 6, `shift_period=25` |
  | C1 | mean 1.0 |
  | C2 | mean 6, `static=True` |
  | C3 | S2 settings, n=50 |
  | C4 | S2 settings, n=80 |
  | C5 | mean 6, `n_gateways=4` |

- `SUITE_A = S1..S6`, `SUITE_B = B1..B5`, `SUITE_C = C1..C5`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.scenarios import make_scenario, SUITE_A, SUITE_B, SUITE_C
from mesh.sim.env import MeshEnv, Action

def test_every_suite_scenario_runs():
    for s in SUITE_A + SUITE_B + SUITE_C:
        cfg = make_scenario(s); env = MeshEnv(cfg, 0); obs = env.reset()
        for _ in range(3):
            obs, r, _ = env.step(Action(obs.channels.copy(), obs.next_hop.copy()))
        assert np.isfinite(r) and len(obs.node_feat) == cfg.n
    assert make_scenario("C4").n == 80 and make_scenario("C5").n_gateways == 4

def test_moving_hotspot_moves():
    env = MeshEnv(make_scenario("B1"), 0); obs = env.reset()
    a = env.traffic.scale.copy()
    for _ in range(41):
        obs, _, _ = env.step(Action(obs.channels.copy(), obs.next_hop.copy()))
    assert not np.array_equal(a, env.traffic.scale)

def test_static_traffic_is_constant():
    from mesh.sim.traffic import Traffic
    tr = Traffic(5, np.random.default_rng(0), 6.0, static=True)
    assert np.allclose(tr.demand(0), 6.0) and np.allclose(tr.demand(7), 6.0)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → 3 passed; full suite green.
- [ ] **Step 5: Commit** `feat: scenario suite B1-B5 (baseline weaknesses) and C1-C5 (our risks)`.

---

### Task 4: Suite runner and composite score

**Files:** Create `mesh/eval/suite.py` and `mesh/eval/composite.py`. Test `tests/test_composite.py`.

**Interfaces:**
- **`suite.run_suite(methods: dict[str, Callable[[], policy]], scenarios: list[str], seeds: range, out: str) -> dict`**
  - Result shape: `{method: {scenario: {metric: [per-seed values]}}}`, where the metrics are the `run_episode` keys.
  - Written as JSON with `{"commit": <git rev-parse HEAD>, "results": ...}`.
- **CLI:**
  ```bash
  python -m mesh.eval.suite --suite A,B,C --seeds 5 --out runs/suite_ref.json --method m0 --method ko --method yuan=runs/yuan_s0.pt ...
  ```
  `name=path` loads a learner checkpoint; a bare `name` means M0 or Ko.
- **`composite.WEIGHTS`** = `{"aggregate": .25, "jain": .20, "p5": .20, "loss_rate": .20, "stability": .15}`. `LOWER_BETTER = {"loss_rate", "stability"}`.
- **`composite.delta(ours, ref, lower_better) -> float`**:
  - higher-is-better: `(ours − ref)/|ref|`
  - lower-is-better: `(ref − ours)/(|ref| + 1)`
  - clipped to ±1; a reference of 0 with ours also 0 gives 0.
- **`composite.score(results, method, ref, scenario, weights=WEIGHTS) -> float`** uses per-scenario means. `stability = n_channel_switch + n_route_change`.
- **`composite.report(path) -> str`** returns markdown with:
  - the composite matrix (rows: methods; columns: each reference; values: mean over suite A+B, plus a per-scenario table against Ko);
  - average rank across the 5 metrics;
  - weight sensitivity: each weight ×0.5 and ×1.5 with renormalisation, reporting the min and max of EGAT's composite against each baseline.

  CLI: `python -m mesh.eval.composite runs/suite_ref.json --ours egat`.

- [ ] **Step 1: Write the failing tests**

```python
import json, numpy as np
from mesh.eval.composite import delta, score, report, WEIGHTS

def test_composite_edge_cases():
    assert delta(1.0, 0.0, False) == 1.0 and delta(0.0, 0.0, False) == 0.0
    assert delta(0.0, 0.0, True) == 0.0 and delta(5.0, 0.0, True) == -1.0
    assert abs(delta(110.0, 100.0, False) - 0.1) < 1e-12

def _fake(path):
    def m(agg, jain, p5, loss, sw, rc):
        return dict(aggregate=[agg], jain=[jain], p5=[p5], loss_rate=[loss], n_channel_switch=[sw], n_route_change=[rc])
    res = {"ko": {"S2": m(100, .6, 1.0, .1, 100, 20)}, "egat": {"S2": m(110, .66, 1.1, .09, 50, 10)}}
    json.dump({"commit": "x", "results": res}, open(path, "w"))
    return res

def test_score_and_report(tmp_path):
    p = tmp_path / "r.json"; res = _fake(p)
    s = score(res, "egat", "ko", "S2")
    assert s > 0.09 and abs(sum(WEIGHTS.values()) - 1.0) < 1e-12
    md = report(str(p), ours="egat")
    assert "ko" in md and "sensitivity" in md.lower()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_composite.py -v` → FAIL.
- [ ] **Step 3: Implement both modules.**
- [ ] **Step 4: Run** → 2 passed.
- [ ] **Step 5: Commit** `feat: suite runner and pre-registered composite score`.

---

### Task 5: Retrain on the new simulator and build the reference table

**Files:** none new. Test: add `test_suite_runs_zero_shot_sizes` to `tests/test_suite.py`.

```python
def test_suite_runs_zero_shot_sizes(tmp_path):
    from mesh.rl.train import train
    from mesh.eval.run import run_episode
    for name in ("yuan", "li", "egat", "q_neutral"):
        ag = train(name, steps=300, seed=0, n=10, log_every=10**9, **({"bc_steps": 200} if name == "egat" else {}))
        m = run_episode(MeshEnv(make_scenario("C4"), 0), ag)
        assert np.isfinite(m["reward"])
```

- [ ] **Step 1: Run the test above.** Fix any size assumption it exposes (e.g. a GRU state built for n=20), then commit `fix: zero-shot network sizes`.
- [ ] **Step 2: Retrain all learners on the new simulator.** Run these in the background, three at a time, then EGAT:
```bash
python -m mesh.rl.train --method q_neutral --steps 100000 --seed 0 --out runs/v6/q_neutral_s0.pt
```
```bash
python -m mesh.rl.train --method yuan --steps 100000 --seed 0 --out runs/v6/yuan_s0.pt
```
```bash
python -m mesh.rl.train --method li --steps 100000 --seed 0 --out runs/v6/li_s0.pt
```
```bash
python -m mesh.rl.train --method egat --steps 100000 --seed 0 --out runs/v6/egat_s0.pt
```
- [ ] **Step 3: Run the suite** on A+B+C with 5 seeds, every method, `--out runs/v6/suite_ref.json`. Then:
```bash
python -m mesh.eval.composite runs/v6/suite_ref.json --ours egat
```
- [ ] **Step 4: Report** the composite matrix, the average ranks and the per-scenario table against Ko. This table is the reference that Plan 6b's v1 and v2 must beat. There is no assertion.
- [ ] **Step 5: Commit** any code fixes. The results under `runs/` stay out of git.

---

## Done when

- `python -m pytest -q` all green.
- `runs/v6/suite_ref.json` exists for M0, Ko, Q-neutral, Yuan, Li and EGAT (Plan 4 design) on A+B+C with 5 seeds, together with the composite report.
- Spec addendum §2, §5.1 and §5.2 implemented.
