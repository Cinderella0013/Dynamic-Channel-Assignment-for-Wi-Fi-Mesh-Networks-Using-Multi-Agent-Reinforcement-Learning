# Plan 3: RL Baselines (Q-neutral, Yuan GQN, Li GAT-A2C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt three learning baselines to the Plan 1 mesh env (joint channel + routing), give them one shared action/observation/reward interface and one training CLI, and make trained checkpoints runnable through `mesh.eval.run`.

**Architecture:** One per-agent interface in `mesh/rl/agents_io.py` (agent = mesh node; discrete action = downlink channel × parent slot; radio 0 follows the parent's downlink, the same tree convention Ko uses), used identically by all three learners and later by EGAT-MAPPO (Plan 4). Each learner is one file exposing `act(obs, explore)`, `learn(...)`, `save/load`. `mesh/rl/train.py` runs every learner with the same env-step budget on episodes drawn from S1–S6 with fresh seeds.

**Tech Stack:** Python ≥ 3.11, NumPy, PyTorch (CUDA 12.8 build for the RTX 5060; CPU fallback), pytest. No PyTorch Geometric in this plan: n ≤ 80, so dense masked attention is simpler than a new dependency.

**Spec:** `docs/superpowers/specs/2026-11-03-egat-mappo-mesh-design.md` (§1.2 baselines, §4.4 same env for all, §5.5 fair comparison: same env steps, same observation where possible, adaptation notes).

**Sources:**
- Yuan: S. Yuan, Y. Zhang, T. Ma, Z. Cheng, D. Guo, "Graph convolutional reinforcement learning for resource allocation in hybrid overlay–underlay cognitive radio network with network slicing", IET Communications, 2022, doi:10.1049/cmu2.12527 — read from the user's PDF (§4.2–4.3, Table 2).
- Li: Z. Li et al., "Dynamic Allocation of C-V2X Communication Resources Based on Graph Attention Network and Deep Reinforcement Learning", Sensors 25(16):5209, 2025 — read from PMC. **Not yet confirmed by the user as "Li"**; if a different paper is meant, only Task 5 changes.

## Global Constraints

- Agent = every alive mesh node (gateways included: they choose a downlink channel; their parent slot is masked to slot 0 and ignored).
- `K_SLOTS = 4` parent slots = the 4 link neighbours with highest SNR (ties → lower id), fixed per topology; fewer neighbours → remaining slots masked.
- Action index `a = c * K_SLOTS + j`, `c ∈ {0..5}` ↦ downlink channel index `c + 1` (channels 1..6), `j` = parent slot. `N_ACTIONS = 24`.
- Decoded `Action`: `channels[u, 1] = c_u + 1`; `channels[u, 0] = channels[parent(u), 1]` (gateways and unrouted nodes: `0`); `next_hop[u] = slot_j(u)` (gateways: −1).
- Per-agent reward (all learners): `r_i = 0.5 · local_utility_i + 0.5 · reward_global / n_alive − γ · sw_i − δ · rc_i` with γ=0.1, δ=0.05 (Plan 1 `RewardWeights`), `sw_i` = own radio-channel changes, `rc_i` = own `next_hop` change (0/1).
- Same budget for every learner: `--steps` env steps (default 100 000); episodes sample scenario uniformly from S1–S6, `kind="rgg"`, `n=20`, episode seed from the training RNG.
- Discount γ_RL = 0.98 for all learners (Yuan Table 2; Li's value is ambiguous in the paper).
- Device: `cuda` if `torch.cuda.is_available()` else `cpu`; results must not depend on device beyond float noise.
- Every adaptation choice that departs from a paper is listed in `docs/baselines-adaptation.md` (Task 7) — spec §5.5 appendix.

## Review Focus

1. A node whose every parent slot is masked (isolated after S6 failure) → action decode gives `next_hop = −1`, no exception; masked actions are never chosen even when exploring. Test: Task 2 `test_mask_isolated_node`.
2. Dead nodes in the observation (S6) → they get no reward, no transition is learned for them, attention never attends to them. Test: Task 2 `test_dead_node_excluded`, Task 4 `test_attention_ignores_masked`.
3. Episode boundary → bootstrapped targets use `done` (no value bootstrapped across episodes/topologies). Test: Task 4 `test_td_target_respects_done`.
4. Checkpoint saved on GPU loads and acts on CPU. Test: Task 6 `test_checkpoint_roundtrip_cpu`.
5. The env's deferred Plan 1 minors would corrupt learning (one-step interferer lookahead, `hop_to_gw = −1`, `step()` past `done`, out-of-range `next_hop` counted as a route change) → fixed in Task 1 with one test each.

---

## File Structure

```
mesh/sim/env.py            # Task 1: hardening for learners
mesh/rl/__init__.py
mesh/rl/agents_io.py       # slots, masks, features, decode, per-agent reward
mesh/rl/q_neutral.py       # tabular Q-learning, shared table
mesh/rl/yuan_gqn.py        # GQN: MLP encoder + 2 dot-product attention layers + Q head, replay, KL reg
mesh/rl/li_gat_a2c.py      # GAT (additive attention, conflict graph) + actor/critic, 1-step A2C
mesh/rl/train.py           # shared training loop + CLI
mesh/eval/run.py           # Task 6: load checkpoints as policies
docs/baselines-adaptation.md
tests/test_env_hardening.py, test_agents_io.py, test_q_neutral.py,
tests/test_yuan_gqn.py, test_li_gat_a2c.py, test_train.py
```

---

### Task 1: Env hardening for learners (Plan 1 deferred minors)

**Files:**
- Modify: `mesh/sim/env.py`
- Test: `tests/test_env_hardening.py`

**Interfaces:**
- Produces (changed behaviour, same signatures):
  - `MeshEnv.reset(seed: int | None = None) -> Observation` — `None` keeps `self.seed`; an int replaces it.
  - Observation `rate_mbps` uses the external noise of the step **just executed** (`self.t - 1`), not the next one; at reset, no noise.
  - `hop_to_gw` feature = `n` (not −1) for unreachable nodes.
  - `step()` after `done` raises `RuntimeError("episode finished; call reset()")`.
  - `_validate` maps any `next_hop` outside `[0, n)` to −1 before counting route changes.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, pytest
from mesh.sim.env import MeshEnv, Action
from mesh.sim.scenarios import make_scenario

def _hold(obs):
    return Action(obs.channels.copy(), obs.next_hop.copy())

def test_reset_with_new_seed_changes_topology():
    env = MeshEnv(make_scenario("S1"), 0)
    a = env.reset().edge_attr
    b = env.reset(seed=1).edge_attr
    assert env.seed == 1 and not (a.shape == b.shape and np.array_equal(a, b))
    assert np.array_equal(env.reset(seed=0).edge_attr, a)

def test_no_interferer_lookahead():
    env = MeshEnv(make_scenario("S4"), 0); obs = env.reset()
    for _ in range(50):                          # steps t = 0..49 executed, interferer starts at t = 50
        obs, _, _ = env.step(_hold(obs))
    before = obs.edge_attr[:, 3].copy()
    obs, _, _ = env.step(_hold(obs))             # t = 50 executed with interferer
    assert before.sum() > obs.edge_attr[:, 3].sum()

def test_unreachable_hop_feature_is_n():
    env = MeshEnv(make_scenario("S1"), 0); env.reset()
    env.alive[:] = False; env.alive[list(env.gw)] = True     # every non-gateway unreachable
    obs = env._observe(None)
    nongw = [i for i in range(20) if i not in env.gw]
    assert (obs.node_feat[nongw, 3] == 20).all() and (obs.node_feat[list(env.gw), 3] == 0).all()

def test_step_after_done_raises():
    cfg = make_scenario("S1"); env = MeshEnv(cfg, 0); obs = env.reset()
    for _ in range(cfg.episode_len):
        obs, _, _ = env.step(_hold(obs))
    with pytest.raises(RuntimeError):
        env.step(_hold(obs))

def test_out_of_range_next_hop_not_a_route_change():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    a = _hold(obs); g = obs.gateways[0]; a.next_hop[g] = -5
    _, _, info = env.step(a)
    assert info["n_route_change"] == 0
```

- [ ] **Step 2: Run** `python -m pytest tests/test_env_hardening.py -v` → 5 FAIL.
- [ ] **Step 3: Implement the five changes in `mesh/sim/env.py`.** Keep `_ext_noise(t)`; `_observe` takes the noise of `self.t - 1` (or `None` when `self.t == 0`).
- [ ] **Step 4: Run** the same command → 5 passed; then `python -m pytest -q` → all pass (Plan 1 tests unchanged).
- [ ] **Step 5: Commit** `fix: env hardening for learners (reset seed, no lookahead, hop feature, done guard, next_hop range)`.

---

### Task 2: Shared agent interface

**Files:**
- Create: `mesh/rl/__init__.py`, `mesh/rl/agents_io.py`
- Test: `tests/test_agents_io.py`

**Interfaces:**
- Consumes: `Observation`, `Action`, `edge_matrix`, `NODE_FEATS`, `RewardWeights` (Plan 1 / Task 1).
- Produces (`mesh/rl/agents_io.py`):
  - constants `K_SLOTS = 4`, `N_DL_CH = 6`, `N_ACTIONS = 24`
  - `neighbor_slots(obs) -> np.ndarray int (n, K_SLOTS)` — link neighbours by SNR desc, −1 padding.
  - `action_mask(obs, slots) -> np.ndarray bool (n, N_ACTIONS)` — action valid iff its slot holds an alive neighbour (for gateways: only slot 0, any channel). A node with no valid slot: only slot-0 actions valid (decoded to `next_hop = −1`). Dead nodes: all actions valid (ignored downstream).
  - `decode(obs, slots, a: np.ndarray int (n,)) -> Action` — per Global Constraints; a slot that is −1 or holds a dead node decodes to `next_hop = −1`; parents processed in tree-depth order so `channels[u,0]` reads the parent's new downlink.
  - `agent_features(obs, slots) -> tuple[np.ndarray (n, F_O), np.ndarray (n, F_M)]`
    - `o_i` (shared with neighbours, Yuan's o): `[offered_load, unmet, is_gateway, hop_to_gw / n, alive, onehot(radio1 channel, 7), onehot(current parent slot, K_SLOTS+1)]` → `F_O = 17` (last one-hot cell = "no parent").
    - `m_i` (local only, Yuan's m): per slot `[is_valid, snr_db / 40, rate_mbps / 143.4, load_frac]` → `F_M = 16`.
  - `per_agent_reward(prev_obs, action, info, reward_global, w: RewardWeights) -> np.ndarray (n,)` — formula in Global Constraints; dead nodes → 0.
  - `alive_mask(obs) -> np.ndarray bool (n,)`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.sim.graph import common_channel
from mesh.rl.agents_io import (K_SLOTS, N_ACTIONS, neighbor_slots, action_mask, decode,
                               agent_features, per_agent_reward, alive_mask)
from mesh.sim.env import RewardWeights

def _obs(name="S2", seed=0):
    env = MeshEnv(make_scenario(name), seed); return env, env.reset()

def test_slots_sorted_by_snr_and_padded():
    _, obs = _obs()
    s = neighbor_slots(obs)
    assert s.shape == (20, K_SLOTS) and (s[:, 0] >= 0).all()

def test_decode_tree_channels_share_with_parent():
    env, obs = _obs(); slots = neighbor_slots(obs)
    rng = np.random.default_rng(0); m = action_mask(obs, slots)
    a = np.array([rng.choice(np.flatnonzero(m[i])) for i in range(20)])
    act = decode(obs, slots, a); cc = common_channel(act.channels)
    for u in range(20):
        if act.next_hop[u] >= 0:
            assert cc[u, act.next_hop[u]] >= 1     # parent link uses a downlink channel
    assert (act.next_hop[list(obs.gateways)] == -1).all()

def test_mask_isolated_node():
    env, obs = _obs("S2", 0)
    slots = neighbor_slots(obs)
    u = next(i for i in range(20) if i not in obs.gateways)
    obs.node_feat[:, 4] = 0
    obs.node_feat[u, 4] = 1                       # u alive, every neighbour of u dead
    m = action_mask(obs, slots)
    assert m[u].sum() == 6 and m[u, 0::4].all()   # only slot-0 actions, any channel
    a = np.zeros(20, int); a[u] = 4 * 3           # channel index 3, slot 0
    assert decode(obs, slots, a).next_hop[u] == -1

def test_features_shapes():
    _, obs = _obs(); o, m = agent_features(obs, neighbor_slots(obs))
    assert o.shape == (20, 17) and m.shape == (20, 16) and np.isfinite(o).all() and np.isfinite(m).all()

def test_dead_node_excluded():
    env, obs = _obs("S6", 0)
    for _ in range(102):                          # failure at t=100; t=101 starts from an obs that shows it
        prev = obs
        act = decode(prev, neighbor_slots(prev), np.zeros(20, int))
        obs, r, info = env.step(act)
    dead = ~alive_mask(prev)
    rew = per_agent_reward(prev, act, info, r, RewardWeights())
    assert dead.sum() == 1 and rew[dead].item() == 0.0 and np.isfinite(rew).all()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_agents_io.py -v` → FAIL (import).
- [ ] **Step 3: Implement `mesh/rl/agents_io.py`.**
- [ ] **Step 4: Run** → 5 passed.
- [ ] **Step 5: Commit** `feat: shared per-agent action/observation/reward interface`.

---

### Task 3: Q-neutral (tabular Q-learning)

**Files:**
- Create: `mesh/rl/q_neutral.py`
- Test: `tests/test_q_neutral.py`

**Interfaces:**
- Consumes: Task 2 functions.
- Produces:
  - `discretize(o: np.ndarray (F_O,), m: np.ndarray (F_M,)) -> tuple` — `(load_bin ∈ 0..2 [<1, <5, ≥5 Mbps], unmet>0.1, is_gateway, min(hop,3), radio1 channel, best valid slot by rate, rate_bin of that slot ∈ 0..2 [<30, <80, ≥80])`.
  - `class QNeutral: __init__(self, alpha=0.1, gamma=0.98, eps_start=0.6, eps_end=0.05, eps_decay_steps=50_000, seed=0)`
    - `act(self, obs, explore=False) -> Action`
    - `learn(self, obs, action_idx: np.ndarray (n,), rewards: np.ndarray (n,), next_obs, done: bool) -> dict` — one shared table `dict[tuple, np.ndarray(N_ACTIONS)]` (parameter sharing across agents; unseen states init 0); masked actions excluded from `max`.
    - `last_action_idx: np.ndarray` set by `act` (the trainer reads it).
    - `save(path)`, `QNeutral.load(path)` (pickle).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.rl.q_neutral import QNeutral, discretize
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.rl.agents_io import N_ACTIONS

def test_discretize_is_hashable_and_small():
    o = np.zeros(17); m = np.zeros(16); k = discretize(o, m)
    assert hash(k) is not None and len(k) == 7

def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    q = QNeutral(eps_start=1.0, eps_end=1.0, seed=0)
    for _ in range(300):
        q.act(obs, explore=True); a = q.last_action_idx
        r = np.where(a == 5, 1.0, 0.0)
        q.learn(obs, a, r, obs, done=True)
    q.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (q.last_action_idx[nongw] == 5).mean() > 0.8
```

- [ ] **Step 2: Run** `python -m pytest tests/test_q_neutral.py -v` → FAIL.
- [ ] **Step 3: Implement.** Action 5 (= channel 2, slot 1) may be masked for some nodes; the test's threshold 0.8 allows that.
- [ ] **Step 4: Run** → 2 passed.
- [ ] **Step 5: Commit** `feat: Q-neutral tabular baseline`.

---

### Task 4: Yuan GQN (GAT + DQN)

**Files:**
- Create: `mesh/rl/yuan_gqn.py`
- Test: `tests/test_yuan_gqn.py`

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - `class GQNNet(nn.Module): __init__(self, f_o=17, f_m=16, hidden=64, heads=4, layers=2, tau=0.25)`; `forward(o: Tensor (B, n, f_o), m: Tensor (B, n, f_m), nbr: Tensor long (B, n, 4), nbr_valid: Tensor bool (B, n, 4)) -> tuple[Tensor (B, n, N_ACTIONS), Tensor (B, n, heads, 4)]`
    - `nbr[:, i] = [i, slot0, slot1, slot2]` (self + Yuan's 3 neighbours, Table 2), invalid entries masked.
    - Encoder: `MLP(f_o → hidden → hidden)` on `o` only (Yuan: only the communicated state is encoded).
    - Each conv layer (Eq. 11–12): per head `α_ij = softmax_j(τ · (W_m h_i) · (W_m h_j))` over valid `j`, `h'_i = ReLU(concat_m Σ_j α_ij W_m h_j)`, head dim `hidden // heads`.
    - Q head: `MLP([o, m, h, h'_1, h'_2] → 128 → 128 → N_ACTIONS)` (paper uses a CNN; MLP — see adaptation notes).
    - Returns Q values and the last layer's attention weights.
  - `class YuanAgent: __init__(self, lr=1e-3, gamma=0.98, batch=10, buffer=10_000, soft_tau=0.01, kl_lambda=0.03, eps_start=0.6, eps_end=0.05, eps_decay_steps=50_000, seed=0, device=None)`
    - `act(obs, explore=False) -> Action`, sets `last_action_idx`; masked Q = −inf; ε-random picks among valid actions only.
    - `learn(obs, action_idx, rewards, next_obs, done) -> dict` — pushes the graph transition (arrays, not tensors) into a ring buffer; once ≥ `batch`, samples `batch` graphs (all same n) and minimises Eq. 13:
      `mean_i [r_i + γ (1−done) max_{a' valid} Q⁻(s', a') − Q(s, a_i)]² + λ · mean_heads KL(W(s) ‖ W(s'))` over alive agents, with `W(s')` from the **online** net under `no_grad` (paper §4.2). Soft target update `θ⁻ ← τθ + (1−τ)θ⁻` every learn call.
    - `save(path)` (`torch.save` state_dict + hparams), `YuanAgent.load(path, device=None)` mapping to `device`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, torch
from mesh.rl.yuan_gqn import GQNNet, YuanAgent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.rl.agents_io import N_ACTIONS

def _batch(n=5):
    o = torch.randn(2, n, 17); m = torch.randn(2, n, 16)
    nbr = torch.arange(n).view(1, n, 1).repeat(2, 1, 4)
    nbr[..., 1] = (nbr[..., 0] + 1) % n
    valid = torch.ones(2, n, 4, dtype=torch.bool); valid[..., 2:] = False
    return o, m, nbr, valid

def test_shapes_and_attention_normalised():
    q, att = GQNNet()(*_batch())
    assert q.shape == (2, 5, N_ACTIONS) and att.shape == (2, 5, 4, 4)
    assert torch.allclose(att.sum(-1), torch.ones(2, 5, 4))

def test_attention_ignores_masked():
    _, att = GQNNet()(*_batch())
    assert (att[..., 2:] == 0).all()

def test_td_target_respects_done():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    ag = YuanAgent(batch=1, kl_lambda=0.0, seed=0)
    assert np.allclose(ag.td_target(obs, np.ones(20), obs, done=True), 1.0)
    assert not np.allclose(ag.td_target(obs, np.ones(20), obs, done=False), 1.0)

def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    ag = YuanAgent(batch=4, eps_start=1.0, eps_end=1.0, seed=0, lr=3e-3)
    for _ in range(400):
        ag.act(obs, explore=True); a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    ag.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (ag.last_action_idx[nongw] == 5).mean() > 0.8
```

  `td_target(obs, rewards, next_obs, done) -> np.ndarray (n,)` is a public helper on `YuanAgent` used by `learn`.

- [ ] **Step 2: Run** `python -m pytest tests/test_yuan_gqn.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → 4 passed.
- [ ] **Step 5: Commit** `feat: Yuan GQN baseline (GAT + DQN + temporal relation KL)`.

---

### Task 5: Li GAT-A2C

**Files:**
- Create: `mesh/rl/li_gat_a2c.py`
- Test: `tests/test_li_gat_a2c.py`

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - `class GATA2CNet(nn.Module): __init__(self, f_in=33, heads=8, hidden=8, out=16, dropout=0.6)`; `forward(x: Tensor (B, n, f_in), adj: Tensor bool (B, n, n), mask: Tensor bool (B, n, N_ACTIONS)) -> tuple[Tensor logits (B, n, N_ACTIONS), Tensor value (B, n)]`
    - `x = concat(o, m)` z-scored with running mean/std (Li Eq. 10; buffers updated in `train()` mode only).
    - Graph: `adj` = conflict graph (Li: edge iff mutual interference) + self-loops; dead nodes have no edges.
    - GAT layer 1: 8 heads × 8, additive attention `LeakyReLU(aᵀ[Wh_i‖Wh_j])` (0.2 slope), softmax over `adj`, concat, ELU; layer 2: 1 head → `out = 16`. Dropout 0.6 on inputs and attention.
    - Actor `MLP([x, z] → 500 → 250 → N_ACTIONS)`, logits masked to −1e9; critic `MLP([x, z] → 500 → 250 → 1)` (state value V).
  - `class LiAgent: __init__(self, lr=5e-3, lr_min=1e-4, lr_decay=0.96, lr_decay_every=1000, gamma=0.98, entropy=0.01, value_coef=0.5, seed=0, device=None)`
    - `act(obs, explore=False) -> Action` — sample when exploring, argmax otherwise; sets `last_action_idx`.
    - `learn(obs, action_idx, rewards, next_obs, done) -> dict` — 1-step on-policy A2C per alive agent: `A_i = r_i + γ(1−done)V(s')_i − V(s)_i`; loss `−mean(logπ(a_i)·A_i.detach()) + value_coef·mean(A_i²) − entropy·mean(H)`; Adam; lr step decay as above.
    - `adjacency(obs) -> np.ndarray bool (n, n)` public helper.
    - `save(path)`, `LiAgent.load(path, device=None)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, torch
from mesh.rl.li_gat_a2c import GATA2CNet, LiAgent
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.rl.agents_io import N_ACTIONS

def test_shapes_and_mask():
    net = GATA2CNet().eval()
    x = torch.randn(1, 6, 33); adj = torch.eye(6, dtype=torch.bool)[None]
    mask = torch.ones(1, 6, N_ACTIONS, dtype=torch.bool); mask[..., 0] = False
    logits, v = net(x, adj, mask)
    assert logits.shape == (1, 6, N_ACTIONS) and v.shape == (1, 6)
    assert (torch.softmax(logits, -1)[..., 0] < 1e-6).all()

def test_adjacency_is_conflict_graph_with_self_loops():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    adj = LiAgent.adjacency(obs)
    assert adj.diagonal().all() and (adj == adj.T).all() and adj.sum() > 20

def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    ag = LiAgent(seed=0, lr=3e-3)
    for _ in range(400):
        ag.act(obs, explore=True); a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    ag.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (ag.last_action_idx[nongw] == 5).mean() > 0.8
```

- [ ] **Step 2: Run** `python -m pytest tests/test_li_gat_a2c.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → 3 passed.
- [ ] **Step 5: Commit** `feat: Li GAT-A2C baseline`.

---

### Task 6: Training loop, CLI, checkpoint policies in eval

**Files:**
- Create: `mesh/rl/train.py`
- Modify: `mesh/eval/run.py`, `pyproject.toml`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Tasks 2–5; `make_scenario`, `MeshEnv`, `run_episode`.
- Produces:
  - `LEARNERS = {"q_neutral": QNeutral, "yuan": YuanAgent, "li": LiAgent}`
  - `train(method: str, steps: int, seed: int = 0, n: int = 20, out: str | None = None, log_every: int = 5000) -> object` — loop: draw scenario name uniformly from S1–S6 and an episode seed from `np.random.default_rng(seed)`; `obs = env.reset()`; each step `agent.act(obs, explore=True)` → `env.step` → `per_agent_reward` → `agent.learn`; on `done` start a new episode; stops after exactly `steps` env steps; prints mean episode reward every `log_every` steps; saves to `out` if given; returns the agent.
  - CLI: `python -m mesh.rl.train --method yuan --steps 100000 --seed 0 --out runs/yuan_s0.pt`
  - `mesh/eval/run.py`: `--policy` also accepts `q_neutral|yuan|li` together with `--checkpoint PATH`; loads with `LEARNERS[name].load(path)`; `run_episode` calls `policy.act(obs)` (explore defaults to False).
  - `pyproject.toml`: add `torch` to dependencies.

- [ ] **Step 1: Write the failing tests**

```python
import subprocess, sys, numpy as np
from mesh.rl.train import train, LEARNERS
from mesh.eval.run import run_episode
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario

def test_train_smoke_all_methods(tmp_path):
    for name in LEARNERS:
        ag = train(name, steps=450, seed=0, n=10, out=str(tmp_path / f"{name}.pt"), log_every=10**9)
        m = run_episode(MeshEnv(make_scenario("S1", n=10), 3), ag)
        assert all(np.isfinite(v) for v in m.values())

def test_checkpoint_roundtrip_cpu(tmp_path):
    for name, cls in LEARNERS.items():
        p = str(tmp_path / f"{name}.pt")
        a = train(name, steps=250, seed=0, n=10, out=p, log_every=10**9)
        b = cls.load(p, device="cpu") if name != "q_neutral" else cls.load(p)
        obs = MeshEnv(make_scenario("S1", n=10), 5).reset()
        assert np.array_equal(a.act(obs).next_hop, b.act(obs).next_hop)

def test_eval_cli_with_checkpoint(tmp_path):
    p = str(tmp_path / "li.pt"); train("li", steps=250, seed=0, n=10, out=p, log_every=10**9)
    out = subprocess.run([sys.executable, "-m", "mesh.eval.run", "--scenario", "S1", "--policy", "li",
                          "--checkpoint", p, "--seeds", "1", "--n", "10"],
                         capture_output=True, text=True, check=True)
    assert "mean" in out.stdout
```

  `load(path, device=None)` on `YuanAgent`/`LiAgent`; `QNeutral.load(path)`.

- [ ] **Step 2: Run** `python -m pytest tests/test_train.py -v` → FAIL.
- [ ] **Step 3: Implement `train.py` and the `run.py` changes.**
- [ ] **Step 4: Run** → 3 passed; `python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: shared RL training loop and checkpoint evaluation`.

---

### Task 7: Install CUDA torch, adaptation notes, first real training runs

**Files:**
- Create: `docs/baselines-adaptation.md`
- Modify: none (results are reported, not committed as data)

**Interfaces:** consumes Task 6 CLI.

- [ ] **Step 1: Install the CUDA build.**
```bash
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```
Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → contains `True` and `RTX 5060`. If CUDA is still unavailable, continue on CPU and say so in the report (training is slower, results unchanged).

- [ ] **Step 2: Write `docs/baselines-adaptation.md`.** One section per baseline with: source, what was kept, what changed, and why. Must include at least:
  - All: agent = mesh node; action = downlink channel × parent slot (papers: channel × power); radio 0 follows parent; per-agent reward from Global Constraints (papers: interference-temperature × efficiency for Yuan, V2N/V2V mix for Li); no edge features in attention (spec: edge features are EGAT-MAPPO's contribution).
  - Q-neutral: definition assumed = tabular Q-learning with a reward that weights no objective specially; Yuan also compares against Q-learning; state discretisation listed.
  - Yuan: kept Eq. 11–13, Table 2 values (3 neighbours, 2 layers, batch 10, λ 0.03, τ 0.25, lr 0.001, soft 0.01, ε 0.6, γ 0.98, ReLU, Adam); changed: Q head MLP instead of CNN; heads = 4 and buffer = 10 000 (not given in the paper); no power action.
  - Li: kept conflict-graph GAT, z-scored inputs, 2 layers / 8 heads / dropout 0.6, actor/critic 500-250, lr 0.005→1e-4; changed: critic is V (paper inconsistent Q vs V), standard 1-step on-policy A2C instead of replay (paper ambiguous), auxiliary GAT MSE loss dropped (ill-defined), γ = 0.98 (paper value ambiguous), entropy 0.01 and value coef 0.5 (not given).
  - Pending: hyperparameter tuning with Optuna (20 trials each) belongs to Plan 5.

- [ ] **Step 3: Train each learner once** (run in background, ~minutes each):
```bash
python -m mesh.rl.train --method q_neutral --steps 100000 --seed 0 --out runs/q_neutral_s0.pt
```
```bash
python -m mesh.rl.train --method yuan --steps 100000 --seed 0 --out runs/yuan_s0.pt
```
```bash
python -m mesh.rl.train --method li --steps 100000 --seed 0 --out runs/li_s0.pt
```
Add `runs/` to `.gitignore`.

- [ ] **Step 4: Evaluate all five baselines on S1–S6** with `python -m mesh.eval.run --scenario <S> --policy <p> [--checkpoint runs/<p>_s0.pt] --seeds 5` and put the table of `aggregate / jain / p5 / utility` in the final report. No assertion: this is the first look, not the paper result (Plan 5 does seeds, tuning and statistics). If a learner is below M0 everywhere, report it as a finding with its training curve, do not tune it here.

- [ ] **Step 5: Commit** `docs: baseline adaptation notes; ignore runs/`.

---

## Done when

- `python -m pytest -q` all green.
- `runs/{q_neutral,yuan,li}_s0.pt` exist and evaluate through the CLI on all six scenarios.
- `docs/baselines-adaptation.md` lists every departure from the papers.
- Spec milestone M4 (Q-neutral, Yuan, Li runnable with equal budget, adaptation recorded) satisfied, minus tuning (Plan 5).
