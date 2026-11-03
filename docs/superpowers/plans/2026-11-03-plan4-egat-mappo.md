# Plan 4: EGAT-MAPPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the proposed model — edge-aware GATv2 + GRU encoder, two-timescale channel/routing heads, Ko-hybrid (BC warm start, residual prior, safe fallback), trained with MAPPO — as a learner that plugs into the Plan 3 training loop and evaluation CLI, with every component switchable for the Plan 5 ablations.

**Architecture:** Pure PyTorch, dense attention (n ≤ 80, same ruling as Plan 3). `mesh/model/ko_prior.py` exposes Ko's costs/actions in the shared action space. `mesh/model/egat.py` is the network. `mesh/model/agent.py` is `EGATAgent` with the Plan 3 learner interface (`act`, `learn`, `save`, `load`): the first `bc_steps` env steps act as Ko and collect data, then BC + V_Ko fitting, then PPO on rollouts. Registered as `"egat"` in `LEARNERS`, so `python -m mesh.rl.train --method egat` and `python -m mesh.eval.run --policy egat --checkpoint ...` work unchanged.

**Tech Stack:** Python ≥ 3.11, NumPy, PyTorch 2.11 + CUDA 12.8, pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-egat-mappo-mesh-design.md` (§2 architecture, §3 reward, §5.6 ablations).

## Global Constraints

- Action space, slots, decode, features, reward function: reuse `mesh/rl/agents_io.py` unchanged in meaning (`K_SLOTS=4`, `N_ACTIONS=24`, `a = c*K_SLOTS + j`). Same env, same scenarios, same budget as the baselines: `--steps` counts **every** env step, including the Ko data-collection phase.
- Every component has a flag in `EGATConfig` (Task 2) with these defaults: `use_edge_feat=True, use_gru=True, two_timescale=True, t_c=10, progress_mask=True, use_bc=True, use_ko_prior=True, use_fallback=True`.
- Hyperparameter defaults: `hidden=64, heads=4, layers=2, lr=3e-4, gamma=0.98, gae_lambda=0.95, clip=0.2, epochs=4, rollout=1000 (env steps), minibatch=50 (env steps), entropy=0.01, value_coef=0.5, max_grad=0.5, bc_steps=10_000, bc_epochs=20, bc_lr=1e-3, lam_start=0.5, lam_end=0.1, fallback_margin=0.0, fallback_entropy=2.0`.
- Reward: `per_agent_reward(..., lam=λ)` with λ annealed linearly from `lam_start` to `lam_end` over the run (spec §3). Baselines keep λ = 0.5. This difference is disclosed in `docs/egat-mappo.md`.
- Spec departures decided here (all disclosed in `docs/egat-mappo.md`):
  - One shared centralized critic instead of one critic per head. The channel head uses the GAE advantage at its decision step.
  - GRU state is stored with each transition and not back-propagated through time (truncation length 1).
  - Edge features come from `EDGE_FEATS` (`is_link, is_conflict, snr_db, rate_mbps, load_frac, same_channel`). Spec §2.1 also lists ETX and interference power, which the env does not expose.

## Review Focus

1. The progress mask leaves no valid parent for a routable node (all slot neighbours have hop ≥ own hop) → that cannot happen by construction (Plan 3 hop-first slots); if it ever does, the node falls back to slot 0 and decodes to −1 instead of crashing. Test: Task 3 `test_progress_mask_routes_are_loop_free`.
2. Fallback replaces a parent's action → its children's radio 0 must follow the parent's **final** channel. Test: Task 3 `test_fallback_merge_keeps_tree_channels`.
3. A rollout spans an episode boundary → GAE must not bootstrap across it, and GRU state must reset at `obs.t == 0`. Test: Task 4 `test_gae_cuts_at_done`, Task 3 `test_gru_state_resets_each_episode`.
4. Training stops during BC collection (e.g. tiny smoke runs) → the agent still acts (as Ko) and saves/loads. Test: Task 6 smoke via `test_train_smoke_all_methods`.
5. Off-decision steps in two-timescale mode → the channel head's log-prob and entropy must not enter the loss for those steps. Test: Task 4 `test_channel_logprob_only_on_decision_steps`.

---

## File Structure

```
mesh/baselines/ko.py        # Task 1: extract ko_route_via() (behaviour unchanged)
mesh/rl/agents_io.py        # Task 1: per_agent_reward(..., lam=0.5)
mesh/rl/train.py            # Task 1: agent kwargs, reward λ hook; Task 6: register "egat"
mesh/model/__init__.py
mesh/model/ko_prior.py      # Ko costs + Ko actions in the shared action space
mesh/model/egat.py          # EGATConfig, EdgeGATv2 layer, EGATNet
mesh/model/agent.py         # EGATAgent: act/fallback, rollout buffer, PPO, BC, save/load
docs/egat-mappo.md          # model card + departures from spec
tests/test_ko_prior.py, test_egat_net.py, test_egat_act.py, test_egat_ppo.py, test_egat_bc.py
```

---

### Task 1: Ko prior + loop hooks

**Files:**
- Modify: `mesh/baselines/ko.py`, `mesh/rl/agents_io.py`, `mesh/rl/train.py`
- Create: `mesh/model/__init__.py`, `mesh/model/ko_prior.py`
- Test: `tests/test_ko_prior.py`

**Interfaces:**
- Produces:
  - `mesh/baselines/ko.py`: `ko_route_via(obs, alive) -> np.ndarray (n, n)`. It returns Ko's `w + dist` matrix, with `inf` for unusable links. `KoLocalCost._route` calls it; the behaviour is identical and the existing Ko tests stay green.
  - `mesh/model/ko_prior.py`:
    - `ko_slot_costs(obs, slots) -> np.ndarray (n, K_SLOTS)`: `via[u, slots[u, j]]`. Invalid slots get `+inf`. Each row is then z-scored over its finite entries, giving 0 for a single finite entry and `+inf` kept where the slot is invalid.
    - `ko_channel_costs(obs) -> np.ndarray (n, N_DL_CH)`: Ko's local channel cost for downlink channels 1..6, summed over conflict neighbours that are not the node's children, using the **current** `obs.channels`. Each row is z-scored.
    - `ko_action_idx(obs, slots, ko: KoLocalCost) -> np.ndarray int (n,)`. It calls `ko.act(obs)`, then maps the result to `c*K + j`: `c = ko_channels[u,1] − 1`, and `j` is the slot holding Ko's parent. If that parent is not in a slot, `j` is the first valid slot, or 0.
  - `agents_io.per_agent_reward(prev_obs, action, info, reward_global, w, lam=0.5)`: `lam·local + (1−lam)·global/n_alive − penalties`. The default keeps the baselines bit-identical.
  - `train.train(method, steps, seed=0, n=20, out=None, log_every=5000, **agent_kwargs)`:
    - It passes `agent_kwargs` to the learner constructor.
    - It calls `per_agent_reward(..., lam=agent.reward_lam(step, steps))` when the agent has `reward_lam`, and uses `0.5` otherwise.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.env import MeshEnv, RewardWeights
from mesh.sim.scenarios import make_scenario
from mesh.baselines.ko import KoLocalCost
from mesh.rl.agents_io import neighbor_slots, decode, per_agent_reward, K_SLOTS
from mesh.model.ko_prior import ko_slot_costs, ko_channel_costs, ko_action_idx

def _obs(s="S2", seed=0):
    env = MeshEnv(make_scenario(s), seed); return env, env.reset()

def test_slot_cost_order_matches_ko_via():
    # Ko's hysteresis may keep an older parent, so compare cost ORDER with Ko's via matrix
    from mesh.baselines.ko import ko_route_via
    _, obs = _obs(); slots = neighbor_slots(obs); alive = obs.node_feat[:, 4] > 0
    via, cost = ko_route_via(obs, alive), ko_slot_costs(obs, slots)
    for u in range(20):
        ok = np.isfinite(cost[u])
        if ok.sum() > 1:
            assert np.array_equal(np.argsort(cost[u][ok], kind="stable"),
                                  np.argsort(via[u, slots[u][ok]], kind="stable"))

def test_costs_shapes_and_invalid_inf():
    _, obs = _obs(); slots = neighbor_slots(obs)
    sc, cc = ko_slot_costs(obs, slots), ko_channel_costs(obs)
    assert sc.shape == (20, K_SLOTS) and cc.shape == (20, 6) and np.isfinite(cc).all()
    assert np.isinf(sc[slots < 0]).all()

def test_ko_action_idx_reproduces_ko_through_interface():
    env, obs = _obs(); slots = neighbor_slots(obs); ko = KoLocalCost()
    a = ko_action_idx(obs, slots, ko); act = decode(obs, slots, a); raw = KoLocalCost().act(obs)
    assert (act.channels[:, 1] == raw.channels[:, 1]).all()
    assert (act.next_hop == raw.next_hop).mean() > 0.9

def test_lam_default_matches_old_reward():
    env, obs = _obs(); slots = neighbor_slots(obs); act = decode(obs, slots, np.zeros(20, int))
    _, r, info = env.step(act)
    a = per_agent_reward(obs, act, info, r, RewardWeights())
    b = per_agent_reward(obs, act, info, r, RewardWeights(), lam=0.5)
    c = per_agent_reward(obs, act, info, r, RewardWeights(), lam=0.1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_ko_prior.py -v` → FAIL (import).
- [ ] **Step 3: Implement** the three file changes and `ko_prior.py`.
- [ ] **Step 4: Run** → 4 passed; `python -m pytest -q` → all pass (Ko and baseline tests unchanged).
- [ ] **Step 5: Commit** `feat: Ko prior in the shared action space; reward λ and agent kwargs hooks`.

---

### Task 2: EGAT network

**Files:**
- Create: `mesh/model/egat.py`
- Test: `tests/test_egat_net.py`

**Interfaces:**
- Produces:
  - `@dataclass class EGATConfig`: all flags and hyperparameters from Global Constraints, as fields with those defaults.
  - `class EdgeGATv2(nn.Module): __init__(self, d_in, d_edge, heads, d_head)`; `forward(h: (B,n,d_in), e: (B,n,n,d_edge), adj: bool (B,n,n)) -> (B,n,heads*d_head)`.
    - GATv2 attention: `s_ij = aᵀ LeakyReLU(W_s h_i + W_t h_j + W_e e_ij)` per head, softmaxed over `j` with `adj[i,j]`.
    - Output: `Σ_j α_ij W_t h_j`.
    - With `d_edge = 0`, the `W_e` term is absent.
  - `class EGATNet(nn.Module): __init__(self, cfg: EGATConfig, f_node=F_O+F_M, d_edge=len(EDGE_FEATS))`
    - `forward(x: (B,n,f_node), e: (B,n,n,d_edge), adj: (B,n,n), h_prev: (B,n,hidden), ko_slot: (B,n,K), ko_chan: (B,n,6), slot_mask: (B,n,K)) -> dict` with keys:
      - `chan_logits (B,n,6)`, `route_logits (B,n,K)`
      - `value (B,n)`, `value_ko (B,n)`
      - `h (B,n,hidden)`
    - Encoder: `Linear(f_node→hidden)`, then `layers × [EdgeGATv2 → +residual → LayerNorm → ReLU]`. When `use_edge_feat=False`, `d_edge=0` and `e` is ignored.
    - GRU: `GRUCell(hidden, hidden)` on the encoder output with `h_prev`. When `use_gru=False`, it is the identity and `h = encoder output`.
    - Heads: `chan = MLP(hidden→64→6)` and `route = MLP(hidden→64→K)`.
    - Residual prior when `use_ko_prior`:
      - `chan_logits −= softplus(β_c)·nan_to_num(ko_chan, posinf=0)` and `route_logits −= softplus(β_r)·nan_to_num(ko_slot, posinf=0)`.
      - `β_c`, `β_r` are learnable and initialised so that `softplus(β) = 1`.
    - `route_logits` are masked to `−1e9` where `~slot_mask`.
    - Critic: `MLP([h_i, mean_j h_j] → 128 → 1)` (centralized). `value_ko` is a separate `MLP(hidden→64→1)` head.
  - `adj = link | conflict | self` from the observation.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from mesh.model.egat import EGATConfig, EdgeGATv2, EGATNet

def _inputs(n=6, B=2, d_edge=6):
    x = torch.randn(B, n, 33); e = torch.randn(B, n, n, d_edge)
    adj = torch.eye(n, dtype=torch.bool).repeat(B, 1, 1); adj[:, 0, 1] = adj[:, 1, 0] = True
    h = torch.zeros(B, n, 64); ks = torch.zeros(B, n, 4); kc = torch.zeros(B, n, 6)
    sm = torch.ones(B, n, 4, dtype=torch.bool); sm[..., 3] = False
    return x, e, adj, h, ks, kc, sm

def test_shapes_and_route_mask():
    out = EGATNet(EGATConfig())(*_inputs())
    assert out["chan_logits"].shape == (2, 6, 6) and out["route_logits"].shape == (2, 6, 4)
    assert out["value"].shape == (2, 6) and out["h"].shape == (2, 6, 64)
    assert (out["route_logits"][..., 3] < -1e8).all()

def test_attention_respects_adjacency():
    layer = EdgeGATv2(8, 6, 2, 4); h = torch.randn(1, 4, 8); e = torch.randn(1, 4, 4, 6)
    adj = torch.eye(4, dtype=torch.bool)[None]
    out1 = layer(h, e, adj); h2 = h.clone(); h2[0, 3] += 10.0
    assert torch.allclose(layer(h2, e, adj)[0, :3], out1[0, :3])

def test_edge_features_matter_only_when_enabled():
    torch.manual_seed(0); x, e, adj, h, ks, kc, sm = _inputs()
    on = EGATNet(EGATConfig(use_edge_feat=True)).eval(); off = EGATNet(EGATConfig(use_edge_feat=False)).eval()
    assert not torch.allclose(on(x, e, adj, h, ks, kc, sm)["route_logits"], on(x, e * 0, adj, h, ks, kc, sm)["route_logits"])
    assert torch.allclose(off(x, e, adj, h, ks, kc, sm)["route_logits"], off(x, e * 0, adj, h, ks, kc, sm)["route_logits"])

def test_ko_prior_shifts_logits_toward_low_cost():
    x, e, adj, h, ks, kc, sm = _inputs()
    net = EGATNet(EGATConfig(use_ko_prior=True)).eval()
    base = net(x, e, adj, h, ks, kc, sm)["route_logits"]
    ks2 = ks.clone(); ks2[..., 0] = -3.0
    assert (net(x, e, adj, h, ks2, kc, sm)["route_logits"][..., 0] > base[..., 0]).all()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_egat_net.py -v` → FAIL.
- [ ] **Step 3: Implement `mesh/model/egat.py`.**
- [ ] **Step 4: Run** → 4 passed.
- [ ] **Step 5: Commit** `feat: EGAT network (edge-aware GATv2, GRU, two heads, Ko residual prior, critics)`.

---

### Task 3: Acting — masks, two timescales, fallback, GRU state

**Files:**
- Create: `mesh/model/agent.py` (acting half)
- Test: `tests/test_egat_act.py`

**Interfaces:**
- Consumes: Task 1, Task 2, `agents_io`.
- Produces (`EGATAgent`):
  - `__init__(self, seed=0, device=None, **cfg_overrides)`. It builds `self.cfg = EGATConfig(**cfg_overrides)`, the net, Adam, and a `KoLocalCost`; sets `self.steps = 0`, `self.phase = "bc" if cfg.use_bc else "ppo"`, `self.last_action_idx = None`, and `self.fallback_rate = 0.0`.
  - `progress_mask(obs, slots) -> np.ndarray bool (n, K)`: `agents_io` slot validity AND `hop[slot] < hop[u]` (when `cfg.progress_mask`). Gateways get slot 0 only. A row with nothing valid gets slot 0 only.
  - `prepare(obs) -> dict`: numpy inputs for one step, with keys `x, e, adj, ko_slot, ko_chan, slot_mask, slots, alive, is_chan_step`.
    - `is_chan_step = (not cfg.two_timescale) or obs.t % cfg.t_c == 0`.
    - `e` is the dense `(n,n,6)` edge-feature tensor built from `obs.edge_index/edge_attr`.
  - `act(obs, explore=False) -> Action`:
    - It resets the GRU state to zeros when `obs.t == 0`.
    - **In BC phase with explore=True:** it returns Ko's action through the interface and records it for BC.
    - **Otherwise**, for each node:
      - Routing slot: sampled (explore) or argmax.
      - Channel: on chan steps, sampled or argmax. On other steps it keeps the current downlink `obs.channels[:,1] − 1`.
      - The action index is `c*K + j`.
    - **Fallback** (only when `explore=False` and `cfg.use_fallback`): node i takes Ko's index when `value_i < value_ko_i − fallback_margin` or the joint entropy of the heads `> fallback_entropy`. It updates `fallback_rate` (fraction of alive nodes) and then decodes once. The merge happens at the index level, so the tree channels stay consistent.
    - It stores what PPO needs for this step in `self._pending` (inputs, `h_prev`, action, log-probs, value, `is_chan_step`) and sets `last_action_idx`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.sim.graph import common_channel
from mesh.rl.agents_io import neighbor_slots, decode
from mesh.model.agent import EGATAgent
from mesh.model.ko_prior import ko_action_idx
from mesh.baselines.ko import KoLocalCost

def _env(s="S2", seed=0):
    env = MeshEnv(make_scenario(s), seed); return env, env.reset()

def test_progress_mask_routes_are_loop_free():
    env, obs = _env(); ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, device="cpu")
    for _ in range(30):
        act = ag.act(obs, explore=True)
        for u in range(20):
            v, k = u, 0
            while act.next_hop[v] >= 0 and k <= 20:
                v, k = act.next_hop[v], k + 1
            assert k <= 20
        obs, _, _ = env.step(act)

def test_two_timescale_holds_channels_off_decision_steps():
    env, obs = _env(); ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, device="cpu")
    obs, _, _ = env.step(ag.act(obs, explore=True))           # t=0 decision step
    for _ in range(8):                                          # t=1..8: no channel decisions
        before = obs.channels[:, 1].copy()
        obs, _, _ = env.step(ag.act(obs, explore=True))
        assert (obs.channels[:, 1] == before).all()

def test_fallback_always_equals_ko():
    env, obs = _env(); ag = EGATAgent(seed=0, use_bc=False, fallback_entropy=-1.0, device="cpu")
    slots = neighbor_slots(obs)
    a = ag.act(obs, explore=False); ko = decode(obs, slots, ko_action_idx(obs, slots, KoLocalCost()))
    assert np.array_equal(a.next_hop, ko.next_hop) and np.array_equal(a.channels, ko.channels)
    assert ag.fallback_rate == 1.0

def test_fallback_merge_keeps_tree_channels():
    env, obs = _env(); ag = EGATAgent(seed=0, use_bc=False, fallback_entropy=2.5, device="cpu")
    a = ag.act(obs, explore=False); cc = common_channel(a.channels)
    for u in range(20):
        if a.next_hop[u] >= 0:
            assert cc[u, a.next_hop[u]] >= 1

def test_gru_state_resets_each_episode():
    env, obs = _env(); ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, device="cpu")
    first = ag.act(obs).next_hop.copy()
    for _ in range(5):
        obs, _, _ = env.step(ag.act(obs))
    assert np.array_equal(ag.act(env.reset()).next_hop, first)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_egat_act.py -v` → FAIL.
- [ ] **Step 3: Implement the acting half of `mesh/model/agent.py`.**
- [ ] **Step 4: Run** → 5 passed.
- [ ] **Step 5: Commit** `feat: EGAT acting — progress mask, two-timescale channels, Ko fallback, GRU state`.

---

### Task 4: MAPPO learner

**Files:**
- Modify: `mesh/model/agent.py`
- Test: `tests/test_egat_ppo.py`

**Interfaces:**
- Produces (`EGATAgent`):
  - `learn(obs, action_idx, rewards, next_obs, done) -> dict`:
    - **BC phase:** it records `(prepare(obs), action_idx, rewards, done)`. Once `steps == bc_steps`, it runs `bc_fit()` (Task 5) and switches to `"ppo"`.
    - **PPO phase:** it appends `self._pending` plus `rewards`, `done` and the alive mask to the rollout. When the rollout holds `cfg.rollout` steps, it runs `update(next_obs)` and clears the rollout.
    - It always increments `self.steps`.
  - `gae(rewards (T,n), values (T,n), dones (T,), last_value (n,)) -> (adv (T,n), ret (T,n))`: standard GAE with `γ, λ`. `done` cuts the bootstrap at episode ends.
  - `update(last_obs) -> dict`: `cfg.epochs` passes over shuffled minibatches of `cfg.minibatch` env steps. Each pass:
    - Recomputes log-probs from stored inputs and `h_prev`.
    - Policy loss: PPO-clip on `logp = logp_route + is_chan_step·logp_chan`, with advantages normalised over alive agents.
    - Value loss: `value_coef · MSE(value, ret)`.
    - Entropy: `entropy · (H_route + is_chan_step·H_chan)`.
    - Masked to alive agents; gradients clipped at `max_grad`.
    - Returns `{"policy_loss", "value_loss", "entropy", "approx_kl"}`.
  - `reward_lam(step, total) -> float`: linear from `lam_start` to `lam_end`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, torch
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario
from mesh.model.agent import EGATAgent

def test_gae_cuts_at_done():
    ag = EGATAgent(seed=0, use_bc=False, device="cpu")
    r = np.ones((3, 2)); v = np.zeros((3, 2)); d = np.array([0.0, 1.0, 0.0])
    adv, ret = ag.gae(r, v, d, last_value=np.full(2, 100.0))
    assert np.allclose(ret[1], 1.0)                      # episode ended at t=1: no bootstrap
    assert ret[2, 0] > 50                                # t=2 bootstraps from last_value

def test_channel_logprob_only_on_decision_steps():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, rollout=4, minibatch=4, device="cpu")
    flags = []
    for _ in range(4):
        act = ag.act(obs, explore=True); flags.append(ag._pending["is_chan_step"])
        nxt, r, info = env.step(act); ag.learn(obs, ag.last_action_idx, np.zeros(20), nxt, False); obs = nxt
    assert flags == [True, False, False, False]

def test_learns_constant_best_action():
    env = MeshEnv(make_scenario("S1"), 0); obs = env.reset()
    ag = EGATAgent(seed=0, use_bc=False, use_fallback=False, use_ko_prior=False, two_timescale=False,
                   progress_mask=False, rollout=20, minibatch=10, lr=3e-3, device="cpu")
    for t in range(600):
        ag.act(obs, explore=True); a = ag.last_action_idx
        ag.learn(obs, a, np.where(a == 5, 1.0, 0.0), obs, done=True)
    ag.act(obs, explore=False)
    nongw = [i for i in range(20) if i not in obs.gateways]
    assert (ag.last_action_idx[nongw] == 5).mean() > 0.8

def test_reward_lam_anneals():
    ag = EGATAgent(seed=0, device="cpu")
    assert ag.reward_lam(0, 100) == 0.5 and abs(ag.reward_lam(100, 100) - 0.1) < 1e-9
```

  Note for `test_learns_constant_best_action`: `learn` with `obs` identical to `next_obs` and `done=True` makes the GAE return equal the reward.

- [ ] **Step 2: Run** `python -m pytest tests/test_egat_ppo.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → 4 passed.
- [ ] **Step 5: Commit** `feat: MAPPO learner for EGAT (rollouts, GAE, PPO-clip, λ anneal)`.

---

### Task 5: BC warm start + V_Ko

**Files:**
- Modify: `mesh/model/agent.py`
- Test: `tests/test_egat_bc.py`

**Interfaces:**
- Produces:
  - `EGATAgent.bc_fit() -> dict`. It runs over the recorded BC steps, which are Ko's own trajectories:
    - **Actor:** `bc_epochs` epochs of cross-entropy on Ko's indices (route head on Ko's slot, channel head on Ko's channel, decision steps only), using Adam at `bc_lr` with minibatches of `minibatch` env steps. It stops early once alive-agent agreement on the training data reaches ≥ 0.9.
    - **V_Ko:** `value_ko` is fitted by MSE to Ko's per-agent discounted Monte-Carlo returns, cut at `done`. The critic `value` is fitted to the same returns, as a warm start. `value_ko` is frozen afterwards (`requires_grad_(False)`).
    - It then clears the BC buffer.
    - Returns `{"agreement", "bc_loss", "vko_loss"}`.
  - `EGATAgent.agreement(obs) -> float`: the fraction of alive nodes whose greedy action index, with fallback off, equals `ko_action_idx`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mesh.rl.train import train
from mesh.sim.env import MeshEnv
from mesh.sim.scenarios import make_scenario

def test_bc_imitates_ko_on_unseen_topology():
    ag = train("egat", steps=1200, seed=0, n=10, log_every=10**9, bc_steps=1200, bc_epochs=30,
               device="cpu")
    assert ag.phase == "ppo"
    obs = MeshEnv(make_scenario("S2", n=10), 99).reset()
    assert ag.agreement(obs) > 0.7

def test_value_ko_frozen_after_bc():
    ag = train("egat", steps=400, seed=0, n=10, log_every=10**9, bc_steps=400, bc_epochs=2, device="cpu")
    assert not any(p.requires_grad for p in ag.net.value_ko.parameters())
```

  `train("egat", ...)` needs Task 6's registration. Implement the one-line `LEARNERS["egat"] = EGATAgent` registration in this task, and ledger that it was pulled forward from Task 6.

- [ ] **Step 2: Run** `python -m pytest tests/test_egat_bc.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → 2 passed.
- [ ] **Step 5: Commit** `feat: EGAT behaviour cloning from Ko and frozen V_Ko`.

---

### Task 6: Registration, checkpoints, first training run, model card

**Files:**
- Modify: `mesh/rl/train.py`, `mesh/eval/run.py` (`--policy` choices add `egat`)
- Create: `docs/egat-mappo.md`
- Test: existing `tests/test_train.py` (the loops over `LEARNERS` now include `egat`)

**Interfaces:**
- `EGATAgent.save(path)` and `EGATAgent.load(path, device=None)`. The checkpoint holds the cfg dict, net state, optimiser state, phase, steps and fallback stats. Unfinished BC data is **not** saved: a checkpoint saved during BC loads in BC phase with an empty buffer.
- `LEARNERS["egat"] = EGATAgent`.

- [ ] **Step 1: Run** `python -m pytest tests/test_train.py -v`. Expect it to FAIL on `egat` (save/load missing) or PASS if Task 5 already covered everything. Either way, finish `save`/`load` until 3 pass.
- [ ] **Step 2: Write `docs/egat-mappo.md`.** It is the model card. It lists:
  - each component and its flag;
  - the hyperparameter defaults;
  - every departure from the spec listed in Global Constraints;
  - the reward-λ difference from the baselines;
  - the fallback rule and the `fallback_rate` metric.
- [ ] **Step 3: Full suite** `python -m pytest -q` → all pass. Commit `feat: register EGAT-MAPPO learner, checkpoints, model card`.
- [ ] **Step 4: Train** (background, same budget as baselines):
```bash
python -m mesh.rl.train --method egat --steps 100000 --seed 0 --out runs/egat_s0.pt
```
- [ ] **Step 5: Evaluate** on S1–S6:
```bash
python -m mesh.eval.run --scenario S2 --policy egat --checkpoint runs/egat_s0.pt --seeds 5
```
  Run it for each of S1–S6. Report agg / Jain / p5 / utility / switches / route changes / drops / reward next to the Plan 3 table (M0, Ko, Q-neutral, Yuan, Li), and include `fallback_rate`.
  - There is no assertion. This is one seed with default hyperparameters; Plan 5 does seeds, tuning, statistics and ablations.
  - If EGAT is below Ko everywhere, report the training curve, BC agreement and fallback rate as findings. Do not tune here.

---

## Done when

- `python -m pytest -q` all green.
- `runs/egat_s0.pt` trains on the same 100k-step budget and evaluates through the CLI on S1–S6.
- `docs/egat-mappo.md` lists every component flag (for Plan 5 ablations) and every departure from the spec.
- Spec milestone M5 (EGAT-MAPPO trains, BC → MAPPO) satisfied. The "≥ Ko in S1" check is reported, not asserted.
