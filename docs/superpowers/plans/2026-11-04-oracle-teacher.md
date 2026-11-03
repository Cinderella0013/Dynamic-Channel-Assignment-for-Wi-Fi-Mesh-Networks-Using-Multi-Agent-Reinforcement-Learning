# Oracle Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EGAT v1/v2 imitate a clairvoyant lookahead oracle instead of Ko during their BC phase.

**Architecture:** A new privileged policy `LookaheadOracle` is added in `mesh/baselines/oracle.py`. `train()` gets a `teacher` option. While the agent's phase is "bc", it sets `agent.ko` to a fresh oracle bound to each episode's env, and restores `KoLocalCost()` after BC. The agents' BC code path stays unchanged, because both already label with `ko_action_idx(obs, slots, self.ko)`.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-oracle-teacher-design.md`

## Global Constraints

- No changes to `mesh/sim`, the reward, the checkpoint selection (validation reward), `mesh/eval` or the pre-registered composite.
- Oracle values: horizon=10, t_c=10, one coordinate pass per round starting from Ko's action, objective = summed env reward over the horizon on `copy.deepcopy(env)`.
- `teacher="ko"` (the default) must reproduce today's behaviour exactly.
- The oracle is never used at evaluation time. The Ko prior and fallback after BC remain `KoLocalCost`.
- Training code changes must not disturb runner jobs that are already running. Python loads modules at start, so editing files is safe; do not switch branches.

## Review Focus

- **Dead node or gateway:** the oracle must skip nodes whose `next_hop < 0`. Owned by Task 1 (valid-action test on S4, which has node failures).
- **Parent changed to a node whose downlink differs:** the child's uplink radio must follow the new parent. Owned by Task 1 (radio-consistency assert).
- **Episode boundary inside BC:** each new env must get a new oracle; a stale env would leak the previous episode's future. Owned by Task 2 (`agent.ko.env is env` assert).
- **BC finishing mid-episode:** `agent.ko` must be Ko immediately after the phase flips, not only at the next episode. Owned by Task 2 (check after training finishes, plus a check inside the loop).
- **Agents without a `ko` / `phase` attribute (q_neutral, yuan, li) given `teacher="oracle"`:** this is an error, not a silent no-op. Owned by Task 2.

---

### Task 1: LookaheadOracle

**Files:**
- Create: `mesh/baselines/oracle.py`
- Test: `tests/test_oracle.py`

**Interfaces:**
- Produces: `LookaheadOracle(env: MeshEnv, horizon: int = 10, t_c: int = 10)` with `.act(obs) -> Action` and attribute `.env`.
- Port the spike at `C:\Users\qoqao\AppData\Local\Temp\claude\a24f837d081b3773\b56590b2-fa09-4d31-9d6a-90eca00bfb47\scratchpad\oracle.py`: classes `Oracle`, `rollout`, `with_channel`, `with_parent`. Use objective "reward" only, and replace the module globals with constructor arguments.

- [ ] **Step 1: Write failing tests**

```python
def test_oracle_action_valid_and_radio_consistent():
    for scen in ("S2", "S4"):
        env = MeshEnv(make_scenario(scen), 0); obs = env.reset(); o = LookaheadOracle(env)
        for _ in range(12):
            a = o.act(obs)
            assert a.channels.min() >= 0 and a.channels.max() < N_CHANNELS
            kids = np.flatnonzero(a.next_hop >= 0)
            assert (a.channels[kids, 0] == a.channels[a.next_hop[kids], 1]).all()
            obs, _, _ = env.step(a)

def test_oracle_round_not_worse_than_ko():
    env = MeshEnv(make_scenario("S2"), 0); obs = env.reset()
    o = LookaheadOracle(env)
    assert rollout(env, o.act(obs), 10) >= rollout(env, KoLocalCost().act(obs), 10) - 1e-9
```

`rollout(env, action, horizon) -> float` is a module function and is exported.

- [ ] **Step 2:** Run `python -m pytest tests/test_oracle.py -q`. Expected: ImportError.
- [ ] **Step 3:** Implement `mesh/baselines/oracle.py` (port of the spike; module docstring says "privileged: reads future traffic; training-time teacher only").
- [ ] **Step 4:** Run `python -m pytest tests/test_oracle.py -q`. Expected: 2 passed.
- [ ] **Step 5:** Commit with message `feat: lookahead oracle (privileged teacher)`.

### Task 2: Teacher option in train

**Files:**
- Modify: `mesh/rl/train.py` (`train()` signature, the episode-start block at `if obs is None:`, and `main()` argparse)
- Test: `tests/test_train_teacher.py`

**Interfaces:**
- Consumes: `LookaheadOracle` (Task 1), `KoLocalCost`.
- Produces: `train(method, steps, seed=0, n=20, out=None, log_every=5000, val_every=None, teacher="ko", **agent_kwargs)` and CLI `--teacher {ko,oracle}`.

- [ ] **Step 1: Write failing tests**

```python
def test_oracle_teacher_only_during_bc(monkeypatch):
    seen = []
    orig = EGATv1Agent.learn
    def spy(self, *a, **k):
        seen.append((self.phase, type(self.ko).__name__))
        return orig(self, *a, **k)
    monkeypatch.setattr(EGATv1Agent, "learn", spy)
    ag = train("egat_v1", steps=260, teacher="oracle", bc_steps=200, bc_epochs=1)
    assert {k for p, k in seen if p == "bc"} == {"LookaheadOracle"}
    assert {k for p, k in seen if p != "bc"} <= {"KoLocalCost"} and isinstance(ag.ko, KoLocalCost)

def test_ko_teacher_default_unchanged():
    ag = train("egat_v2", steps=60, bc_steps=40, bc_epochs=1)
    assert isinstance(ag.ko, KoLocalCost)

def test_oracle_teacher_rejects_learners_without_bc():
    with pytest.raises(ValueError):
        train("yuan", steps=10, teacher="oracle")
```

Also add an assert inside the loop, `agent.ko.env is env`, through a monkeypatched `LookaheadOracle` subclass that records the env it was given and checks it against the env returned by the patched `MeshEnv` in the same episode. Keep it to one test, `test_oracle_bound_to_current_episode_env`, with steps=420 and bc_steps=400, so the BC phase spans 3 episodes.

- [ ] **Step 2:** Run `python -m pytest tests/test_train_teacher.py -q`. Expected: FAIL (`unexpected keyword argument 'teacher'`).
- [ ] **Step 3:** Implement the following.
  - In `train()`, if `teacher == "oracle"` and the agent lacks `phase` or `ko`, raise `ValueError`.
  - At each episode start, while `agent.phase == "bc"`, set `agent.ko = LookaheadOracle(env)`.
  - After every `agent.learn(...)`, if `teacher == "oracle"` and the phase is no longer "bc" and `agent.ko` is not a `KoLocalCost`, set `agent.ko = KoLocalCost()`.
  - Add the argparse `--teacher` flag with choices `("ko", "oracle")` and default "ko".
- [ ] **Step 4:** Run `python -m pytest tests/test_train_teacher.py tests/test_train_val.py tests/test_train.py -q`. Expected: all pass.
- [ ] **Step 5:** Commit with message `feat: --teacher oracle for EGAT BC phase`.

### Task 3: Variant labels in the suite, then launch

**Files:**
- Modify: `mesh/eval/suite.py` `_factory` (currently `LEARNERS[name].load(path)`, which raises KeyError for `egat_v1_oracle`)
- Modify: `docs/egat-mappo.md` (add an "Oracle teacher" section: what the teacher is, that it is privileged at training time only, the ~75 min of extra BC time, and the checkpoint names)
- Test: `tests/test_suite.py`

**Interfaces:**
- Produces: `--method <label>=<ckpt>`, where `<label>` is a learner name or a learner name plus a suffix. The class is the longest `LEARNERS` key that prefixes the label, so `egat_v1_oracle` loads `EGATv1Agent`. The results are keyed by the full label.

- [ ] **Step 1: Write failing test** `test_factory_resolves_variant_label`:
  - Save an untrained `EGATv1Agent(seed=0)` to `tmp_path/"m.pt"`.
  - `name, f = _factory(f"egat_v1_oracle={path}")`.
  - Assert `name == "egat_v1_oracle"` and `type(f()).__name__ == "EGATv1Agent"`.
- [ ] **Step 2:** Run `python -m pytest tests/test_suite.py -q`. Expected: FAIL with KeyError.
- [ ] **Step 3:** Implement the longest-prefix lookup. An unknown label raises `ValueError` naming the label.
- [ ] **Step 4:** Run the full test suite with `python -m pytest -q`. Expected: all pass. Do not pipe the output through `tail`; read the summary line.
- [ ] **Step 5:** Write the docs section, then commit with message `feat: suite variant labels; docs: oracle teacher`.
- [ ] **Step 6:** Send the runner session (SendMessage to `local_4e2b88ea-677c-4bdb-bd24-ca0474681f8d`) these jobs:
  - 6 runs: `python -m mesh.rl.train --method egat_v{1,2} --teacher oracle --steps 100000 --seed k --val-every 10000 --out runs/v7/ckpt/egat_v{1,2}_oracle_s{k}.pt` for k in 0, 1, 2. Start them as soon as GPU slots allow, at most 3 GPU training jobs at a time, and give them priority over seeds that have not started yet.
  - Evaluate each run in settings (a) and (b) at 20 seeds into `runs/v7/t<k>/{a,b}/suite_egat_v{1,2}_oracle.json`, using `--method egat_v1_oracle=runs/v7/ckpt/egat_v1_oracle_s<k>.pt`.
