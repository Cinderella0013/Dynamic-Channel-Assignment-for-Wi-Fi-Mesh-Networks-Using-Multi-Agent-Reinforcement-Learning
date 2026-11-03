# Oracle teacher for EGAT v1/v2 - design addendum

Date: 2026-11-04. Extends `2026-11-03-egat-v1-v2-design.md`. Deadline: 2026-11-13 (sooner is better).

## 1. Why

Seed-0 gate result: v1 +0.109, v2 +0.112 composite over Ko on A+B. About 75% of the lead is stability; the throughput, fairness and p5 deltas are about 0.

An oracle spike was run on the switching-cost simulator (A+B, 3 seeds; `runs/oracle/oracle_h10_reward.json`). The oracle knows the next 10 steps of traffic, starts from Ko's action, makes one coordinate pass over the nodes, and scores each candidate by rolling out a copy of the env. Result vs Ko:

| | composite | aggregate | p5 | Jain | loss rate | switches + route changes |
|---|---|---|---|---|---|---|
| oracle | +0.287 | +15% | +26% | +1% | -66% | -36% |

The oracle wins every scenario, by +0.11 to +0.36. The channel + parent action space therefore has enough headroom; the action space stays as it is.

Hypothesis: the learners are capped by their teacher. v1's BC and v2's DQfD imitate Ko, so the policies start at Ko's level and RL adds little beyond it.

## 2. Change

Swap the teacher used during the BC phase from Ko to the oracle. Everything else stays fixed: the simulator, the RL phase, the reward, the checkpoint selection and the pre-registered evaluation.

1. **`mesh/baselines/oracle.py`.** `LookaheadOracle(env, horizon=10, t_c=10)` with `act(obs) -> Action`.
   - Behaviour is identical to the spike: every t_c steps it starts from Ko's action and makes one pass over the nodes. Per node it tries every downlink channel and every valid parent slot, and keeps a candidate if the summed env reward over `horizon` steps on a `copy.deepcopy(env)` improves.
   - Between rounds it holds its action, except that it follows Ko's immediate reroute when a parent breaks.
   - Changing a node's downlink channel also updates its children's uplink radio.
   - It is privileged: it reads the live env, including future traffic. It is never used at evaluation time.
2. **Teacher hook.** BC in both agents already labels with `ko_action_idx(obs, slots, self.ko)`, which calls `self.ko.act(obs)`.
   - `train(..., teacher="ko"|"oracle")` (CLI `--teacher`): with "oracle", at each episode start while `agent.phase == "bc"`, set `agent.ko = LookaheadOracle(env)`.
   - When the phase leaves "bc", restore `KoLocalCost()`. The Ko residual prior and fallback used at test time stay Ko.
   - Default "ko" means no behaviour change.
3. **Naming.** Checkpoints are named `egat_v1_oracle_s<k>.pt` and `egat_v2_oracle_s<k>.pt`. Evaluation reuses the `egat_v1` / `egat_v2` policies through `--method name=ckpt`; it needs no new eval code.

Online rather than pre-generated demos. The oracle costs about 0.45 s per step, so a 10k-step BC phase adds about 75 min per training run. A shared demo file would save about 2 h of wall time over 6 runs. However, it needs a replay path for observations, rewards, infos and the GRU state through `train`, which is more code and more risk. Online generation reuses the existing BC code path unchanged.

## 3. Fairness

- The teacher sees the future only during training ("learning using privileged information"). At evaluation every method sees the same observations.
- Baselines are not retrained, because their published designs have no teacher.
- Ablation, at no extra cost: v1/v2 with the Ko teacher (runner seeds 0-2) vs v1/v2 with the oracle teacher, same seeds.
- Checkpoint selection stays on validation reward for every learner. Composite-based selection was considered and rejected: it would favour our methods unless every baseline were retrained. This is listed as a limitation.

## 4. Experiments

- Train `egat_v1` and `egat_v2` with `--teacher oracle` at seeds 0, 1 and 2: 100k steps, `--val-every 10000`.
- Evaluate at 20 seeds in settings (a) A+B+C and (b) A+B, reusing runner's heuristic and baseline suites for the same training seed.
- Report the composite with paired significance (`mesh/eval/composite.py`), the ablation, and the oracle as a ceiling row for A+B at 3 seeds, labelled as privileged.
- Success uses the pre-registered criteria unchanged:
  - mean composite ≥ +10% vs every baseline on A+B;
  - better with p<0.05 in ≥60% of scenarios;
  - no scenario below -5% vs Ko.
- If the oracle-teacher models still fall short, B (DAgger rounds) is the next option. It needs its own approval.

## 5. Tests

- `LookaheadOracle` returns a valid Action: channels in range, and every child's radio 0 equals its parent's radio 1. On S2 it scores at least Ko's summed reward over one round.
- `train(..., teacher="oracle")` uses the oracle during BC: `agent.ko` is a `LookaheadOracle` while phase is "bc" and a `KoLocalCost` afterwards. Run with tiny `bc_steps` and steps.
- `teacher="ko"` leaves `agent.ko` as Ko throughout (regression).
