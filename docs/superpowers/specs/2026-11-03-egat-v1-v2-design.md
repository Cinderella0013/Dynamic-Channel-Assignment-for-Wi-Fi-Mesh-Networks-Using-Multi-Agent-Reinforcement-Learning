# EGAT v1 / v2 + Switching-Cost Simulator + Pre-registered Evaluation (Addendum)

- Date: 2026-11-03
- Status: design. v2 approved by the user. v1 waits on the 300k-step diagnosis (§3).
- Extends: `2026-11-03-egat-mappo-mesh-design.md`. Sections here replace that spec's §5.7–5.8 (statistics, success criteria) and add §2 to its simulator.

---

## 1. What the spike measured (why this addendum exists)

| Measurement | S2 | S4 | S5 |
|---|---|---|---|
| Ko reward (5 seeds) | 2714 | 2077 | 1820 |
| One-step clairvoyant oracle over Ko (2 seeds) | +7.9% / +9.2% | +2.1% / +5.0% | +4.7% / +7.7% |
| EGAT BC-only (10k steps, imitate Ko) | 2769 | 2079 | 1829 |
| EGAT BC + PPO 90k (Plan 4 result) | 2708 | 2043 | 1784 |
| Same, Ko prior switched off at eval | 2586 | 1967 | 1704 |

- **Little room above Ko in the current simulator.** An oracle that knows next-step traffic only beats Ko by about 2–9%. A 10–20% reward gain over Ko is not reachable while switching costs nothing but a reward penalty.
- **PPO makes the BC policy worse.** BC-only already beats Ko by about 2% on S2, and PPO drops below it. The learned prior weights fall from 1.0 to 0.62 (routes) and 0.72 (channels).
- **The decision:** model the real cost of switching in the simulator (§2), fix how the learner improves on Ko (v1, §3), and add the planner the user specified (v2, §4). Evaluation is locked before any Plan 5 run (§5).

---

## 2. Simulator: queues, switching downtime, ACK feedback (applies to every method)

### 2.1 What changes

1. **Per-node queues.** `backlog_t+1 = backlog_t + demand_t − sent_t`, capped at `Q_MAX = 5 × mean_mbps × 0.1 s` per node; overflow is lost. A flow's offered load for allocation becomes `backlog + demand`. Throughput is what is actually sent.
2. **Switching downtime.** For one step (100 ms):
   - a node that changes its radio-1 channel cannot send on links that use that channel;
   - a node that changes its parent cannot send its own flow.
   Rationale: 802.11 channel switch plus re-association or path update takes tens to hundreds of ms. Cite 802.11h/s CSA and HWMP path-update timings in the paper.
3. **Queue loss on unprepared switches.** When a node switches without queue preservation, the backlog it holds at that moment is lost. Queue preservation is CSA-style: announce the switch, buffer, then resume. The `queue_preserve` flag sits in the env and applies to the actions of whichever method has it enabled.
4. **ACK feedback.** New edge feature `ack_ratio` = sent / offered on that link in the last step (1.0 when idle). New node feature `backlog`.
5. **New metric `loss_rate`** = (queue overflow + switch loss) / total demand, per episode.

### 2.2 Fairness rules

- Every method is retrained and re-evaluated on the new simulator: M0, Ko, Q-neutral, Yuan, Li, EGAT v1 and v2.
- The old results stay in the paper as the no-switching-cost setting.
- Queue preservation is a protocol feature any method could use. Report two settings:
  - **(a)** only EGAT v2 has `queue_preserve`, because it is part of the v2 mechanism;
  - **(b)** every method has it.

  Setting (b) separates the gain of RAES plus migration control from the gain of queue preservation.
- The downtime length (1 step) and `Q_MAX` are fixed now, before any v1/v2 result exists. A sensitivity row reports downtime 0 and 2 steps.

---

## 3. EGAT v1: Claude's design (MAPPO line, improved)

The aim is to make PPO improve on the BC policy instead of eroding it.

1. **KL anchor to the BC policy.** After BC, freeze a copy of the policy as `π_BC`. Add `κ · KL(π_θ ‖ π_BC)` to the PPO loss, with κ annealed from 0.1 to 0.01. This is the AlphaStar-style "stay near the supervised policy" regulariser.
2. **Train on the evaluation reward.** Keep λ fixed at 0.1: mostly the global reward, matching the env reward used for every reported metric. The λ anneal becomes an ablation row. Per-agent switch penalties stay.
3. **Pick checkpoints on validation.**
   - Every 10k steps, evaluate greedily on fixed validation topologies (S1–S6, seeds 1000–1002, never used for testing).
   - Keep the best checkpoint by mean validation reward. The BC-only checkpoint is a candidate, so the selected model can never fall below BC.
   - Test seeds remain untouched.
4. **Semi-MDP channel credit.** On decision steps the channel head's advantage uses the reward summed over the next `t_c` steps, as the original spec asked.
5. **Lower learning rate** after BC: 1e-4. Every component keeps its ablation flag.

**Pending:** the 300k-step run (same code as Plan 4). If longer training recovers on its own, item 5 may be enough. This section gets finalised with that number.

---

## 4. EGAT v2: the user's design (GAT-Q + RAES + migration control + queue preservation)

Each decision round runs this pipeline:

```
input (obs + ACK + backlog)
  → GAT scores Q(u,c), Q(u,j)
  → RAES reserves channels in order
  → migration refinement
  → execute moves with queue preservation
  → ACK feeds the next round
```

1. **Scoring network.** The edge-aware GATv2 + GRU encoder from Plan 4 (same features plus `ack_ratio` and `backlog`). Two Q heads:
   - `Q_ch(u, c)` for downlink channels 1..6;
   - `Q_rt(u, j)` for parent slots, with the progress mask.
2. **Training.** Double DQN, off-policy, on the actions actually executed after RAES and migration refinement. Off-policy learning is used because the planner always rewrites the network's argmax, so on-policy log-probs would be wrong.
   - Replay 50k steps, batch 32 env steps, soft target update τ = 0.01, γ = 0.98.
   - ε-greedy on Q before RAES, ε from 0.3 to 0.02 over 50k steps.
   - BC warm start: regress Q toward Ko's actions with a margin loss (DQfD-style). The first 10k steps act as Ko and count toward the budget.
3. **RAES: ordered channel reservation.**
   - Order: decreasing backlog (congested nodes reserve first); ties go to the smaller hop count. The ablation compares parents-first order.
   - Each node, in turn, reserves `c* = argmax_c Q_ch(u,c) − a·I(u,c) − b·L(u,c) − g·M(u,c)`, where:
     - `I(u,c)` = Σ over already-reserved conflict neighbours on c of (1 + their backlog), normalised;
     - `L(u,c)` = summed offered load on c within u's conflict neighbourhood (current plan);
     - `M(u,c)` = `1[c ≠ current]` × (downtime × own backlog), i.e. the throughput the move costs;
     - default `a = b = g = 1` after per-row z-scoring of each term; tuned in Plan 5.
   - Routing: `j* = argmax_j Q_rt(u,j)`, with the same migration cost applied to a parent change.
4. **Migration refinement.** A proposed move (channel or parent) is kept only if all of these hold:
   - its estimated gain `Q(new) − Q(current)` exceeds `M + margin` (margin 0.1 after z-score);
   - at most `k_max = ceil(0.1·n)` moves per round, highest net gain first;
   - no two conflicting nodes move in the same round;
   - the node is not in a 3-round cooldown after its last move.

   Rejected moves keep the current assignment.
5. **Execute with queue preservation.** Moves are executed with `queue_preserve=True`, so backlog survives the downtime (§2.1.3).
6. **ACK.** The env's `ack_ratio` and `backlog` features carry each round's outcome into the next.

Ablation flags: `raes` (off = per-node argmax), `migration_control`, `queue_preserve`, `raes_order`, plus every Plan 4 encoder flag.

---

## 5. Pre-registered evaluation (fixed before Plan 5 runs)

### 5.1 Composite score

Per scenario and per metric k, the relative improvement over a reference method R is:
- higher-is-better metrics: `Δ_k = (ours − R)/|R|`;
- lower-is-better metrics: `Δ_k = (R − ours)/(|R| + 1)`, where +1 smooths count metrics that can be 0.

Each `Δ_k` is clipped to ±1.

`Composite = Σ w_k Δ_k`, with weights:

| Metric | Weight |
|---|---|
| Throughput | 0.25 |
| Jain | 0.20 |
| p5 | 0.20 |
| loss_rate (lower is better) | 0.20 |
| Stability = channel switches + route changes (lower is better) | 0.15 |

Report all of these alongside each other:
- the composite against **each** baseline (one matrix), with Ko as the headline reference;
- the weight-free average rank across the 5 metrics;
- the per-metric table;
- a sensitivity check that scales each weight by ±50% and renormalises, stating whether the conclusion still holds.

### 5.2 Scenario suite (each with its rationale, fixed now)

- **A, standard:** S1–S6 (existing).
- **B, known weaknesses of the baselines:**

  | ID | Scenario | Weakness tested |
  |---|---|---|
  | B1 | Moving hotspot: the hot gateway region changes every 40 steps | Ko is myopic and decides every t_c steps |
  | B2 | Periodic traffic, 60-step cycle | Li's near-static policy |
  | B3 | Sparse topology: 25 m less area per node, so fewer neighbours | Yuan and Q-neutral form routing loops |
  | B4 | Unseen grid topology | Q-neutral's tabular state |
  | B5 | Rapid regime shifts every 25 steps | Every slow-adapting method |

- **C, where EGAT may lose (reported as limitations if it does):**

  | ID | Scenario |
  |---|---|
  | C1 | Very light load (mean 1 Mbps) |
  | C2 | Static traffic (no on/off) |
  | C3 | N = 50 zero-shot |
  | C4 | N = 80 zero-shot |
  | C5 | 4 gateways |

### 5.3 Success criteria (pre-registered)

1. Mean composite against **every** baseline is ≥ +10% over suites A+B, with a paired Wilcoxon test (Holm-corrected) at p < 0.05 per scenario in ≥ 60% of A+B scenarios.
2. No A or B scenario has composite below −5% against Ko.
3. Group C is reported in full whatever the outcome.
4. Weight sensitivity: criterion 1 still holds for all ±50% weight perturbations, or the paper states that it does not.

Statistics follow the original spec §5.7: training seeds, 20 test topologies, 3 traffic seeds, bootstrap CIs.

---

## 6. Order of work

1. **Plan 6a.** Simulator §2: queues, downtime, `queue_preserve`, ACK features, `loss_rate`. Then retrain all baselines and the Plan 4 EGAT (1 seed) to get a new reference table.
2. **Plan 6b.** EGAT v1 (§3) and v2 (§4), 1 seed each, evaluated with the §5.1 composite on suites A + B.
3. **Gate.** Plan 5's full statistical run starts only if v1 or v2 is on track for criterion 1 (≥ +10% mean composite against Ko on A+B at one seed). Otherwise iterate on 6b first.
4. **Plan 5.** Optuna, seeds, statistics, ablations, group C.
