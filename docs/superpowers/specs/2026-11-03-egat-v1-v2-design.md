# EGAT v1 / v2 + Switching-Cost Simulator + Pre-registered Evaluation (Addendum)

- Date: 2026-11-03
- Status: design approved (v1 and v2). Review fixes to §2 and §5.1 pre-registered.
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
6. **1 s windowed fairness (user-approved after review).**
   - Utility, Jain and p5 use each node's throughput averaged over the last `utility_window = 10` steps. The average is taken only over the steps in which the node had something to send.
   - The flow set is method-independent: alive non-gateways with demand at some step of the window.
   - Reason: with switching downtime, a per-step `log(throughput)` scores a 100 ms outage, during which the queue is preserved, as starvation. Ko then lost to M0 on reward even though it beat M0 on throughput by 46%.
7. **Switching scope (user-approved after review).**
   - A parent change blocks the whole uplink for the downtime, so traffic relayed through that node waits as well.
   - On an unprepared switch, every source whose planned path crosses an interrupted link loses its queued backlog. The fluid model has no relay queues, so the loss is attributed at the source.
   - This applies to every method in the same way.
8. **`n_drop` counts routing failures only** (no planned path to a gateway). Flows cut off for a step by switching downtime are counted separately as `n_interrupted`.

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

**300k-step diagnosis (done, Plan 4 code, old simulator):** longer PPO makes things worse.

| Model | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|
| BC-only | 241 | 2769 | 2757 | 2079 | 1829 | 2047 |
| PPO 100k | 243 | 2708 | 2662 | 2043 | 1784 | 2031 |
| PPO 300k | 243 | 2623 | 2602 | 2014 | 1743 | 1969 |

Longer training is therefore not a fix. All five v1 items stand, and the KL anchor and checkpoint selection are required.

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

Correction after review: the `+1` smoothing applies only to the count metric (stability). `loss_rate` uses `Δ = (R − ours)/max(R, 0.01)`, so a 10% reduction in loss scores +0.10, matching a 10% throughput gain. With `+1`, loss carried about a tenth of its stated weight.

Average rank uses mid-ranks, so tied methods share the average position.

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
  | B3 | Sparse topology: rgg spacing 37.5 m instead of 30 m (larger area per node, fewer neighbours) | Yuan and Q-neutral form routing loops |
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

**Amendment, 2026-11-05, made after the seed-0 and 3-seed results were seen.** Criterion 1's per-scenario test is reported twice, and both versions are Holm-corrected across the 11 A+B scenarios:
- **(i) As pre-registered:** a paired Wilcoxon test on per-run composites.
- **(ii) Added:** a paired bootstrap of the pre-registered composite itself. It is computed on metric means, with the runs resampled together with their pairs, and gives a two-sided bootstrap p-value.

Reason: Wilcoxon tests the median of per-run composites, which is a different quantity from the mean-based composite that criterion 1 is defined on. Against a reference whose runs vary widely (Li's per-run composite against EGAT ranged from −0.46 to +0.72), the two disagree.

Both results are reported. Any claim that rests only on (ii) is stated as such.

Implementation note: Holm correction was missing from the first implementation (commit 087f27e). It was added at the same time as this amendment.

Evaluation seeds: 20 env seeds per scenario (each seed draws both the topology and the traffic) × 3 training seeds. This replaces "20 test topologies × 3 traffic seeds".

**Amendment, 2026-11-06, made after the 5-training-seed results were seen.**

Training seeds rose to 5. The headline CI is now hierarchical: training-seed files are resampled first, then runs within each file. Results from the 5 seeds:
- EGAT: per-seed composite vs Ko ranged from −0.043 to +0.179.
- v1: hierarchical CI vs Ko was [+0.078, +0.116].

Diagnosis: the chosen checkpoint's validation reward did not rank seeds by their test composite. Two causes:
- The validation set was small (S1–S6 × 3 topologies), and a validation pass moved by about ±100 reward.
- Selection used raw env reward, while evaluation uses the composite.

Changes, applied to **every learner** (q_neutral, yuan, li, egat, egat_v1, egat_v2), with all of them retrained under these rules:
- **Selection metric:** the pre-registered composite vs Ko, averaged over the A+B scenarios. It is computed on validation topologies, env seeds 1000–1003, which are disjoint from the evaluation seeds 0–19. This replaces mean env reward on S1–S6 × seeds 1000–1002.
- **Weight averaging:** after BC, the validated candidate carries an EMA of the network weights (decay 0.9995 per env step) instead of the raw weights. Q-neutral is tabular and keeps raw values.
- **Candidates saved:** every validated candidate is saved next to the chosen checkpoint (`<out>_val<step>.pt`). The selection rule above is fixed. Saved candidates are for diagnosis only, not for re-selection.

Reported side by side: the earlier 5-seed results (old selection) and the new ones.

**Amendment, 2026-11-06 (second), made after the 5-training-seed v8 results were seen.** Under the new selection, v1 beats Ko (+0.109) but loses to Li (−0.054) on A+B setting (a). It wins 4 of 5 metrics against Li and loses stability: 308 vs 122 switches per episode, a delta clipped at −1. Part of this comes from cascades: radio 0 follows the parent's channel.

Change, for **egat_v1 only**, at greedy decoding only (training and exploration are unchanged):
- **Switch hysteresis.** A logit bonus `switch_margin` is added to each node's current channel and its current parent. A node switches only when the new choice wins by more than the margin. The decision stays local to each node.
- **Choosing the margin.** The grid is {0, 0.5, 1, 2, 4}. The margin is the one with the best mean validation composite vs Ko over the 5 training seeds, using the same validation set as above (A+B × seeds 1000–1003). It is fixed before any test evaluation, and test evaluation is then run once.
- **Fairness.** The baselines keep their own published decoding. Hysteresis is part of the proposed method and is disclosed as a post-hoc addition. An ablation that gives Li and Yuan the same hysteresis is not run yet; it is open if a reviewer asks for it.

Reported side by side: v1 without and with hysteresis.

---

## 6. Order of work

1. **Plan 6a.** Simulator §2: queues, downtime, `queue_preserve`, ACK features, `loss_rate`. Then retrain all baselines and the Plan 4 EGAT (1 seed) to get a new reference table.
2. **Plan 6b.** EGAT v1 (§3) and v2 (§4), 1 seed each, evaluated with the §5.1 composite on suites A + B.
3. **Gate.** Plan 5's full statistical run starts only if v1 or v2 is on track for criterion 1 (≥ +10% mean composite against Ko on A+B at one seed). Otherwise iterate on 6b first.
4. **Plan 5.** Optuna, seeds, statistics, ablations, group C.
