# Response to manuscript v5 comments 5 (§6.4) and 7 (§7)

Scope: reconstruct the Ko, Yuan+H and ns-3 comparisons and Tables 5–9 from the recorded outputs. No new simulation was run.

- **Regenerate:** `python docs/paper/review/reconstruct_tables.py` (run from the repo root with `PYTHONPATH=.`).
- **Output:** written to `docs/paper/review/reconstruct_tables.out.md`, with every value printed to 6 decimals and the run/configuration of each table taken from the suite files themselves (commit, scenario overrides, agent overrides, evaluation seeds).
- **Code path:** all values go through `mesh/eval/composite.py`, the module that produced the reports cited in the manuscript.

## 1. N_rep (reconciled first)

N_rep is the field `stability`. Its definition is in the code:

- **Per step** (`mesh/sim/env.py`, step function):
  - `n_channel_switch` counts radio retunes over the n×2 channel matrix (uplink and downlink radios), alive nodes only.
  - `n_route_change` counts parent changes of traffic sources.
- **Per episode** (`mesh/eval/run.py`): both counters are summed over the 200 steps.
- **Score** (`mesh/eval/composite.py`, `_means`): `stability = n_channel_switch + n_route_change`.

**What one decision costs.** A downlink change counts once for the node's downlink radio. Each child's uplink radio follows the parent's downlink, so each child that stays on that parent adds one more count. A parent change counts once as a route change, plus one uplink retune if the new parent uses a different channel.

**Bound.** The bound is therefore T(2n + n_src), not 2nT:
- n = 20 with 2 gateways: 200 × (40 + 18) = 11,600.
- n = 80 (C4): 200 × (160 + 78) = 47,600.

The largest recorded values are within it: Q-neutral S4 8,219.9 and C4 34,885.

**Common start state.** Every episode starts from the same assignment: all nodes on channel pair (0, 1) with min-hop parents (`MeshEnv.reset`). Step 1 therefore counts each method's departure from that common start. M0 keeps it, so M0 shows 0 except after the S6 relay failure.

**Same counter for every method.** The counter is the same for every method:
- Every simulator method (M0, Ko, Q-neutral, Li, Yuan, Li+H, Yuan+H, EGAT) is evaluated through `run_episode` on `MeshEnv`.
- The ns-3 environment uses the identical expression (`mesh/ns3/env.py`, `n_sw = ((channels != self.channels) & alive).sum()`).

No score changes. The paper only needs the definition above in place of "one count per binary action change". Split of the reported 0–19 means:

| method | channel retunes | route changes | N_rep |
|---|---|---|---|
| Ko | 456.90 | 158.61 | 615.51 |
| Yuan+H | 40.26 | 3.06 | 43.32 |
| EGAT+H | 39.48 | 2.76 | 42.25 |
| Q-neutral | 6263.29 | 1787.22 | 8050.51 |

## 2. Aggregation order and the zero-reference rule

**Steps 1–2: per episode** (one topology, one trained model):
1. Throughput, Jain and P5 are computed every step. Each is then averaged over the 200 steps. Jain and P5 use the 1-s window throughput of flows that had demand in the window.
2. Loss is total lost traffic over total demand. N_rep is a sum over the episode, as in section 1.

**Steps 3–6: one comparison, one scenario.**

3. **Pooled-mean estimand** (Table 7, the "mean" column of the reports): each metric is averaged over all runs of the method in the scenario.
4. Relative terms (13) are formed from those scenario means and clipped to [−1, 1].
5. The terms are weighted (14) and summed. Scenarios are then averaged with equal weight over the 11 A+B scenarios.
6. **Seed-pair estimand** (Tables 5, 8, 9 and every interval):
   - For each pair (our training seed i, reference training seed j), metrics are averaged over the 20 topologies of that pair.
   - Runs are matched by evaluation topology. The score is formed as in steps 4–5 and averaged over scenarios, giving a 5×5 matrix.
   - The reported value is the matrix mean.
   - M0 and Ko are deterministic: their single output file enters once per training seed, so their matrix columns are identical.

**Zero-reference rule** (`delta` in `composite.py`):
- For a higher-is-better metric with reference value exactly 0, the term is 0 if ours is also 0, otherwise sign(ours), i.e. +1 after clipping.
- Q-neutral's mean P5 is exactly 0 in every A+B scenario (printed in the output file), so its P5 term is +1 × 0.20 for every method.
- Loss uses max(L_R, 0.01).
- N_rep uses N_R + 1.
- No other epsilon is applied.

## 3. Interval calculation and its dependence assumptions

The interval is `seed_matrix_ci` in `composite.py`. With m the 5×5 matrix:

- v_o = var(row means)/5 and v_r = var(column means)/5.
- Half-width = t_{0.975, df} √(v_o + v_r), with Welch–Satterthwaite df from the two variance terms.
- For deterministic references v_r = 0, so df = 4.

Assumptions, which the manuscript text already describes correctly:
- **What the variance counts.** The row and column means are treated as two independent samples of training-seed effects, one per method, with additive effects. The 25 pairs are not treated as independent. Only between-seed variance enters, through the 5 row means and 5 column means.
- **What is fixed.** The interval is conditional on the evaluated topology cohort; topology-sampling variance is excluded.
- **Reused checkpoints.** Seeds 0–2 of our model are the ablation checkpoints, so the 40–59 interval is conditional on those trained models.
- **No joint coverage.** Each interval is a separate 95% interval; there is no simultaneous adjustment across comparisons.
- **Per-scenario tests.** The Wilcoxon and bootstrap tests are diagnostics conditional on the trained models.

## 4. Reconstructed key comparisons (full precision)

| comparison | point | 95% interval | df |
|---|---|---|---|
| vs Ko, 0–19 | +0.260296 | [+0.230390, +0.290203] | 4.00 |
| vs Ko, 40–59 | +0.245951 | [+0.215532, +0.276371] | 4.00 |
| vs Yuan+H, 0–19 | +0.085097 | [+0.042557, +0.127636] | 8.00 |
| vs Yuan+H, 40–59 | +0.066841 | [+0.031338, +0.102345] | 7.42 |
| ns-3 RTS/CTS vs Ko | +0.230892 | [+0.154543, +0.307241] | 4.00 |
| ns-3 RTS/CTS vs Yuan+H | +0.164388 | [+0.003731, +0.325044] | 5.22 |
| ns-3 base MAC vs Ko | +0.218016 | [+0.143806, +0.292227] | 4.00 |
| ns-3 base MAC vs Yuan+H | +0.183524 | [+0.021179, +0.345870] | 5.02 |

Every value in manuscript v5 Tables 5, 6, 8 and 9 rounds to these outputs. The only differences are in Table 7, section 5.

## 5. Table 7 and the 0.093 vs 0.094 value

The full-precision pooled-mean composite against Yuan+H is +0.092766. That is the 0.093 of the standalone summary and of `runs/v10/final/composite_a_nogru_test.md`.

The 0.094 in the DOCX decomposition came from `docs/paper/figures/make_figures.py` (`decomposition()`):
- That function read the per-metric scenario means from the markdown report, where they are printed to 3 decimals, and recomputed the terms from those rounded means.
- Recomputing that way gives +0.093741, which displays as 0.094.

The error is rounding of the inputs, not a different estimator. The terms recomputed from the full-precision outputs sum exactly to the report value for every reference (output file, Table 7).

Corrections to Table 7 (from full precision; other rows are unchanged at 3 decimals):

| reference | Throughput | Jain | P5 | Loss | N_rep term | Total |
|---|---|---|---|---|---|---|
| Yuan | +0.018 | +0.005 | +0.035 | **+0.060** (was 0.061) | +0.126 | +0.244 |
| Yuan+H | +0.016 | −0.000 | +0.024 | **+0.049** (was 0.050) | +0.004 | **+0.093** (was 0.094) |

The §7.2 text follows from this. The N_rep term is 0.138740 of 0.260296 against Ko, which is **53.3%** (manuscript says 53.5%). The remainder is **0.122** (manuscript says 0.121).

The fix in the code: `decomposition()` should load the suite JSONs, as `reconstruct_tables.py` does, instead of parsing the rounded markdown.

## 6. Run/configuration per table

Each table's file list, commit, overrides and margins are printed under the table in the output file.

Common configuration:
- **Simulator:** `airtime=domain, mac_eff=0.65, contention=0.0` (`runs/v10/calib.env`), setting (a) unless stated.
- **Margins:** EGAT+H `switch_margin=2`, EGAT+GRU+H 2, Li+H and Yuan+H 1.

| Table | Outputs used |
|---|---|
| 5 | `runs/v10/t{0-4}/a`, `runs/v10/fresh/t{0-4}/a`, `runs/v10/fresh2/t{0-4}/a`, heuristic file `t0/a/suite_heur.json` of each set |
| 6, 7 | `runs/v10/t{0-4}/a` |
| 8 | ablation checkpoints evaluated at margin 2 (`runs/v10/abl_m2.sh`): `runs/v10/t{0-2}/a/suite_egat_{noedge,nogru,noko,nobc}.json`, full GRU = `suite_egat_h2` seeds 0–2, Yuan+H seeds 0–4 |
| 9 | simulator = Table 5 files restricted to S2/B1/B3; ns-3 = `runs/ns3/v10` (RTS/CTS) and `runs/ns3/v10base` (base MAC), topologies 0–4, `suite_{egat_nogru_h,egat_h2,rl,li_h,yuan_h}.json`, heuristic file `suite_base.json` |

The suites record the commits they ran at (8aaaf1c, 75d17f8, 65c49b6, c7c16c7). The `runs/` folder is not in git, so the supplement for comment 6 must archive it.

## 7. Unfavourable results kept

- **Jain:** EGAT+H 0.786, below Li+H (0.802) and M0 (0.893), and marginally below Yuan+H (0.787).
- **High-load loss, group C:**

  | scenario | EGAT+H loss | EGAT+H throughput (Mb/s) | Ko loss | Yuan+H loss |
  |---|---|---|---|---|
  | C3 | 0.656 | 73.6 | 0.738 | 0.671 |
  | C4 | 0.776 | 71.0 | 0.835 | 0.789 |

  The +0.465 and +0.469 composites coexist with severe congestion.
- **ns-3 S2:**
  - With RTS/CTS, Ko has higher Jain (0.722 vs 0.693) and P5 (1.318 vs 1.292).
  - Without RTS/CTS, Ko also has higher throughput (66.355 vs 63.816 Mb/s) and P5 (1.084 vs 0.955).
- **Yuan+H margin:**
  - The ns-3 RTS/CTS lower bound against Yuan+H is +0.0037.
  - The 40–59 lower bound against Yuan+H (+0.031) is below +0.05.
  - The +0.10 criterion is not met against Yuan+H.
- **Fresh set 20–39:** the comparison with the GRU model spans zero: +0.0396 [−0.0117, +0.0909].

No outcome changes, so the abstract and conclusions need no change from this reanalysis. Only the Table 7 values above and the N_rep definition change.
