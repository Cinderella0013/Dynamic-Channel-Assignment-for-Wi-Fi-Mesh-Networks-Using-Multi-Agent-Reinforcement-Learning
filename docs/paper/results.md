# Results

<!-- Draft v1, 2026-11-05. Every number comes from runs/v8/final/*.md, runs/v8/fresh/final/*.md or
docs/paper/figures/make_figures.py (decomposition, metric means). Regenerate figures with
`python docs/paper/figures/make_figures.py`. -->

## A. Evaluation protocol

Every learned method was trained with five seeds. Each trained model was then evaluated on 20 test topologies (seeds 0-19) in each of the 11 scenarios of suites A (S1-S6) and B (B1-B5). An episode lasts 200 steps of 100 ms. The heuristics Ko and M0 are deterministic given the topology. Unless stated otherwise, all methods run in setting (a), where a node that switches channel or parent loses its queued packets. In setting (b), every method keeps its queue across a switch.

We report the pre-registered composite score. For each scenario, it is the weighted sum of the relative improvements of our method over a reference in aggregate throughput (0.25), Jain fairness (0.20), 5th-percentile flow throughput (0.20), loss rate (0.20) and stability, counted as channel plus route switches per episode (0.15). Each term is clipped to [-1, 1]. The composite is then averaged over scenarios, so +0.10 reads as "10% better".

The uncertainty of this mean has two sources: the test topologies and the training seeds of both methods. We therefore compute the composite for every pair of (our training seed, reference training seed). We then report the mean of this seed-pair matrix with a 95% t-interval whose variance includes the training-seed variance of both methods (Welch-Satterthwaite degrees of freedom). Per-scenario significance uses a Wilcoxon signed-rank test and a paired bootstrap, both Holm-corrected over the 11 scenarios. These per-scenario tests are conditional on the trained models: they do not count training-seed variance, which the t-interval does.

**Disclosure of post-hoc decisions.** Switch hysteresis was added after the first test-set evaluation, as were the rule that selects its margin and the seed-pair interval. The margin is selected only on validation topologies (seeds 1000-1003). The rule picks the margin whose worst composite over all baselines is highest. It gave margin 3 for EGAT. The test set was then evaluated once. To guard against selection on the test seeds, Section D repeats the evaluation on fresh topologies (seeds 20-39) that no decision used. All amendments are dated in the design specification.

## B. Main comparison

Table I compares EGAT-MAPPO with switch hysteresis (EGAT+H) against the baselines in their published form (Fig. 1, blue bars).

**Table I.** Composite improvement of EGAT+H over each baseline, setting (a), suites A+B, test seeds 0-19.

| Reference | Pooled mean | Seed-pair mean [95% t-interval] | Scenarios better (Wilcoxon / bootstrap, Holm) | Min over weights ±50% |
|---|---|---|---|---|
| Ko | +0.195 | +0.195 [+0.165, +0.225] | 100% / 100% | +0.136 |
| Li (GAT-A2C) | +0.201 | +0.160 [+0.061, +0.260] | 91% / 100% | +0.169 |
| Yuan (GAT+DQN) | +0.259 | +0.256 [+0.228, +0.283] | 100% / 100% | +0.209 |
| M0 (static) | +0.368 | +0.368 [+0.356, +0.380] | 100% / 100% | +0.273 |
| Q-neutral | +0.732 | +0.732 [+0.724, +0.740] | 100% / 100% | +0.649 |

Against every baseline, EGAT+H meets all four pre-registered criteria in setting (a):

1. The mean composite is at least +10%, and the lowest is +19.5% against Ko.
2. The improvement is significant in at least 60% of scenarios under both tests; the lowest is 91%, against Li under Wilcoxon.
3. No scenario is worse than Ko by more than 5%. Against Ko every scenario improves, with S1 the smallest at +0.125.
4. The +10% margin holds when each weight is halved or increased by half.

Every seed-pair t-interval excludes zero. The interval against Li is wide because Li varies strongly across its training seeds. Our per-seed composites against Li range from +0.025 to +0.289.

**Where the gain comes from.** Table II splits the composite into the weighted contribution of each metric, and Table III gives the metric means. Against Ko, +0.138 of the +0.196 comes from the stability term. EGAT+H switches 43 times per episode, against 580 for Ko. The rest comes mainly from lower loss (+0.039) and higher throughput (+0.014). Jain fairness is essentially tied (-0.001). Against Li and Yuan, the non-stability terms alone sum to +0.111 and +0.127, so the advantage there does not rest on stability. In absolute terms, EGAT+H raises aggregate throughput by 6.5% over Ko (55.8 vs 52.4 Mb/s), raises 5th-percentile throughput by 3.2%, and cuts the loss rate from 0.195 to 0.156.

**Table II.** Weighted contribution of each metric to the composite of EGAT+H, setting (a), A+B, seeds 0-19 (pooled).

| Reference | Aggregate | Jain | p5 | Loss | Stability | Total |
|---|---|---|---|---|---|---|
| Ko | +0.014 | -0.001 | +0.006 | +0.039 | +0.138 | +0.196 |
| Li | +0.016 | +0.007 | +0.028 | +0.060 | +0.088 | +0.200 |
| Yuan | +0.016 | +0.006 | +0.037 | +0.068 | +0.132 | +0.259 |
| Li+H | +0.015 | +0.007 | +0.017 | +0.054 | -0.035 | +0.057 |
| Yuan+H | +0.011 | +0.006 | +0.020 | +0.045 | +0.003 | +0.086 |

**Table III.** Metric means over A+B, setting (a), seeds 0-19.

| Method | Aggregate (Mb/s) | Jain | p5 (Mb/s) | Loss rate | Switches / episode |
|---|---|---|---|---|---|
| M0 | 32.98 | 0.652 | 0.828 | 0.446 | 0.2 |
| Ko | 52.40 | 0.761 | 1.701 | 0.195 | 580.2 |
| Li | 52.30 | 0.730 | 1.529 | 0.200 | 122.3 |
| Yuan | 52.34 | 0.733 | 1.471 | 0.203 | 420.4 |
| Li+H | 52.52 | 0.730 | 1.617 | 0.196 | 35.0 |
| Yuan+H | 53.37 | 0.732 | 1.586 | 0.186 | 44.6 |
| **EGAT+H** | **55.83** | 0.755 | **1.756** | **0.156** | 43.4 |

Q-neutral delivers almost no traffic (0.19 Mb/s, loss rate 0.976), so its high Jain index (0.919) reflects uniformly near-zero throughput. It is omitted from the table. EGAT+H has the best average rank over the five metrics (2.04, weight-free). The next best are Yuan+H (3.45), Li+H (3.61) and Ko (4.00).

Fig. 3 shows that hysteresis trades a small amount of throughput for stability. Without hysteresis, EGAT reaches 56.3 Mb/s with about 208 switches per episode. With margin 3, it reaches 55.8 Mb/s with 43 switches. On the same seeds, EGAT without hysteresis improves on Ko by +0.168, but on Li by only +0.024 (not significant, 9% of scenarios).

## C. Fairness ablation: hysteresis for every learner

Hysteresis is a decoding rule that any policy with logits can use. A fair comparison therefore gives the same rule to the strongest learned baselines. We added it to Li and Yuan and selected their margins with the same worst-case validation rule. The rule never used EGAT as a reference. It chose margin 16 for Li, the edge of the grid {0, 0.25, ..., 16}, and margin 1 for Yuan. With hysteresis, Li improves on Ko by +0.097 and Yuan by +0.115, against +0.195 for EGAT+H.

Against these strengthened baselines, EGAT+H still wins, by a smaller margin (Table I continued, Fig. 1):

| Reference | Seed-pair mean [95% t-interval] | Scenarios better (Wilcoxon / bootstrap) | Min over weights ±50% |
|---|---|---|---|
| Li+H | +0.057 [+0.020, +0.094] | 27% / 36% | +0.034 |
| Yuan+H | +0.080 [+0.017, +0.143] | 91% / 91% | +0.070 |

On test seeds 0-19, both intervals exclude zero. Neither reaches the +10% criterion, however, and against Li+H the improvement is significant in only a minority of scenarios. Table II locates the remaining advantage. Li+H switches slightly less than EGAT+H (35 vs 43), so the stability term turns negative (-0.035). The gain comes from lower loss (+0.054), higher 5th-percentile throughput (+0.017) and higher aggregate throughput (+0.015). In short, most of the gap over the published Li and Yuan comes from suppressing unnecessary switches. What remains attributable to the EGAT architecture and training is about 6-8%, and it shows up in loss and throughput rather than in stability.

## D. Confirmation on fresh topologies

Test seeds 0-19 had been seen when hysteresis and its selection rule were introduced. We therefore re-evaluated every trained model once on 20 new topologies per scenario (seeds 20-39). No decision used these seeds (Fig. 1, hatched bars).

| Reference | Seed-pair mean [95% t-interval] | Scenarios better (Wilcoxon / bootstrap) | Min over weights ±50% |
|---|---|---|---|
| Ko | +0.151 [+0.111, +0.192] | 100% / 91% | +0.089 |
| Li | +0.161 [+0.075, +0.248] | 100% / 100% | +0.161 |
| Yuan | +0.236 [+0.197, +0.275] | 100% / 100% | +0.190 |
| M0 | +0.315 [+0.289, +0.340] | 100% / 100% | +0.224 |
| Q-neutral | +0.725 [+0.717, +0.733] | 100% / 100% | +0.641 |
| Li+H | +0.044 [-0.005, +0.092] | 27% / 27% | +0.026 |
| Yuan+H | +0.073 [+0.007, +0.139] | 82% / 82% | +0.062 |

The main claim holds on fresh topologies. EGAT+H improves on every published baseline by more than 10%, and every interval excludes zero; against Ko, the lower bound is +11.1%. No scenario is worse than Ko; the smallest gain is +0.065, in S4. The effect is smaller than on seeds 0-19: +0.151 against Ko instead of +0.195. Robustness to the weights also fails against Ko, where the minimum is +0.089. Against Li+H, the advantage is no longer significant (+0.044, interval crossing zero). Against Yuan+H it remains significant (+0.073).

## E. Stress suite C

Suite C contains scenarios chosen in advance as ones where a learned policy might lose: very low load (C1), a static network (C2), larger networks with 50 and 80 nodes (C3, C4), and four gateways (C5). The models were trained on 20-node networks only. In setting (a), EGAT+H improves on Ko by +0.293 on average over C. The gain grows with network size: +0.409 at 50 nodes and +0.424 at 80 nodes. The smallest gain is at very low load (C1, +0.112), where almost all offered traffic is delivered by every method except Q-neutral. Against Li+H and Yuan+H, the averages over C are +0.039 and +0.136.

## F. Component ablation

We retrained EGAT with one component removed at a time, using three training seeds per variant. Every variant was evaluated with hysteresis margin 3 on test seeds 0-19 (setting a, A+B).

| Variant | vs Ko | vs Li+H | vs full EGAT+H [95% t-interval] |
|---|---|---|---|
| Full EGAT+H (seeds 0-2) | +0.199 | +0.063 | - |
| No edge features | +0.160 | +0.015 | -0.056 [-0.149, +0.036] |
| No GRU memory | +0.188 | +0.067 | -0.006 [-0.110, +0.099] |
| No Ko residual prior | +0.155 | +0.000 | -0.072 [-0.168, +0.025] |
| No behaviour cloning from Ko | +0.144 | +0.008 | -0.067 [-0.169, +0.036] |

Removing edge features, the Ko residual prior or the behaviour-cloning warm start lowers the composite by 0.06-0.07. Without any one of them, the advantage over Li+H disappears. The GRU has no measurable effect. With three seeds per variant, none of these differences is statistically significant, so we report them as trends. Setting (b) points the same way.

## G. Queue-preserving setting (b)

When every method keeps its queue across a switch, switching is cheap. Ko's frequent re-optimisation then costs little. On test seeds 0-19, EGAT+H improves on Ko by +0.144 [+0.116, +0.172] and on Li by +0.165 [+0.067, +0.263] (Fig. 2). On fresh seeds 20-39, the improvement on Ko is +0.095 [+0.051, +0.138].

Two scenarios fail criterion 3. On seeds 0-19, B4 is worse than Ko by 0.084. B4 uses a grid topology, a family never seen in training. On seeds 20-39, B4 (-0.074) and S4 (-0.055) both fail. The B4 loss comes from the loss-rate term alone. In setting (b), Ko's loss rate on B4 is very low, so the relative loss term saturates at -1 for small absolute differences. The deficit is not caused by hysteresis: EGAT without hysteresis also loses to Ko on B4 in setting (b) in four of five training seeds. A congestion-aware variant of hysteresis did not pass the validation rule, and the validation topologies do not reveal the B4 deficit. We therefore report it as a limitation. Generalising to unseen topology families when switching is free remains open.

## Figures

- **Fig. 1** (`figures/fig_main.pdf`): composite improvement of EGAT+H over Ko, Li, Yuan, Li+H and Yuan+H, with 95% seed-pair t-intervals. Shown on test seeds 0-19 and on fresh seeds 20-39; the dashed line marks the +10% target.
- **Fig. 2** (`figures/fig_scenarios.pdf`): composite against Ko per scenario, in settings (a) and (b), seeds 0-19; the dashed line marks the -5% floor of criterion 3.
- **Fig. 3** (`figures/fig_switching.pdf`): aggregate throughput against switches per episode (log scale), A+B mean, setting (a), seeds 0-19.
