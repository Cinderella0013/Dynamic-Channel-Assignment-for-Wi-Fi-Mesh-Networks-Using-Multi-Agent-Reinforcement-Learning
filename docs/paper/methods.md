# Methods

<!-- Draft v1, 2026-11-05. Values come from the code at the commit this file was added in:
mesh/sim/{radio,graph,env,traffic,topology,scenarios}.py, mesh/model/{egat,agent}.py, mesh/rl/{agents_io,train}.py,
mesh/baselines/{ko,m0}.py, mesh/eval/composite.py; adaptations from docs/baselines-adaptation.md.
Citation keys in [brackets] are placeholders: no reference list yet, every key must be verified (citation-check)
before submission. [Ko] and [Li] in particular are unconfirmed (see TODO notes). -->

## A. System model

**Topology.** A mesh has n nodes, two of which are gateways chosen at random. By default n = 20 and the nodes are placed uniformly at random in a square of side 30·√n m. Placements are redrawn until the link graph is connected. One unseen family (B4) uses a square grid with 35 m spacing instead. A directed link (u, v) exists when the received power is at least -76 dBm. Two nodes conflict, i.e. are within carrier-sense range, when it is at least -82 dBm.

**PHY.** Received power follows the log-distance path-loss model with ns-3 defaults: 20 dBm transmit power, 46.68 dB reference loss at 1 m and exponent 3. Noise is -94 dBm. The SNR maps to the highest 802.11ax MCS it supports, for a 20 MHz channel, one spatial stream and 0.8 µs guard interval: from 8.6 Mb/s at 2 dB up to 143.4 Mb/s at 37 dB. External interference (scenario S4) adds -85 dBm of noise on one channel for steps 50-150.

**Radios and channels.** Each node has two radios. Radio 1 is the node's downlink, tuned to one of six channels (2.4 GHz channels 6 and 11, 5 GHz channels 36-48); this is the per-node channel decision. Radio 0 is the uplink and follows the downlink channel of the node's parent, so every tree link has a channel both ends share; a node without a parent keeps radio 0 on channel 1. This is the tree convention of Ko's channel assignment [Ko].

**MAC and flows.** Time advances in steps of 100 ms; an episode has 200 steps (20 s). Each non-gateway node is an uplink traffic source routed to a gateway along its parent chain. Airtime is shared by counting contenders: link (u, v) obtains capacity rate(u, v)/k, where k counts the active links on the same channel within conflict range of u or v, itself included. A flow's throughput is its demand, scaled down at every hop whose total offered load exceeds the capacity (a proportional-share fluid model).

**Traffic.** Each source follows an on/off Markov chain that keeps its state with probability 0.9 per step. While on, its demand is drawn from an exponential distribution with the scenario's mean rate. Unsent traffic is queued, up to five times the mean rate per node, and the excess is lost to overflow.

**Switching cost.** A node that retunes its downlink cannot use the links on that channel for one step. A node that changes parent cannot use its uplink for one step. In setting (a), a flow whose planned path crosses an interrupted hop also loses its queued backlog. In setting (b), every method keeps the queue across the switch, in the style of a channel-switch announcement (CSA). The loss rate counts switch losses and overflow, divided by total demand.

## B. Problem formulation

We cast joint channel assignment and routing as a decentralised partially observable Markov game. Every node, gateways included, is an agent and acts every step. The joint objective is the global reward

R_t = Σ_f log(x̄_f + ε) - γ_s·N_sw - δ·N_rc - η·N_drop,

where x̄_f is the throughput of flow f averaged over a 1 s window (10 steps), counting only the steps in which f had traffic, and ε = 0.01. N_sw, N_rc and N_drop count channel switches, parent changes and unroutable flows in the step, with γ_s = 0.1, δ = 0.05 and η = 0.1. The log utility gives proportional fairness, and the window keeps a 100 ms outage with a preserved queue from being scored as starvation.

**Observation.** Node i observes its own offered load, unmet demand, backlog, gateway flag, hop count to the nearest gateway, liveness, current downlink channel and current parent. It also observes, for each of its K = 4 parent candidates, the candidate's validity, SNR, achievable rate and load fraction. The candidates are its link neighbours, ranked first by hop count to a gateway and then by SNR. Ranking by hop count first guarantees that every routable node has a candidate closer to a gateway. Edges of the attention graph join nodes that are linked or in conflict. Each edge carries seven features: link flag, conflict flag, SNR, rate, load fraction, shared-channel flag, and the ratio of delivered to offered traffic on the link in the previous step (a link-level acknowledgement signal).

**Action.** Each agent picks a downlink channel (6 choices) and a parent slot (4 choices), 24 joint actions in total. Gateways have no parent.

**Per-agent reward.** For training, every learner receives

r_i = λ·u_i + (1 - λ)·R_t / n_alive - 0.1·s_i - 0.05·c_i,

where u_i is the log utility of the flows whose planned path crosses i, and s_i and c_i are i's own channel switches and parent changes. A child's radio-0 retune that follows its parent's retune is charged to the parent. The baselines use λ = 0.5 throughout. EGAT-MAPPO anneals λ linearly from 0.5 to 0.1 over training, so its training signal moves toward the global objective. All learners use γ = 0.98.

## C. EGAT-MAPPO

**Encoder.** Node inputs are embedded by a linear layer and passed through two edge-aware GATv2 layers [GATv2] with four heads and hidden size 64. The attention score of node i on neighbour j is

s_ij = aᵀ LeakyReLU(W_s h_i + W_t h_j + W_e e_ij),

normalised by a softmax over i's neighbours, self-loop included. Each layer adds a residual connection and LayerNorm. A per-node GRU cell carries memory across steps and is reset at the start of each episode. The parameters are shared by all agents.

**Heads and timescales.** A channel head and a route head map the node embedding to logits over the six channels and the four parent slots. The route head decides every step. The channel head decides every t_c = 10 steps, and nodes keep their downlink channel in between. A progress mask allows only parents strictly closer to a gateway, which makes every route loop-free.

**Ko residual prior.** Both heads subtract the z-scored cost that Ko's local-cost rule assigns to each option, weighted by a learnable scale: logits ← logits - softplus(β)·z(cost_Ko), with softplus(β) = 1 at initialisation. The policy therefore starts close to Ko and learns corrections to it.

**Training.** Training runs for 100 000 environment steps. Each episode draws a scenario uniformly from S1-S6 and a fresh n = 20 topology.

- **Behaviour-cloning warm start.** For the first 10 000 steps the agent acts as Ko and records the trajectories. These steps count toward the budget. It then fits the actor by cross-entropy on Ko's parent choices, for at most 20 epochs, stopping once agreement on the training data reaches 0.9. Channel labels are interchangeable, so matching Ko's exact channel is not an imitation target. The critic is warm-started on Ko's Monte-Carlo returns.
- **MAPPO.** After behaviour cloning, the actor is trained with MAPPO [MAPPO]: PPO-clip [PPO] with clip 0.2, GAE [GAE] with λ_GAE = 0.95, rollouts of 1000 steps, 4 epochs, learning rate 3·10⁻⁴, entropy coefficient 0.01 and gradient-norm clip 0.5. The centralised critic sees each node's embedding concatenated with the mean embedding over all nodes. It is used only in training.
- **Checkpoint selection.** An exponential moving average of the weights (decay 0.9995 per step) is kept from the end of behaviour cloning. Every 10 000 steps, at the end of behaviour cloning and at the end of training, the averaged policy is scored greedily on validation topologies (env seeds 1000-1003, disjoint from every test seed) over all A and B scenarios. The score is the composite of Section F against Ko. The best candidate is kept. Validation steps do not count toward the budget.

## D. Decentralised execution

At test time each node acts greedily from its own heads, with three additions.

- **Sequential channel decoding.** Parents decide first. In tree order, each node takes its highest-scoring channel that no already-decided conflict neighbour uses. Without this rule, interchangeable channel labels would make nodes in symmetric states pick the same channel. It needs one coordination round per decision, as Ko's channel assignment does. All learned methods, baselines included, use the same decoder.
- **Switch hysteresis.** A bonus m is added to the logit of the node's current parent and of its current downlink channel before the argmax. A node therefore moves only when another option beats the current one by more than m. Training is unchanged. We select m by a worst-case rule on the validation topologies: from the grid {0, 0.5, 1, 2, 3, 4}, take the margin whose smallest composite against any baseline (mean over all training-seed pairs) is largest. For EGAT-MAPPO this gives m = 3.
- **Safe fallback.** A node whose policy entropy exceeds 2 nats takes Ko's action instead. On channel steps the entropy is route plus channel entropy, otherwise route entropy alone. The entropy is computed after the hysteresis bonus, so a node that holds its current choice is not treated as uncertain. The rule uses only the node's own heads, so execution stays decentralised.

## E. Baselines

- **M0** keeps fixed channels and min-hop routing. It never switches.
- **Ko** is our adaptation of the local-cost channel assignment of [Ko] (TODO: confirm the reference). Every t_c = 10 steps, and immediately when a parent link breaks, it routes by Bellman-Ford toward the gateways with link cost (1 + load fraction)/rate, and switches parent only if the new cost is below 0.9 of the current one. Visiting nodes in hop order, it then gives each node the downlink channel with the least load-weighted use among its non-child conflict neighbours.
- **Q-neutral** is tabular Q-learning on a discretised local state with the shared reward and no specially weighted objective. The project brief names this baseline without defining it, so its definition is our assumption.
- **Yuan** is the graph-attention Q-network (GQN) of Yuan et al. [Yuan] (doi:10.1049/cmu2.12527): dot-product attention over the agent and three neighbours, DQN with a soft target update, and the temporal-relation KL regulariser, with the paper's hyperparameters.
- **Li** is the GAT-based advantage actor-critic of Li et al. [Li] (TODO: confirm Sensors 25(16):5209): a two-layer GAT over the interference graph with z-scored inputs and 500-250 actor and critic layers.

All learned baselines use the same observation, action space, per-agent reward, discount, training budget and greedy channel decoder as EGAT-MAPPO. Their attention uses no edge features, which are part of our contribution. Both papers optimise channel and transmit power with no routing, so their action is replaced by ours. Every departure from the source papers, with its reason, is listed in Appendix [baseline adaptations]. Examples: an MLP Q-head where the paper's CNN is unspecified, on-policy A2C where the paper's replay variant is under-specified, and a lower learning rate for Li because the paper's rate saturated the softmax under per-step updates. Every learned method is trained with five seeds and selected on the same validation topologies.

**Hysteresis baselines (Li+H, Yuan+H).** For the fairness ablation, the same logit bonus is applied to Li's and Yuan's greedy decoding. Their margins are chosen by the same worst-case rule over the grid {0, 0.25, 0.5, 1, 2, 4, 8, 16}, with the original baselines (never EGAT-MAPPO) as references. This gives m = 16 for Li and m = 1 for Yuan.

## F. Scenarios, metrics and statistics

**Scenarios.** Suite A covers the nominal conditions and suite B targets known weaknesses of the baselines. Suite C, specified in advance, contains conditions under which a learned policy might lose to a heuristic. Unless listed, the mean load is 6 Mb/s per source, n = 20 and there are two gateways.

| Scenario | Condition |
|---|---|
| S1 | Light load, 2 Mb/s |
| S2 | Heavy load, 10 Mb/s |
| S3 | Hotspot: nodes nearest one gateway generate 3× demand |
| S4 | External interference on one channel, steps 50-150 |
| S5 | Load shift at step 100: half the nodes triple their demand (4 Mb/s base) |
| S6 | Failure of the highest-degree relay at step 100 |
| B1 | Moving hotspot: the hot gateway region changes every 40 steps |
| B2 | Periodic load: demand modulated by 1 + 0.8·sin(2πt/60) |
| B3 | Sparse topology: rgg spacing 37.5 m |
| B4 | Unseen grid topology |
| B5 | Rapid regime shifts: the boosted half is redrawn every 25 steps |
| C1 | Very light load, 1 Mb/s |
| C2 | Static traffic: every source always on at the mean rate |
| C3 | Larger network, n = 50, 10 Mb/s |
| C4 | Larger network, n = 80, 10 Mb/s |
| C5 | Four gateways |

**Metrics.** We report five metrics per episode:

- aggregate throughput of all flows (Mb/s);
- Jain's fairness index of the windowed flow throughputs;
- the 5th percentile of the windowed flow throughputs (p5);
- loss rate;
- stability, the number of channel switches plus parent changes.

**Composite.** For scenario s and reference method R, the pre-registered composite of method M is

C(M, R, s) = Σ_k w_k·clip(d_k, -1, 1),

with weights w = 0.25 (aggregate), 0.20 (Jain), 0.20 (p5), 0.20 (loss rate) and 0.15 (stability). The relative improvements d_k are computed on metric means over the test topologies:

- higher-is-better metrics: d_k = (x_M - x_R)/|x_R|;
- loss rate: d_k = (x_R - x_M)/max(x_R, 0.01);
- stability, which can be zero: d_k = (x_R - x_M)/(x_R + 1).

The composite is averaged over the scenarios of a suite, so +0.10 means 10% better.

**Pre-registered criteria.** EGAT-MAPPO succeeds if, on suites A+B:

1. its mean composite against every baseline is at least +0.10;
2. it is significantly better (Holm-corrected, α = 0.05) in at least 60% of scenarios;
3. no scenario is worse than Ko by more than 0.05;
4. criterion 1 still holds when any one weight is halved or increased by half, with the weights renormalised.

**Statistics.** Each trained model is evaluated on 20 topologies per scenario: test seeds 0-19, plus fresh seeds 20-39 for confirmation.

- **Per-scenario tests.** We use a Wilcoxon signed-rank test on per-run composites and a paired bootstrap of the composite, both Holm-corrected over scenarios. These tests are conditional on the trained models.
- **Uncertainty of the mean.** For the mean over scenarios, we compute the composite for every pair of training seeds of the two methods. We then report the mean of this matrix with a 95% t-interval whose variance sums the between-seed variances of both methods (Welch-Satterthwaite degrees of freedom).
- **Disclosure.** Hysteresis, its margin-selection rule and this interval were introduced after the first test evaluation. They are recorded as dated amendments in the design specification and are discussed with the results.
