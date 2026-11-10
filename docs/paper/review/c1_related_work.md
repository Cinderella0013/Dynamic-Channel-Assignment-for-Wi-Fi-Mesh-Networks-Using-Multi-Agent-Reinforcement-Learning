# C1 — Mechanism-level comparison with [15]–[17] (answer to the `//` comment after Table 2, §2, v5 line ~75)

Prepared 10 Nov 2026. Our reference point: each mesh node is an agent that picks its downlink channel (6 choices) and its parent (4 candidate slots). The model is an edge-aware GATv2 with a Ko local-cost residual prior, warm-started by behaviour cloning from Ko and trained with MAPPO (centralised critic, decentralised execution). Execution adds switch hysteresis (margin 2) and sequential channel decoding (parents before children). Neighbours exchange observations, about one coordination round per channel decision. Transmit power is fixed at 20 dBm.

Access summary:

| Paper | Full text read? | What was read |
|---|---|---|
| [16] Shin et al., IEEE Access 2024 | **Yes** (open access, CC BY-NC-ND) | Full 13-page PDF, pp. 14751–14763 |
| [15] Sun et al., IEEE TWC 2024 | **Yes** (full PDF supplied by the authors' library access, 11 Nov 2026) | Full 13-page article, pp. 10501–10513 |
| [17] Tang et al., JNCA 2025 | **Yes** (full PDF supplied by library access, 11 Nov 2026) | Full 13-page article, art. 104177, pp. 1–13 |

---

## 1. Per-paper notes

### [16] Shin, Lee, Choi — "A Conflict-Aware Channel Assignment in Multi-Radio Multi-Channel Wireless Mesh Networks", IEEE Access 12:14751–14763, 2024

- Source: https://doi.org/10.1109/ACCESS.2024.3357142 (IEEE Xplore document 10411895). Full PDF retrieved from IEEE Xplore and read in full.
- **Action space.** One channel per *link* of a given communication graph, chosen from the intersection of the two endpoint routers' channel lists, L(v) = L(n_i) ∩ L(n_j) (Sec. IV-A, Eq. (8), p. 14757). Routing and parent choice are not decided. The algorithm "preserves the network topology" (Sec. IV-B, p. 14758), and routes come from OLSR in the evaluation (Sec. V, p. 14760). Transmit power is not a decision variable.
- **Information used / exchanged, and when.** The input is the whole communication graph plus per-router channel lists. Links become vertices of a weighted conflict graph with edges between links within 3 hops. Weights are set by hop distance: 1-hop for interference, 2-hop for mainly hidden terminals, 3-hop for mainly exposed terminals (Sec. III-B, pp. 14755–14756). The weights are tuned to (1, 2, 2) by simulation (Sec. V, Table 2, p. 14761). The paper describes no message-passing protocol or distributed exchange. It is a global graph algorithm run once before operation. Traffic load is not an input.
- **Reconfiguration/switching rule.** None. The assignment is one-shot and static. There is no traffic- or time-triggered reassignment and no switching cost. GA has two steps: first it fixes vertices with a single available channel, then it assigns the remaining vertices greedily, one at a time, to the channel that adds the least monochromatic conflict weight (Fig. 6–7, Sec. IV-B, pp. 14757–14758). It is a 2-approximation for the dual Max list-Cut and runs in O(|E_C|²) (Theorem 2, p. 14758). The related-work section notes dynamic channel switching by others [9] but does not adopt it (Sec. II, p. 14752).
- **Training/execution.** No learning. Deterministic combinatorial algorithm. Evaluation: ns-3, 802.11, RTS/CTS disabled, OLSR with 30 s initialisation, 100 s runs, 5 random UDP source–sink pairs at 16–1024 kb/s. Topologies: lattices of 100 nodes, a 25-node chain, and 20–50 random nodes. 3 or 12 channels. Compared with a greedy heuristic GH (pp. 14760–14762).
- **Difference from ours.** [16] solves a static, traffic-agnostic, centralised channel-colouring problem on a fixed routing topology. We solve a load-adaptive, per-node, decentralised joint channel-and-parent decision over time, with explicit control of reconfiguration frequency. [16]'s conflict model (hidden or exposed terminals up to 3 hops) is richer than our interference-edge model. Its approximation guarantee has no counterpart in our work.

### [15] Sun, Lv, Xiao, Liu, Tang, Li, Mu — "Multi-Agent Reinforcement Learning for Dynamic Topology Optimization of Mesh Wireless Networks", IEEE TWC 23(9):10501–10513, 2024

- Source: https://doi.org/10.1109/TWC.2024.3372694. Full text read.
- **Action space.** Each AP is one actor with three actions on its own transmit power: increase by ΔS, decrease by ΔS, or keep (Sec. II, item 3 and Eq. (2), p. 10503). All APs operate on the same channel (Sec. II assumptions, p. 10503). Channels and parents are not chosen: the topology, and so the multi-hop paths, follow from which APs can reach each other at the chosen powers under the Friis receive-sensitivity condition (Eq. (1), p. 10503).
- **Information exchange.** The actor's input is only the AP's own transmit power (one input neuron, Sec. IV-B, p. 10507). A centralized critic gathers global information, computes the rewards, and sends each AP its advantage value H(S_it) as a 4-byte float over the mesh; that signalling delay is not modelled (Sec. IV-A, pp. 10505–10506). APs do not exchange observations with each other.
- **Reconfiguration rule.** No switching cost, hold time or rate limit. Power moves by at most one step ΔS per decision, and "keep" is one of the three actions (Eq. (2), p. 10503). The reward balances E2E delay against channel efficiency (Sec. III, pp. 10504–10505).
- **Training/execution.** MACRL: centralized critic with distributed actors, policy-gradient actor update (Sec. IV, pp. 10505–10507). Evaluated in the authors' simulation and in QualNet (Sec. V-B, p. 10509), with an AP-failure test during training (Sec. V-C, p. 10510) and a comparison with distributed PGRL (Sec. V-D, p. 10511).
- **Difference from ours.** Same training regime family (centralised critic, decentralised actors). The control variable differs: [15] reshapes connectivity through transmit power on one shared channel, while we choose a channel among six and an explicit parent at fixed power. Its actors see only their own power; ours use neighbour features and edge features. It has no model of reconfiguration cost, while switching cost is central to our setting.

### [17] Tang, Sun, Liu, Li, Yuan — "Multi-agent reinforcement learning based dynamic self-coordinated topology optimization for wireless mesh networks", JNCA 239:104177, 2025

- Source: https://doi.org/10.1016/j.jnca.2025.104177. Full text read. Sun and Tang co-author both [15] and [17]; [17] extends the same power-control formulation.
- **Action space.** Per-node transmit power with the same three actions as [15]: +ΔS, −ΔS or keep (Sec. 3.1, p. 3). All nodes share one channel (Sec. 3.1 assumption (1), p. 3). S_min is set so the topology stays connected (Sec. 3.1, p. 4). The reward trades E2E delay against transmit power through a Gaussian-shaped term (Sec. 3.2, pp. 4–5).
- **Information exchange.** A learned all-to-all communication step. Each agent encodes its local observation with an MLP and sends the encoding as a signal. Every agent concatenates the signals of all n agents and turns them into a softmax "self-coordinated communication matrix", which weights whose messages to use (Sec. 4.2, Eqs. (12)–(13), p. 6). The central critic is an LSTM over global state (Sec. 4.3, p. 6).
- **Reconfiguration rule.** No switching cost, hold time or rate limit; power changes by one step per decision.
- **Training/execution.** Actor–critic with centralized LSTM critic and decentralized actors that exchange the learned signals (Sec. 4, pp. 5–7). Compared with MADDPG, MAPPO and MAAC, and reports parameters and FLOPs (Sec. 5.2–5.3, Table 2, p. 7). Throughput reported for 10 nodes (Fig. 5, p. 8), scaling with the number of nodes (Sec. 5.6, Fig. 11, p. 10), and validation on a hardware mesh platform (Fig. 12, p. 10; Sec. 6, pp. 11–12).
- **Difference from ours.** Like [15], it controls transmit power on a single channel and does not choose channels or parents. Its communication is all-to-all among the n agents, so each decision needs signals from every agent; ours uses neighbour-only attention with edge features. It models no reconfiguration cost, while ours limits switching with hysteresis and parent-first channel decoding. It reports a hardware validation, which our study does not have.

### Does any of [15]–[17] warrant an added control?

- [15] and [17] act on transmit power on a single shared channel (verified from the full texts). Our simulator fixes power at 20 dBm and acts on channel and parent. Reproducing them would require a different action space and objective, which would make a new problem rather than a control. Their numbers would not test our retained claims.
- [16] is the only one expressible in our action space: a static channel plan per link, with parents from Ko or minimum hop. It is static and traffic-agnostic, however, while the retained claim concerns load-adaptive reconfiguration under the stated dynamics. It therefore does not challenge that claim directly. **No benchmark campaign started.** If a reviewer later asks, one feasible single-configuration control would be [16]'s GA run once per topology at t = 0 on the existing 20-topology set. That would cost one evaluation run, not a campaign. This is noted, not executed.

---

## 2. Replacement Table 2 (§2)

Table 2. Mechanism-level position of the study relative to representative prior work.
Work | Decision/action | Information exchange | Reconfiguration rule | Training/execution | Difference from ours
Ko [1] | Per-node channel from local cost; forwarding handled separately | Local neighbour channel usage | Local cost rule re-evaluated each round; no learned trade-off | Distributed heuristic, no training | Supplies our prior, BC teacher and baseline; does not choose parents
Hyacinth [11]; Alicherry et al. [12] | Channel assignment coupled with routing / load-aware tree | Neighbour load and channel information | Protocol- or optimisation-driven reassignment | Heuristic or centralised optimisation | Establish the channel–routing coupling we also address
Ramachandran et al. [9] | Interference-aware channel assignment | Measured interference, centralised server | Reassignment with default-channel redirection during the change | Centralised heuristic | Earlier handling of reconfiguration disruption
Yuan [2] | Channel/power allocation, cognitive radio | Attention over interference graph | None specific | Attention + DQN | Mesh adaptation compared numerically, with and without hysteresis
Li [3] | Resource-block/power allocation, C-V2X | GAT over vehicle graph | None specific | GAT + A2C | Mesh adaptation compared numerically, with and without hysteresis
Sun et al. [15] | Per-AP transmit power: +ΔS, −ΔS or keep (Eq. 2, p. 10503); one shared channel; paths follow from the resulting connectivity | Actor sees only its own power (p. 10507); centralized critic sends each AP a 4-byte advantage value over the mesh (pp. 10505–10506) | No switching cost or hold rule; one power step per decision | MACRL: centralized critic, distributed actors; own simulator and QualNet; AP-failure test; vs PGRL (pp. 10509–10511) | Controls power and connectivity on one channel, not channel and parent; no neighbour features; no reconfiguration cost
Shin et al. [16] | One channel per link from endpoint list intersection; topology and OLSR routes unchanged (Sec. IV-A–B, pp. 14757–14758) | Global communication graph and channel lists; weighted conflict graph up to 3 hops (Sec. III-B, p. 14756); no run-time exchange | None: one-shot static assignment, traffic-agnostic | No learning; greedy 2-approximation for Max list-Cut (Theorem 2, p. 14758); ns-3 + OLSR | Static and centralised; no parent choice, load adaptation or switching control; richer hidden/exposed-terminal conflict model than ours; no numerical comparison
Tang et al. [17] | Per-node transmit power: +ΔS, −ΔS or keep (Sec. 3.1, p. 3); one shared channel | Learned all-to-all signals weighted by a softmax communication matrix (Sec. 4.2, p. 6); LSTM central critic (Sec. 4.3, p. 6) | No switching cost or hold rule; one power step per decision | Actor–critic with learned communication; vs MADDPG, MAPPO, MAAC (pp. 7–8); node scaling (p. 10); hardware validation (Fig. 12, p. 10) | Power control on one channel with all-to-all messaging; ours selects channel and parent with neighbour-only attention and explicit switching limits
This work | Per-node downlink channel (6) and parent (4 slots, progress-masked), fixed 20 dBm | Neighbour observation exchange, ~1 coordination round per channel decision; edge features in GATv2 | Switch hysteresis (margin 2); sequential parent-before-child channel decoding | BC warm start from Ko, then MAPPO (centralised critic, decentralised execution) | Joint channel–parent control with explicit reconfiguration limits; simulation, topology confirmation and selected ns-3 tests

Note for authors: rows [1], [2], [3], [9], [11], [12] are condensed from the existing v5 text and table, not re-read for this comment. Check them against the sources before submission, or keep the v5 wording for those rows.

---

## 3. Replacement §2 paragraph (replaces "Recent mesh studies sharpen that comparison…", ≤ 200 words)

Recent mesh studies differ from this work mainly in what they control. Shin et al. [16] compute a one-shot channel assignment for a given topology. Links become vertices of a weighted conflict graph spanning up to three hops, and a greedy 2-approximation assigns each link a channel without changing routes or responding to traffic (Sections III–IV). Sun et al. [15] and Tang et al. [17] optimize mesh topology through transmit power: each node raises, lowers or keeps its power by one step, all nodes share one channel, and paths follow from the resulting connectivity ([15], Eq. (2), p. 10503; [17], Section 3.1). In [15] each actor sees only its own power and a centralized critic sends it a scalar advantage (pp. 10505–10507). [17] adds learned all-to-all signalling weighted by a softmax communication matrix (Section 4.2) and reports a hardware validation. Neither models a cost for changing the configuration. The present controller differs in three respects: it chooses channel and parent jointly at fixed power across six channels, it exchanges information only with graph neighbours through edge-aware attention, and it limits reconfiguration through hysteresis and parent-first channel decoding. Because the action spaces differ, [15]–[17] are not numerical baselines, and our superiority claims remain limited to the Ko, Yuan and Li comparisons.

(Word count ≈ 195.)

---

## 4. Status line

// Authors' status, 11 Nov 2026 — RESOLVED – full texts of [15], [16] and [17] read; actions, information exchange and reconfiguration rules cited by page/section; Table 2 and Section 2 rewritten at mechanism level; no control added because [15] and [17] act on transmit power on one channel and [16] is static, so the retained claims stay within the Ko/Yuan/Li scope.
