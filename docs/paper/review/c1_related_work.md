# C1 — Mechanism-level comparison with [15]–[17] (answer to the `//` comment after Table 2, §2, v5 line ~75)

Prepared 10 Nov 2026. Our reference point: each mesh node is an agent that picks its downlink channel (6 choices) and its parent (4 candidate slots). The model is an edge-aware GATv2 with a Ko local-cost residual prior, warm-started by behaviour cloning from Ko and trained with MAPPO (centralised critic, decentralised execution). Execution adds switch hysteresis (margin 2) and sequential channel decoding (parents before children). Neighbours exchange observations, about one coordination round per channel decision. Transmit power is fixed at 20 dBm.

Access summary:

| Paper | Full text read? | What was read |
|---|---|---|
| [16] Shin et al., IEEE Access 2024 | **Yes** (open access, CC BY-NC-ND) | Full 13-page PDF, pp. 14751–14763 |
| [15] Sun et al., IEEE TWC 2024 | **No** (closed; no preprint found) | Abstract, IEEE index terms, author keywords, captions of all 13 figures (IEEE Xplore metadata) |
| [17] Tang et al., JNCA 2025 | **No** (closed; ScienceDirect blocked automated access with a CAPTCHA; no preprint found) | Crossref metadata only (title, authors, vol. 239, art. 104177). Abstract content seen only as search-engine summaries, not read directly |

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

- Source: https://doi.org/10.1109/TWC.2024.3372694. IEEE Xplore document 10466475 (https://ieeexplore.ieee.org/document/10466475): abstract, keywords and figure captions read via the page's metadata and the `/rest/document/10466475/figures` endpoint. **Full text not accessible** (closed access; no arXiv or author preprint found via OpenAlex, Semantic Scholar or web search). **No page or section numbers can be given.**
- **Action space (partly verified, from captions only).** Fig. 1 contrasts low vs high *transmission power* of an AP when sending a frame to the gateway. Fig. 7 compares "transmission distance" across cases. Fig. 9 shows "optimized topologies". IEEE index terms include "Transmission Power". This indicates per-AP transmit-power (range) control that implicitly reshapes the multi-hop topology. *Unverified:* the exact action set, whether channels or next hops are chosen explicitly, and the decision period.
- **Information exchange.** *Unverified.* The abstract states only a "centralized-critic-distributed-actor scheme" (Fig. 3 caption says the same). The index term "Global Information" suggests the critic uses global state, but what actors observe or exchange at run time is not visible.
- **Reconfiguration rule.** *Unverified.* No switching-cost, hysteresis or rate-limit mechanism is mentioned in the abstract or captions. Fig. 11–12 show dynamic behaviour and recovery after an AP failure. Absence from the abstract does not establish absence from the paper.
- **Training/execution (verified at abstract level).** Multi-agent actor–critic (MACRL), centralised critic with distributed actors. A multi-objective reward balancing method trades end-to-end delay against channel efficiency. Validation in QualNet (Fig. 10, 12). Compared with a PGRL baseline (Fig. 13).
- **Difference from ours.** Similar CTDE training regime. The verified control variable (transmit power, hence connectivity) differs from ours (channel and explicit parent at fixed power). We could not verify whether [15] restricts reconfiguration frequency.

### [17] Tang, Sun, Liu, Li, Yuan — "Multi-agent reinforcement learning based dynamic self-coordinated topology optimization for wireless mesh networks", JNCA 239:104177, 2025

- Source: https://doi.org/10.1016/j.jnca.2025.104177 (ScienceDirect PII S1084804525000748). **Full text and abstract not accessible.** ScienceDirect returned HTTP 403 or a CAPTCHA, which was not bypassed. Crossref and OpenAlex carry no abstract. No preprint was found. Verified: title, authors, volume, article number (Crossref). Note: Sun and Tang co-author both [15] and [17].
- **Action space.** *Unverified.* Search-engine summaries of the ScienceDirect abstract (not read directly) describe a trade-off between end-to-end delay and *terminal transmit power*. This is consistent with v5's wording ("transmit-power control, targeting delay and energy"), but it is not confirmed from the paper.
- **Information exchange.** *Unverified.* The same summaries mention terminal agents coordinating through signals exchanged among agents ("self-coordinated"). Content, timing and size are unknown.
- **Reconfiguration rule.** *Unverified.* Nothing visible.
- **Training/execution.** *Unverified* beyond "multi-agent reinforcement learning" in the title.
- **Difference from ours.** Provisional: delay and power objective via power control, versus our channel-and-parent selection at fixed power with switching-aware execution. To be confirmed when the full text is obtained (library access or a request to the authors).

### Does any of [15]–[17] warrant an added control?

- [15] and [17] act, as far as can be verified, on transmit power. Our simulator fixes power at 20 dBm and acts on channel and parent. Reproducing them would require a different action space and objective, which would make a new problem rather than a control. Their numbers would not test our retained claims (EGAT+H vs Ko, Yuan+H and Li+H on the service–reconfiguration trade-off). **No control added; scope statement kept.**
- [16] is the only one expressible in our action space: a static channel plan per link, with parents from Ko or minimum hop. It is static and traffic-agnostic, however, while the retained claim concerns load-adaptive reconfiguration under the stated dynamics. It therefore does not challenge that claim directly. **No benchmark campaign started.** If a reviewer later asks, one feasible single-configuration control would be [16]'s GA run once per topology at t = 0 on the existing 20-topology set. That would cost one evaluation run, not a campaign. This is noted, not executed.

---

## 2. Replacement Table 2 (§2)

Table 2. Mechanism-level position of the study relative to representative prior work. "Unverified" marks items we could not confirm because the full text was inaccessible; it does not imply the feature is absent.
Work | Decision/action | Information exchange | Reconfiguration rule | Training/execution | Difference from ours
Ko [1] | Per-node channel from local cost; forwarding handled separately | Local neighbour channel usage | Local cost rule re-evaluated each round; no learned trade-off | Distributed heuristic, no training | Supplies our prior, BC teacher and baseline; does not choose parents
Hyacinth [11]; Alicherry et al. [12] | Channel assignment coupled with routing / load-aware tree | Neighbour load and channel information | Protocol- or optimisation-driven reassignment | Heuristic or centralised optimisation | Establish the channel–routing coupling we also address
Ramachandran et al. [9] | Interference-aware channel assignment | Measured interference, centralised server | Reassignment with default-channel redirection during the change | Centralised heuristic | Earlier handling of reconfiguration disruption
Yuan [2] | Channel/power allocation, cognitive radio | Attention over interference graph | None specific | Attention + DQN | Mesh adaptation compared numerically, with and without hysteresis
Li [3] | Resource-block/power allocation, C-V2X | GAT over vehicle graph | None specific | GAT + A2C | Mesh adaptation compared numerically, with and without hysteresis
Sun et al. [15] | Per-AP transmit power, implicitly reshaping multi-hop topology (Fig. 1, 7, 9 captions); exact action set unverified | Unverified; centralised critic, distributed actors (abstract, Fig. 3) | Unverified; dynamic and AP-failure experiments (Fig. 11–12) | MARL actor–critic, CTDE, reward balancing delay vs channel efficiency; QualNet | Acts on power and range, not on channel and explicit parent; switching control unverified; no numerical comparison
Shin et al. [16] | One channel per link from endpoint list intersection; topology and OLSR routes unchanged (Sec. IV-A–B, pp. 14757–14758) | Global communication graph and channel lists; weighted conflict graph up to 3 hops (Sec. III-B, p. 14756); no run-time exchange | None: one-shot static assignment, traffic-agnostic | No learning; greedy 2-approximation for Max list-Cut (Theorem 2, p. 14758); ns-3 + OLSR | Static and centralised; no parent choice, load adaptation or switching control; richer hidden/exposed-terminal conflict model than ours; no numerical comparison
Tang et al. [17] | Unverified (secondary summaries: terminal transmit power, delay–power trade-off) | Unverified (secondary summaries: signals exchanged among terminal agents) | Unverified | MARL (title); details unverified | Provisionally a power-control problem; full text not obtained; no numerical comparison
This work | Per-node downlink channel (6) and parent (4 slots, progress-masked), fixed 20 dBm | Neighbour observation exchange, ~1 coordination round per channel decision; edge features in GATv2 | Switch hysteresis (margin 2); sequential parent-before-child channel decoding | BC warm start from Ko, then MAPPO (centralised critic, decentralised execution) | Joint channel–parent control with explicit reconfiguration limits; simulation, topology confirmation and selected ns-3 tests

Note for authors: rows [1], [2], [3], [9], [11], [12] are condensed from the existing v5 text and table, not re-read for this comment. Check them against the sources before submission, or keep the v5 wording for those rows.

---

## 3. Replacement §2 paragraph (replaces "Recent mesh studies sharpen that comparison…", ≤ 200 words)

Recent mesh studies differ from this work mainly in what they control. Shin et al. [16] compute a one-shot channel assignment for a given topology. Links become vertices of a weighted conflict graph spanning up to three hops, and a greedy 2-approximation assigns each link a channel without changing routes or responding to traffic (Sections III–IV). Sun et al. [15] and Tang et al. [17] apply multi-agent reinforcement learning to mesh topology. The material available for [15] indicates per-access-point transmit-power control trained with a centralized critic and distributed actors. Secondary summaries of [17] describe a trade-off between end-to-end delay and terminal transmit power. In the material we could access, neither is reported to choose channels or parents directly. The present controller differs in three respects: it changes channel and parent jointly at fixed transmit power, its policy conditions on edge features and exchanged neighbour observations, and its execution limits reconfiguration through hysteresis and parent-first channel decoding. The action spaces do not coincide, so [15]–[17] are not numerical baselines, and our superiority claims remain limited to the Ko, Yuan and Li comparisons under the stated model. The full texts of [15] and [17] were unavailable to us.

(Word count ≈ 190.)

---

## 4. Status line

// Authors' status, 10 Nov 2026 — PARTLY – full text of [16] read and cited by section/page; [15] verified only from abstract, keywords and figure captions; [17] verified only from bibliographic metadata (full texts of [15] and [17] inaccessible); Table 2 and §2 rewritten at mechanism level; no control added because [15]/[17] act on transmit power and [16] is static, so the retained claims stay within the Ko/Yuan/Li scope.
