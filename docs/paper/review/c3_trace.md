# Comment 3 (§4): training transition, evaluation epoch, fallback entropy and baseline resources

Source of every statement: `mesh/model/agent.py` (`EGATAgent.prepare/act/learn/update/gae/bc_fit`), `mesh/model/egat.py` (`EGATNet.forward`), `mesh/model/ko_prior.py`, `mesh/rl/agents_io.py`, `mesh/rl/train.py`, the baseline files, and the trace `docs/paper/review/c3_trace.py`. The trace's output is in `c3_trace.out.txt`. Setting: final model `runs/v10/ckpt/egat_nogru_s0.pt` (`egat`, `use_gru=false`), n=20, scenario S2, environment seed 0, `airtime=domain, mac_eff=0.65, contention=0.0`, evaluated with `switch_margin=2.0`. Every number below was printed by the trace. The trace rebuilds `EGATNet.forward` one stage at a time and asserts that the result equals the network output. It also asserts that its reconstructed evaluation action equals the output of `agent.act` at every step.

---

## A. Manuscript text for §4 (replaces the "to be confirmed" parts of §4.3–4.6 and Algorithm 1)

### A.1 Candidate slots, prior and mask (common to training and evaluation)

Each node u has four parent slots. They hold link neighbours ordered by hop count to the nearest gateway (ascending), then SNR (descending), then node id, with −1 as padding. A slot is *valid* when it is not padding and its neighbour is alive. A gateway has no valid slot. The progress mask keeps the valid slots whose neighbour is strictly closer to a gateway (hop(v) < hop(u)). When no slot survives (a gateway, or a node that is cut off or dead), slot 0 is forced open so that the softmax is defined. The executed parent for that slot is then decoded to −1 (no parent), because the slot is not valid.

The two heads are computed at every step in this fixed order:

1. Shared encoder → route logits z_r ∈ ℝ⁴ and channel logits z_c ∈ ℝ⁶.
2. Residual prior: q_r = z_r − κ_r·z_K(c_r^Ko) and q_c = z_c − κ_c·z_K(c_c^Ko). The two scales are separate parameters κ_r = softplus(β_r) and κ_c = softplus(β_c), both initialised to 1 and shared by all nodes. In the s0 checkpoint they are κ_r = 0.703 and κ_c = 0.723. z_K is a per-node z-score taken over the node's finite costs. The slot cost is Ko's Bellman–Ford cost to the gateway through that neighbour, and an invalid slot gets +∞. Before the subtraction, +∞ is replaced by 0.
3. Progress mask: route logits outside the mask are set to −10⁹. **The prior is applied before the mask.** The channel head has no mask, so all six channels are always selectable.

Example (training, t=10, node 0): the slots are [4, 7, 14, 9] at hops [3, 3, 3, 4] and the node's own hop is 4, so the mask is [1, 1, 1, 0]. The raw logits [1.21, −6.94, 8.54, −1.32] become [1.91, −6.24, 7.78, −1.96] after the prior and [1.91, −6.24, 7.78, −10⁹] after the mask, giving π_p = [0.003, 0, 0.997, 0]. For node 1 (slots [2, 12, 19, −1] at hops [1, 2, 2, –], own hop 2), only slot 0 makes progress, so π_p = [1, 0, 0, 0] with H_route = 0 and log π_p = 0. Gateway 10 behaves the same way: slot 0 is forced, H_route = 0, and its executed next hop is −1.

### A.2 One training transition (`act(explore=True)` → `env.step` → `per_agent_reward` → `learn`)

- **Proposal.** Each node samples j ~ π_p and c ~ π_c independently, from the two masked, prior-adjusted heads. There is no sequential coordination between nodes during training.
- **Clock.** The channel head is evaluated at every step. On a channel step (t mod 10 = 0, including t=0) the sampled c is executed. On every other step the sample is discarded and c is set to the current radio-1 channel (`cur_chan`).
- **Executed action.** During training the executed action equals the proposal, apart from that channel hold. **Hysteresis, sequential channel decoding and the Ko fallback are all gated by `explore=False` and never act during training.** The only map from proposal to executed configuration is `decode`: slot → neighbour id (or −1 when the slot is invalid), radio 1 = c+1, and radio 0 = the parent's radio-1 channel. This map does not depend on the parameters. The trace confirms that the executed parent equals slots[j] for every node that has a valid slot.
- **Stored log-probability.** The trace stores log π_old = log π_p(j) + 1[channel step]·log π_c(c), where both factors come from the masked distribution in A.1 under the parameters used during collection. Node 0 at t=10: −0.0028 + (−1.3272) = −1.3300. The same node at t=11 (a hold step) stores −0.0001, because the channel factor is dropped. `_logp_entropy` recomputes the same quantity in the update, and its value differs from the stored one by at most 5×10⁻⁷. The PPO ratio therefore starts at 1. It refers to the proposal, and the proposal is the action that was executed.
- **Entropy.** The loss uses H_route + 1[channel step]·H_chan per node, with coefficient 0.01. Node 0 gives 1.605 nats at t=10 and 0.0013 nats at t=11.
- **Inactive channel head.** On non-channel steps the channel head contributes nothing to the ratio, the surrogate or the entropy bonus, because its log-probability and entropy are multiplied by is_chan = 0. On channel steps it enters through the joint factorised log-probability inside one clipped surrogate. The semi-MDP n-step channel advantage is disabled in this model (`semi_mdp=False`), so both heads share one GAE advantage.
- **Reward.** r_i uses λ_t, which is annealed linearly from 0.5 to 0.1 over the 100,000-step budget. The trace uses the end-of-training value 0.1. Rewards of dead nodes and their loss weights are set to 0.
- **Critic, reset and time limit.** v_i = V([h_i, mean_j h_j]). GAE uses b = c = 1 − done. The flag done is raised only at the 200-step episode limit, which is a time limit and not a terminal state of the network. **The time limit is therefore treated as termination:** in the trace, node 0's last-step return target is 1.163, which equals r_T and contains no V(s_T) term. The previous step bootstraps normally (target 3.791). The rollout length is 1,000 and the BC phase is 10,000 steps, both multiples of 200, so every rollout ends with done = 1 and the `_last_value` bootstrap branch never runs. A new episode starts from a new environment and topology, and the GAE trace is cut at done, so no credit crosses a reset. The first state of an episode receives an ordinary GAE target. The recurrent state is reset at t=0, which has no effect when `use_gru=false`. Yuan, Li and Q-neutral likewise multiply their bootstrap by (1 − done), so all learners treat the time limit the same way.
- **Behaviour cloning (first 10,000 of the 100,000 steps).** Ko acts and its action is stored. The parent target is the slot holding Ko's parent. If that parent is not one of the four slots, or Ko returns no parent, the target is the first valid slot. Ties are broken inside Ko by `argmin`, which picks the lowest node id for parents and the lowest channel index for channels. The route cross-entropy is multiplied by the progress mask at the teacher slot, so a Ko parent that the mask forbids (an equal-hop parent) produces **no** route target. In the trace episode this applied to 50 of 4,000 live-node targets (1.2%). **The channel head is also cloned**, by cross-entropy to Ko's channel on channel steps. The critic is fitted to Monte-Carlo returns of Ko's trajectories, again cut at the time limit. Fitting stops after 20 epochs or when the joint route-and-channel argmax agreement reaches 90%.

### A.3 One deterministic evaluation epoch (`act(explore=False)`, m = 2)

Order of operations at every step:

1. Compute q_r and q_c as in A.1, with the prior applied before the mask.
2. Hysteresis: add +m to the route logit of the incumbent parent's slot, but only if the incumbent is one of the slots. A masked incumbent stays near −10⁹, so a failed or non-progress incumbent cannot be kept. Add +m to the channel logit of the current radio-1 channel. The margins are added per head, not to a 24-action joint.
3. Parent: j = argmax of the biased route logits. Ties go to the lowest slot index (torch argmax), but no exact ties occurred: 0 in the S2 epoch.
4. Channel, on channel steps only: sequential decoding. Nodes are processed in order of depth in the tree defined by the chosen parents, then by node id. Each node takes its highest biased channel score among the channels not already taken by decided, live conflict neighbours. If all six are taken, it takes the least-used one (score − 10³·count). On other steps c = current channel.
5. Fallback: for each live node, H = H(π_r^m) + 1[channel step]·H(π_c^m), computed on the **biased** distributions from step 2. If H > 2.0, the node's whole action (parent and channel) is replaced by Ko's action. Ko's action is computed on Ko's own tree and is not re-coordinated with the sequential decoder.
6. `decode` maps the action to the executed (next hop, radio channels). A slot with no valid neighbour becomes −1.

Example (S2, t=10, node 0): the route logits [3.96, −8.17, 4.66, −10⁹] become [5.96, −8.17, 4.66, −10⁹] after the +2 on the incumbent slot 0. Hysteresis therefore keeps parent 4, which would otherwise have been replaced by slot 2. On the channel side, argmax picks c=3, but the sequential decoder moves the node to c=1 because a decided conflict neighbour holds channel 3. The biased entropy is H = 2.141 > 2, so **the fallback fires**. Ko's action is executed instead (next hop 14, channels [6, 2]), which changes the parent that hysteresis had just held.

Counts for the S2 epoch: 4,000 live node-decisions. The fallback fired once, at t=10 on node 0. Hysteresis changed the argmax 6 times for parents and 131 times for channels. The sequential decoder moved 98 channel choices away from the per-node argmax. There were no exact ties.

### A.4 Which entropy, and can the fallback fire?

The code computes the entropy as the **sum of the two factorised head entropies**, evaluated after mask and hysteresis. On channel steps this is H_route + H_chan. On the other nine steps out of ten it is H_route alone (`agent.py` line 126). For independent heads this sum equals the entropy of the 24-action product distribution. Its bound depends on the number k of progress-valid slots: H ≤ ln k + ln 6, which is at most ln 24 = 3.18.

- **Non-channel steps (90% of decisions):** H ≤ ln 4 = 1.386 < 2. **The fallback cannot fire.** The largest value observed was 0.69 nats.
- **Channel steps with k = 1** (most nodes, and all gateways): H ≤ ln 6 = 1.792 < 2. **It cannot fire.**
- **Channel steps with k ≥ 2:** the bound is ln 12 = 2.48 or higher, so the fallback can fire, and it does, rarely.

Across suites A+B (11 scenarios, evaluation seed 0, m = 2), the fallback fired on **9 of 43,901 live node-decisions (0.02%)**. All 9 events happened on channel steps and on the same node (node 0, k = 3). Seed 0 gives the same topology in nine of the eleven scenarios, which explains why the events cluster. Ko's action differed from the policy's action in 6 of the 9 events. Ko's parent never fell outside the progress mask (0 of 9), but the code does not guarantee this. The largest biased entropy was 2.20. Without the margin, 24 node-decisions would have exceeded 2 nats, so hysteresis lowers the entropy and suppresses most potential fallbacks. In practice the fallback is a rare, channel-step-only event, not an active safety mechanism.

### A.5 Baseline decision clocks, information, warm start and training resources

| Method | Channel decision | Parent decision | Observation used at execution | Warm start | Training interactions |
|---|---|---|---|---|---|
| M0 | never (fixed radio channels (0,1)) | every step, min-hop (lowest id on ties); changes only when the alive set changes | link adjacency, alive flags, gateways | – | none |
| Ko | every 10 steps (t mod 10 = 0); at any step if some parent breaks | same clock; switches parent only if new cost < 0.9 × current cost | link SNR→rate, load fraction, offered load, conflict graph, current channels; **network-wide Bellman–Ford distances** | – | none |
| Q-neutral | every step (joint 24 actions) | every step | own 7-tuple discretisation (load bin, unmet, gateway flag, hop bin, radio-1 channel, best-rate slot, rate bin); validity mask without progress | none | 100,000 env steps, 500 episodes (2.0 M agent transitions); tabular update for every live agent; ε 0.6→0.05 over 50 k steps |
| Yuan | every step (joint) | every step | own 18+20 features, plus dot-product attention over self and the first 3 slots, 2 layers; validity mask without progress | none | 100,000 env steps, 500 episodes; one DQN gradient step (batch 10 from a 10 k replay) per env step; ε 0.6→0.05 over 50 k |
| Li | every step (joint) | every step | 38 z-scored features, 2-layer GAT over the **conflict** graph; validity mask without progress | none | 100,000 env steps, 500 episodes; one A2C update per env step |
| Li+H, Yuan+H | as Li / Yuan | as Li / Yuan | as Li / Yuan; at evaluation, +1 for keeping the channel and +1 for keeping the parent on the joint scores | none | same checkpoint as Li / Yuan; m = 1 chosen on validation from {0, 0.25, 0.5, 1, 2, 4, 8, 16} |
| EGAT+H | every 10 steps (t mod 10 = 0), in training and evaluation | every step | 38 node features + 7 edge features, 2 GATv2 layers over link ∪ conflict; Ko slot and channel costs as prior (**network-wide Bellman–Ford**); progress mask; Ko action as fallback; mean-pooled critic in training only | 10,000 steps of Ko behaviour + BC of both heads and the critic (inside the 100 k) | 100,000 env steps, 500 episodes = 10,000 BC + 90,000 PPO (90 rollouts × 4 epochs × 20 minibatches); m = 2 from {0, 0.5, 1, 2, 3, 4} |

Shared by every learner: training scenarios are drawn uniformly from S1–S6 with a new topology for each episode, n = 20, using the same simulator settings. Checkpoints are validated every 10,000 steps plus once at the end, 11 validations × (11 scenarios × 4 validation seeds × 200 steps) = **96,800 non-learning env steps each**. The checkpoint with the best validation composite is kept. The selected s0 checkpoints are at step 100,000 (EGAT no-GRU), 60,000 (Yuan), 20,000 (Li) and 30,000 (Q-neutral). Weight averaging (EMA with decay 0.9995 per step) is applied to EGAT after BC, and to Li and Yuan from step 1; Q-neutral has none. All learned methods use the same sequential channel decoder at evaluation and none use it in training. The two methods with a 10-step channel clock, Ko and EGAT, have one tenth of the channel-change opportunities of Yuan, Li and Q-neutral, which may change channel at every step. Only Ko and EGAT use network-wide routing distance and enforce strict progress, so the decoder is the only shared component. The comparison is between systems that differ in information, decision clock and warm start, not between encoders.

---

## B. Discrepancies with the v6 text and corrections

1. **§4.6:** "The system also falls back to Ko's action when policy entropy exceeds 2 nats after hysteresis. … A joint 24-action distribution, or a specified sum of head entropies, can exceed it."
   **Correction:** "At evaluation only, a node falls back to Ko's full action (parent and channel) when H(π_r^m) + H(π_c^m) > 2 nats. Both entropies are computed after the mask and the hysteresis margin, and the channel term is included only on channel steps. Because the bound is ln k + ln 6, the fallback cannot fire on non-channel steps or for nodes with a single progress candidate. Across suites A+B (seed 0) it fired on 9 of 43,901 node-decisions (0.02%), all on channel steps."

2. **§4.3:** "The parent head is evaluated every step, while the channel head is evaluated every ten steps."
   **Correction:** "Both heads are computed at every step. A channel decision is taken only at t ≡ 0 (mod 10), including t = 0. In between, the radio-1 channel is held, and the channel head's output is excluded from the action, the log-probability and the entropy."

3. **§4.5, Eq. (9) and** "At a time-limit boundary, bootstrapping and continuing a trace need not have the same convention."
   **Correction:** "The implementation uses b = c = 1 − done, and done is set only at the 200-step time limit. The time limit is therefore treated as termination, with no bootstrap from V(s_T). Each 1,000-step rollout ends on an episode boundary, so no rollout-end bootstrap occurs. All learners use the same convention."

4. **§4.5:** "The described actor warm start fits parent choices by cross-entropy … This formulation does not imply that the channel head is cloned in the same way."
   **Correction:** "BC fits the parent head by cross-entropy to Ko's slot, except where the progress mask forbids it (1.2% of targets in the trace). It also fits the channel head by cross-entropy to Ko's channel on channel steps, and the critic to Ko's Monte-Carlo returns. The 90% stopping rule applies to joint route-and-channel agreement. The 10,000 BC steps are part of the 100,000-step budget."

5. **§4.4:** "The initial scale is described as one; its exact sharing across heads or contexts must follow the released implementation."
   **Correction:** "There are two scales, κ_r = softplus(β_r) and κ_c = softplus(β_c), each initialised to 1 and shared by all nodes. After training (s0) they are 0.703 and 0.723. z_K is a per-node z-score over finite costs. The prior is applied before the progress mask."

6. **§4.6, Eq. (10)**, q_H(a) = q(a) + m·1{a = a_old}: for EGAT the margin is added separately to the incumbent parent slot and the incumbent channel. For Li and Yuan it is added as +m (keep channel) and +m (keep parent) on the joint scores. Both are additive per component, but the per-step channel opportunity differs: every step for Li and Yuan, every tenth step for EGAT.

7. **Proposition 1 (§5.1)**, "no later operation overrides that condition": the Ko fallback is such a later operation. It can install a parent that is not progress-valid (equal hop). This did not happen in the trace (0 of 9 fallbacks), but the code does not prevent it. Either state the proposition for decisions without fallback or add the mask to the fallback.

8. **§4.6, Algorithm 1:** replace steps 3–4 with the exact order in A.3. Add that in training, steps 2–5 of A.3 (hysteresis, sequential decoding, fallback) are absent, so training is consistent with the action it executes. The policy is then deployed under execution rules it was not trained with, and the effect of those rules is measured only by validation.

9. **§4.5:** "Moving-average weights are maintained": EMA is also applied to Li and Yuan, but not to Q-neutral. Each learner also uses 96,800 non-learning validation env steps for checkpoint selection, plus the margin-selection runs for the +H variants.

// Authors' status: RESOLVED – The trace (`c3_trace.py`/`.out.txt`) gives the exact training and evaluation order from the code. Training executes the sampled factorised proposal, with log π_p(j) + 1[channel step]·log π_c(c) stored and the inactive channel head excluded, and the time limit treated as termination. The 2-nat fallback uses the sum of the two biased head entropies, so it can fire only on channel steps for nodes with at least two progress candidates (9 of 43,901 decisions); §4.3–4.6, Eq. (9), BC and the baseline-resource table are corrected accordingly.
