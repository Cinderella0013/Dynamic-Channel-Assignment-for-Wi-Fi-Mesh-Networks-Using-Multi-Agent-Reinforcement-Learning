# Comment 2: model definitions from the code

Scope: the items of comment 2 that v6 still lists as open (observation, queue and update order, potential and mask, gateway/no-parent handling, u_i, n_alive, reward counters, overlapping-domain airtime). The N_rep reconciliation already in v6 §3.2 is not repeated. All references are to the `ns3-validation` branch. The calibrated simulator settings used for every v10 result are `airtime="domain"`, `mac_eff=0.65`, `contention=0.0` (`runs/v10/calib.env`, passed with `--set`, `mesh/rl/train.py:164`). The `ScenarioConfig` defaults (`airtime="static"`, `mac_eff=1.0`, `mesh/sim/scenarios.py:25-27`) are not used for any reported number.

`docs/paper/review/c2_obs_dump.py` prints the vector below for one node of `MeshEnv(S2, seed 0)` and asserts the 18 + 20 = 38 layout.

---

## 1. Observation vector (38 entries per node)

The actor input is x_i = [o_i, m_i] (`mesh/model/agent.py:51,59`). Here o_i has F_O = 6 + |ℂ_all| + (K+1) = 6 + 7 + 5 = 18 entries and m_i has F_M = 5K = 20 entries, with K = 4 (`mesh/rl/agents_io.py:12-16`). All entries describe the state after the previous step (`mesh/sim/env.py:217,220-237`). Rates and queues are in Mb/s per 100 ms step. A queue of q therefore holds 0.1·q Mb.

### 1a. Own state, indices 0–17 (`agents_io.py:69-80`, `env.py:224-225`)

| Idx | Name | Definition (code) | Normalisation |
|---|---|---|---|
| 0 | offered_load | d_i(t−1): new demand generated at i in the last step, zero for gateways and dead nodes (`env.py:133`) | none (Mb/s) |
| 1 | unmet | o_i − x_i = (q_i + d_i) − x_i: offered minus served in the last step. Equals the new queue plus switch loss plus overflow (`env.py:134,224`) | none (Mb/s) |
| 2 | is_gateway | 1{i ∈ G} | binary |
| 3 | hop_to_gw | φ_i / n, with φ_i defined in §3 (unreachable or dead: φ_i = n, so the entry is 1) | ÷ n (`agents_io.py:75`) |
| 4 | alive | 1{i alive} | binary |
| 5 | backlog | q_i(t), the source queue after the last step (`env.py:181,209`) | none (Mb/s) |
| 6–12 | downlink channel | one-hot of the radio-1 channel over the 7 labels (1, 6, 11, 36, 40, 44, 48) (`radio.py:10`, `agents_io.py:76`) | one-hot |
| 13–17 | current parent | one-hot over {slot 0, …, slot 3, none}. "None" covers no parent and a parent that is not among the four candidates (`agents_io.py:77-79`) | one-hot |

Index 6 (channel 1) is structurally zero. Radio 1 is reset to index 1 (`env.py:75`), and every controller sets it to c+1 ∈ {1,…,6} (`agents_io.py:63`, `ko.py:93`). Channel 1 appears only on radio 0, the uplink, of gateways and parentless nodes (§4). The network therefore receives 37 informative entries and one constant.

### 1b. Candidate slots, indices 18–37 (`agents_io.py:82-91`)

Slot j ∈ {0,…,3} of node i holds candidate v = s_{ij} and occupies indices 18 + 5j + k:

| k | Name | Definition | Normalisation |
|---|---|---|---|
| 0 | valid | 1{slot filled and v alive} | binary |
| 1 | snr | static link SNR, P_rx(i,v) − N_0 in dB. It omits the S4 external interferer (`env.py:231`) | ÷ 40. Not clipped: about 0.45 at the −76 dBm link threshold, above 1 below ≈8 m |
| 2 | rate | MCS rate on the channel i and v currently share, including external noise; 0 if they share no channel (`env.py:227`, `graph.py:38-44`) | ÷ 143.4, in [0,1] |
| 3 | load_frac | load_ℓ / cap_ℓ on link i→v. Under domain airtime this equals D_ℓ/η, the domain airtime demand relative to η (§7); 0 if the link is unloaded (`env.py:228`, `flows.py:97`) | none, ≥ 0, may exceed 1 |
| 4 | ack | delivered/offered on link i→v in the last step, sent_ℓ / load_ℓ; 1 if the link is unloaded (`env.py:229`) | in [0,1] |

All five values are multiplied by `valid`, so empty slots and dead candidates read as zeros (`agents_io.py:90`). **The fifth descriptor is the per-link delivery (ACK) ratio**, the same quantity as edge feature 7. The four descriptors named in the text (feasibility, SNR, rate, load) are k = 0–3.

Slots are filled with the link neighbours of i (P_rx ≥ −76 dBm, any liveness), sorted by φ ascending, then SNR descending, then node id. The first four are kept and missing slots are padded with −1 (`agents_io.py:23-37`). The `valid` feature does **not** apply the progress mask or the gateway rule. Gateways therefore show valid = 1 on their filled slots, although their parent is always decoded to −1 (§4).

---

## 2. Queue limit and update order within one step

**Queue limit.** q_max = 5·μ, with μ the scenario mean on-rate (`env.py:121`): 10 in S1, 50 in S2, 30 in S3/S4/S6 and B1–B5, 20 in S5. Queues exist only at traffic sources. Relays hold no queue: a flow is served end-to-end at one rate along its path (fluid model).

**Order of operations in `MeshEnv.step`** (`env.py:116-218`). The action (C(t), P(t)) is applied to state at t:

1. **Validate.** Out-of-range parents become −1 (`env.py:113`).
2. **Failure** (S6, t = failure_step only). The non-gateway with the most links dies. Its queue counts as lost (`env.py:123-128`).
3. **Exogenous inputs.** Interferer noise (S4) is applied, and the hotspot moves (B1) (`env.py:129-132`).
4. **Demand.** d_i(t) is drawn from the on/off chain: persistence 0.9, initial on-probability 0.7, on-rate Exp(μ) times scenario multipliers (`traffic.py:5-6,28-48`). Gateways and dead nodes get zero. Offered load o_i = q_i(t) + d_i(t) (`env.py:133-134`).
5. **Switch detection.** sw_ch(i) = 1{radio-1 channel changed} for live nodes, and sw_rt(i) = 1{parent changed} for live sources. Each sets a one-step downtime counter (`env.py:139-143`).
6. **Rates.** The shared channel of each pair is the largest common channel index (`graph.py:24-35`); dead nodes share none. MCS rate follows (`env.py:144-145`). **Interrupted links**: every link of a retuned node on its new radio-1 channel, in both directions, which includes its children's uplinks; and the uplink of a node that changed parent. These links get rate 0 for this step (`env.py:146-150`).
7. **Routing.** The planned path uses rate > 0 and ignores downtime; the actual path uses this step's rate after interruptions. Both follow next_hop until a gateway is reached; otherwise the path is None (`env.py:152-153`, `flows.py:32-48`). The active flows are the live sources with o_i > 0 (`env.py:154`).
8. **Airtime/capacity and service.** `domain_allocate` gives the served rate x_i for each flow from the actual paths and o_i (§7, `env.py:160-161`). Per-link delivered traffic is accumulated for the ACK feature (`env.py:165-169`).
9. **Switch loss**, setting (a) only. A flow whose *planned* path contains an interrupted hop loses q_i(t), the queue carried into the step. The current step's arrivals d_i(t) are kept (`env.py:172-178`). Such a flow also has no actual path, so x_i = 0.
10. **Queue and overflow.** b_i = o_i − x_i − ℓ^sw_i, overflow = max(b_i − q_max, 0), q_i(t+1) = min(max(b_i, 0), q_max) (`env.py:179-181`). Loss = switch loss + failure loss + overflow (`env.py:215`).
11. **Downtime counters** decrease by one (`env.py:182-183`).
12. **Throughput window, utility, counters, reward** (§5–6, `env.py:189-205`).
13. **Commit** C(t), P(t), demand, queues and loads; t ← t+1; observe (`env.py:207-218`).

In one step, service draws on the queue and the new arrivals together. Switch loss and overflow are applied after service.

---

## 3. Progress potential φ and progress mask

**Potential.** φ_i is the multi-source BFS hop count from the live gateways on the physical link graph E_L restricted to live nodes (`flows.py:7-20`, `env.py:222`):

    φ_g = 0 for live g ∈ G;   φ_i = 1 + min_{v ∈ N_L(i), v alive} φ_v;   φ_i = n if unreachable or dead (`env.py:223`).

E_L contains the pairs with P_rx ≥ −76 dBm. It is distance-based and independent of channels and load (`graph.py:17-21`). φ therefore changes only when a node fails. Every agent reads φ from the same observation, so all decisions in a step use one common potential, as Proposition 1 requires.

**Mask** (`mesh/model/agent.py:40-47`). Let s_{ij} be slot j of node i. Then

    M_ij = 1{s_ij ≠ −1} · 1{s_ij alive} · 1{i ∉ G} · 1{φ_{s_ij} < φ_i},

and if no slot qualifies, M_i0 = 1. The mask is applied to the route logits after the Ko prior: logits − softplus(β_r)·z(Ko cost), masked with −10⁹ (`egat.py:116-118`). Channels are not masked. Slots are sorted by φ, so slot 0 is a progress parent whenever φ_i < n. The forced slot 0 therefore arises only for gateways, dead nodes and nodes cut off from every gateway.

---

## 4. Gateway and no-parent handling

Action a = 4c + j encodes downlink channel c+1 and parent slot j (`agents_io.py:3-5`). `decode` (`agents_io.py:56-66`) sets:

- **Parent**: p_i = s_ij if slot j passes `_valid_slots` (filled, candidate alive, i not a gateway; `agents_io.py:40-44`), otherwise −1. Gateways always get p_g = −1.
- **Radio 1 (downlink)**: c_i + 1 from the node's own action. This includes gateways and parentless nodes, which still choose a downlink channel.
- **Radio 0 (uplink)**: the parent's radio-1 channel if p_i ≥ 0, else index 0 (channel 1). Gateways and parentless nodes therefore park their uplink on channel 1. Because `common_channel` takes the largest shared index (`graph.py:24-35`), a parked uplink never displaces a dedicated channel.

Effects on traffic: a gateway generates no demand (`env.py:133`) and terminates paths. A source without a path to a gateway is not served. Its offered load stays in its queue up to q_max, it counts in N_unrouted (§6), and its windowed throughput enters the utility as log(0 + ε). Dead nodes produce no demand, share no channel and are excluded from every counter.

**Code note (cut-off nodes).** The comment at `agent.py:46` says forced slot 0 is "decoded to −1". `decode` uses `_valid_slots`, not the progress mask, so a live node cut off from all gateways but with a live neighbour is decoded to that neighbour. In S6 (seeds 0–19, steps 101–199, Ko driving), 297 of 495 cut-off node-steps decoded to a parent. The flow is unrouted either way, so throughput and loss are unchanged. Any cycle stays inside the cut-off component, and such parent changes do enter N_rc.

---

## 5. u_i, n_alive and the local reward (3)

**Windowed throughput** (`env.py:189-194`). With W = 10 steps (1 s, `scenarios.py:19`) and a_i(τ) = 1{o_i(τ) > 0}:

    x̄_i(t) = Σ_{τ∈W} x_i(τ)a_i(τ) / Σ_{τ∈W} a_i(τ)   (0 if the denominator is 0),
    F(t) = {i alive, i ∉ G : d_i(τ) > 0 for some τ ∈ W}.

**Global utility.** U(t) = Σ_{f∈F(t)} log(x̄_f + ε) with ε = 0.01 (`env.py:204`, `metrics.py:74-75`).

**Local utility** (`env.py:195-199`):

    u_i(t) = Σ_{f ∈ F(t) : i ∈ π̂_f(t)} log(x̄_f + ε),   with π̂_f = {f} if the planned path is None.

Here π̂_f is the planned path (step 7). It includes the source and the terminating gateway. Relays and gateways therefore receive the utility of every flow they carry, and an unroutable flow is charged at its source.

**n_alive.** This is the number of live nodes, **gateways included**, in the observation on which the action was taken: n_alive = max(Σ_i alive_i(t−1), 1) (`agents_io.py:95,105`). At the failure step the dying node is still counted and still receives a reward for that step.

**Local reward** (`agents_io.py:94-107`):

    r_i(t) = λ u_i(t) + (1−λ) R_t / n_alive − 0.1 s_i(t) − 0.05 r^p_i(t),   and r_i = 0 for dead i.

For EGAT, λ_t = 0.5 + (0.1 − 0.5)·min(k/K_total, 1), where k is the environment step including behaviour cloning (`agent.py:140-142`, `egat.py:45-46`). The baselines use λ = 0.5 (`train.py:120`).

**Local penalty terms.**

- s_i = 1{c_{i,1} changed} + 1{c_{i,0} changed, not explained by following}, plus one for each child whose uplink was retuned only because i retuned its downlink. A child's radio-0 change counts as "following" when the child kept the same parent and that parent's radio 1 changed. The change is then charged to the parent (`agents_io.py:98-103`).
- r^p_i = 1{p_i(t) ≠ p_i(t−1)} (`agents_io.py:104`).

With every node alive, Σ_i s_i = N_sw(t) and Σ_i r^p_i = N_rc(t). The local penalties are therefore an exact attribution of the two global counters in (2): the global penalty is split among the nodes that caused it, not copied to each node.

---

## 6. Reward counters (`env.py:200-205`, reported in `info`, `env.py:212-216`)

| Counter | Definition | In reward |
|---|---|---|
| N_sw(t) (`n_channel_switch`) | Σ_{i alive} Σ_{k∈{0,1}} 1{c_{i,k}(t) ≠ c_{i,k}(t−1)} | −0.1 (γ) |
| N_rc(t) (`n_route_change`) | Σ_{i alive, i∉G} 1{p_i(t) ≠ p_i(t−1)} | −0.05 (δ) |
| N_unrouted(t) (`n_drop`) | active flows (o_i > 0) whose **planned** path is None, i.e. a routing failure; a downtime interruption does not count | −0.1 (η) |
| N_int(t) (`n_interrupted`) | active flows with a planned path but no actual path (switch downtime) | not in reward |

R_t = U(t) − 0.1 N_sw − 0.05 N_rc − 0.1 N_unrouted (`env.py:205`, weights `env.py:43-48`). N_rep = Σ_t (N_sw + N_rc), as v6 §3.2 states.

---

## 7. Airtime allocation with overlapping domains (`flows.py:71-102`)

The allocation is a one-pass computation without iteration. Inputs are the actual paths π_f, the offered loads o_f of active flows, this step's rates R_ℓ (zero on interrupted links) and the shared channel ch(ℓ).

1. **Link load**: y_ℓ = Σ_{f: ℓ∈π_f} o_f. Every hop carries the flow's full offered load (`flows.py:77-81`).
2. **Airtime demand** of each loaded directed link: a_ℓ = y_ℓ / R_ℓ (`flows.py:83`).
3. **Neighbourhood**: k ∈ 𝒩(ℓ) iff ch(k) = ch(ℓ) and some endpoint of k equals, or lies within carrier-sense range (P_rx ≥ −82 dBm) of, some endpoint of ℓ. The relation is symmetric and includes ℓ itself (`flows.py:84-89`). Links on a relay's two radios use different channels and do not interact.
4. **Domain load** centred on ℓ: D_ℓ = Σ_{k∈𝒩(ℓ)} a_k (`flows.py:90`).
5. **Budget**: η_ℓ = mac_eff / (1 + κ(n_tx(ℓ) − 1)) = 0.65 with κ = 0 (`flows.py:91`).
6. **Domain factor**: f_ℓ = min(1, η / D_ℓ) (`flows.py:92`).
7. **Overlap rule.** A link belongs to the domain centred on every neighbour and takes the tightest factor: g_ℓ = min_{k∈𝒩(ℓ)} f_k (`flows.py:93`).
8. **Flow service**: x_f = o_f · min_{ℓ∈π_f} g_ℓ (`flows.py:98-101`).

Since every k ∈ 𝒩(ℓ) has g_k ≤ f_ℓ, each domain satisfies Σ_{k∈𝒩(ℓ)} a_k g_k ≤ f_ℓ D_ℓ ≤ η, so no domain exceeds 65% airtime. The rule is work-conserving only in one sense: unloaded co-channel links cost nothing, unlike the static 1/k share (`graph.py:47-64`). Airtime freed when a flow is throttled at another bottleneck is **not** redistributed. Loads are not reduced for upstream throttling, and there is no progressive filling. The allocation is therefore conservative where domains chain. The reported link capacity is cap_ℓ = y_ℓ η / D_ℓ (`flows.py:97`), so the observed load_frac equals D_ℓ/η.

---

## 8. Corrections to the v6 text

1. §3.1, ‘‘Its downlink radio chooses a channel from ℂ={6,11,36,40,44,48}; its uplink follows the downlink channel of its parent.’’
   → Add: ‘‘A gateway or a node without a parent parks its uplink radio on channel 1. All episodes start with uplink channel 1, downlink channel 6 and minimum-hop parents.’’ (`agents_io.py:65`, `env.py:75-76`.) v6 §3.2 calls this start ‘‘channel pair (0,1)’’; that is the index pair for channels (1, 6).

2. §3.1, ‘‘ordered by hop count and then SNR’’ → ‘‘the four link neighbours with the lowest hop count φ, ties broken by higher SNR and then lower node id.’’ ‘‘A progress mask allows only feasible parents that are closer to a gateway under a potential φ’’ → give φ and M from §3. ‘‘Gateways and nodes with no feasible parent require separate execution rules’’ → replace with the §4 rules.

3. §3.1 writes E_L(t). In the code E_L is fixed by node positions and changes only through node failure. Write E_L, restricted to live nodes.

4. §3.2, ‘‘Reconfiguration interrupts the affected link for one step. In setting (a), affected flows also lose the queues waiting on that path’’ → ‘‘A downlink retune blocks all of the node's links on its new channel, including its children's uplinks, for one step. A parent change blocks the new uplink for one step. In setting (a), a flow whose planned path crosses a blocked link loses the queue it carried into the step; the step's new arrivals are kept.’’ Add q_max = 5μ and the order in §2.

5. §3.3, ‘‘When demand exceeds that limit, the described model reduces the service of affected flows proportionally and allows idle airtime to be reused.’’ → ‘‘An over-subscribed domain scales the offered load of every flow through it by η/D. A link in several domains takes the smallest factor, and a flow takes the smallest factor on its path. Idle co-channel links consume no airtime, but airtime released by throttling elsewhere is not reassigned.’’ Then give steps 1–8 of §7, which also resolve the sentence ‘‘Overlapping domains and queue updates require a precise scheduling rule.’’

6. §3.4, ‘‘x̄_f is throughput averaged over one second’’ → ‘‘averaged over those of the last ten steps (1 s) in which the source had traffic to send.’’ The flow set is the live non-gateways with demand in the window. N_unrouted counts routing failures only, not switch-downtime interruptions.

7. §3.4, ‘‘u_i is the utility associated with flows traversing node i and n_alive is the number of live or available nodes under the implemented convention’’ → use the §5 definitions. u_i sums log(x̄_f + 0.01) over flows whose planned path includes i, gateways included; an unroutable flow is charged at its source. n_alive counts live nodes including gateways. ‘‘the mapping of the local terms in (3) is still to be stated’’ → ‘‘s_i and r^p_i partition N_sw and N_rc. An uplink retune caused by the parent's downlink retune is charged to the parent.’’

8. §4.2, ‘‘The described observations include … candidate feasibility, SNR, rate and load … the text names four candidate descriptors’’ → ‘‘Each slot holds five values: availability, SNR/40, current-channel rate/143.4, domain utilisation and the previous-step delivery ratio (Table C2).’’ Also state that the own-state block is 6 scalars, a 7-way downlink one-hot and a 5-way parent one-hot, and that loads and queues enter unnormalised in Mb/s.

9. §5.1, Proposition 1 assumes ‘‘no later operation overrides that condition’’. The entropy fallback replaces an unsure node's action with Ko's (`agent.py:123-130`). Ko's parent is mapped to a slot without the progress mask (`ko_prior.py:42-53`) and can be an equal-hop neighbour. When Ko drives S6 (seeds 0–9), 2,935 of 35,010 live non-gateway decisions (8.4%) fall outside the mask. The proposition covers the policy's own choices only; fallback nodes need Ko's own loop handling (`ko.py:71-72`). This belongs to comment 3, but the condition is stated here. A related point for the 2-nat threshold: the fallback entropy is H(route) + H(channel) on channel steps and H(route) alone otherwise (`agent.py:126`). With a maximum of log 4 ≈ 1.39 < 2, fallback can fire only on channel steps, every t_c = 10.

---

// Authors' status: RESOLVED – The 38-entry observation (fifth slot descriptor = per-link delivery ratio), q_max = 5μ and the step order, the BFS hop potential and progress mask, the gateway/no-parent rules, u_i, n_alive, the reward counters and the min-over-domains airtime rule are now defined from the code with file:line references, and nine v6 sentences are corrected accordingly.
