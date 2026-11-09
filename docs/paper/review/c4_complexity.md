# Comment 4: complexity of the controller, checked against the code and measured

Scope: Section 5.3 of the revised manuscript (`egat-mappo-v6.txt`, lines 280–295) and the related sentences in Sections 4 and 8.3. Model: final checkpoint `runs/v10/ckpt/egat_nogru_s0.pt` (EGAT, GRU off, two layers, H = 4, d = 64), executed with switch margin m = 2.0. Benchmark: `docs/paper/review/c4_bench.py`; raw output: `docs/paper/review/c4_bench.out.txt`.

Notation: n nodes; E_L directed link edges; E_C directed conflict edges; E = |link ∪ conflict| + n self-links (the attention neighbourhood); L = 2 layers; d = 64; C = 6 downlink channels; P = K = 4 parent slots; h = height of the parent tree; I_BF = Bellman–Ford rounds until the distances stop changing (≤ n).

## 1. What the implementation does

| Item | Implementation (file:line) | Cost per decision |
|---|---|---|
| Edge tensor | Dense. `prepare` allocates `e = zeros(n, n, 7)` and scatters the sparse `edge_index` into it (`mesh/model/agent.py:54-55`). The adjacency is a dense n×n boolean mask with self-loops (`agent.py:56-59`). | O(n²) time and storage |
| Observation views | `edge_matrix` builds a dense n×n array for each feature it is asked for (`mesh/sim/env.py:35-40`). It is called about 10 times per decision: `neighbor_slots`, `agent_features`, `ko_route_via`, `ko_channel_costs` and `sequential_channels`. The edge list itself comes from `np.nonzero` over n×n (`env.py:230`). | T_obs = O(n²) |
| Attention layer | Dense GATv2. `z = s + t` broadcasts to (B, n, n, H, d/H) (`mesh/model/egat.py:80`). The edge projection runs on the full n×n×7 tensor (`egat.py:82`). The softmax runs over all n columns, with non-edges set to −∞ (`egat.py:84`). Aggregation is a dense einsum over j (`egat.py:85`). | O(nd² + n²d) per layer; activations O(n²d) |
| Heads and critic | Two MLPs (64 → 64 → C or K). The centralised critic is also evaluated on every inference call (`egat.py:119-121`), although execution does not use its output. | O(nd² + nd(C+K)), plus O(nd·128) for the critic |
| Ko prior, routing part | `ko_slot_costs` → `ko_route_via` runs dense synchronous Bellman–Ford (`mesh/baselines/ko.py:31-36`) with up to n rounds of an n×n min. | O(I_BF·n²), worst case O(n³) |
| Ko prior, channel part | Python loops over nodes and channels (`mesh/model/ko_prior.py:35-38`), each doing a dense row mask. | O(n²C) |
| Ko cost normalisation | Per node, a z-score over that node's finite candidate costs, with ε = 1e-6 (`ko_prior.py:9-17`). Invalid slots stay at +∞, become 0 through `nan_to_num`, and are then removed by the slot mask (`egat.py:116-118`). The two scales are separate, softplus(β_c) and softplus(β_r), each initialised to 1 and shared across nodes (`egat.py:101-103`). | O(n(C+K)), negligible |
| Progress potential φ | Hop count to the nearest gateway, from BFS in `hops_to_gateway` (`mesh/sim/flows.py:7-20`). It scans dense rows (`np.flatnonzero(adj[u])`) and is recomputed in every observation (`env.py:222`), although it only changes when a node fails. The progress mask is then O(nK) (`agent.py:40-47`). | O(n²) as implemented; O(n + E_L) with adjacency lists |
| Parent decoding | Hysteresis bias, then argmax over K slots (`agent.py:105-114`). | O(nK) |
| Channel decoding | `sequential_channels` (`mesh/rl/agents_io.py:110-133`): depth by pointer chasing (`:122-127`), a sort by (depth, id) (`:130`), then for each node a dense conflict-row scan with `bincount` (`:131`). It runs on every step, including the nine of ten steps where its output is discarded (`agent.py:115-120`). | O(n·h + n log n + n² + nC) |
| Entropy fallback | If any node's entropy exceeds 2 nats, the full Ko action is computed for the whole network (`agent.py:126-130`). On channel steps that is another Bellman–Ford and Ko's tree-ordered channel pass. | O(I_BF·n² + n·h + n²C), only when triggered (measured rate ≈ 10⁻⁴) |
| Encoder caching | None. Each 100 ms step reruns `prepare` and the full forward pass (`agent.py:86-90`). The GRU is disabled in the final model, so `h_prev` is unused. Embeddings are not reused across the nine route-only steps between channel decisions. | T_episode = T_steps × T_exec (200 steps) |

Corrected per-decision bound (centralised simulation of all n agents):

    T_exec = O( L(nd² + n²d) + nd² + nd(C+K) + I_BF·n² + n²C + n·h + n log n + n² )
           = O( n²(Ld + I_BF + C) + nd²(L+1) ),   with I_BF ≤ n (worst case O(n³)).

Storage, excluding weights: O(n²(d + 7)) at inference, because the two layers run one after the other under `no_grad` and so only one layer's n×n×d buffer is alive at a time. In training it is O(B·L·n²d) for a minibatch of B = 50 transitions, since autograd keeps every layer's buffers. Weights: 29,132 actor parameters and 16,641 critic parameters (45,773 in total), which matches O(Ld² + 38d + 7Ld + d(C+K)).

Training term: O(K_ep·B·[L(nd² + n²d) + nd² + nd(C+K)]) per update. The Ko prior is computed once per stored transition, not once per epoch.

Sequential decoding is one O(n) centralised loop, not n separate forward passes. It is still inherently sequential, however. Node u can only choose after every conflict neighbour that comes earlier in the total (depth, id) order. So in a distributed realisation the number of rounds equals the longest chain in that precedence DAG. That chain is generally longer than the tree height, because conflicting nodes at the same depth are serialised by id (Section 2).

## 2. Measurements

Hardware: 13th Gen Intel Core i5-13450HX (10 cores: 6 P + 4 E, 16 threads), 31.7 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB). Windows 11 (10.0.26300), AC power, Balanced power plan. Python 3.12.10, PyTorch 2.11.0+cu128, NumPy 2.4.3.

Protocol: n = 20 (S2), 50 (C3) and 80 (C4). All are random geometric graphs with 2 gateways and a mean load of 10 Mb/s. For each n: 2 episodes × 200 steps (seeds 0 and 1), with the first 10 steps of each episode discarded, giving 380 decisions. Latency is wall time of `agent.act(obs)`: observation tensors, Ko prior, progress mask, encoder, heads, hysteresis, sequential decoding and fallback. `env.step` is excluded. All n agents are computed in one centralised call.

CPU runs used one thread (`torch.set_num_threads(1)`, OMP/MKL = 1), and the process was pinned to one P-core logical CPU. Without pinning, medians varied 3–6× between repeated runs as Windows moved the thread between core types. Each configuration ran in its own subprocess. No other Python process was running, and system CPU load was 11–32%, as logged in the output file.

| n | Device | Median (ms) | p95 (ms) | Max (ms) | Prepare, of which Ko prior (ms) | Forward (ms) | Peak RSS (MB) | CUDA peak (MB) | NumPy heap peak (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 20 | CPU, 1 thread | 10.6 | 12.7 | 49.9 | 5.0 / 4.2 | 2.9 | 547 | – | 0.10 |
| 50 | CPU, 1 thread | 20.3 | 24.9 | 65.9 | 11.1 / 9.8 | 5.3 | 551 | – | 0.43 |
| 80 | CPU, 1 thread | 32.1 | 36.9 | 47.6 | 18.1 / 15.9 | 10.1 | 555 | – | 0.94 |
| 20 | RTX 5060 | 19.0 | 26.7 | 77.5 | 5.2 / 4.1 | 9.9 | 1150 | 10.0 | 0.10 |
| 50 | RTX 5060 | 24.8 | 35.2 | 55.4 | 11.5 / 10.7 | 8.0 | 1151 | 11.6 | 0.43 |
| 80 | RTX 5060 | 33.0 | 41.5 | 63.0 | 18.3 / 16.5 | 7.5 | 1153 | 14.6 | 0.95 |

Notes on the table:
- The prepare, Ko prior and forward columns are medians over a subsample of every 20th decision.
- Peak RSS is the peak working set (`psutil`). Most of it is the PyTorch import baseline (about 545 MB) or the CUDA context (about 1.15 GB). The change from n = 20 to n = 80 is +8 MB.
- The NumPy heap peak comes from `tracemalloc`, which does not trace PyTorch tensors. The CUDA peak is `torch.cuda.max_memory_allocated` and counts every tensor allocation.
- The analytical sizes of the dense buffers are 0.011, 0.067 and 0.171 MB for the edge tensor (n²·7·4 B), and 0.10, 0.61 and 1.56 MB for one GAT pairwise buffer (n²·d·4 B).

Scaling: from n = 20 to n = 80 (4×), CPU latency grows 3.0×. The Ko prior grows 3.8× and the encoder 3.5×. The Ko prior and observation construction take about 55% of the step; the network forward takes about 30%. At these sizes the GPU is no faster: the forward pass is dominated by launch and transfer overhead, and the NumPy preparation stays on the CPU.

Topology and communication (graph statistics: evaluation seeds 0–19; tree and decoding statistics: the decoded parent trees of the 380 timed decisions):

| Quantity | n = 20 | n = 50 | n = 80 |
|---|---|---|---|
| E (attention edges incl. self) / n² | 0.57 | 0.26 | 0.18 |
| Attention neighbours per node, mean (max), excl. self | 10.3 (19) | 12.2 (23) | 13.2 (30) |
| Link neighbours per node, mean | 5.0 | 5.6 | 5.8 |
| Conflict neighbours per node, mean (max) | 10.3 (19) | 12.2 (23) | 13.2 (30) |
| Hop eccentricity to gateway (BFS rounds for φ), median (max) | 3.5 (5) | 7 (11) | 8 (12) |
| Bellman–Ford rounds for the Ko prior, median (max) | 4.5 (6) | 8 (12) | 9 (13) |
| Decoded tree levels (incl. gateway level), observed values | {3, 5} | {7, 11} | {7, 11} |
| Sequential channel-decoding rounds (critical path), median (max) | 13 (15) | 31.5 (32) | 30.5 (37) |
| Fallback rate (fraction of node decisions) | 1.3×10⁻⁴ | 5×10⁻⁵ | 3×10⁻⁵ |

The tree is constant within each evaluated episode, so the tree-level values are per seed (2 seeds). The hop eccentricity over 20 seeds is the broader estimate of tree height.

Messages per node per decision, for a decentralised realisation of the same computation (each round is one local broadcast per node):
- Encoder: L = 2 rounds. In each round a node broadcasts its 64-float embedding (256 B in fp32) and receives one from each attention neighbour: about 21, 24 and 26 received vectors per decision at n = 20, 50 and 80.
- Ko prior: I_BF distance-vector rounds of one scalar from each link neighbour, plus one round of channel and load state from conflict neighbours.
- Channel decoding (channel steps, every 10th step): critical-path rounds, with one channel announcement per node, received by its 10–13 conflict neighbours on average.
- Potential φ: BFS rounds equal to the eccentricity, needed only after a topology change.

Dependent message rounds per decision (median): route-only steps 7.5, 11 and 12; channel steps 20.5, 42.5 and 42.5 at n = 20, 50 and 80.

## 3. Is a real-time claim supported (100 ms decision step)?

- **Computation: yes.** The centralised simulation of all n agents takes 10.6 / 20.3 / 32.1 ms median and 12.7 / 24.9 / 36.9 ms p95 on one CPU thread at n = 20 / 50 / 80, with no step above 66 ms. That is within the 100 ms step, with margin, for this checkpoint and these graph sizes.
- **Deployment: no.** A decentralised channel decision at n = 50–80 needs about 42 dependent message rounds, about 30 of them for the sequential channel decoding. That leaves roughly 2.4 ms per multi-hop wireless round, including channel access. We have not measured round latency over the air, so a deployment-level real-time claim is not supported.
- **Scalability claim:** the code's cost is quadratic in n, through the dense edge tensor, dense attention and dense Bellman–Ford. The decoding critical path grows with conflict density, not only with tree height. The measured growth (3.0× latency for 4× n) is consistent with this quadratic cost at small n. It does not establish near-linear scaling.

## 4. Discrepancies with the v6 text and corrections

1. v6: "Given graph features and precomputed node projections, a sparse GAT layer costs O(nd²+Ed), including edge projection up to the fixed input width."
   Correction: "The implemented GATv2 layer is dense. Edge features are scattered into an n×n×7 tensor, and attention is evaluated over all n² pairs with non-edges masked to −∞. One layer therefore costs O(nd² + n²d), including the 7→d edge projection on all n² entries. A sparse implementation would cost O(nd² + Ed)."

2. v6: "T_enc=O(L(nd²+Ed))." (11)
   Correction: "T_enc = O(L(nd² + n²d)) for the implementation (O(L(nd² + Ed)) only for a sparse one)."

3. v6: "Under a decoder that sorts nodes once and scans edges using channel-used sets, its work is O(n log n+E+n(C+P))."
   Correction: "The implemented decoder computes each node's tree depth by following parent pointers (O(nh)), sorts nodes by (depth, id) once (O(n log n)), and scans a dense conflict row for each node (O(n²)). Its work is O(nh + n log n + n² + n(C+P)). It runs on every step, but its channel output is applied only on channel steps."

4. v6: Eq. (12), "T_exec=O(L(nd²+Ed)+nd²+nd(C+P)+n log n+E+n(C+P))+T_K+T_φ+T_obs."
   Correction: "T_exec = O(L(nd² + n²d) + nd² + nd(C+P) + nh + n log n + n²) + T_K + T_φ + T_obs. Here T_K = O(I_BF·n² + n²C), from Ko's dense Bellman–Ford over I_BF ≤ n rounds plus the channel-cost loop, and is paid on every step. T_φ = O(n²), from BFS hop counts over dense adjacency rows, recomputed every step. T_obs = O(n²). With I_BF ≤ n, the worst case is O(n³); in the evaluated topologies I_BF ≤ 13."

5. v6: "Episode cost depends on when embeddings are recomputed or cached."
   Correction: "Embeddings are not cached. Observation construction, the Ko prior and the full forward pass are recomputed at every 100 ms step, including the nine route-only steps between channel decisions, so episode cost is 200·T_exec."

6. v6: "T_obs can be quadratic if all node pairs are examined, even when the retained graph is sparse."
   Correction: "T_obs is quadratic in the implementation: the edge list and every per-feature view are built from n×n arrays, about ten times per decision."

7. v6: "A BFS potential on an unweighted graph would cost O(n+|E_L|), but that bound cannot be assigned to a different routing routine merely because both produce parent candidates."
   Correction: "The progress potential φ is the BFS hop count to the nearest gateway on the link graph. It costs O(n²) as implemented (dense rows) and O(n + |E_L|) with adjacency lists. It is recomputed every step but changes only on node failure. The Ko prior uses a separate Bellman–Ford routine with load-dependent costs, O(I_BF·n²)."

8. v6: "With fixed layer/head counts, sparse feature and activation storage is O(nd+E(d+7)+n(C+P)), apart from model weights."
   Correction: "Storage is O(n²(d+7) + n(C+P)) at inference, from the dense edge tensor and one layer's pairwise buffer, and O(B·L·n²d) in training, apart from model weights."

9. v6: "The supplied architecture depicts an n×n×7 edge tensor; materializing such a tensor requires quadratic storage regardless of the number of useful edges. Dense attention can similarly require O(Ln²d) activations."
   Correction: "The code does materialise this n×n×7 tensor and evaluates attention densely. At n = 80, E/n² = 0.18, so about 82% of the pairwise work is masked."

10. v6: "Shared projection/head weights contribute O(Ld²+38d+7Ld+d(C+P)), plus any separate prior and training critic."
    Correction: add "29,132 actor and 16,641 critic parameters. The prior has two scalar parameters. The critic is evaluated at inference in the released code but is not used for execution."

11. v6: "where z_K denotes the implemented cost normalization. The initial scale is described as one; its exact sharing across heads or contexts must follow the released implementation."
    Correction: "z_K is a per-node z-score over that node's finite candidate costs, (c − mean)/(std + 10⁻⁶). Infeasible candidates receive no prior and are masked. Constant rows give 0. The channel and parent heads have separate scales, softplus(β_c) and softplus(β_r), each initialised to 1 and shared across nodes."

12. v6: "Two graph layers entail two aggregation dependencies, not necessarily two physical message rounds: neighbor discovery, potential dissemination and sequential decoding can require further communication."
    Correction: append "In the evaluated topologies a decision needs 2 embedding rounds, 1 state round and 4.5–9 Bellman–Ford rounds (median) for the Ko prior. On channel steps it also needs 13–31.5 sequential decoding rounds (median, critical path of the (depth, id) order over conflict edges). That is three to four times the tree height, because conflicting nodes at equal depth are serialised."

13. v6 (8.3): "The study also does not measure radio reconfiguration on hardware, coordination-message cost or wall-clock inference deadlines."
    Correction: "The study measures simulated inference latency: 10.6 / 20.3 / 32.1 ms median and 12.7 / 24.9 / 36.9 ms p95 for all agents at n = 20 / 50 / 80, on one thread of an Intel i5-13450HX. It also counts coordination rounds, about 42 dependent rounds per channel decision at n = 50–80. It does not measure radio reconfiguration on hardware or over-the-air message latency, so no deployment real-time claim is made."

// Authors' status: RESOLVED – the bounds now follow the dense implementation (O(n²) attention and observation, an O(I_BF·n²) Ko prior, no caching, and a decoding critical path longer than the tree height), and the 20/50/80-node latency, memory and communication measurements show that computation fits the 100 ms step while deployment real-time is not claimed.
