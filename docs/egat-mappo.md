# EGAT-MAPPO Model Card

EGAT-MAPPO is the proposed model for joint channel assignment and routing in Wi-Fi mesh networks. Its code is in `mesh/model/`. It trains through `python -m mesh.rl.train --method egat` and is evaluated with `python -m mesh.eval.run --policy egat --checkpoint ...`.

## Components and ablation flags

Each component can be switched off with a field of `EGATConfig`, which is how the spec §5.6 ablations are run.

| Component | What it does | Flag (default) |
|---|---|---|
| Edge-aware GATv2 | Attention score `aᵀ LeakyReLU(W_s h_i + W_t h_j + W_e e_ij)` over link edges, conflict edges and self-loops. Edge features are `is_link, is_conflict, snr_db, rate_mbps, load_frac, same_channel`. Two layers, each with a residual connection and LayerNorm. | `use_edge_feat=True` |
| GRU memory | A per-node `GRUCell` on the encoder output. Its state resets at `t = 0`. | `use_gru=True` |
| Two timescales | The channel head decides every `t_c` steps; between decisions, nodes keep their downlink channel. The routing head decides every step. | `two_timescale=True`, `t_c=10` |
| Progress mask | A parent must have a strictly smaller hop count to a gateway, which makes every route loop-free. | `progress_mask=True` |
| BC warm start | For the first `bc_steps` env steps the agent acts as Ko and records the trajectories. It then fits the actor by cross-entropy, and fits the critic and `V_Ko` to Ko's Monte-Carlo returns. Those steps count toward the training budget. | `use_bc=True`, `bc_steps=10000` |
| Ko residual prior | `logits −= softplus(β)·z(cost_Ko)` for both heads. β is learnable and starts at 1. | `use_ko_prior=True` |
| Safe fallback | At evaluation only, a node uses Ko's action when the joint entropy of its two heads exceeds `fallback_entropy`. It also falls back when `V_π < V_Ko − fallback_margin`, but this check only applies after `V_Ko` has been fitted. `fallback_rate` is the fraction of alive nodes that fell back. | `use_fallback=True`, `fallback_entropy=2.0`, `fallback_margin=0.0` |
| MAPPO | All nodes share one actor. A centralized critic sees each node's embedding plus the mean over all nodes. Training uses PPO-clip and GAE. | — |

## Hyperparameter defaults

| Group | Setting | Value |
|---|---|---|
| Network | hidden size | 64 |
| | attention heads | 4 |
| | layers | 2 |
| PPO | learning rate | 3e-4 |
| | γ | 0.98 |
| | GAE λ | 0.95 |
| | clip | 0.2 |
| | epochs per rollout | 4 |
| | rollout | 1000 env steps |
| | minibatch | 50 env steps |
| | entropy coefficient | 0.01 |
| | value coefficient | 0.5 |
| | max gradient norm | 0.5 |
| BC | epochs | at most 20; stops early when agreement on the training data reaches 0.9 |
| | learning rate | 1e-3 |

## Reward

EGAT-MAPPO uses the same per-agent reward as the baselines (`mesh/rl/agents_io.per_agent_reward`), with one difference. The weight λ between local and global utility is annealed linearly from 0.5 to 0.1 over the run, as spec §3 asks. The baselines keep λ fixed at 0.5. Disclose this in the paper.

## Departures from the spec

- **One critic, not one per head.** The channel head uses the GAE advantage at its decision step, and its log-prob and entropy count only on decision steps.
- **No backpropagation through time.** The GRU state is stored with each transition, so the truncation length is 1.
- **Edge features.** The spec also lists ETX and interference power. The env does not expose these, so they are not used.
- **Greedy (evaluation) channel decoding is sequential and conflict-aware.** Parents decide first. Each node takes its highest-scoring channel that no already-decided conflict neighbour is using. Channel labels are interchangeable, so a per-node argmax would give every node the same channel whenever they see the same state. Training samples from the policy and is unchanged.
- **BC agreement counts the route slot only.** Matching Ko's exact channel label is not a meaningful imitation target, for the same reason. BC skips the route target wherever Ko picks a parent the progress mask forbids (one with the same hop count).
