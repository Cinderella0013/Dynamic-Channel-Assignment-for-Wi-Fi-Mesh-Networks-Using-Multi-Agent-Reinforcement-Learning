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
| BC warm start | For the first `bc_steps` env steps the agent acts as Ko and records the trajectories. It then fits the actor by cross-entropy and warm-starts the critic on Ko's Monte-Carlo returns. Those steps count toward the training budget. | `use_bc=True`, `bc_steps=10000` |
| Ko residual prior | `logits −= softplus(β)·z(cost_Ko)` for both heads. β is learnable and starts at 1. | `use_ko_prior=True` |
| Safe fallback | At evaluation only, a node uses Ko's action when its policy entropy exceeds `fallback_entropy`. On channel-decision steps this is route plus channel entropy; on other steps it is route entropy alone. It uses only the node's own heads, so execution stays decentralised. `fallback_rate` is the fraction of alive nodes that fell back on the last step, and is 0 when the flag is off. | `use_fallback=True`, `fallback_entropy=2.0` |
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
- **Greedy (evaluation) channel decoding is sequential and conflict-aware, for every learned method.** Parents decide first. Each node takes its highest-scoring channel that no already-decided conflict neighbour is using (`agents_io.sequential_channels`). Channel labels are interchangeable, so a per-node argmax would give every node the same channel whenever they see the same state. The same decoder is applied to Q-neutral, Yuan and Li (flag `seq_decode`, default on), so it gives EGAT no advantage. Like Ko, it needs one sequential coordination round at execution. Training samples from the policy and is unchanged. Report the plain-argmax variant (`seq_decode=False`) as an ablation.
- **No `V_Ko` value check in the fallback.** The spec's rule `V_π < V_Ko − m` was removed after review. `V_Ko` was fitted with λ≈0.5 while `V_π` ends training at λ≈0.1, and the shared encoder kept training after `V_Ko` was frozen. The check therefore fired on about 7.6% of nodes and lowered S2 reward by about 2.7%. It also needs the centralized critic at execution.
- **No semi-MDP credit for the channel head.** The spec asks for reward accumulated over `t_c`; the implementation uses the per-step GAE advantage at the decision step.
- **BC agreement counts the route slot only.** Matching Ko's exact channel label is not a meaningful imitation target, for the same reason. BC skips the route target wherever Ko picks a parent the progress mask forbids (one with the same hop count).

## EGAT v1 (`--method egat_v1`)

Same network and components as above. Changes, from spec addendum §3:

- **KL anchor.** After BC the policy is frozen as `π_BC`. PPO adds `κ·KL(π_θ ‖ π_BC)`, with κ annealed from 0.1 to 0.01 over the training budget. The channel-head term counts on decision steps only.
- **Fixed λ = 0.1.** This keeps the training reward close to the global env reward used for evaluation.
- **Semi-MDP channel credit.** The channel head uses an n-step advantage with n = `t_c`, cut at episode end. The route head keeps GAE, and the two heads have separately clipped PPO surrogates.
- **Learning rate 1e-4 after BC.**
- **Checkpoint selection on validation.** This applies to every learner (§ below).

Reason: with Plan 4 code, PPO lowered the BC policy's reward. S2: BC 2769, PPO 100k 2708, PPO 300k 2623.

## EGAT v2 (`--method egat_v2`): the user's design

Each step runs this pipeline:

```
input (obs + ACK + backlog)
  → GAT scores Q_ch(u,c), Q_rt(u,j)
  → RAES ordered channel reservation
  → migration refinement
  → execute with queue preservation
  → ACK feeds the next step
```

- **Training.** Double DQN on the executed action, with additive joint Q = Q_ch + Q_rt. Replay buffer 50k, batch 32, τ = 0.01, ε from 0.3 to 0.02 over 50k steps. The warm start is DQfD on Ko demonstrations: the first 10k steps act as Ko and count toward the budget.
- **RAES.** Nodes take turns in decreasing-backlog order. Each one reserves the channel with the highest `z(Q) − a·z(interference from reserved neighbours) − b·z(co-channel load) − g·z(move cost)`. The parent is chosen by `z(Q_rt) − g·z(move cost)` under the progress mask.
- **Migration refinement.** A proposed move is kept only if all of these hold:
  - its gain beats `margin` = 0.1;
  - at most `ceil(0.1·n)` nodes move per step;
  - no two conflicting nodes move in the same step;
  - the node is not in its 3-step cooldown.
- **Queue preservation.** Moves are executed with `queue_preserve=True`, i.e. CSA-style buffering.
- **Scaling of the terms (changed after review).**
  - Q, interference and co-channel load are z-scored per node, over the 6 channels or over the allowed parent slots.
  - The move cost M is z-scored **globally**, over all nodes. A node with a large backlog therefore pays more to move than one with a tiny backlog. A per-node z-score would charge every node with any backlog the same fixed amount (spec addendum §4.3 departure).
  - Refinement scores parent moves on the same scale as RAES, including M, so channel and parent moves follow one rule.
- **Cooldown.** A node that moved cannot move again for the next 3 rounds.
- **Exploration.** During training, nodes chosen for ε-exploration execute their RAES proposal without refinement. Otherwise refinement would filter out most exploratory moves (k_max ≈ 2 of 20).
- **k_max.** Uses the number of alive nodes.
- **Flags:** `raes`, `migration_control`, `queue_preserve`, `raes_order`, plus all encoder flags.
- **Fairness setting (b).** Every method runs with `queue_preserve`. This separates the gain from planning and migration control from the gain from queue preservation alone.

## Checkpoint selection (all learners)

`--val-every 10000` evaluates the current policy greedily on S1–S6 with validation seeds 1000–1002, which are never used for testing. The checkpoint kept is the one with the best mean validation reward. Candidates are every validation point, the end of BC and the final policy. Validation runs only between training episodes, and its env steps do not count toward the training budget.
