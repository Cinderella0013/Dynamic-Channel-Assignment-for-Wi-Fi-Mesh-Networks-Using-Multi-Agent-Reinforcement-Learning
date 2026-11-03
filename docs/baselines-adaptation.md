# Baseline Adaptation Notes

Appendix material for spec §5.5: every way the RL baselines differ from their source papers when run on the EGAT-MAPPO Wi-Fi mesh environment, and why.

## Common to all learned baselines

- **Agent:** each mesh node is an agent, gateways included. Parameters are shared across agents.
- **Action:** a downlink channel index (6 choices, channels 1–6 on radio 1) combined with a parent slot (4 choices: link neighbours ranked by hop count to a gateway, then by SNR), giving 24 discrete actions. Ranking by SNR alone left 22% of nodes with no slot closer to a gateway (Ko expressed through the interface fell from +2714 to −1592 on S2); hop-first ranking restores 2682. Radio 0 follows the parent's downlink channel. Ko uses the same tree convention, so every parent link has a channel both ends share. The papers act on channel × transmit power and have no routing. Power control is outside this project's scope (spec §1.4).
- **Observation:**
  - `o_i` is shared with neighbours: offered load, unmet demand, is_gateway, hop/n, alive, radio-1 channel, current parent slot.
  - `m_i` stays local: per slot, whether it is valid, its SNR, rate and load fraction.
  - **No edge features enter attention.** Edge-aware attention is EGAT-MAPPO's contribution (spec §1.3).
- **Reward:** `r_i = 0.5·local_utility_i + 0.5·reward_global/n_alive − 0.1·switches_i − 0.05·route_change_i`. `switches_i` counts the node's own radio-1 change; a child's radio-0 change caused by its parent retuning is charged to the parent. Both terms use the env's proportional-fair log utility. The papers use different rewards: Yuan uses interference-temperature × efficiency, and Li uses a V2N/V2V rate mix. One shared reward keeps the comparison fair.
- **Budget:** each learner gets the same number of env steps. Every episode draws a scenario uniformly from S1–S6 with a fresh topology seed, n = 20.
- **Discount:** γ = 0.98 for all learners.
- **Gradient clipping:** gradient norm is clipped at 10 for both neural baselines. Neither paper mentions it. It keeps early Q-values and advantages from blowing up under log-utility rewards.
- **Tuning:** hyperparameters are paper defaults where given. The Optuna search (20 trials per method) belongs to Plan 5.

## Q-neutral

- **Definition:** tabular Q-learning with the shared reward above, with no objective weighted specially. This is an assumption, because the project brief names "Q-neutral" without defining it. Yuan et al. also compare against tabular Q-learning.
- **Setup:** one Q-table shared by all agents. Settings: α = 0.1, γ = 0.98, ε from 0.6 to 0.05 over 50 000 steps.
- **State discretisation:** `(load bin {<1, <5, ≥5 Mbps}, unmet > 0.1, is_gateway, hop/n bin {0, ≤0.1, ≤0.2, >0.2}, radio-1 channel, best valid slot, best slot rate bin {<30, <80, ≥80 Mbps})`.

## Yuan et al. 2022: GQN (doi:10.1049/cmu2.12527)

**Kept from the paper:**
- An MLP encoder on the communicated state `o_i` only (§4.2).
- Two multi-head dot-product attention layers over the agent itself and 3 neighbours: `α = softmax(τ·W h_i·(W h_j)ᵀ)`, then concatenate the heads and apply ReLU (Eq. 11–12).
- A Q head that receives `[o, m, h, h'_1, h'_2]`.
- DQN with replay and a soft target update.
- The temporal-relation KL regulariser between the current and next attention weights. The next-state attention comes from the online network, as in §4.2 (Eq. 13).
- Table 2 values:

  | Hyperparameter | Value |
  |---|---|
  | Neighbours | 3 |
  | Attention layers | 2 |
  | Mini-batch size | 10 |
  | KL weight λ | 0.03 |
  | Attention scale τ | 0.25 |
  | Learning rate | 0.001 |
  | Soft-update coefficient | 0.01 |
  | Initial ε | 0.6 |
  | γ | 0.98 |
  | Activation | ReLU |
  | Optimizer | Adam |

**Changed:**
- **Q head:** an MLP (128-128) instead of the paper's CNN. The CNN's shape is not given, and the input is a flat vector.
- **Not specified in the paper:** 4 attention heads, hidden size 64, replay buffer of 10 000 graph transitions, ε floor 0.05 over 50 000 steps.
- **Neighbours:** the first 3 parent slots (link neighbours ranked hop-first, then SNR), i.e. the most gateway-ward neighbours, not the nearest ones.
- **Actions:** no power action.

## Li et al. 2025: GAT-A2C (Sensors 25(16):5209)

This is the assumed source for "Li". The user has not yet confirmed it.

**Kept from the paper:**
- The interference (conflict) graph with self-loops as the attention graph.
- z-scored node inputs (Eq. 10).
- A 2-layer GAT with additive LeakyReLU attention: 8 heads in the first layer, 16-dimensional output, dropout 0.6.
- Actor and critic on `[x ‖ z]` (Eq. 19), both with 500-250 hidden layers.
- Learning rate from 0.005 decaying to 1e-4 (×0.96 every 1000 updates).

**Changed:**
- **Critic outputs V(s).** The paper uses Q in Eq. 24 and V in the text.
- **Standard 1-step on-policy A2C** (`A = r + γV(s') − V(s)`) instead of the paper's replay-based variant, whose target mixes target networks and argmax-π and is not fully specified.
- **No auxiliary GAT MSE loss.** Eq. 16–17 compare a scalar label with a vector embedding, so the loss is ill-defined.
- **z-score statistics** use a running EMA with momentum 0.01, updated only during learning (once per update; the bootstrap pass runs in eval mode). The standard deviation is floored at 0.1 and z is clipped to ±5, so features that stay constant for a long time do not explode on rare events (S4/S6). The paper's fixed training-set statistics do not exist for an online learner.
- **γ = 0.98.** The paper's value is ambiguous: it lists λ = 0.5 under "discount factor".
- **Not specified in the paper:** entropy 0.01, value coefficient 0.5, Adam optimizer, GAT hidden size 8 per head.
- **Actions:** no power action.
