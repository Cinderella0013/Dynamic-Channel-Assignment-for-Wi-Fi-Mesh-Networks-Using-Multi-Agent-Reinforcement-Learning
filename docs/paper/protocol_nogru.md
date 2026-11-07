# Pre-registered protocol: EGAT without GRU as the main model (written 2026-11-07, before any run below)

## Why
In the margin-2 ablation (3 seeds, test topologies 0-19, seen after the main results), EGAT without the GRU
beat Yuan+H by +0.087 [+0.048, +0.125], while full EGAT+H ties Yuan+H (+0.007 [-0.041, +0.055]).
That result is post hoc. This protocol checks it on topologies no decision has used.

## Fixed before running
- **Model:** `egat` with `--agent-set use_gru=false`, v10 sim (airtime=domain, mac_eff 0.65, contention 0),
  100k steps, same recipe as the ablation. Training seeds 0-4. Seeds 0-2 reuse the existing ablation checkpoints
  (`runs/v10/ckpt/egat_nogru_s{0,1,2}.pt`); seeds 3 and 4 are trained with the same command.
- **Margin:** grid {0, 0.5, 1, 2, 3, 4}, worst-case rule of the main model (`fair_pick.py`): validation seeds
  1000-1003, A+B, worst composite over Ko, M0, Li, Yuan, Q-neutral. Li+H and Yuan+H are not used for selection.
- **Confirmation topologies:** seeds 40-59, suites A+B. Never used by any earlier run.
- **Methods evaluated on 40-59:** no-GRU+H, EGAT+H (m 2), Ko, M0, Li, Yuan, Q-neutral, Li+H (m 1), Yuan+H (m 1);
  5 training seeds each for learners; settings (a) and (b).
- **Statistic:** composite lead, mean over training-seed pairs, 95% t-interval (as in the thesis).

## Criteria (setting a, seeds 40-59, A+B)
- **P1 (primary):** no-GRU+H vs Yuan+H: mean >= +0.05 and CI lower bound > 0.
- **P2:** no-GRU+H vs Ko, Li, Yuan, M0, Q-neutral and Li+H: CI lower bound > 0 for each.
- **P3:** the thesis's four criteria pass vs the published-method baselines.

## Decision (fixed now)
- P1 and P2 pass: no-GRU+H becomes the main model in the thesis, paper and slides. The change is disclosed
  (chosen after an ablation seen on test topologies; confirmed on 40-59 under this protocol).
- Otherwise: EGAT+H stays the main model; this run is reported as a negative follow-up in 5.4.4.
- No model, margin or scenario changes after the 40-59 results are seen. Whatever comes out is reported.

## Reported, no criterion
Setting (b); vs full EGAT+H; ns-3 (S2/B1/B3, topologies 0-4, RTS/CTS and base MAC, same as the existing ns-3 table);
test topologies 0-19 for all 5 seeds.
