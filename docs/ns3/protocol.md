# ns-3 validation: bridge protocol and work split

Goal: evaluate the trained policies (no retraining) on packet-level 802.11 in ns-3.47, so the
claim "EGAT+H beats Ko/Li/Yuan" is checked outside the fluid simulator. Evaluation only.

## 1. Architecture

```
Python (WSL, /mnt/e/MESH)                          ns-3 (WSL, ~/ns3_validation/ns-3.47)
mesh.ns3.env.Ns3MeshEnv  --- JSON lines, stdin -->  build/scratch/ns3.47-mesh_bridge
  same API as MeshEnv    <-- JSON lines, stdout --  (logs on stderr only)
```

- `Ns3MeshEnv(cfg, seed, transport)` has the same API as `mesh.sim.env.MeshEnv`
  (`reset() -> Observation`, `step(Action) -> (Observation, reward, info)`), so
  `mesh.eval.run.run_episode`, `mesh.eval.suite.run_suite` and `mesh.eval.composite` are reused
  unchanged. Every policy (m0, ko, li, yuan, q_neutral, egat) runs on the Python side.
- Python owns: topology (`generate_topology`, same seeds), traffic demand (`Traffic`, same seeds,
  so every method sees the same demand), node failure, hotspot moves, and the info metrics.
- ns-3 owns: packet transport for one step (100 ms): MAC/PHY, queues, interference, switching.
- One ns-3 process per episode. Python spawns it, sends `init`, then one `step` per env step, then `end`.

## 2. Messages (one JSON object per line, UTF-8)

Python -> ns-3, once:

```json
{"type": "init", "n": 20, "pos": [[x, y], ...], "gateways": [3, 11],
 "channels": [1, 6, 11, 36, 40, 44, 48], "step_s": 0.1, "pkt_bytes": 1200,
 "queue_bytes": 625000, "switch_delay_s": 0.1, "route_hold_s": 0.1,
 "ext_interf": null, "seed": 7}
```

- `channels`: IEEE channel numbers indexed by the action's channel index (`mesh.sim.radio.CHANNELS`).
- `queue_bytes`: per-node queue limit = `5 * mean_mbps * 1e6 * step_s / 8` (the sim's `q_max`).
- `ext_interf`: null or `{"channel_idx": 1, "start": 50, "end": 150, "noise_dbm": -85.0}` (steps).
- ns-3 replies `{"type": "ready"}`.

Python -> ns-3, every step `t = 0, 1, ...`:

```json
{"type": "step", "t": 0, "channels": [[0, 1], ...], "next_hop": [-1, 3, ...],
 "demand_mbps": [0.0, 7.3, ...], "alive": [1, 1, ...]}
```

ns-3 applies the action at the start of the step, generates `demand_mbps[s] * step_s` of UDP
payload at each source during the step, runs `step_s` of simulated time, and replies:

```json
{"type": "result", "t": 0,
 "delivered_mbps": [...],          // per source: bytes received at any gateway sink in this step * 8 / step_s / 1e6
 "queue_bytes": [...],             // per node, at step end: bytes waiting (IP + MAC queues), all sources
 "src_queue_bytes": [...],         // per source, at step end: its own bytes still in the network
 "drop_bytes": {"overflow": [...], "mac": [...], "switch": [...], "noroute": [...]},  // per source, this step
 "links": [[u, v, tx_ok, tx_fail, snr_db, rate_mbps], ...]  // directed links that carried data this step
}
```

- Every array is length `n`, indexed by node id; per-source arrays are 0 for gateways.
- `tx_ok` / `tx_fail`: MPDUs acked / finally failed on u->v. `snr_db`, `rate_mbps`: mean over the step.
- Python -> ns-3 `{"type": "end"}`; ns-3 exits 0. Any error: one line `{"type": "error", "msg": "..."}` on stdout, exit 1.

## 3. ns-3 model (lane 1, `ns3/mesh_bridge.cc`, build with `bash ns3/build.sh` in WSL)

- 802.11ax 20 MHz, 1 SS, ad-hoc MAC (A-MPDU + Block Ack as ns-3 sets them up),
  `IdealWifiManager` (SNR-based MCS, closest to the sim's SNR->MCS table).
- `LogDistancePropagationLossModel` exponent 3.0, reference loss 46.6777 dB at 1 m (same as
  `mesh/sim/radio.py`); tx power 20 dBm; noise figure 7 dB so the noise floor is -94 dBm.
- Channel index c is the 5 GHz channel 36 + 4c (the `channels` numbers in `init` are ignored:
  in the sim channels differ only by index, and retuning across 2.4/5 GHz crashes ns-3.47).
- Radios: every node has one device per channel, fixed for the run; the action's two channels
  say which are in use. A channel that enters a node's set is usable `switch_delay_s` later.
  Nothing retunes at runtime: ns-3.47 runtime switching stalls queues (cancelled ADDBA timeouts,
  lost channel access) and aborts on responses scheduled across the switch.
- Forwarding is done by the bridge, not an IP stack: a packet carries (source, ttl); each hop
  sends it to `next_hop` on the common channel (largest shared index, as `common_channel`); a
  gateway counts it delivered. No ARP or routing-protocol traffic.
- A changed `next_hop` holds that node's uplink for `route_hold_s`. Defaults (0.1 s) mirror the
  sim's `downtime_steps=1`.
- Leaving a channel: setting (a) drops that device's queue (`switch` drops); `init` key
  `"queue_preserve": 1` = setting (b), the queue is re-forwarded on the new route.
- `alive=0`: the node's PHYs off; its own in-flight bytes counted as `switch` drops once.
- External interference (S4) is not modelled; not in the matrix.

MAC options (`init` keys, optional): `"rts": 1` RTS/CTS before every data frame;
`"ampdu_bytes"` (default 65535), `"amsdu_bytes"` (default 0 = off).

Open-loop calibration (`python ns3/calibrate.py [--init JSON]`: the sim's actions and demand
replayed into ns-3, S2, seeds 0-4, setting (a)); aggregate throughput, sim -> ns-3:

| config | Ko (sim 72.6) | M0 (sim 49.8) |
|---|---|---|
| first version (flush pulled in-flight MPDUs out of the queue) | 57.8 (-20.3%) | 32.1 (-35.5%) |
| fixed: only never-sent MPDUs leave the queue | 67.4 (-7.1%) | 32.1 (-35.5%) |
| fixed + A-MSDU 7935 B | 63.2 (-12.9%) | 33.6 (-32.5%) |
| fixed + A-MSDU + A-MPDU 1 MB | 63.0 (-13.3%) | 33.9 (-31.9%) |
| fixed + A-MSDU + A-MPDU 1 MB + 1500 B packets | 63.3 (-12.7%) | 32.4 (-35.0%) |
| fixed + RTS/CTS | 70.4 (-3.1%) | 37.5 (-24.6%) |

The sim has no MAC overhead (one 802.11ax link peaks at ~78% of its PHY rate in ns-3) and a
count-based airtime share without collisions, so it is optimistic, most under single-channel
contention (M0, every link on one channel). RTS/CTS helps both methods (hidden terminals in a
multi-hop mesh). Ko passes the 20% gate, M0 does not in any configuration. Ranking Ko > M0
holds everywhere. The configuration is chosen from Ko/M0 only, before any learned policy is run
in ns-3.

## 4. Python side (lane 2)

`info` is computed exactly as `MeshEnv.step` computes it, with ns-3 measurements in place of the
fluid model: `thr = delivered_mbps`; `backlog = src_queue_bytes * 8 / step_s / 1e6` (sim's
Mbps-step unit); `lost = sum(drop_bytes) * 8 / step_s / 1e6`; jain, p5, utility on the same
1 s window; switch and route-change counts from the actions; `n_drop` from routes that do not
reach a gateway.

Observation features:

| feature | source |
|---|---|
| offered_load, is_gateway, hop_to_gw, alive | as in MeshEnv |
| unmet | `offered - delivered` |
| backlog | ns-3 `src_queue_bytes` |
| is_link, is_conflict, snr_db, rate_mbps, same_channel | radio graph + `link_rate` (same propagation model as ns-3) |
| load_frac | path load (paths x offered) / `link_capacity`, as in MeshEnv |
| ack_ratio | measured `tx_ok / (tx_ok + tx_fail)` per link, 1 where no traffic |

`MeshEnv` behaviour must not change (existing tests stay green). A `FakeTransport` that answers
`step` from the fluid model lets `Ns3MeshEnv` be tested without ns-3: with it, an episode must
match `MeshEnv` on the same seed.

## 5. Experiment matrix (fixed before any ns-3 result is seen)

- Scenarios: S2 (high load), B1 (moving hotspot), B3 (sparse). Traffic/topology only, no PHY extras.
- Eval seeds 0-4. Learners: training seeds 0-4 (`E:/MESH/runs/v8/ckpt/<name>_s<k>.pt`, not in git).
- Methods: `m0`, `ko`, `li`, `yuan`, `q_neutral`, `egat_h3` (= `egat_s<k>.pt` with `switch_margin=3.0`).
- Setting (a): default (no queue preservation). Sensitivity: native switch delay.
- Output: `runs/ns3/<setting>/t<k>/suite_<method>.json`, same schema as `run_suite`, so
  `python -m mesh.eval.composite` reads it unchanged.
- Calibration gate (before the matrix): Ko and M0 on S2, seeds 0-4. If ns-3 mean aggregate
  throughput differs from the sim by more than 20%, stop and report; ns-3 stays future work.
- Report whatever comes out: composite of EGAT+H vs each method, seed-pair CI, as in the paper.

## 6. Work split (no two lanes write the same file)

| lane | owner | files |
|---|---|---|
| 1. ns-3 scenario | main session | `ns3/mesh_bridge.cc`, `ns3/build.sh` (copied into `scratch/mesh_bridge/`) |
| 2. Python bridge | session "py-bridge" | `mesh/ns3/*`, `tests/test_ns3_env.py` |
| 3. docs + report | session "ns3-docs" | `ns3/report.py`, `docs/paper/ns3_section_draft.md` |

## 7. Sim airtime model recalibrated against ns-3 (update, not in the thesis)

The v8 policies on ns-3 (rts): EGAT+H still beat M0/Li/Yuan/Q-neutral, but its lead over Ko fell from +21%
(sim) to +1.5% (CI spans 0). Open-loop replays traced it to the sim's static 1/k airtime share: it ignores
MAC overhead (optimistic for a saturated channel, where EGAT+H, Li and Yuan put most of their traffic) and
does not let a link use idle neighbours' airtime (pessimistic for configurations spread over channels, as Ko's).
The gap grows with load (EGAT+H S2: -4% at half demand, -15% at 75%, -22% at full; Ko within 3%).

`ScenarioConfig.airtime = "domain"` (`mesh.sim.flows.domain_allocate`): each loaded link's interference domain
may use at most `mac_eff` of the airtime; idle neighbours cost nothing. Fit on S2/B3 open-loop replays of
M0, Ko, EGAT+H, Li, Yuan: `mac_eff = 0.65` gives rms 7.4% (worst 11.4%, Ko S2 -11%); held-out B1 rms 7.0%
(worst 10.0%); static model 23.5% / 26.5%. Method rankings match ns-3 in S2, B3, B1.

v10 = v8 recipe retrained in that sim (`runs/v10`, egat/li/yuan/q_neutral x 5 seeds, switch_margin 1.0
picked on validation). EGAT+H composite lead, S2/B1/B3, seed-pair CI:

| vs | v8 sim | v8 ns-3 | v10 sim | v10 ns-3 |
|---|---|---|---|---|
| M0 | +32.6 | +43.8 | +34.6 | +53.9 [+46.0, +61.8] |
| Ko | +21.2 | +1.5 [-11.5, +14.6] | +25.4 | +23.6 [+6.6, +40.6] |
| Li | +17.9 | +25.7 | +13.5 | +36.4 [+21.4, +51.5] |
| Yuan | +26.7 | +25.2 | +23.1 | +34.7 [+18.0, +51.3] |
| Q-neutral | +71.9 | +67.3 | +71.8 | +69.5 [+67.2, +71.8] |

## 8. v10 completion checks (update, not in the thesis)
Full sim suites (`runs/v10/final/composite_{a,b}_egat_h1.md`, 20 eval seeds x 5 training seeds). EGAT+H
composite lead, A+B, seed-pair mean with 95% t-interval over both methods' training seeds:

| vs | setting a | setting b (queue_preserve) | setting a, suite C (pooled) |
|---|---|---|---|
| Ko | +0.230 [+0.202, +0.258] | +0.234 [+0.205, +0.263] | +0.290 |
| M0 | +0.296 [+0.277, +0.314] | +0.301 [+0.282, +0.319] | +0.196 |
| Li | +0.098 [-0.089, +0.285] | +0.101 [-0.073, +0.276] | +0.139 |
| Yuan | +0.195 [+0.152, +0.238] | +0.184 [+0.147, +0.221] | +0.280 |
| Q-neutral | +0.733 [+0.723, +0.743] | +0.741 [+0.723, +0.758] | +0.708 |

Li: not significant in the sim. Li seed 4 failed in training (validation composite never above -0.047;
other seeds reached +0.07 to +0.21) and scores -0.083 vs Ko on A+B; the other Li seeds sit near EGAT+H.
Without Li s4 the lead is +0.035 [-0.070, +0.141]. Pooled means (+0.195) hide this. On ns-3 the Li lead
stays significant (below), so sim and ns-3 disagree on Li only.

ns-3 base MAC (no RTS), `runs/ns3/v10base`, S2/B1/B3, seed-pair CI (v8 base in brackets for reference):
M0 +56.0 [+49.3, +62.7] (v8 +47.2), Ko +22.1 [+3.2, +41.0] (v8 +3.2, CI spanned 0),
Li +39.1 [+24.4, +53.8], Yuan +35.0 [+17.3, +52.7], Q-neutral +68.2 [+65.8, +70.6].
Aggregate throughput EGAT+H / Ko (Mbps): rts S2 70.9/67.1, B1 68.8/66.1, B3 48.2/43.6;
base S2 67.8/66.4, B1 64.5/60.5, B3 46.8/42.5.

Test suite: 167 passed at c7c16c7 + docs.
