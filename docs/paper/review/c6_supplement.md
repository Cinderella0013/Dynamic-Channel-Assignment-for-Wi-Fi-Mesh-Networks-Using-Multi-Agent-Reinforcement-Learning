# Comment 6: versioned supplement

> // Please provide one versioned supplement containing the exact code, configuration, model checkpoints, environment/build records, full-precision outputs and dated protocol history. [...] The private repository listed in the README was not accessible during this review, so access or an archive is still needed.

## Manuscript text (§6.5 / Supplement S1)

**Supplement S1 (study archive).** All results in this paper come from one archive, `NW14_EGAT_supplement_fb58cad.zip` (about 247 MB), which can be read without access to the project repository. No model was retrained to build it. It contains:

- `code/`: a `git archive` of commit `fb58cad`, covering the simulator, training and evaluation code, the ns-3 bridge (`ns3/mesh_bridge.cc`, `ns3/build.sh`, `ns3/calibrate.py`), the protocol documents and the scripts that build the tables and figures.
- `runs/v10/`: the full-precision outputs of the study. These are all model checkpoints (`ckpt/*.pt`, final and every 10k-step validation checkpoint, with training logs), the simulator calibration (`calib.env`: `airtime=domain`, `mac_eff=0.65`, `contention=0`), every evaluation-suite JSON for settings (a) and (b), the composite reports (`final/`), the timestamped event log (`events.log`) and the shell scripts that launched each stage.
- `runs/ns3/v10/` (RTS/CTS) and `runs/ns3/v10base/` (base MAC, no RTS/CTS): the ns-3 suite JSONs and logs, together with the ns-3 run scripts and event log. Each suite JSON records the commit, scenario overrides, agent overrides and evaluation seeds. The ns-3 files also record the bridge binary and MAC options.
- `MANIFEST.md`: the commit, a SHA-256 hash for every checkpoint and suite JSON, the environment (Python 3.12.10, torch 2.11.0+cu128, full `pip freeze`, ns-3.47 built with g++ 13.3.0 and CMake 3.28.3 under WSL Ubuntu, the SHA-256 of the bridge binary, and the build script), the seed map, settings, ablation files, MAC settings, the dated protocol history, and the command that regenerates each table and figure.

**Seed map.** Every learner uses training seeds 0–4. The three-seed ablation uses seeds 0–2. Checkpoint and hysteresis-margin selection use validation topologies 1000–1003 only. Topologies 0–19 are the exploratory test set, 20–39 are fresh topologies, and 40–59 are the pre-registered confirmation set. The simulator airtime model was fitted against ns-3 on S2 and B3, with B1 held out. The ns-3 evaluation covers S2, B1 and B3 on topologies 0–4. The heuristics M0 and Ko are deterministic, so one output file per setting is reused for every training seed.

**Settings and MAC configurations.** Setting (a) drops the queue of a device that leaves a channel. Setting (b) adds `queue_preserve=true`, which keeps queues across a channel switch and re-forwards them on the new route. The ns-3 runs use 802.11ax at 20 MHz with A-MPDU (65,535 B) and A-MSDU off. The RTS/CTS configuration is `{"rts": 1}` and the base configuration is `{}`. The three-seed ablation (margin 2) consists of checkpoints `egat_{noedge,nogru,noko,nobc}_s{0,1,2}.pt` and suites `t{0,1,2}/{a,b}/suite_egat_{noedge,nogru,noko,nobc}.json`, compared with the full-GRU model `suite_egat_h2.json` on the same seeds.

**Regeneration.** After the archived `runs/` is merged into `code/runs/`, `PYTHONPATH=. python docs/paper/review/reconstruct_tables.py` recomputes Tables 5–9 from the suite JSONs. We ran this command on the extracted archive, and its output was byte-identical to the recorded `reconstruct_tables.out.md`. `python docs/paper/figures/make_figures.py` and `make_architecture.py` rebuild the figures. Tables 1–4 and 10 are descriptive.

**Protocol history.** The confirmation of EGAT without GRU followed the protocol `docs/paper/protocol_nogru.md`. That protocol fixes the model, the margin rule, the confirmation topologies 40–59, the criteria and the decision rule. Four records date it:

1. File-system time: the protocol file was created at 2026-11-07 02:31:58 (+07:00) and has not been modified since.
2. Launch: `runs/v10/events.log` records `nogru start` at 02:59:19. The launching script, `runs/v10/nogru.sh`, was created at 02:59:13 and cites the protocol on its second line.
3. Confirmation outputs: all 82 suite files evaluated on topologies 40–59 were written between 04:16 and 05:11 the same morning. No earlier file uses those seeds; topologies 20–39 were the latest previously used.
4. Commit fields: those 82 files record `8aaaf1c-dirty`. Commit `8aaaf1c` (2026-11-06 23:17) does not contain the protocol, so the protocol then existed only as an uncommitted working-tree file.

The protocol was first committed in `be22ad0` at 09:42:30, after the runs. Its committed blob (`de8b3b5`) is identical to the file created at 02:31, so the text was not edited after the results were seen. The pre-registration date therefore rests on file-system timestamps and the event log, which are local records. Git does not independently timestamp it. We report this limitation as it stands.

// Authors' status: RESOLVED – A self-contained, commit-stamped archive (`NW14_EGAT_supplement_fb58cad.zip`) now provides the code, configuration, all checkpoints, environment and ns-3 build records, full-precision outputs, seed map, settings (a)/(b), the three-seed ablation, both ns-3 MAC settings, table-regeneration commands verified to reproduce Tables 5–9 exactly, and the dated protocol history, which shows the protocol file was written before the 40–59 evaluation but committed to git only afterwards.
