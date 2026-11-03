# EGAT-MAPPO: Edge-aware GAT + Multi-agent PPO สำหรับ Joint Channel Assignment และ Routing ใน Wi-Fi Mesh

- วันที่: 2026-11-03
- สถานะ: Design (อนุมัติรายส่วนแล้ว — รอรีวิว spec)
- ประเภทงาน: งานวิจัย (โมเดล + การทดลองเปรียบเทียบ)

---

## 1. เป้าหมายและขอบเขต

### 1.1 เป้าหมาย
พัฒนาโมเดลที่มี GAT เป็นแกนหลักแบบ hybrid เพื่อตัดสินใจ **channel assignment และ routing ร่วมกัน** ใน Wi-Fi Mesh ให้ได้ **throughput รวมสูงขึ้นและ fairness ดีขึ้น** เทียบกับ baseline 5 ตัว โดยแต่ละองค์ประกอบของโมเดลต้องอ้างอิงจุดเด่นหรือแก้จุดอ่อนของ baseline ที่ระบุได้ชัด

### 1.2 Baselines
| Baseline | สมมติฐานความหมาย (ต้องยืนยัน) | จุดเด่นที่ยืมมา | จุดอ่อนที่งานนี้แก้ |
|---|---|---|---|
| M0 | ไม่ optimize (ค่าเริ่มต้น/static) | ใช้เป็น floor อ้างอิง | — |
| Ko local-cost adaptation | heuristic แบบกระจายที่ใช้ local cost | กระจาย เร็ว อธิบายได้ ไม่ต้องเทรน — ใช้เป็น teacher, prior, fallback | myopic ไม่เห็นภาพรวม |
| Q-neutral | tabular Q-learning, reward neutral | ความเรียบง่าย | generalize ข้าม topology ไม่ได้ state ระเบิด |
| Yuan GAT+DQN adaptation | GAT encoder + DQN | GAT encoder, off-policy | action space ระเบิด, Q overestimation, ไม่ใช้ edge feature (ต้องตรวจ) |
| Li GAT-A2C adaptation | GAT encoder + A2C | โครง actor-critic | variance สูง, on-policy ไม่เสถียร, ไม่ใช้ edge feature (ต้องตรวจ) |

### 1.3 Contributions ที่คาดหวัง
1. GATv2 ที่ใช้ **edge feature** (RSSI/SNR, airtime, interference, ETX) และ **conflict edge** เพื่อให้ attention เห็น interference จริง
2. **Two-timescale policy**: หัว channel (ช้า) + หัว routing (เร็ว) บน encoder เดียว
3. **Ko-hybrid**: BC warm-start + residual prior + safe fallback → ไม่มี cold start และมี worst-case guarantee เทียบ Ko
4. **MAPPO (CTDE)** แทน DQN/A2C → เสถียรกว่าและ execute แบบกระจาย
5. Protocol ประเมินผลแบบ Python-train / ns-3-validate พร้อมรายงาน sim-to-sim gap

### 1.4 นอกขอบเขต
- Packet-level simulation ใน Python (ns-3 ทำหน้าที่นี้)
- Node mobility
- TX power / rate control, client steering
- 802.11s/HWMP (ใช้ ad-hoc + static routing ใน ns-3 เพื่อไม่ให้ protocol routing เดิมแย่ง decision — ระบุเป็น limitation)

### 1.5 สิ่งที่ต้องตรวจก่อนเริ่ม implementation
- [ ] อ่าน paper Yuan และ Li: ทำ joint channel+routing หรือใช้ edge feature อยู่แล้วหรือไม่ ถ้าใช่ ปรับน้ำหนัก contribution ไปที่ข้อ 2, 3, 5
- [ ] ยืนยันนิยาม M0 และ Q-neutral
- [ ] บันทึกการ adapt baseline แต่ละตัว (อะไรต่างจาก paper ต้นฉบับ)

---

## 2. สถาปัตยกรรมโมเดล

### 2.1 Graph input
- **Node** = mesh router
- **Node feature**: queue length, offered load, จำนวน client, channel ปัจจุบัน (one-hot ต่อ radio), role (gateway/relay), hop-to-gateway
- **Link edge** (เชื่อมกันได้จริง): RSSI, SNR, airtime utilization, co-channel interference, ETX, current rate
- **Conflict edge**: ระหว่างโหนดที่อยู่ใน carrier-sense range กันแม้ไม่มี link — feature = estimated interference power, same-channel flag
- Edge type ใส่เป็น one-hot ใน edge feature (ใช้ GATv2 ตัวเดียว ไม่แยก relation)

### 2.2 Encoder
- `GATv2Conv(edge_dim=...)` 2–3 ชั้น (PyTorch Geometric), multi-head, residual + LayerNorm
- GRU ต่อ node embedding ข้าม timestep (จำ trend traffic)
- Weight shared ทุก node → ขนาด graph เปลี่ยนได้

### 2.3 Policy heads (actor)
- **Channel head**: ตัดสินทุก `T_c` step (ค่าเริ่ม 10), output distribution ต่อ radio เหนือ channel set, มี action mask (channel ที่ไม่อนุญาต)
- **Routing head**: ตัดสินทุก step, เลือก next-hop ต่อ destination-gateway จาก neighbor ที่มี link เท่านั้น (pointer-style score ต่อ edge), mask กัน loop (ห้ามเลือก neighbor ที่ hop-to-gateway มากกว่าตัวเอง + 1)
- Routing head รับ channel ปัจจุบันเป็น input (channel เป็น context ของ routing)

### 2.4 Critic
- Centralized critic (ใช้ตอนเทรนเท่านั้น): global mean/attention pooling ของ node embeddings + global features → V(s)
- Critic แยกต่อหัว (V_channel บน semi-MDP timescale, V_route ทุก step)

### 2.5 Ko-hybrid
1. **BC warm-start**: เก็บ trajectory จาก Ko ในทุก scenario เทรน actor ด้วย cross-entropy จน agreement ≥ 90% (สูงสุด 20 epoch) แล้วจึงเริ่ม PPO
2. **Residual prior**: `logits = f_θ(h) − β · cost_Ko(a)` โดย `β` เรียนได้ (init 1.0) → policy เรียนเฉพาะส่วนปรับแก้
3. **Safe fallback** (execution): ถ้า `V_π(s) < V_Ko(s) − m` (margin `m`) หรือ entropy ของ policy สูงเกิน threshold → ใช้ action ของ Ko ใน step นั้น; บันทึก fallback rate เป็น metric
   - `V_π` = critic ของ policy, `V_Ko` = value network แยกที่ fit กับ return ของ Ko ระหว่างช่วง BC (encoder เดียวกัน, head แยก, freeze หลัง BC)
   - `m` และ entropy threshold เป็น hyperparameter ใน Optuna search
   - ตอน execution แบบกระจาย ใช้ local version: ประเมินจาก node embedding ของตัวเอง (critic head ขนาดเล็กต่อ node) เพื่อไม่ต้องพึ่ง centralized critic

### 2.6 Training
- MAPPO, parameter sharing, GAE(λ), clip 0.2, entropy bonus anneal
- Parallel envs (vectorized Python sim)
- Curriculum: N เล็ก → ใหญ่, load เบา → หนัก

---

## 3. Reward

```
r_t = Σ_f log(x_f + ε)            # proportional-fair utility (throughput + fairness)
      − γ · n_channel_switch
      − δ · n_route_change
      − η · n_drop
```

- เหตุผล: log utility รวม throughput และ fairness ในเทอมเดียว ไม่ต้องจูนน้ำหนัก throughput/Jain
- Per-agent: `r_i = λ · r_local_i + (1 − λ) · r_global`, `λ` anneal 0.5 → 0.1; `r_local_i` = utility ของ flow ที่ผ่าน node i
- Channel head ได้ reward สะสมตลอดช่วง `T_c` (semi-MDP); routing head ได้ทุก step
- γ, δ, η, ε อยู่ใน config และอยู่ใน Optuna search

---

## 4. Simulators

### 4.1 Python fluid simulator (เทรน)
- Time step ≈ 100 ms, flow-level (fluid) ไม่ใช่ packet-level
- **Topology generators**: random geometric graph, grid, indoor floor plan; train N = 10–30, test ถึง 80; gateway 1–3; 2 radios/node
- **Channels**: 2.4 GHz {1, 6, 11}, 5 GHz 4–8 channels (configurable)
- **Propagation**: log-distance path loss — พารามิเตอร์เดียวกับ `ns3::LogDistancePropagationLossModel`
- **Rate**: SINR → MCS ตาม 802.11ax table
- **Contention**: conflict graph จาก carrier-sense range; link ใน conflict set แบ่ง airtime; capacity = rate × airtime share; routing → flow allocation บน path
- **Traffic**: flows ไป/กลับ gateway, on/off + Poisson, regime shift ระหว่าง episode
- **Events**: external interferer (ยึด channel ช่วงเวลา), node/link failure
- **Interface**: Gym-style multi-agent env (`reset`, `step(actions) → obs_graph, rewards, done, info`)

### 4.2 Calibration กับ ns-3 (ทำก่อนเทรน)
- รัน ns-3 บน topology เล็ก (5–10 nodes) หลาย configuration
- ปรับพารามิเตอร์ contention/overhead ของ Python sim ให้ per-link throughput error ≤ 10–15%
- บันทึก error ไว้รายงาน

### 4.3 ns-3 (validate)
- `ns3-ai` (shared memory) เชื่อม Python policy
- Wi-Fi ad-hoc MAC, 802.11ax, channel ตั้งผ่าน `WifiPhy::ChannelSettings`
- Routing ผ่าน `Ipv4StaticRouting` อัปเดตตาม action
- Flow monitor เก็บ throughput, delay, PDR

### 4.4 Baselines implementation
- ทุก baseline รันบน Python sim **และ** ns-3 ด้วย seed เดียวกัน ผ่าน interface เดียวกับโมเดลเรา
- ห้ามใช้ตัวเลขจาก paper ต้นฉบับเปรียบเทียบตรง

---

## 5. Evaluation Protocol

### 5.1 Research Questions
- RQ1: EGAT-MAPPO ชนะ baseline ทั้ง 5 ด้าน throughput + fairness หรือไม่
- RQ2: แต่ละ component contribute เท่าไร (ablation)
- RQ3: generalize ไป topology/ขนาดที่ไม่เคยเห็นได้แค่ไหน
- RQ4: ผลบน Python sim คงอยู่บน ns-3 หรือไม่

### 5.2 Scenarios
| ID | Scenario | ทดสอบ |
|---|---|---|
| S1 | Light load | ไม่แย่กว่า heuristic |
| S2 | Heavy load (ใกล้อิ่มตัว) | ความสามารถหลัก |
| S3 | Hotspot gateway | load balancing ผ่าน routing |
| S4 | External interference | channel adaptation |
| S5 | Regime shift | การปรับตัว (GRU) |
| S6 | Node/link failure | ความทนทาน + fallback |

### 5.3 Generalization (RQ3)
- Train: random geometric, N = 10–30
- Test: (a) unseen topology ขนาดเดียวกัน (b) N = 50, 80 (c) grid / indoor (d) จำนวน channel ต่างจาก train
- Q-neutral ต้องเทรนใหม่ต่อ topology — รายงานต้นทุนนี้

### 5.4 Metrics
- **หลัก**: aggregate throughput, Jain fairness index, 5th-percentile flow throughput
- **รอง**: p95 delay, PDR, channel switches/ชั่วโมง, route flaps, episodes-to-Ko-level, inference latency/decision, signaling overhead (bytes ระหว่าง node ตอน execution), fallback rate

### 5.5 Fair comparison
- RL baselines ทุกตัวได้ env steps เท่ากัน
- Hyperparameter tuning: Optuna 20 trials/วิธี
- Observation เท่ากันเท่าที่ทำได้; ส่วนที่ต่าง (เช่น edge feature) ระบุชัดว่าเป็น contribution
- บันทึกการ adapt baseline ทุกตัวใน appendix

### 5.6 Ablations
| Variant | ตัดอะไร | คาดว่ากระทบ |
|---|---|---|
| −EdgeFeat | edge feature + conflict edge | throughput |
| −TwoTimescale | channel+routing ทุก step | stability (switches ↑) |
| −BC | ไม่มี warm-start | convergence speed |
| −KoPrior | ไม่มี residual + fallback | worst-case vs Ko |
| −GRU | ไม่มี temporal memory | S5 regime shift |
| A2C instead of MAPPO | เปลี่ยน trainer | variance |

### 5.7 Statistics
- 5 training seeds × 20 test topologies × 3 traffic seeds
- Mean ± 95% CI (bootstrap)
- Paired Wilcoxon signed-rank (ต่อ topology) vs baseline ที่ดีที่สุดในแต่ละ scenario, Holm correction
- RQ4: Kendall τ ของ ranking วิธีระหว่าง Python กับ ns-3 + absolute gap

### 5.8 Success criteria (pre-registered)
1. ชนะ baseline ที่ดีที่สุดทั้ง throughput และ Jain อย่างมีนัยสำคัญใน ≥ 4/6 scenarios
2. ไม่มี scenario ที่แย่กว่า Ko เกิน 5%
3. Zero-shot ที่ N = 50 ลดลงไม่เกิน 15% เทียบ in-distribution
4. ns-3 Kendall τ ≥ 0.8

ไม่ถึงเกณฑ์ → รายงานตามจริงเป็น finding

---

## 6. โครงสร้างโค้ด (เบื้องต้น)

```
mesh/
  sim/          # Python fluid simulator: topology, propagation, contention, traffic, env
  baselines/    # m0, ko_local_cost, q_neutral, yuan_gat_dqn, li_gat_a2c
  model/        # encoder (GATv2+GRU), heads, critic, ko_prior
  train/        # bc pretrain, mappo trainer, configs
  eval/         # scenarios, metrics, statistics, ablation runner
  ns3/          # ns-3 scenario script + ns3-ai bridge
  calib/        # Python vs ns-3 calibration
tests/
```
- Python 3.11, PyTorch, PyTorch Geometric, Optuna, NumPy/SciPy
- แต่ละ module มี test เล็กสุดที่ fail ถ้า logic พัง (เช่น contention: link 2 ตัว conflict กันต้องแบ่ง airtime)

---

## 7. Milestones

| # | งาน | Deliverable / Done when |
|---|---|---|
| M1 | Python sim + tests | env รันได้, sanity check (1 link, 2 conflicting links, multi-hop) ผ่าน |
| M2 | ns-3 scenario + calibration | error ≤ 15% บน topology เล็ก |
| M3 | M0 + Ko | รันครบ 6 scenarios บน Python sim |
| M4 | Q-neutral, Yuan, Li | รันครบด้วย budget เท่ากัน, บันทึกการ adapt |
| M5 | EGAT-MAPPO (BC → MAPPO) | เทรน converge, ≥ Ko ใน S1 |
| M6 | Main results + ablation | ตาราง RQ1, RQ2 พร้อมสถิติ |
| M7 | Generalization | ตาราง RQ3 |
| M8 | ns-3 validation | ตาราง RQ4 |

ลำดับนี้ทำให้ baseline น่าเชื่อถือก่อนเริ่มงานของเรา

---

## 8. Risks

| Risk | ผลกระทบ | การรับมือ |
|---|---|---|
| Yuan/Li ทำ joint หรือใช้ edge feature อยู่แล้ว | contribution อ่อน | ตรวจก่อน (1.5); เน้น two-timescale + Ko-hybrid + ns-3 validation |
| Sim-to-sim gap ใหญ่ | ผล ns-3 ไม่ตาม | calibration M2 ก่อนเทรน; รายงาน gap |
| MAPPO ไม่ converge บน joint action | ไม่มีผล | BC warm-start, curriculum, ablation ทีละหัว |
| Fluid model ไม่จับ MAC dynamics | over-estimate throughput | ns-3 เป็นผลสุดท้ายที่รายงาน |
| Baseline adaptation ถูกมองว่าอ่อน | reviewer ไม่เชื่อ | tuning budget เท่ากัน + appendix การ adapt |
