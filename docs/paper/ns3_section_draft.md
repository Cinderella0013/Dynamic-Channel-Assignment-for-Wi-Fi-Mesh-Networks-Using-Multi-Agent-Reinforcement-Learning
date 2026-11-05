# ns-3 validation section: draft with placeholders

Fill the `{{...}}` from `python ns3/report.py --ns3 runs/ns3/default --sim <sim dir>`. Run order and the 20% calibration gate are in `docs/ns3/protocol.md`. If the gate fails, use the "gate failed" variants at the end instead of the results text.

## A. Thesis (Thai), chapter 4, new subsection "4.x  การทดสอบบน ns-3"

**4.x.1  การตั้งค่า**

เพื่อตรวจสอบว่าข้อสรุปจากสภาพแวดล้อมจำลองระดับ flow ยังคงอยู่เมื่อการส่งข้อมูลเป็นระดับแพ็กเก็ต ผู้วิจัยนำนโยบายที่ฝึกเสร็จแล้วไปประเมินบน ns-3 รุ่น 3.47 โดยไม่ฝึกใหม่และไม่ปรับพารามิเตอร์ใด ๆ ทุกวิธีทำงานฝั่ง Python ด้วยนโยบายที่ตรึงไว้ (frozen policy) ส่วน ns-3 ทำหน้าที่ขนส่งแพ็กเก็ตในแต่ละขั้นเวลา 100 ms ผ่านท่อข้อความ JSON ทีละบรรทัด โครงสร้างโหนด ความต้องการทราฟฟิก การล้มของโหนด และการย้ายจุดร้อนมาจากตัวสร้างชุดเดียวกับสภาพแวดล้อมจำลองและค่าเมล็ดสุ่มเดียวกัน ทุกวิธีจึงเผชิญความต้องการทราฟฟิกเดียวกัน

แต่ละโหนดมีวิทยุสองตัวแบบ ad-hoc ใช้ 802.11ax ความกว้างช่อง 20 MHz หนึ่งสตรีมเชิงพื้นที่ และ IdealWifiManager เลือก MCS จาก SNR ซึ่งใกล้เคียงตาราง SNR–MCS ของสภาพแวดล้อมจำลองที่สุด การสูญเสียตามระยะใช้ log-distance ค่าเลขชี้กำลัง 3.0 กำลังส่ง 20 dBm และ noise floor −94 dBm ตรงกับแบบจำลอง Python เส้นทางเป็นแบบ static route ที่สร้างใหม่ทุกขั้นเวลาจากโหนดแม่ที่นโยบายเลือก การเปลี่ยนช่องของวิทยุใช้ channel switch delay 100 ms ซึ่งสอดคล้องกับการหยุดชะงัก 1 ขั้นเวลาของสภาพแวดล้อมจำลอง และเพิ่มการทดสอบความไวด้วยค่า native ของ ns-3 (250 µs) การเปลี่ยนโหนดแม่ทำให้ uplink ของโหนดนั้นรอ 100 ms

สถานการณ์ที่ใช้คือ S2 (โหลดสูง), B1 (จุดร้อนเคลื่อนที่) และ B3 (เครือข่ายเบาบาง) เลือกก่อนเห็นผลบน ns-3 ใช้ค่าเมล็ดสุ่มประเมิน {{n_eval}} ค่า ({{eval_seeds}}) × ค่าเมล็ดสุ่มการฝึก {{n_train}} ค่าของแต่ละวิธีที่เรียนรู้ คู่เทียบคือ M0, Ko, Li, Yuan และ Q-neutral เทียบกับ EGAT-MAPPO ที่ใช้ switch hysteresis (EGAT+H, switch margin 3.0) และคำนวณคะแนนรวมห้าตัวชี้วัดกับช่วงความเชื่อมั่นแบบคู่ค่าเมล็ดสุ่มด้วยโค้ดชุดเดียวกับหัวข้อ 3.4 ก่อนรันชุดทดลองหลัก กำหนดเกณฑ์ปรับเทียบไว้ว่า หาก throughput รวมเฉลี่ยของ Ko และ M0 บน S2 บน ns-3 ต่างจากสภาพแวดล้อมจำลองเกิน 20% จะหยุดและรายงานว่าการยืนยันบน ns-3 เป็นงานในอนาคต

**4.x.2  ผลการปรับเทียบและผลหลัก**

ผลปรับเทียบบน S2: throughput รวมเฉลี่ยของ Ko บน ns-3 เท่ากับ {{ko_ns3}} Mbps เทียบกับ {{ko_sim}} Mbps ในสภาพแวดล้อมจำลอง (ต่างกัน {{ko_diff}}) และของ M0 เท่ากับ {{m0_ns3}} Mbps เทียบกับ {{m0_sim}} Mbps (ต่างกัน {{m0_diff}}) ซึ่ง{{gate_result_th}}เกณฑ์ 20%

ตารางที่ 4.x  ส่วนนำของคะแนนรวมของ EGAT+H เหนือแต่ละวิธี บน ns-3 เทียบกับสภาพแวดล้อมจำลอง (เฉลี่ย S2, B1, B3; ช่วงความเชื่อมั่น 95% แบบคู่ค่าเมล็ดสุ่ม)

| วิธีอ้างอิง | ส่วนนำบน ns-3 [ช่วง 95%] | ส่วนนำในสภาพแวดล้อมจำลอง [ช่วง 95%] | Throughput ns-3 / จำลอง (Mbps) | อัตราสูญเสีย ns-3 / จำลอง | การสลับช่อง ns-3 / จำลอง |
|---|---|---|---|---|---|
| M0 | {{lead_m0_ns3}} | {{lead_m0_sim}} | {{thr_m0}} | {{loss_m0}} | {{sw_m0}} |
| Ko | {{lead_ko_ns3}} | {{lead_ko_sim}} | {{thr_ko}} | {{loss_ko}} | {{sw_ko}} |
| Li | {{lead_li_ns3}} | {{lead_li_sim}} | {{thr_li}} | {{loss_li}} | {{sw_li}} |
| Yuan | {{lead_yuan_ns3}} | {{lead_yuan_sim}} | {{thr_yuan}} | {{loss_yuan}} | {{sw_yuan}} |
| Q-neutral | {{lead_q_ns3}} | {{lead_q_sim}} | {{thr_q}} | {{loss_q}} | {{sw_q}} |
| EGAT+H | – | – | {{thr_egat}} | {{loss_egat}} | {{sw_egat}} |

บน ns-3 EGAT+H มีคะแนนรวมสูงกว่า Ko {{lead_ko_ns3}} (สภาพแวดล้อมจำลอง {{lead_ko_sim}}) สูงกว่า Li {{lead_li_ns3}} และสูงกว่า Yuan {{lead_yuan_ns3}} {{significance_th}} จำนวนการสลับช่องของ EGAT+H คือ {{sw_egat}} ครั้งต่อ episode เทียบกับ {{sw_ko_only}} ของ Ko ซึ่ง{{switch_trend_th}} เมื่อลด channel switch delay เหลือค่า native ส่วนนำเหนือ Ko เปลี่ยนเป็น {{lead_ko_native}}

**4.x.3  ข้อจำกัด**

การทดสอบนี้เป็นการประเมินเท่านั้น นโยบายถูกฝึกในสภาพแวดล้อมจำลองระดับ flow และไม่ได้ปรับให้เข้ากับ ns-3 จึงอาจเสียเปรียบเมื่อพฤติกรรมระดับแพ็กเก็ต (การชนกันของ MAC การลดอัตราของ IdealWifiManager และคิวของ MAC) ต่างจากแบบจำลอง นอกจากนี้ครอบคลุมเพียงสามสถานการณ์ ไม่มีสัญญาณรบกวนภายนอก (S4) และยังไม่ได้ทดสอบบนอุปกรณ์จริง

## B. Paper (English), section 5, "Packet-level check in ns-3"

We evaluated the frozen policies, without retraining, in ns-3.47. Policies run in Python; ns-3 carries the packets of each 100 ms step (802.11ax, 20 MHz, two radios per node, IdealWifiManager, log-distance loss with exponent 3, static routes rebuilt every step, 100 ms channel-switch delay with a native-delay sensitivity run). Topology, traffic and failures come from the simulator's generators with the same seeds, so all methods see the same demand. We use scenarios S2, B1 and B3, {{n_eval}} evaluation seeds and {{n_train}} training seeds per learner. A gate fixed beforehand required Ko and M0 on S2 to reach within 20% of the simulator's mean aggregate throughput; the differences were {{ko_diff}} and {{m0_diff}}, so the gate {{gate_result_en}}. Table N gives EGAT+H's composite lead over each method in both environments. On ns-3 its lead over Ko is {{lead_ko_ns3}} (simulator {{lead_ko_sim}}), over Li {{lead_li_ns3}} and over Yuan {{lead_yuan_ns3}}; {{significance_en}}. The study covers three scenarios without external interference and the policies were not adapted to ns-3.

Table N. Composite lead of EGAT+H over each method, ns-3 vs simulator (mean of S2, B1, B3; seed-pair 95% interval).

| method | lead, ns-3 [95% interval] | lead, simulator [95% interval] | throughput ns-3 / sim (Mbps) | loss rate ns-3 / sim | channel switches ns-3 / sim |
|---|---|---|---|---|---|
| M0 | {{lead_m0_ns3}} | {{lead_m0_sim}} | {{thr_m0}} | {{loss_m0}} | {{sw_m0}} |
| Ko | {{lead_ko_ns3}} | {{lead_ko_sim}} | {{thr_ko}} | {{loss_ko}} | {{sw_ko}} |
| Li | {{lead_li_ns3}} | {{lead_li_sim}} | {{thr_li}} | {{loss_li}} | {{sw_li}} |
| Yuan | {{lead_yuan_ns3}} | {{lead_yuan_sim}} | {{thr_yuan}} | {{loss_yuan}} | {{sw_yuan}} |
| Q-neutral | {{lead_q_ns3}} | {{lead_q_sim}} | {{thr_q}} | {{loss_q}} | {{sw_q}} |

## C. If the calibration gate fails

Thai: "ผลปรับเทียบบน S2 พบว่า throughput รวมของ {{failed_method}} บน ns-3 ต่างจากสภาพแวดล้อมจำลอง {{diff}} เกินเกณฑ์ 20% ที่กำหนดไว้ล่วงหน้า จึงไม่ดำเนินการชุดทดลองหลัก และถือว่าการยืนยันบน ns-3 เป็นงานในอนาคต"

English: "In the S2 calibration, {{failed_method}}'s aggregate throughput on ns-3 differed from the simulator by {{diff}}, above the 20% gate set beforehand. We therefore did not run the main matrix and leave ns-3 validation to future work."

Placeholder map: `{{lead_*}}`, `{{thr_*}}`, `{{loss_*}}`, `{{sw_*}}` = cells of the report's main table; `{{ko_ns3}}` etc. = calibration table; `{{ko_diff}}`, `{{m0_diff}}` = its "difference" column.
