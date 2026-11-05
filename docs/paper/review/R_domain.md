# Reviewer 2 (Domain: Wi-Fi mesh, channel assignment, routing)

## สรุปงาน
งานนี้เสนอ EGAT-MAPPO สำหรับเลือกช่อง downlink และโหนดแม่ร่วมกันใน multi-radio mesh แบบต้นไม้ (2 วิทยุ โดย uplink ใช้ช่องของโหนดแม่) และประเมินบน fluid simulator ที่เขียนขึ้นเอง ใช้ PHY ของ 802.11ax กับ path loss ตามค่าเริ่มต้นของ ns-3 และแบ่ง airtime ตามจำนวนลิงก์ที่แย่งช่องกัน คู่เทียบคือ Ko (WCNC 2007) กับ GNN-RL สองวิธีที่ดัดแปลงจากโดเมนอื่น ผลหลักคือดีกว่า Ko +19.5% แต่ส่วนต่างราว 70% มาจากจำนวนการสลับ (43 เทียบ 580 ครั้ง)

## จุดแข็ง
1. โจทย์ตรงกับปัญหาจริงของ WMN ที่ช่องกับเส้นทางผูกกัน ผู้เขียนเข้าใจว่าเมื่อโหนดแม่เปลี่ยนช่อง ลูกทุกตัวได้รับผลด้วย (บทที่ 3.2.1) และใช้ progress mask ที่รับประกันว่าเส้นทางไม่วน (4.2.2)
2. พารามิเตอร์ PHY ถูกต้องและตรวจสอบได้ ได้แก่ reference loss 46.68 dB ตาม ns-3, noise −94 dBm และตาราง MCS ของ 802.11ax 20 MHz 1SS (`radio.py:5-16`)
3. ต้นทุนการสลับเกิดในพลวัตของระบบจริง คือหยุดใช้ลิงก์และสูญเสียคิว ไม่ได้มีแค่ค่าปรับใน reward และยังแยก setting (b) ออกมาเพื่อแยกผลของการเก็บคิว
4. รายงานผลด้านลบอย่างตรงไปตรงมา ได้แก่ Li+H ไม่มีนัยสำคัญบน topology ใหม่ แพ้ Ko ใน B4 และผลบน topology ใหม่เล็กลง พร้อมบันทึกการดัดแปลงคู่เทียบไว้ครบ

## ประเด็น (เรียงตามความรุนแรง)

**C1 [CRITICAL] คู่เทียบ Ko ถูกดัดแปลงจนสลับบ่อยผิดธรรมชาติ และไม่ได้รับ hysteresis แบบเดียวกับ Li และ Yuan**
- ตำแหน่ง: PDF หน้า 26–27 §3.4.1, หน้า 34–36 §4.4.2–4.4.3
- หลักฐาน: ใน `ko.py:96-101` ต้นทุนช่องคิดจาก `(1 + load[others])` ซึ่งเป็น offered load ณ ขั้นนั้น ทราฟฟิกเป็น on/off ที่สุ่มอัตราใหม่ทุก 100 ms (`traffic.py:37`) Ko จึงเลือกช่องใหม่ทุก 10 ขั้นตามสัญญาณรบกวนของภาระ และไม่มี hysteresis ฝั่งช่อง (มีเฉพาะเส้นทาง `SWITCH_GAIN=0.9`) Ko ต้นฉบับเป็นอัลกอริทึม self-stabilizing ที่ควรลู่เข้าแล้วหยุดสลับ (หน้า 15: "self-stabilizing") แต่ในงานนี้ Ko สลับ 580 ครั้งต่อ episode นอกจากนี้ Ko ต้นฉบับไม่มี routing ส่วน Bellman-Ford ที่ใช้ต้นทุน (1+load)/rate เป็นส่วนที่ผู้จัดทำเพิ่มเอง
- ผลกระทบ: ข้อได้เปรียบ +13.8 จาก +19.6 จุด (ตารางที่ 7) จึงอาจเป็น artifact ของการดัดแปลง Ko
- แก้ไข: (ก) เพิ่ม Ko+H ด้วยกติกาเลือก margin เดียวกับ Li+H และ Yuan+H (ข) เพิ่ม Ko ที่ใช้ interference cost ตามต้นฉบับซึ่งไม่ขึ้นกับภาระ หรือใช้ภาระที่เฉลี่ยแบบ EWMA (ค) ถ้า EGAT+H เทียบ Ko+H ไม่ถึง 10% ต้องปรับข้อสรุปในบทคัดย่อ

**M1 [MAJOR] แบบจำลองการสูญเสียคิวใน setting (a) ซึ่งเป็น setting หลัก ลงโทษการสลับเกินจริง**
- ตำแหน่ง: หน้า 28 §4.1.2, หน้า 26 §3.4
- หลักฐาน: ใน `env.py:178-183` เมื่อ hop ใดบน planned path ถูกตัด backlog ทั้งหมดของ source จะหายไป แบบจำลองนี้มีคิวเฉพาะที่ต้นทาง ไม่มีคิวราย hop การเปลี่ยนช่องที่ hop ปลายทางจึงลบข้อมูลที่ต้นทางทิ้ง ซึ่งไม่เกิดในจริง เพราะ 802.11h CSA และ 802.11s MCSA เก็บ frame ไว้ระหว่างย้ายช่อง ส่วนอุปกรณ์สลับช่องจริงสูญเสียเพียงเศษของ MAC queue
- ผลกระทบ: เมื่อเก็บคิวได้ (setting b) ข้อได้เปรียบเหนือ Ko ลดเหลือ +9.5% บน topology ใหม่ และแพ้ใน B4/S4 ตัวชี้วัดความเสถียรยังนับต้นทุนการสลับซ้ำอีกครั้ง เพราะต้นทุนนั้นปรากฏในอัตราสูญเสียแล้ว
- แก้ไข: ใช้คิวราย hop หรือทิ้งเฉพาะข้อมูลที่ค้างอยู่บนลิงก์ที่ถูกตัดภายในเวลา downtime ให้ setting (b) หรือ setting กลางที่ใช้แบบจำลองนี้เป็นผลหลัก และอภิปรายเรื่องการนับซ้ำในคะแนนรวม

**M2 [MAJOR] แบบจำลอง MAC และการรบกวนหยาบเกินกว่าจะสรุปว่าผลส่งต่อไปยัง ns-3 หรือ 802.11s ได้**
- หลักฐาน (`graph.py:48-60`):
  (ก) airtime แบ่งเป็น 1/k ตามจำนวนลิงก์ที่ active โดยไม่ขึ้นกับภาระ ลิงก์ที่ส่ง 0.1 Mb/s จึงได้ airtime เท่ากับลิงก์ที่อิ่มตัว
  (ข) k ไม่ได้คิดตาม clique ของ conflict graph
  (ค) อัตราส่งขึ้นกับ SNR ไม่ใช่ SINR จึงไม่มี hidden terminal และไม่มีการรบกวนที่ต่ำกว่า −82 dBm
  (ง) ไม่มี MAC overhead และไม่มี performance anomaly ของ DCF
  (จ) `allocate()` แบ่งตามสัดส่วน ไม่ใช่ max-min และไม่กระจาย capacity ที่เหลือ (`flows.py:52`)
- ข้อเท็จจริงเพิ่มเติม: เกณฑ์ลิงก์ −76 dBm เท่ากับ SNR 18 dB ทุกลิงก์จึงได้ MCS ≥ 5 (68.8 Mb/s) ช่วง "8.6 ถึง 143.4 Mb/s" ในหน้า 28 จึงชวนให้เข้าใจผิด
- ข้อสังเกต: คอมเมนต์ในโค้ดเองระบุให้เปลี่ยนเป็น clique-based "if Plan 2 calibration error > 15%" แต่ยังไม่ได้ทำ calibration
- แก้ไข: ใช้เงื่อนไข airtime Σ load/rate ≤ 1 บน clique ใส่ค่า MAC efficiency และ calibrate กับ ns-3 บน topology เล็ก 2–3 แบบ ซึ่งผู้จัดทำมีโครงสร้างพื้นฐานจากระยะที่ 1 แล้ว อย่างน้อยควรแสดงว่าลำดับของวิธีไม่เปลี่ยน

**M3 [MAJOR] ระบบที่จำลองไม่ใช่ "Wi-Fi Mesh" ที่ใช้จริง และกรอบเวลาไม่สมจริง**
- ตำแหน่ง: หน้า 28 §4.1.1–4.1.2, ตารางที่ 3
- หลักฐาน:
  (ก) ระบบเป็นสถาปัตยกรรมต้นไม้แบบ Hyacinth (Raniwala & Chiueh) ไม่ใช่ 802.11s ซึ่งใช้ HWMP และ airtime metric บนช่องเดียว ลูกเปลี่ยนช่องตามแม่ทันทีโดยไม่มี re-peering (`agents_io.py:65`) ขณะที่ 802.11s ต้องทำ mesh peering/SAE ใหม่ ซึ่งนานกว่า 100 ms มาก
  (ข) "การผสม 2.4/5 GHz" มีแค่ในชื่อ ทุกช่องใช้ reference loss 46.68 dB ซึ่งเป็นค่า Friis ที่ 5.15 GHz ช่อง 6 และ 11 จึงมีพฤติกรรมทางฟิสิกส์เหมือนช่อง 5 GHz ไม่มีระยะที่ไกลกว่าและไม่มีการรบกวนภายนอกที่หนาแน่นกว่า
  (ค) ทราฟฟิกมีเฉพาะ uplink ไม่มีโหลดจาก client access และ gateway ไม่มีขีดจำกัด backhaul ขณะที่ทราฟฟิกจริงส่วนใหญ่เป็น downlink
  (ง) ระบบตัดสินใจช่องทุก 1 วินาที และ episode ยาว 20 วินาที แต่ระบบจริงเช่น ACS/RRM ปรับช่องในระดับนาทีถึงชั่วโมง ข้อค้นพบว่า "สลับน้อยแล้วดีกว่า" จึงเป็นสิ่งที่วงการรู้อยู่แล้ว
- แก้ไข: เรียกระบบว่า "multi-radio tree-based WMN (Hyacinth-style)" ใช้ path loss แยกตามย่านความถี่ เพิ่มสถานการณ์ downlink และ client load และอภิปรายว่าผลส่งต่อไปยัง 802.11s ได้มากน้อยเพียงใด

**M4 [MAJOR] คู่เทียบที่เรียนรู้มาจากโดเมนอื่น และไม่มีคู่เทียบมาตรฐานของ WMN**
- ตำแหน่ง: หน้า 16 §2.2.11–2.2.12, หน้า 27 §3.4.1
- หลักฐาน: Yuan เป็นงาน CRN slicing และ Li เป็นงาน C-V2X ทั้งสองไม่มี routing และถูกตัด power action กับ reward ต้นฉบับออก (`docs/baselines-adaptation.md`) เอกสารนั้นยังระบุว่าที่มาของ Li "has not yet confirmed" ส่วน Q-neutral ผู้จัดทำ "ตีความจากชื่อ" (หน้า 42) คู่เทียบเหล่านี้จึงวัดความสามารถของ GNN encoder ไม่ใช่วิธีแก้ joint CA+routing
- แก้ไข: เพิ่มอย่างน้อยหนึ่งวิธีจากแต่ละกลุ่มต่อไปนี้
  - load-aware distributed CA + routing แบบ Hyacinth
  - static CA + airtime/WCETT routing ซึ่งใกล้กับ 802.11s
  - upper bound แบบรวมศูนย์ เช่น simulated annealing หรือ ILP บน fluid model ตามแนว Alicherry et al. เพื่อแสดงว่ายังมีช่องว่างให้ปรับปรุงเท่าใด
  - ถ้าเป็นไปได้ ควรมี GNN-RL ที่ทำในบริบท WLAN/WMN โดยตรง

**M5 [MAJOR] รายการอ้างอิงและทฤษฎีในบทที่ 2 ไม่ถึงมาตรฐานวิชาการ**
- ตำแหน่ง: หน้า 11–12 §2.1, หน้า 44–45
- หลักฐาน: [1]–[6] เป็นเว็บหรือบล็อก (TechTarget, GeeksforGeeks, Medium, SiamWireless, Synopsys, builtin) และเป็นแหล่งเดียวของทฤษฎี §2.1.1–2.1.6 ส่วน [8], [11], [12] ไม่มีชื่อผู้แต่งหรือ venue ([12] มาจาก Bohrium และไม่ถูกอ้างในเนื้อหา) และ [9] ใช้ชื่อมหาวิทยาลัยแทน venue ไม่มีงาน WMN คลาสสิกเลย แม้แต่ Ramachandran INFOCOM 2006 ที่บทความอ้าง ([9]) ก็ไม่ปรากฏในเล่ม นอกจากนี้หมายเลข [1]–[13] ในตารางที่ 1 ชนกับหมายเลขในรายการอ้างอิง เช่น "[1] Adaptive Channel..." ขณะที่อ้างอิง [1] คือ TechTarget และหน้า 19 เป็นหน้าว่าง
- แก้ไข: แทนเว็บด้วย survey หรือ textbook (ดูรายการด้านล่าง) เติม metadata ให้ครบ และใช้หมายเลขอ้างอิงเดียวกันทั้งเล่ม

**m1 [MINOR] ข้อความเชิงโดเมนในบทที่ 2 ไม่ถูกต้องหรือล้าสมัย**
- §2.1.2 (หน้า 11) นิยาม DCA แบบเซลลูลาร์ ("blocking probability") ซึ่งไม่เกี่ยวกับ WMN
- §2.1.4 ใช้คำ "การชนกันของช่องสัญญาณ" ปน interference กับ collision และไม่แยก co-channel จาก adjacent-channel รวมทั้งไม่ระบุว่า 2.4 GHz มีช่องที่ไม่ทับกันเพียง 3 ช่อง
- §2.1.5 (หน้า 12) "งานนี้ประยุกต์ใช้ Q-Learning ... กับ Access Point" เป็นข้อความที่ค้างมาจากระยะที่ 1
- §2.1.6 อ้างว่า GNN "แก้ปัญหา...Herd Behavior" แต่ §5.3.3 แสดงว่าต้องใช้ sequential decoder จึงแก้ได้
- §2.2.5 พิมพ์ "QaulNet" ผิด ที่ถูกคือ QualNet
- แก้ไข: เขียนใหม่โดยอิงแหล่งในรายการด้านล่าง

**m2 [MINOR] คำศัพท์และนิยามตัวชี้วัด**
- "downlink" ในงานนี้หมายถึง backhaul ไปยังลูก ซึ่งชนกับความหมาย AP→STA
- §4.1.3 ใช้ "(ช่อง downlink, ช่องโหนดแม่)" ซึ่งคำว่า "ช่อง" ปนความหมายระหว่าง channel กับ slot
- `env.py:198` นับการสลับทั้งสองวิทยุ การที่ลูกเปลี่ยนวิทยุ 0 ตามแม่จึงถูกนับเพิ่ม ทำให้ค่าของวิธีที่เปลี่ยนช่องที่โหนดแม่ซึ่งมีลูกมากพองตัว
- `radio.py:10` มีช่องที่ 7 (ช่อง 1) ซ่อนอยู่สำหรับวิทยุ 0 ของ gateway แต่ตารางที่ 3 ระบุ 6 ช่อง
- แก้ไข: นิยามคำให้ชัด และระบุวิธีนับการสลับในหัวข้อ 3.4

## งานที่ควรอ้างเพิ่ม
- Raniwala, A., Chiueh, T. "Architecture and Algorithms for an IEEE 802.11-Based Multi-Channel Wireless Mesh Network." IEEE INFOCOM 2005. (Hyacinth ใช้ convention เดียวกับงานนี้)
- Raniwala, A., Gopalan, K., Chiueh, T. "Centralized Channel Assignment and Routing Algorithms for Multi-Channel Wireless Mesh Networks." ACM MC2R 2004.
- Alicherry, M., Bhatia, R., Li, L. "Joint Channel Assignment and Routing for Throughput Optimization in Multi-radio Wireless Mesh Networks." ACM MobiCom 2005.
- Kodialam, M., Nandagopal, T. "Characterizing the Capacity Region in Multi-Radio Multi-Channel Wireless Mesh Networks." ACM MobiCom 2005.
- Draves, R., Padhye, J., Zill, B. "Routing in Multi-Radio, Multi-Hop Wireless Mesh Networks." ACM MobiCom 2004. (WCETT)
- Ramachandran, K. N. et al. "Interference-Aware Channel Assignment in Multi-Radio Wireless Mesh Networks." IEEE INFOCOM 2006.
- Subramanian, A. P., Gupta, H., Das, S. R., Cao, J. "Minimum Interference Channel Assignment in Multiradio Wireless Mesh Networks." IEEE Trans. Mobile Computing, 2008.
- Akyildiz, I. F., Wang, X., Wang, W. "Wireless Mesh Networks: A Survey." Computer Networks 47(4), 2005. (ใช้แทน [1])
- Si, W., Selvakennedy, S., Zomaya, A. Y. "An Overview of Channel Assignment Methods for Multi-Radio Multi-Channel Wireless Mesh Networks." J. Parallel Distrib. Comput. 70(5), 2010.
- Jain, K., Padhye, J., Padmanabhan, V. N., Qiu, L. "Impact of Interference on Multi-hop Wireless Network Performance." ACM MobiCom 2003. (conflict graph)
- Bianchi, G. "Performance Analysis of the IEEE 802.11 Distributed Coordination Function." IEEE JSAC 18(3), 2000.
- Heusse, M. et al. "Performance Anomaly of 802.11b." IEEE INFOCOM 2003.
- Hiertz, G. R. et al. "IEEE 802.11s: The WLAN Mesh Standard." IEEE Wireless Communications 17(1), 2010.
- Shen, Y., Shi, Y., Zhang, J., Letaief, K. B. "Graph Neural Networks for Scalable Radio Resource Management." IEEE JSAC 39(1), 2021.
- (ควรตรวจรายละเอียด) Nakashima, K. et al. "Deep Reinforcement Learning-Based Channel Allocation for Wireless LANs With Graph Convolutional Networks." IEEE Access, 2020.
- (ควรตรวจ venue และปี) Almasan, P. et al. "Deep Reinforcement Learning Meets Graph Neural Networks: Exploring a Routing Optimization Use Case." Computer Communications, ~2022.

## คำแนะนำโดยรวม: **Major Revision**
งานนี้มีความโปร่งใสด้านระเบียบวิธีสูงกว่าโครงงานระดับปริญญาตรีทั่วไป แต่ในมุมของโดเมน ข้อสรุปหลักว่า "ดีกว่า Ko ≥10%" ยังพึ่งพาสามสิ่ง ได้แก่ (1) Ko ที่ถูกดัดแปลงจนสลับบ่อยและไม่ได้รับ hysteresis (2) แบบจำลองการสูญเสียคิวที่ลงโทษการสลับเกินจริง และ (3) MAC แบบ 1/k ที่ยังไม่ได้ calibrate ส่วนคู่เทียบที่เรียนรู้มาจากโดเมนอื่น และไม่มีคู่เทียบมาตรฐานของ WMN ถ้าเพิ่ม Ko+H ให้ queue loss สมจริงขึ้น ทำ calibration ขั้นต่ำกับ ns-3 และเปลี่ยนแหล่งอ้างอิงเว็บเป็นงานวิชาการ งานนี้จะมีสถานะเหมาะสำหรับเวทีระดับชาติ
