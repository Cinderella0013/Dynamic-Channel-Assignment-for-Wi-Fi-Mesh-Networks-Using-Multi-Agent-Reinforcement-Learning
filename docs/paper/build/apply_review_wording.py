# -*- coding: utf-8 -*-
"""One-off: wording fixes from the simulated review (docs/paper/review/editorial_decision.md, plan items 1-7 and minors).
Edits thesis_content.py and paper_content.py in place; every target must be found exactly once."""
import re
from pathlib import Path

HERE = Path(__file__).parent
GLOBAL = [  # (old, new) applied to both files, longest first
    ("ที่ลงทะเบียนไว้ล่วงหน้า", "ที่กำหนดไว้ล่วงหน้า"),
    ("ที่ลงทะเบียนล่วงหน้า", "ที่กำหนดไว้ล่วงหน้า"),
    ("ที่ลงทะเบียนไว้ก่อน", "ที่กำหนดไว้ก่อน"),
    ("ที่ลงทะเบียนไว้", "ที่กำหนดไว้"),
    ("ตามรูปแบบที่ตีพิมพ์", "ที่ดัดแปลงจากงานเผยแพร่"),
    ("observation, action, reward และงบการฝึกเดียวกัน", "observation, action, รูปแบบ reward และงบการฝึกเดียวกัน"),
]


def set_par(s, prefix, new):
    """Replace the whole ("p", "...") string whose text starts with prefix."""
    pat = re.compile(r'\("p", "' + re.escape(prefix) + r'[^"]*"\)')
    hits = pat.findall(s)
    assert len(hits) == 1, (prefix, len(hits))
    return pat.sub(lambda m: f'("p", "{new}")', s)


def sub1(s, old, new):
    assert s.count(old) == 1, (old[:50], s.count(old))
    return s.replace(old, new)


def thesis(s):
    s = set_par(s, "EGAT-MAPPO ดีกว่าคู่เทียบทุกวิธีที่ดัดแปลงจากงานเผยแพร่",
        "ในสภาวะหลัก (setting a) บน topology ทดสอบ EGAT-MAPPO มีคะแนนรวมเฉลี่ยสูงกว่าคู่เทียบที่ดัดแปลงจากงานเผยแพร่ทุกวิธีเกิน 10% (Ko +19.5%, Li +16.0%, Yuan +25.6%) ดีกว่าอย่างมีนัยสำคัญทุกวิธี และผ่านเกณฑ์ที่กำหนดไว้ทั้ง 4 ข้อ บน topology ใหม่ที่ไม่เคยใช้ตัดสินใจยังดีกว่า Ko +15.1% และไม่แพ้ Ko ในสถานการณ์ใด แต่เกณฑ์ความทนต่อการปรับน้ำหนักไม่ผ่าน ข้อได้เปรียบเหนือ Ko ส่วนใหญ่มาจากการสลับที่น้อยลง (43 เทียบ 580 ครั้งต่อ episode) ขณะที่ throughput สูงกว่า 6.5% เมื่อให้ Li และ Yuan ใช้ hysteresis เช่นเดียวกัน ส่วนต่างเหลือ 4–8% และเทียบ Li ที่มี hysteresis ไม่มีนัยสำคัญบน topology ใหม่ ข้อจำกัดสำคัญคือยังไม่ได้ยืนยันบน ns-3 ยังไม่ได้เทียบกับ Ko ที่มี hysteresis และเมื่อเก็บคิวระหว่างสลับได้ยังแพ้ Ko บางสถานการณ์")
    s = set_par(s, "EGAT-MAPPO improves on every baseline",
        "In the main setting on the test topologies, EGAT-MAPPO's mean composite exceeds every adapted published baseline by more than 10% (Ko +19.5%, Li +16.0%, Yuan +25.6%); every difference is significantly above zero, and all four criteria fixed before the main experiments are met. On fresh topologies never used for any decision it still improves on Ko by 15.1% and loses no scenario to Ko, but the weight-robustness criterion fails. Most of the gain over Ko comes from fewer switches (43 versus 580 per episode); throughput rises 6.5%. When Li and Yuan receive the same hysteresis, the margin falls to 4-8%, and against Li with hysteresis it is not significant on fresh topologies. The results are not yet validated in ns-3, Ko has not been tested with hysteresis, and with queue-preserving switches the model still loses to Ko in some scenarios.")
    s = sub1(s, "(5)  ด้านตัวชี้วัด ใช้คะแนนรวมห้าตัวชี้วัดและเกณฑ์ความสำเร็จ 4 ข้อที่กำหนดไว้ก่อนการทดลองหลัก การเปลี่ยนแปลงหลังเห็นผลทุกครั้งบันทึกพร้อมวันที่ และยืนยันด้วยชุด topology ใหม่",
        "(5)  ด้านตัวชี้วัด ใช้คะแนนรวมห้าตัวชี้วัดและเกณฑ์ความสำเร็จ 4 ข้อที่กำหนดไว้ก่อนการทดลองหลัก การเปลี่ยนแปลงหลังเห็นผลทุกครั้งบันทึกพร้อมวันที่ (รายละเอียดในหัวข้อ 3.4) และยืนยันด้วยชุด topology ใหม่")
    s = sub1(s, "และบันทึกการดัดแปลงทุกจุด\"),",
        "และบันทึกการดัดแปลงทุกจุด ความต่างเดียวด้าน reward คือ EGAT-MAPPO ปรับน้ำหนัก λ ระหว่างฝึก (หัวข้อ 4.1.4)\"),")
    s = set_par(s, "การเปิดเผยการตัดสินใจหลังเห็นผล",
        "การเปิดเผยการตัดสินใจหลังเห็นผล: เอกสารข้อกำหนดของการทดลองถูกแก้ไข 7 ครั้งระหว่างการทำงาน ทุกครั้งบันทึกพร้อมวันที่ ได้แก่ การเพิ่มต้นทุนการสลับในสภาพแวดล้อมจำลองหลังการทดลองนำร่องพบว่าแทบไม่มีช่องให้ดีกว่า Ko, การเพิ่ม switch hysteresis และกติกาเลือก margin และการเปลี่ยนวิธีคำนวณช่วงความเชื่อมั่น ซึ่งสามข้อหลังเกิดหลังการประเมินชุดทดสอบครั้งแรก นอกจากนี้ได้ประเมินแบบจำลองหลายรุ่น (EGAT, v1, v2 ทั้งที่มีและไม่มี hysteresis) บน topology ทดสอบ 0–19 แล้วรายงานรุ่นที่ดีที่สุด margin เลือกจาก topology validation (ค่าเมล็ดสุ่ม 1000–1003) เท่านั้น แล้วรันชุดทดสอบครั้งเดียว และเพื่อยืนยัน ได้ประเมินซ้ำบน topology ใหม่ (ค่าเมล็ดสุ่ม 20–39) ที่ไม่เคยใช้ตัดสินใจใด ผลบน topology ใหม่จึงเป็นผลที่ควรใช้อ้างอิงมากที่สุด")
    s = set_par(s, "การทดสอบรายสถานการณ์ใช้ Wilcoxon",
        "การทดสอบรายสถานการณ์ใช้ Wilcoxon signed-rank และ paired bootstrap ทั้งสองปรับด้วย Holm ข้ามสถานการณ์ การทดสอบนี้ถือว่าแต่ละ run เป็นอิสระและไม่นับความแปรปรวนของการฝึก จึงอาจประเมินนัยสำคัญสูงกว่าความจริง สำหรับค่าเฉลี่ยข้ามสถานการณ์ จึงคำนวณคะแนนรวมของทุกคู่ (ค่าเมล็ดสุ่มการฝึกของเรา, ของคู่เทียบ) แล้วรายงานค่าเฉลี่ยพร้อมช่วงความเชื่อมั่น 95% แบบ t ที่รวมความแปรปรวนระหว่างค่าเมล็ดสุ่มของทั้งสองวิธี (Welch–Satterthwaite) ช่วงนี้ยังไม่นับความแปรปรวนจากการสุ่ม topology ทดสอบ")
    s = set_par(s, "ใน ablation ความเป็นธรรม ได้ใส่ hysteresis",
        "ใน ablation ความเป็นธรรม ได้ใส่ hysteresis ให้ Li และ Yuan ด้วยกติกาเลือก margin เดียวกัน (Li+H, Yuan+H) โดยใช้คู่เทียบเดิม ไม่ใช้ EGAT-MAPPO เป็นตัวอ้างอิงในการเลือก grid ของ margin ต่างกันตามวิธี คือ {0, 0.5, 1, 2, 3, 4} สำหรับ EGAT-MAPPO และ {0, 0.25, 0.5, 1, 2, 4, 8, 16} สำหรับ Li และ Yuan ซึ่ง Li ได้ margin 16 ที่ขอบของ grid ค่าที่ดีที่สุดของ Li จึงอาจสูงกว่านี้ ส่วน Ko ยังไม่ได้ทดสอบแบบมี hysteresis")
    s = set_par(s, "EGAT+H ผ่านเกณฑ์ที่กำหนดไว้ครบ 4 ข้อ",
        "ใน setting (a) บน topology 0–19 EGAT+H ผ่านเกณฑ์ที่กำหนดไว้ครบ 4 ข้อเทียบทุกคู่เทียบ (1) คะแนนรวมเฉลี่ยเกิน +10% โดยต่ำสุดคือ Ko +19.5% (2) ดีกว่าอย่างมีนัยสำคัญอย่างน้อย 91% ของสถานการณ์ (3) ดีกว่า Ko ทุกสถานการณ์ ต่ำสุดคือ S1 +0.125 และ (4) ยังเกิน +10% เมื่อปรับน้ำหนักทุกแบบ ช่วงความเชื่อมั่นทุกตัวไม่คร่อม 0 คือดีกว่าอย่างมีนัยสำคัญ แต่ขอบล่างของบางคู่ต่ำกว่า 10% เช่น เทียบ Li ขอบล่างคือ +6.1% จึงยืนยันได้ว่าดีกว่า แต่ยังยืนยันไม่ได้ว่าดีกว่าเกิน 10% ช่วงเทียบ Li กว้างเพราะ Li แปรปรวนมากระหว่างค่าเมล็ดสุ่มการฝึก (+0.025 ถึง +0.289 ราย seed) ผลแยกตามชุด topology และ setting สรุปในตารางที่ 12")
    s = sub1(s, "ข้อได้เปรียบเหนือสองวิธีนี้จึงไม่ได้มาจากการสลับน้อยเพียงอย่างเดียว",
        "ข้อได้เปรียบเหนือสองวิธีนี้จึงไม่ได้มาจากการสลับน้อยเพียงอย่างเดียว แต่เทียบ Ko ส่วนที่ไม่ใช่ความเสถียรรวมกันเหลือเพียง +5.8% ผลรวมของส่วนประกอบในตารางนี้อาจต่างจากตารางที่ 6 ได้ ±0.1% จากการปัดเศษ")
    s = sub1(s, "ส่วนที่มาจากสถาปัตยกรรมและการฝึกของ EGAT-MAPPO อยู่ที่ราว 6–8% และปรากฏในอัตราสูญเสียและ throughput",
        "ส่วนที่เหลือราว 6–8% บน topology ทดสอบ และ 4–7% บน topology ใหม่ ปรากฏในอัตราสูญเสียและ throughput แต่ผลการตัดองค์ประกอบ (หัวข้อ 4.4.6) ชี้ว่าส่วนนี้อาจมาจากความรู้ของ Ko ที่ให้กับ EGAT-MAPPO มากกว่าจากตัวสถาปัตยกรรม")
    s = set_par(s, "การตัดคุณลักษณะของขอบ Ko prior หรือ BC ทำให้คะแนนรวมลดลง",
        "การตัดคุณลักษณะของขอบ Ko prior หรือ BC ทำให้คะแนนรวมเฉลี่ยลดลง 6–7% ส่วน GRU ไม่มีผลที่วัดได้ ด้วย 3 ค่าเมล็ดสุ่มต่อแบบ ช่วงความเชื่อมั่นของทุกความต่างคร่อม 0 จึงสรุปได้เพียงว่าเป็นแนวโน้ม ยังไม่ยืนยันว่าองค์ประกอบใดช่วยจริง ข้อสังเกตคือเมื่อตัด Ko prior หรือ BC ข้อได้เปรียบเหนือ Li+H เหลือใกล้ 0 (+0.0% และ +0.8%) จึงเป็นไปได้ว่าส่วนต่างที่เหลือเหนือ Li+H มาจากความรู้ของ Ko ที่ให้กับ EGAT-MAPPO เท่านั้น setting (b) ให้ผลไปทางเดียวกัน")
    s = set_par(s, "ตามวัตถุประสงค์ 1.2.3",
        "ตามวัตถุประสงค์ 1.2.3 ใน setting (a) บน topology ทดสอบ EGAT-MAPPO มีคะแนนรวมเฉลี่ยสูงกว่าคู่เทียบที่ดัดแปลงจากงานเผยแพร่ทุกวิธีเกิน 10% (Ko +19.5%, Li +16.0%, Yuan +25.6%, M0 +36.8%, Q-neutral +73.2%) ดีกว่าอย่างมีนัยสำคัญทุกวิธี และผ่านเกณฑ์ที่กำหนดไว้ครบ 4 ข้อ บน topology ใหม่ค่าเฉลี่ยยังเกิน 10% ทุกวิธี (Ko +15.1%) และไม่แพ้ Ko ในสถานการณ์ใด แต่เกณฑ์ข้อ 4 ไม่ผ่านเทียบ Ko และใน setting (b) ไม่ผ่านเกณฑ์ข้อ 3–4 (ตารางที่ 12) เครือข่าย 50 และ 80 โหนดที่ไม่เคยใช้ฝึกได้ผลดีขึ้นอีก (+41–42% เทียบ Ko)")
    s = set_par(s, "ตามวัตถุประสงค์ 1.2.4",
        "ตามวัตถุประสงค์ 1.2.4 ข้อได้เปรียบเหนือ Ko ส่วนใหญ่มาจากการสลับที่น้อยลง (ส่วนที่ไม่ใช่ความเสถียรเหลือ +5.8%) และเมื่อให้ Li และ Yuan ใช้ hysteresis เดียวกัน ส่วนต่างเหลือ 6–8% บน topology ทดสอบ และ 4–7% บน topology ใหม่ ซึ่งเทียบ Li+H ไม่มีนัยสำคัญ การตัดองค์ประกอบให้แนวโน้มว่าคุณลักษณะของขอบ Ko prior และ BC อาจช่วย ส่วน GRU ไม่ช่วย แต่ยังไม่มีนัยสำคัญ")
    s = set_par(s, "Hysteresis กติกาเลือก margin และวิธีคำนวณช่วงความเชื่อมั่นถูกเพิ่ม",
        "Hysteresis กติกาเลือก margin และวิธีคำนวณช่วงความเชื่อมั่นถูกเพิ่มหลังเห็นผลทดสอบครั้งแรก และได้ประเมินหลายรุ่นบน topology 0–19 ก่อนเลือกรุ่นที่รายงาน แม้ margin จะเลือกบน validation เท่านั้น บันทึกทุกการเปลี่ยนแปลง และยืนยันบน topology ใหม่แล้ว ผลบน topology ใหม่ก็เล็กกว่าเดิม (Ko +15.1% จาก +19.5%) ซึ่งบ่งว่าผลชุดแรกสูงกว่าความจริงบางส่วน")
    # new limitation: Ko without hysteresis
    s = sub1(s, '    ("h2", "5.3  ปัญหาอุปสรรคและแนวทางแก้ไข"),',
        '    ("h3", "5.2.8  คู่เทียบ Ko ที่ไม่มี hysteresis"),\n'
        '    ("p", "Ko ที่ดัดแปลงมาเลือกช่องตามภาระปัจจุบันซึ่งเปลี่ยนทุก 100 ms และไม่มี hysteresis ของช่อง จึงสลับ 580 ครั้งต่อ episode ต่างจาก Ko ต้นฉบับที่ออกแบบให้ลู่เข้าแล้วหยุดสลับ ขณะที่งานนี้ให้ hysteresis กับ EGAT-MAPPO, Li และ Yuan แต่ไม่ได้ให้กับ Ko ส่วนต่างเหนือ Ko ที่มาจากความเสถียรจึงอาจสูงกว่าที่ Ko ที่ปรับให้ดีที่สุดจะให้ งานต่อไปคือทดสอบ Ko ที่มี hysteresis ด้วยกติกาเลือก margin เดียวกัน"),\n'
        '    ("h2", "5.3  ปัญหาอุปสรรคและแนวทางแก้ไข"),')
    # criteria table, end of 4.5
    s = sub1(s, '    ("h2", "4.6  ผลระยะที่ 1: GAT+RAES บน native ns-3 แบบ single-hop"),',
        '    ("p", "ตารางที่ 12 สรุปผลตามเกณฑ์ 4 ข้อแยกตาม setting และชุด topology เทียบคู่เทียบที่ดัดแปลงจากงานเผยแพร่ เกณฑ์ผ่านครบเฉพาะ setting (a) บน topology 0–19"),\n'
        '    ("ct", "ผลตามเกณฑ์ที่กำหนดไว้ แยกตาม setting และชุด topology (เทียบคู่เทียบที่ดัดแปลงจากงานเผยแพร่)"),\n'
        '    ("tbl", [["Setting / topology", "(1) เฉลี่ย ≥ +10%", "(2) นัยสำคัญ ≥ 60%", "(3) ไม่แพ้ Ko เกิน 5%", "(4) ทนการปรับน้ำหนัก"],\n'
        '             ["(a) / 0–19", "ผ่าน (Ko +19.5%)", "ผ่าน (≥ 91%)", "ผ่าน (ต่ำสุด S1 +12.5%)", "ผ่าน (Ko ต่ำสุด +13.6%)"],\n'
        '             ["(a) / ใหม่ 20–39", "ผ่าน (Ko +15.1%)", "ผ่าน (≥ 91%)", "ผ่าน (ต่ำสุด S4 +6.5%)", "ไม่ผ่าน (Ko ต่ำสุด +8.9%)"],\n'
        '             ["(b) / 0–19", "ผ่าน (Ko +14.4%)", "ผ่าน (Ko 91% / 82%)", "ไม่ผ่าน (B4 −8.4%)", "ไม่ผ่าน (Ko ต่ำสุด +7.9%)"],\n'
        '             ["(b) / ใหม่ 20–39", "ไม่ผ่าน (Ko +9.5%)", "ผ่าน (Ko 73% / 64%)", "ไม่ผ่าน (S4 −5.5%, B4 −7.4%)", "ไม่ผ่าน (Ko ต่ำสุด +2.2%)"]],\n'
        '     [1826, 1700, 1700, 2000, 1800]),\n'
        '    ("h2", "4.6  ผลระยะที่ 1: GAT+RAES บน native ns-3 แบบ single-hop"),')
    s = sub1(s, "ผลหลักแสดงในตารางที่ 12", "ผลหลักแสดงในตารางที่ 13")
    s = sub1(s, "ระยะที่ 1 พัฒนา GAT+RAES และรุ่น Hybrid ที่เพิ่ม conflict mask",
        "ระยะที่ 1 พัฒนา GAT+RAES (GAT ร่วมกับ Risk-Aware Execution Selector ซึ่งเลือกช่องตามลำดับโดยถ่วงค่า Q กับความเสี่ยงการรบกวนและต้นทุนการสลับ) และรุ่น Hybrid ที่เพิ่ม conflict mask")
    s = sub1(s, "Hybrid ได้ satisfaction 61.773%",
        "ตัวชี้วัดหลักคือ satisfaction (SAT) ซึ่งเท่ากับทราฟฟิกที่ส่งได้หารด้วยความต้องการราย flow Hybrid ได้ SAT 61.773%")
    s = sub1(s, "สูงกว่า GA-Net และ DGN adaptations ทุก 50 กรณีที่จับคู่ และสูงกว่า BFS-CA-inspired",
        "สูงกว่า GA-Net [16] และ DGN [24] adaptations ทุก 50 กรณีที่จับคู่ และสูงกว่า BFS-CA-inspired [25]")
    s = sub1(s, "สเกล log) ของแต่ละวิธี เฉลี่ย A+B setting (a)\")",
        "สเกล log) ของแต่ละวิธี เฉลี่ย A+B setting (a) แกน y ไม่ได้เริ่มที่ 0\")")
    # classic mesh literature in the chapter-2 synthesis
    s = sub1(s, "งานวิจัยที่ทบทวนชี้ว่า MARL ช่วยจัดสรรช่องสัญญาณได้ดีกว่าวิธีดั้งเดิมในหลายสภาวะ",
        "งานด้าน mesh หลายวิทยุแบบดั้งเดิมศึกษาการจัดสรรช่องร่วมกับการเลือกเส้นทางมานาน เช่น Hyacinth ของ Raniwala และ Chiueh [35], การจัดสรรช่องและเส้นทางร่วมกันเพื่อ throughput ของ Alicherry และคณะ [36], การวิเคราะห์ขีดความสามารถของ Kodialam และ Nandagopal [37] และตัวชี้วัดเส้นทาง WCETT ของ Draves และคณะ [38] งานเหล่านี้ใช้การหาค่าเหมาะสมหรือ heuristic จากภาพรวมของเครือข่าย งานวิจัยที่ทบทวนชี้ว่า MARL ช่วยจัดสรรช่องสัญญาณได้ดีกว่าวิธีดั้งเดิมในหลายสภาวะ")
    s = sub1(s, "DROP_REFS = {13, 15, 19, 20, 21, 22, 23, 24, 25}", "DROP_REFS = {12, 13, 15, 19, 20, 21, 22, 23}")
    s = sub1(s, '    "34.  C.  Yu,',
        '    "35.  A.  Raniwala and T.  Chiueh.  (2005).  \\"Architecture and Algorithms for an IEEE 802.11-Based Multi-Channel Wireless Mesh Network,\\" IEEE INFOCOM.",\n'
        '    "36.  M.  Alicherry,  R.  Bhatia,  and L.  Li.  (2005).  \\"Joint Channel Assignment and Routing for Throughput Optimization in Multi-radio Wireless Mesh Networks,\\" ACM MobiCom.",\n'
        '    "37.  M.  Kodialam and T.  Nandagopal.  (2005).  \\"Characterizing the Capacity Region in Multi-Radio Multi-Channel Wireless Mesh Networks,\\" ACM MobiCom.",\n'
        '    "38.  R.  Draves,  J.  Padhye,  and B.  Zill.  (2004).  \\"Routing in Multi-Radio,  Multi-Hop Wireless Mesh Networks,\\" ACM MobiCom.",\n'
        '    "34.  C.  Yu,')
    return s


def paper(s):
    s = set_par(s, "งานนี้เสนอ EGAT-MAPPO สำหรับ",
        "งานนี้เสนอ EGAT-MAPPO สำหรับการจัดสรรช่องสัญญาณร่วมกับการเลือกเส้นทางในเครือข่าย Wi-Fi Mesh แบบหลาย hop แต่ละโหนดเป็นตัวแทนที่เลือกช่อง downlink และโหนดแม่ของตนเอง แบบจำลองใช้ GATv2 ที่รับคุณลักษณะของขอบ หัวตัดสินใจสองช่วงเวลา และ residual prior จากวิธี local cost ของ Ko ฝึกด้วย behaviour cloning ตามด้วย MAPPO และใช้ switch hysteresis ขณะใช้งานแบบกระจายศูนย์ ประเมินบนสภาพแวดล้อมจำลอง 20 โหนด 6 ช่อง ใน 11 สถานการณ์ เทียบกับ M0, Ko, Q-neutral, Yuan (GAT+DQN) และ Li (GAT-A2C) ด้วยคะแนนรวมห้าตัวชี้วัดที่กำหนดไว้ล่วงหน้า ในสภาวะหลักบน topology ทดสอบ EGAT-MAPPO มีคะแนนรวมเฉลี่ยสูงกว่าทุกวิธีเกิน 10% (Ko +19.5%, Li +16.0%, Yuan +25.6%) อย่างมีนัยสำคัญ และบน topology ใหม่ยังสูงกว่า Ko +15.1% แต่ส่วนใหญ่ของข้อได้เปรียบเหนือ Ko มาจากการสลับที่น้อยลง (throughput +6.5%) เมื่อให้คู่เทียบใช้ hysteresis เดียวกัน ส่วนต่างเหลือ 4–8% และยังไม่ได้เทียบกับ Ko ที่มี hysteresis")
    s = set_par(s, "We propose EGAT-MAPPO",
        "We propose EGAT-MAPPO for joint channel assignment and routing in multi-hop Wi-Fi mesh networks. Every node is an agent that chooses its downlink channel and its parent. The model combines an edge-aware GATv2 encoder, two decision timescales and a residual prior from Ko's local-cost rule; it is warm-started by behaviour cloning, trained with MAPPO and executed in a decentralised way with switch hysteresis. In a 20-node, six-channel simulator over 11 scenarios, we compare it with M0, Ko, Q-neutral, Yuan (GAT+DQN) and Li (GAT-A2C) on a five-metric composite fixed before the main experiments. In the main setting on the test topologies, its mean composite exceeds every adapted baseline by more than 10% (Ko +19.5%, Li +16.0%, Yuan +25.6%), significantly above zero; on fresh topologies it still improves on Ko by 15.1%. Most of the gain over Ko comes from fewer switches (throughput +6.5%); with the same hysteresis for the baselines the margin falls to 4-8%, and Ko has not yet been tested with hysteresis.")
    s = set_par(s, "ตาราง 3 และภาพ 2 แสดงว่า",
        "ตาราง 3 และภาพ 2 แสดงว่าบน topology ทดสอบ EGAT+H มีคะแนนรวมเฉลี่ยสูงกว่าคู่เทียบที่ดัดแปลงจากงานเผยแพร่ทุกวิธีเกิน 10% ช่วงความเชื่อมั่นทุกตัวไม่คร่อม 0 แต่ขอบล่างของบางคู่ต่ำกว่า 10% (Li +6.1%) ดีกว่าอย่างมีนัยสำคัญอย่างน้อย 91% ของสถานการณ์ ดีกว่า Ko ทุกสถานการณ์ (ต่ำสุด S1 +0.125) และยังเกิน 10% เมื่อปรับน้ำหนักทุกแบบ จึงผ่านเกณฑ์ครบ 4 ข้อเฉพาะใน setting (a) บน topology 0–19")
    s = sub1(s, "ค่าจริงแสดงในตาราง 5 EGAT+H มี throughput",
        "ถ้าไม่นับความเสถียร ส่วนต่างเหนือ Ko เหลือ +5.8% ค่าจริงแสดงในตาราง 5 EGAT+H มี throughput")
    s = sub1(s, "ไม่ถึง 10% ข้อได้เปรียบที่เหลือมาจากอัตราสูญเสีย p5 และ throughput",
        "ไม่ถึง 10% ข้อได้เปรียบที่เหลือมาจากอัตราสูญเสีย p5 และ throughput Li+H ได้ margin 16 ที่ขอบของ grid จึงอาจยังไม่ใช่ค่าที่ดีที่สุดของ Li")
    s = sub1(s, "การตัดคุณลักษณะของขอบ Ko prior หรือ BC ลดคะแนน 6–7% และทำให้ข้อได้เปรียบเหนือ Li+H หายเกือบหมด ส่วน GRU ไม่มีผล ความต่างยังไม่มีนัยสำคัญจึงรายงานเป็นแนวโน้ม",
        "การตัดคุณลักษณะของขอบ Ko prior หรือ BC ลดคะแนน 6–7% ส่วน GRU ไม่มีผล ช่วงความเชื่อมั่นคร่อม 0 ทุกตัวจึงเป็นเพียงแนวโน้ม และเมื่อตัด Ko prior หรือ BC ข้อได้เปรียบเหนือ Li+H เหลือใกล้ 0 ส่วนต่างที่เหลือจึงอาจมาจากความรู้ของ Ko มากกว่าตัวสถาปัตยกรรม")
    s = sub1(s, "และ (5) คู่เทียบ Yuan และ Li ต้องดัดแปลงจากโจทย์ต้นฉบับ",
        "(5) คู่เทียบ Yuan และ Li ต้องดัดแปลงจากโจทย์ต้นฉบับ (6) ยังไม่ได้เทียบกับ Ko ที่มี hysteresis ซึ่ง Ko ที่ดัดแปลงมาสลับ 580 ครั้งต่อ episode และ (7) ได้ประเมินหลายรุ่นบน topology 0–19 ก่อนเลือกรุ่นที่รายงาน")
    s = sub1(s, "EGAT-MAPPO ตัดสินใจช่องและเส้นทางร่วมกันแบบกระจายศูนย์ และดีกว่าคู่เทียบที่ดัดแปลงจากงานเผยแพร่ทุกวิธีเกิน 10% อย่างมีนัยสำคัญ ทั้งบน topology ทดสอบและ topology ใหม่",
        "EGAT-MAPPO ตัดสินใจช่องและเส้นทางร่วมกันแบบกระจายศูนย์ และมีคะแนนรวมเฉลี่ยสูงกว่าคู่เทียบที่ดัดแปลงจากงานเผยแพร่ทุกวิธีเกิน 10% โดยดีกว่าอย่างมีนัยสำคัญ ทั้งบน topology ทดสอบและ topology ใหม่")
    s = sub1(s, "งานนี้ใช้ GATv2 [5]",
        "งาน mesh หลายวิทยุแบบดั้งเดิม เช่น [11]–[14] จัดสรรช่องร่วมกับเส้นทางด้วยการหาค่าเหมาะสมหรือ heuristic จากภาพรวม งานนี้ใช้ GATv2 [5]")
    s = sub1(s, 'manual/html/index.html",\n]',
        'manual/html/index.html",\n'
        '    "[11] A. Raniwala and T. Chiueh, “Architecture and Algorithms for an IEEE 802.11-Based Multi-Channel Wireless Mesh Network,” IEEE INFOCOM, 2005.",\n'
        '    "[12] M. Alicherry, R. Bhatia and L. Li, “Joint Channel Assignment and Routing for Throughput Optimization in Multi-radio Wireless Mesh Networks,” ACM MobiCom, 2005.",\n'
        '    "[13] M. Kodialam and T. Nandagopal, “Characterizing the Capacity Region in Multi-Radio Multi-Channel Wireless Mesh Networks,” ACM MobiCom, 2005.",\n'
        '    "[14] R. Draves, J. Padhye and B. Zill, “Routing in Multi-Radio, Multi-Hop Wireless Mesh Networks,” ACM MobiCom, 2004.",\n]')
    return s


if __name__ == "__main__":
    import sys  # optional argument: "thesis" or "paper" to apply one file only
    todo = (("thesis_content.py", thesis), ("paper_content.py", paper))
    for name, fn in [t for t in todo if len(sys.argv) < 2 or t[0].startswith(sys.argv[1])]:
        p = HERE / name
        s = p.read_text(encoding="utf-8")
        for old, new in GLOBAL:
            s = s.replace(old, new)
        p.write_text(fn(s), encoding="utf-8")
        print("updated", name)
