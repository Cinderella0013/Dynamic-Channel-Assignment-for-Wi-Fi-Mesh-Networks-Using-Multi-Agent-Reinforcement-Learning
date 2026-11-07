// Build the re-presentation deck: node docs/paper/build/build_slides.js OUT.pptx
// Needs pptxgenjs (npm). Numbers come from docs/paper/build/thesis_content.py (v10, EGAT without GRU, margin 2).
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2] || "NW14_Presentation_v10.pptx";
const FIG = path.join(__dirname, "..", "figures");
const F = "TH Sarabun New";
const TEAL = "0E8A8F", TINT = "E4F2F2", INK = "1F2933", MUTED = "5B6670", AMBER = "B86E00", AMBERT = "FCF0DC", LINE = "D5E3E3";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: F, bodyFontFace: F };
pres.title = "EGAT-MAPPO: ผลรอบใหม่";
pres.author = "NW14";

const TOTAL = 13;
let n = 0;
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { rect: { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: TEAL } } },
    { text: { text: "NW14   EGAT-MAPPO", options: { x: 0.45, y: 5.22, w: 4, h: 0.3, fontFace: F, fontSize: 12, color: MUTED, margin: 0 } } },
  ],
  placeholders: [],
});

function slide(title, kicker) {
  n += 1;
  const s = pres.addSlide({ masterName: "CONTENT" });
  if (kicker) s.addText(kicker, { x: 0.45, y: 0.22, w: 9, h: 0.3, fontFace: F, fontSize: 14, color: MUTED, margin: 0, isTextBox: true });
  s.addText(title, { x: 0.45, y: 0.48, w: 9.1, h: 0.62, fontFace: F, fontSize: 32, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText(`${String(n).padStart(2, "0")} / ${String(TOTAL).padStart(2, "0")}`,
    { x: 8.3, y: 5.22, w: 1.3, h: 0.3, fontFace: F, fontSize: 12, color: MUTED, align: "right", margin: 0, isTextBox: true });
  return s;
}
const T = (s, text, o) => s.addText(text, Object.assign({ fontFace: F, color: INK, margin: 0, isTextBox: true, valign: "top" }, o));
function card(s, x, y, w, h, fill) { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || TINT }, line: { color: fill || TINT } }); }
function table(s, rows, o, hlRows) {
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    const opt = Object.assign({ fontFace: F, fontSize: 16, color: INK, valign: "middle", align: j === 0 ? "left" : "center",
      border: { type: "solid", pt: 0.75, color: LINE }, margin: [2, 6, 2, 6] }, cell.options || {});
    if (i === 0) Object.assign(opt, { bold: true, fill: { color: TINT } });
    if (hlRows && hlRows.includes(i)) Object.assign(opt, { fill: { color: AMBERT } });
    return { text: cell.text, options: opt };
  }));
  s.addTable(body, Object.assign({ x: 0.45, w: 9.1 }, o));
}
const ok = (t) => ({ text: t, options: { color: TEAL, bold: true } });
const no = (t) => ({ text: t, options: { color: AMBER, bold: true } });

// 1. Title
{
  n += 1;
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: TEAL }, line: { color: TEAL } });
  T(s, "NW14   EGAT-MAPPO", { x: 0.6, y: 0.4, w: 5, h: 0.3, fontSize: 14, color: MUTED });
  T(s, "การจัดสรรช่องสัญญาณแบบไดนามิกในเครือข่าย Wi-Fi Mesh\nโดยใช้เทคนิคการเรียนรู้แบบเสริมกำลังหลายตัวแทน",
    { x: 0.6, y: 0.9, w: 8.8, h: 1.25, fontSize: 32, bold: true, lineSpacingMultiple: 0.9 });
  T(s, "Dynamic Channel Assignment for Wi-Fi Mesh Networks Using Multi-Agent Reinforcement Learning",
    { x: 0.6, y: 2.2, w: 8.8, h: 0.35, fontSize: 16, color: MUTED });
  card(s, 0.6, 2.8, 8.8, 0.85, TINT);
  T(s, [{ text: "ผลรอบใหม่: ", options: { bold: true, color: TEAL } },
        { text: "ชนะคู่เทียบจากงานเผยแพร่อย่างมีนัยสำคัญทางสถิติ พร้อมข้อจำกัดที่ยังเหลือ" }],
    { x: 0.8, y: 2.9, w: 8.4, h: 0.65, fontSize: 22, valign: "middle" });
  T(s, "นายวิศรุต ไตรยวงศ์ 663380363-7\nอาจารย์ที่ปรึกษา ศ. ดร.จักรชัย โสอินทร์\nวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น",
    { x: 0.6, y: 3.95, w: 8.8, h: 1.1, fontSize: 18, color: MUTED, lineSpacingMultiple: 0.95 });
  s.addNotes("รอบนี้มาตอบข้อติจากรอบก่อนโดยตรง: ผลเทียบ Ko รอบก่อนต่างกันน้อยและไม่มีนัยสำคัญ รอบนี้ทดสอบใหม่ทั้งหมดด้วยโปรโตคอลที่กำหนดไว้ล่วงหน้า");
}

// 2. Old vs new
{
  const s = slide("ข้อติจากรอบก่อน และผลรอบนี้", "รอบก่อน 28 ตุลาคม 2569  →  รอบนี้");
  card(s, 0.45, 1.3, 4.4, 3.7, "F3F4F5");
  card(s, 5.15, 1.3, 4.4, 3.7, TINT);
  T(s, "รอบก่อน (GAT+RAES)", { x: 0.7, y: 1.45, w: 4, h: 0.4, fontSize: 20, bold: true, color: MUTED });
  T(s, "+0.893 จุด", { x: 0.7, y: 1.9, w: 4, h: 0.7, fontSize: 44, bold: true, color: MUTED });
  T(s, "SAT เหนือ Ko", { x: 0.7, y: 2.6, w: 4, h: 0.35, fontSize: 18, color: MUTED });
  T(s, [
    { text: "ชนะ Ko 6 จาก 12 คู่", options: { bullet: true, breakLine: true } },
    { text: "2 seed ที่เคยเห็นแล้ว ไม่มีช่วงความเชื่อมั่น", options: { bullet: true, breakLine: true } },
    { text: "single-hop 15 คู่ AP–STA ไม่ใช่ mesh จริง", options: { bullet: true, breakLine: true } },
    { text: "อาจเป็นความคลาดเคลื่อนของระบบ", options: { bullet: true, bold: true } },
  ], { x: 0.7, y: 3.05, w: 4, h: 1.8, fontSize: 18, color: MUTED, paraSpaceAfter: 2 });
  T(s, "รอบนี้ (EGAT-MAPPO)", { x: 5.4, y: 1.45, w: 4, h: 0.4, fontSize: 20, bold: true, color: TEAL });
  T(s, "+24.6%", { x: 5.4, y: 1.9, w: 4, h: 0.7, fontSize: 44, bold: true, color: TEAL });
  T(s, "คะแนนรวมเหนือ Ko  95% CI [+21.6, +27.6]", { x: 5.4, y: 2.6, w: 4, h: 0.35, fontSize: 18, color: INK });
  T(s, [
    { text: "ดีกว่า Ko ทุกสถานการณ์ (ต่ำสุด +13.2%)", options: { bullet: true, breakLine: true } },
    { text: "topology ยืนยันที่ล็อกเกณฑ์ไว้ก่อนรัน", options: { bullet: true, breakLine: true } },
    { text: "mesh หลาย hop เลือกช่องและเส้นทางร่วมกัน", options: { bullet: true, breakLine: true } },
    { text: "ชนะทุกวิธี รวม Yuan+H และบน ns-3", options: { bullet: true, bold: true, color: TEAL } },
  ], { x: 5.4, y: 3.05, w: 4, h: 1.8, fontSize: 18, paraSpaceAfter: 2 });
  s.addNotes("รอบก่อนส่วนต่างเหนือ Ko น้อยกว่า 1 จุด ชนะครึ่งเดียว และไม่มีสถิติรองรับ อาจารย์จึงมองว่าอาจเป็น noise รอบนี้ส่วนต่าง +24.6% บน topology ยืนยัน 40–59 ที่ไม่เคยใช้ และช่วงความเชื่อมั่น 95% อยู่เหนือ 0 ทั้งช่วง");
}

// 3. What changed
{
  const s = slide("สิ่งที่เปลี่ยนเพื่อตอบข้อติ", "ทดสอบใหม่ทั้งหมด ไม่ได้ปรับผลเดิม");
  const items = [
    ["01", "ปัญหาจริงขึ้น", "mesh หลาย hop 20 โหนด 2 gateway\nแต่ละโหนดเลือกช่องและเส้นทางร่วมกัน\nการสลับมีต้นทุนจริง"],
    ["02", "คู่เทียบจากงานเผยแพร่", "Ko (2007) heuristic แบบกระจาย\nYuan (2022) GAT + DQN\nLi (2025) GAT-A2C  และ M0, Q-neutral"],
    ["03", "โปรโตคอลกำหนดไว้ก่อน", "5 seed ฝึกต่อวิธี × 20 topology\nCI 95% แบบ seed-pair, เกณฑ์ผ่าน 4 ข้อ\nยืนยันบน topology ใหม่ 2 ชุด"],
    ["04", "ยืนยันนอก sim", "ns-3 802.11ax ระดับแพ็กเก็ต\nปรับเทียบ sim กับ ns-3\nความคลาด 23.5% → 7.4%"],
  ];
  items.forEach(([num, head, body], i) => {
    const x = 0.45 + i * 2.31;
    card(s, x, 1.35, 2.15, 3.6, TINT);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.55, w: 0.62, h: 0.62, fill: { color: TEAL }, line: { color: TEAL } });
    T(s, num, { x: x + 0.2, y: 1.55, w: 0.62, h: 0.62, fontSize: 20, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    T(s, head, { x: x + 0.2, y: 2.35, w: 1.85, h: 0.7, fontSize: 21, bold: true, lineSpacingMultiple: 0.9 });
    T(s, body, { x: x + 0.2, y: 3.1, w: 1.85, h: 1.75, fontSize: 16, color: INK, lineSpacingMultiple: 0.95 });
  });
  s.addNotes("สี่อย่างที่เปลี่ยน: (1) ปัญหาเป็น mesh จริง (2) คู่เทียบมาจากงานตีพิมพ์ (3) กำหนดวิธีวัดและเกณฑ์ผ่านไว้ก่อนดูผล (4) ยืนยันบน ns-3 ระดับแพ็กเก็ต");
}

// 4. Architecture
{
  const s = slide("EGAT-MAPPO: แบบจำลองที่เสนอ", "แต่ละโหนดตัดสินใจเองขณะใช้งาน ฝึกด้วย critic ส่วนกลาง");
  s.addImage({ path: path.join(FIG, "fig_architecture.png"), x: 0.45, y: 1.25, w: 7.4, h: 3.8 });
  T(s, [
    { text: "GATv2 รับคุณลักษณะของขอบ", options: { breakLine: true, bold: true } },
    { text: "SNR, อัตราส่ง, ภาระ, ack", options: { breakLine: true } },
    { text: "Ko prior + BC", options: { breakLine: true, bold: true } },
    { text: "เริ่มจากความรู้ของ Ko", options: { breakLine: true } },
    { text: "Hysteresis m = 2", options: { breakLine: true, bold: true } },
    { text: "สลับเมื่อดีกว่าเกิน margin", options: { breakLine: true } },
    { text: "ไม่มี GRU", options: { breakLine: true, bold: true } },
    { text: "ตัดออกตาม ablation", options: {} },
  ], { x: 7.95, y: 1.4, w: 1.65, h: 3.6, fontSize: 16, color: INK, paraSpaceAfter: 3 });
  s.addNotes("margin 2 เลือกบน validation ด้วยกติกา worst-case เทียบทุกคู่เทียบ ไม่ได้ดูชุดทดสอบ รุ่นนี้ไม่มี GRU เพราะ ablation พบว่า GRU ไม่ช่วย และยืนยันด้วยโปรโตคอลที่ล็อกไว้ก่อน (สไลด์ 11)");
}

// 5. Evaluation design
{
  const s = slide("การประเมิน: 16 สถานการณ์ 2 setting 3 ชุด topology", "วัด throughput, Jain, p5, อัตราสูญเสีย, ความเสถียร รวมเป็นคะแนนเดียว");
  const groups = [
    ["A  พื้นฐาน", "S1 ภาระต่ำ 2 Mb/s\nS2 ภาระสูง 10 Mb/s\nS3 hotspot\nS4 สัญญาณรบกวนภายนอก"],
    ["B  เปลี่ยนแปลง", "B1 hotspot เคลื่อนที่\nB2 ภาระเป็นคาบ\nB3 topology เบาบาง\nB4 grid ไม่เคยเห็นตอนฝึก"],
    ["C  ยาก / ขยาย", "C1 ภาระต่ำมาก\nC2 ทราฟฟิกคงที่\nC3 50 โหนด\nC4 80 โหนด (ฝึก 20 โหนด)"],
  ];
  groups.forEach(([h, b], i) => {
    const x = 0.45 + i * 3.08;
    card(s, x, 1.3, 2.9, 2.35, TINT);
    T(s, h, { x: x + 0.2, y: 1.42, w: 2.5, h: 0.42, fontSize: 21, bold: true, color: TEAL });
    T(s, b, { x: x + 0.2, y: 1.88, w: 2.6, h: 1.7, fontSize: 17 });
  });
  table(s, [
    ["", "ค่าที่ใช้"],
    ["Setting", "(a) ทิ้งคิวเมื่อสลับ   (b) เก็บคิวระหว่างสลับ"],
    ["Topology", "ทดสอบ 0–19,  ใหม่ 20–39,  ยืนยัน 40–59 (ล็อกเกณฑ์ไว้ก่อนรัน)"],
    ["สถิติ", "5 seed ฝึกต่อวิธี, ค่าเฉลี่ยคู่ seed, CI 95%, Wilcoxon / bootstrap รายสถานการณ์"],
  ], { y: 3.85, colW: [1.5, 7.6], rowH: 0.3, fontSize: 15 });
  s.addNotes("ผลหลักใช้ A+B ส่วน C เป็นการทดสอบขยาย คะแนนรวมเป็นค่าสัมพัทธ์เทียบคู่เทียบ +0.246 อ่านว่าดีกว่า 24.6%");
}

// 6. Headline
{
  const s = slide("ผลหลัก: ชนะ Ko อย่างมีนัยสำคัญ", "setting (a), กลุ่ม A+B, topology ยืนยัน 40–59, ค่าเฉลี่ยคู่ seed และ CI 95%");
  const stats = [
    ["+24.6%", "เหนือ Ko", "CI [+21.6, +27.6]"],
    ["+6.7%", "เหนือ Yuan+H (รองลงมา)", "CI [+3.1, +10.2]"],
    ["4 / 4", "เกณฑ์ที่กำหนดไว้ก่อน", "ผ่านทุก setting และ topology"],
  ];
  stats.forEach(([big, lab, sub], i) => {
    const x = 0.45 + i * 3.08;
    card(s, x, 1.35, 2.9, 2.3, TINT);
    T(s, big, { x: x + 0.2, y: 1.5, w: 2.6, h: 0.95, fontSize: 54, bold: true, color: TEAL });
    T(s, lab, { x: x + 0.2, y: 2.5, w: 2.6, h: 0.4, fontSize: 19, bold: true });
    T(s, sub, { x: x + 0.2, y: 2.92, w: 2.6, h: 0.4, fontSize: 16, color: MUTED });
  });
  T(s, [
    { text: "ไม่มีสถานการณ์ใดแพ้ Ko ", options: { bold: true } },
    { text: "ทั้งสอง setting และทั้งสามชุด topology (ต่ำสุด S1 ภาระต่ำ +13.2%)", options: { breakLine: true } },
    { text: "เครือข่ายใหญ่ที่ไม่เคยฝึก ", options: { bold: true } },
    { text: "50 และ 80 โหนด ดีกว่า Ko +47%", options: { breakLine: true } },
    { text: "เทียบรอบก่อน ", options: { bold: true } },
    { text: "+0.9 จุด ชนะ 6/12 คู่ ไม่มี CI", options: { color: MUTED } },
  ], { x: 0.45, y: 3.9, w: 9.1, h: 1.2, fontSize: 18, paraSpaceAfter: 2 });
  s.addNotes("topology 40–59 ไม่เคยถูกใช้มาก่อน และเกณฑ์ตัดสินเขียนไว้ก่อนรัน ผลใกล้เคียงชุดทดสอบ (Ko +26.0%) แปลว่าไม่ได้ overfit กับชุดทดสอบ");
}

// 7. vs published baselines (native chart)
{
  const s = slide("เทียบคู่เทียบจากงานเผยแพร่ทุกวิธี", "คะแนนรวมที่ EGAT+H ดีกว่า (%), setting (a), A+B");
  s.addChart(pres.charts.BAR, [
    { name: "topology ทดสอบ 0–19", labels: ["Ko", "Li", "Yuan", "M0"], values: [26.0, 20.3, 24.1, 31.1] },
    { name: "topology ยืนยัน 40–59", labels: ["Ko", "Li", "Yuan", "M0"], values: [24.6, 17.9, 24.8, 31.6] },
  ], {
    x: 0.45, y: 1.2, w: 5.2, h: 3.9, barDir: "col", barGapWidthPct: 60, chartColors: [TEAL, "7FC4C6"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: F, dataLabelFontSize: 14, dataLabelColor: INK,
    dataLabelFormatCode: "0.0",
    catAxisLabelFontFace: F, catAxisLabelFontSize: 16, catAxisLabelColor: INK,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 12, valAxisLabelColor: MUTED, valAxisMinVal: 0, valAxisMaxVal: 40,
    valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontFace: F, legendFontSize: 14, legendColor: INK,
  });
  table(s, [
    ["", "ทดสอบ 95% CI", "ยืนยัน 95% CI"],
    ["Ko", ok("[+23.0, +29.0]"), ok("[+21.6, +27.6]")],
    ["Li", ok("[+6.9, +33.7]"), ok("[+3.3, +32.4]")],
    ["Yuan", ok("[+20.1, +28.2]"), ok("[+21.4, +28.2]")],
    ["M0", ok("[+29.1, +33.1]"), ok("[+30.0, +33.2]")],
    ["Q-neutral", ok("+73.6%"), ok("+73.7%")],
  ], { x: 5.85, w: 3.7, y: 1.3, colW: [0.95, 1.375, 1.375], rowH: 0.36, fontSize: 15 });
  T(s, "ช่วงความเชื่อมั่นไม่คร่อม 0 ทุกวิธีทั้งสองชุด ช่วงเทียบ Li กว้างที่สุดเพราะ Li seed 4 ฝึกได้ไม่ดี",
    { x: 5.85, y: 3.65, w: 3.7, h: 1.4, fontSize: 16, color: MUTED });
  s.addNotes("สีเขียวคือช่วงความเชื่อมั่นไม่คร่อม 0 ทุกช่อง Li seed 4 ที่ฝึกได้ไม่ดียังรวมไว้ตามโปรโตคอล");
}

// 8. Criteria
{
  const s = slide("เกณฑ์ผ่าน 4 ข้อที่กำหนดไว้ก่อนทดสอบ", "เทียบคู่เทียบจากงานเผยแพร่ แยกตาม setting และชุด topology");
  table(s, [
    ["Setting / topology", "(1) เฉลี่ย ≥ +10%", "(2) นัยสำคัญ ≥ 60% ของสถานการณ์", "(3) ไม่แพ้ Ko เกิน 5%", "(4) ทนการปรับน้ำหนัก ±50%"],
    ["(a) ทดสอบ 0–19", ok("ผ่าน  Li +20.3%"), ok("ผ่าน  ≥ 91%"), ok("ผ่าน  S1 +13.4%"), ok("ผ่าน  Yuan +19.5%")],
    ["(a) ยืนยัน 40–59", ok("ผ่าน  Li +17.9%"), ok("ผ่าน  ≥ 91%"), ok("ผ่าน  S1 +13.2%"), ok("ผ่าน  Ko +19.1%")],
    ["(b) ทดสอบ 0–19", ok("ผ่าน  Li +19.8%"), ok("ผ่าน  ≥ 91%"), ok("ผ่าน  S1 +13.7%"), ok("ผ่าน  Yuan +17.3%")],
    ["(b) ยืนยัน 40–59", ok("ผ่าน  Li +17.5%"), ok("ผ่าน  ≥ 91%"), ok("ผ่าน  S1 +13.6%"), ok("ผ่าน  Ko +18.8%")],
  ], { y: 1.35, colW: [1.7, 1.8, 2.0, 1.8, 1.8], rowH: 0.52, fontSize: 16 });
  T(s, "ตัวเลขในแต่ละช่องคือค่าที่ต่ำที่สุดของเกณฑ์นั้น เช่น (1) คู่เทียบที่ส่วนต่างน้อยที่สุด, (3) สถานการณ์ที่ส่วนต่างเหนือ Ko น้อยที่สุด  topology ใหม่ 20–39 ผ่านครบเช่นกัน",
    { x: 0.45, y: 4.3, w: 9.1, h: 0.7, fontSize: 16, color: MUTED });
  s.addNotes("เกณฑ์ทั้ง 4 ข้อเขียนไว้ก่อนเห็นผลทดสอบ ผ่านครบทุก setting และทุกชุด topology");
}

// 9. Why it wins
{
  const s = slide("ชนะเพราะอะไร: สลับน้อยลง และส่งได้มากขึ้น", "ค่าจริงเฉลี่ย, setting (a), topology ทดสอบ");
  const stats = [["42 vs 616", "การสลับต่อ episode (EGAT+H vs Ko)"], ["53.2 vs 47.3", "throughput รวม Mb/s (+12.4%)"], ["0.172 vs 0.248", "อัตราสูญเสีย"]];
  stats.forEach(([big, lab], i) => {
    const y = 1.3 + i * 1.22;
    card(s, 0.45, y, 4.3, 1.08, TINT);
    T(s, big, { x: 0.65, y: y + 0.08, w: 4, h: 0.6, fontSize: 34, bold: true, color: TEAL });
    T(s, lab, { x: 0.65, y: y + 0.66, w: 4, h: 0.35, fontSize: 16 });
  });
  s.addImage({ path: path.join(FIG, "fig_switching.png"), x: 5.0, y: 1.3, w: 4.55, h: 2.98 });
  T(s, "จาก +26.0% เหนือ Ko: +13.9% มาจากความเสถียร ส่วนที่ไม่เกี่ยวกับการสลับ (สูญเสีย, throughput, p5) รวม +12.1%",
    { x: 5.0, y: 4.35, w: 4.55, h: 0.75, fontSize: 15, color: MUTED });
  s.addNotes("Ko สลับบ่อยเพราะตัดสินใจจากต้นทุนเฉพาะหน้า ทุกครั้งที่สลับเสียคิวและเส้นทาง EGAT+H ได้ throughput สูงกว่า 12.4% และอัตราสูญเสียต่ำกว่า จึงไม่ได้ชนะจากการอยู่นิ่งเพียงอย่างเดียว");
}

// 10. Fairness: competitors with hysteresis
{
  const s = slide("ทดสอบความเป็นธรรม: ใส่ hysteresis ให้คู่เทียบด้วย", "Li+H และ Yuan+H ใช้กติกาเลือก margin เดียวกัน, setting (a), A+B");
  s.addChart(pres.charts.BAR, [
    { name: "topology ทดสอบ 0–19", labels: ["Li+H", "Yuan+H"], values: [13.3, 8.5] },
    { name: "topology ยืนยัน 40–59", labels: ["Li+H", "Yuan+H"], values: [10.7, 6.7] },
  ], {
    x: 0.45, y: 1.25, w: 4.4, h: 3.8, barDir: "col", barGapWidthPct: 70, chartColors: [TEAL, "7FC4C6"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: F, dataLabelFontSize: 14, dataLabelColor: INK, dataLabelFormatCode: "0.0",
    catAxisLabelFontFace: F, catAxisLabelFontSize: 16, catAxisLabelColor: INK,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 12, valAxisLabelColor: MUTED, valAxisMinVal: 0, valAxisMaxVal: 16,
    valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontFace: F, legendFontSize: 14, legendColor: INK,
  });
  card(s, 5.05, 1.3, 4.5, 2.05, TINT);
  T(s, [{ text: "ชนะทั้งคู่อย่างมีนัยสำคัญ", options: { bold: true, color: TEAL, breakLine: true } },
        { text: "Li+H  +10.7% [+7.5, +14.0]\nYuan+H  +6.7% [+3.1, +10.2]\nสลับพอๆ กัน (42 vs 43) จึงชนะจากการเลือกช่องและเส้นทางที่ดีกว่า" }],
    { x: 5.25, y: 1.42, w: 4.15, h: 1.85, fontSize: 17 });
  card(s, 5.05, 3.5, 4.5, 1.55, AMBERT);
  T(s, [{ text: "ข้อจำกัด: Yuan+H ยังไม่ถึง 10%", options: { bold: true, color: AMBER, breakLine: true } },
        { text: "ส่วนต่าง +6.7% ผ่านเป้า 5% แต่ไม่ถึง 10% และไม่ทนการปรับน้ำหนักที่ระดับ 10%" }],
    { x: 5.25, y: 3.6, w: 4.15, h: 1.4, fontSize: 17 });
  s.addNotes("รอบก่อนแบบที่มี GRU เสมอกับ Yuan+H (+0.7%) รอบนี้ชนะอย่างมีนัยสำคัญ ส่วนต่างมาจากอัตราสูญเสีย p5 และ throughput ไม่ใช่จากการสลับ เพราะสลับเท่ากัน");
}

// 11. GRU removal under a pre-registered protocol
{
  const s = slide("ตัด GRU ออก: ยืนยันด้วยโปรโตคอลที่ล็อกไว้ก่อนรัน", "ablation ชี้ว่า GRU ไม่ช่วย แต่เห็นผลบนชุดทดสอบแล้ว จึงต้องยืนยันบน topology ใหม่");
  const steps = [
    ["1", "Ablation (seed 0–2)", "ตัด GRU ดีกว่าแบบเต็ม +7.0%\nแบบเต็มเสมอ Yuan+H"],
    ["2", "ล็อกโปรโตคอล", "เกณฑ์ P1–P3 และกติกาตัดสิน\nเขียนก่อนรัน 7 พ.ย. 2569"],
    ["3", "ยืนยันบน 40–59", "topology ที่ไม่เคยใช้\n5 seed, margin เลือกใหม่ได้ 2"],
  ];
  steps.forEach(([num, head, body], i) => {
    const x = 0.45 + i * 3.08;
    card(s, x, 1.3, 2.9, 1.75, TINT);
    s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: 1.42, w: 0.5, h: 0.5, fill: { color: TEAL }, line: { color: TEAL } });
    T(s, num, { x: x + 0.18, y: 1.42, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    T(s, head, { x: x + 0.8, y: 1.47, w: 2.0, h: 0.42, fontSize: 19, bold: true });
    T(s, body, { x: x + 0.18, y: 2.05, w: 2.6, h: 0.95, fontSize: 15 });
  });
  table(s, [
    ["เกณฑ์ (setting a, 40–59)", "ต้องได้", "ผลที่ได้", ""],
    ["P1  เทียบ Yuan+H", "≥ +5% และขอบล่าง > 0", "+6.7% [+3.1, +10.2]", ok("ผ่าน")],
    ["P2  เทียบ Ko, Li, Yuan, M0, Q-neutral, Li+H", "ขอบล่าง > 0 ทุกวิธี", "ต่ำสุด Li [+3.3, +32.4]", ok("ผ่าน")],
    ["P3  เกณฑ์ 4 ข้อ", "ผ่านครบ", "ผ่านครบ", ok("ผ่าน")],
  ], { y: 3.25, colW: [3.4, 2.1, 2.4, 1.2], rowH: 0.36, fontSize: 15 });
  s.addNotes("ถ้าไม่ผ่าน P1 และ P2 จะคงแบบที่มี GRU ไว้และรายงานเป็นผลลบ กติกานี้เขียนไว้ก่อนรัน และห้ามเปลี่ยนแบบจำลองหรือสถานการณ์หลังเห็นผล แบบที่ไม่มี GRU ดีกว่าแบบเดิม +4.9% บน 40–59");
}

// 12. ns-3
{
  const s = slide("ยืนยันบน ns-3 ระดับแพ็กเก็ต: ชนะทุกวิธี", "802.11ax, S2 / B1 / B3, topology 0–4, นโยบายเดิมไม่ฝึกใหม่");
  table(s, [
    ["EGAT+H เทียบ", "RTS/CTS  [95% CI]", "ไม่ใช้ RTS/CTS  [95% CI]", "ผล"],
    ["Ko", "+23.1% [+15.5, +30.7]", "+21.8% [+14.4, +29.2]", ok("ชนะ")],
    ["Yuan+H", "+16.4% [+0.4, +32.5]", "+18.4% [+2.1, +34.6]", ok("ชนะ (ช่วงกว้าง)")],
    ["Li+H", "+36.5% [+32.2, +40.8]", "+39.5% [+35.7, +43.4]", ok("ชนะ")],
    ["Yuan", "+36.4% [+22.3, +50.4]", "+37.2% [+22.6, +51.9]", ok("ชนะ")],
    ["Li", "+43.7% [+35.1, +52.2]", "+46.1% [+37.9, +54.3]", ok("ชนะ")],
    ["M0", "+52.6% [+48.6, +56.5]", "+55.5% [+52.3, +58.6]", ok("ชนะ")],
    ["Q-neutral", "+69.3% [+68.2, +70.4]", "+68.0% [+67.0, +69.0]", ok("ชนะ")],
  ], { y: 1.25, colW: [1.6, 2.8, 2.8, 1.9], rowH: 0.36, fontSize: 16 }, [2]);
  T(s, "throughput (RTS/CTS) เท่ากับหรือสูงกว่า Ko: S2 68.1 vs 67.1, B3 49.8 vs 43.6 Mb/s และสลับน้อยกว่า 18 เท่า (45 vs 790)  ข้อยกเว้น: ไม่ใช้ RTS/CTS ใน S2 Ko ได้ throughput สูงกว่า (66.4 vs 63.8)",
    { x: 0.45, y: 4.3, w: 9.1, h: 0.8, fontSize: 16, color: MUTED });
  s.addNotes("รอบก่อนแบบที่มี GRU ชนะ Ko บน ns-3 แค่ +10.7% แบบไม่มีนัยสำคัญ และเสมอ Yuan+H รอบนี้ชนะทุกวิธี แต่ขอบล่างเทียบ Yuan+H เฉียด 0 เพราะมีเพียง 5 topology ns-3 เป็นผลประกอบ ไม่ได้ใช้ตัดสินตามโปรโตคอล");
}

// 13. Conclusion
{
  const s = slide("สรุป", "สิ่งที่ยืนยันได้ และสิ่งที่ยังเหลือ");
  card(s, 0.45, 1.3, 4.45, 3.75, TINT);
  card(s, 5.1, 1.3, 4.45, 3.75, AMBERT);
  T(s, "ยืนยันได้", { x: 0.7, y: 1.42, w: 4, h: 0.45, fontSize: 22, bold: true, color: TEAL });
  T(s, [
    { text: "ชนะคู่เทียบจากงานเผยแพร่ทุกวิธีเกิน 10% อย่างมีนัยสำคัญ บน topology 3 ชุด", options: { bullet: true, breakLine: true } },
    { text: "ชนะ Li+H และ Yuan+H ที่ใช้ hysteresis เดียวกัน", options: { bullet: true, breakLine: true } },
    { text: "เกณฑ์ 4 ข้อและเกณฑ์ P1–P3 ผ่านครบ", options: { bullet: true, breakLine: true } },
    { text: "บน ns-3 ชนะทุกวิธีอย่างมีนัยสำคัญ", options: { bullet: true } },
  ], { x: 0.7, y: 1.92, w: 4.05, h: 3.0, fontSize: 19, paraSpaceAfter: 6 });
  T(s, "ข้อจำกัดและงานต่อ", { x: 5.35, y: 1.42, w: 4, h: 0.45, fontSize: 22, bold: true, color: AMBER });
  T(s, [
    { text: "เหนือ Yuan+H +6.7% ยังไม่ถึง 10%: ลองเรียน hysteresis ระหว่างฝึก", options: { bullet: true, breakLine: true } },
    { text: "การตัด GRU เลือกหลังเห็น ablation (ยืนยันด้วยโปรโตคอลแล้ว)", options: { bullet: true, breakLine: true } },
    { text: "ns-3 มี 5 topology: ช่วงเทียบ Yuan+H ยังกว้าง", options: { bullet: true, breakLine: true } },
    { text: "ยังไม่ได้ทดสอบบนอุปกรณ์จริง", options: { bullet: true } },
  ], { x: 5.35, y: 1.92, w: 4.05, h: 3.0, fontSize: 19, paraSpaceAfter: 6 });
  s.addNotes("ข้อติเรื่องนัยสำคัญเทียบ Ko แก้แล้วทั้งใน sim และ ns-3 และตอนนี้ชนะคู่เทียบที่ใส่ hysteresis ด้วย ส่วนที่ยังเหลือแสดงไว้ตรงๆ");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
