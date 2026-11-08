// Build the re-presentation deck: node docs/paper/build/build_slides.js OUT.pptx
// Needs pptxgenjs (npm). Numbers come from docs/paper/build/thesis_content.py (v10, EGAT without GRU, margin 2).
// Story: how the result moved from "win that spans zero" to "clear win", then each baseline's results and how we beat them.
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2] || "NW14_Presentation_v10.pptx";
const FIG = path.join(__dirname, "..", "figures");
const F = "TH Sarabun New";
const TEAL = "0E8A8F", TINT = "E4F2F2", INK = "1F2933", MUTED = "5B6670", AMBER = "B86E00", AMBERT = "FCF0DC", LINE = "D5E3E3", GRAYT = "F3F4F5";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: F, bodyFontFace: F };
pres.title = "EGAT-MAPPO: จากชนะคร่อม 0 สู่ชนะทิ้งห่าง";
pres.author = "NW14";

const TOTAL = 12;
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
  s.addText(title, { x: 0.45, y: 0.48, w: 9.1, h: 0.62, fontFace: F, fontSize: 30, bold: true, color: INK, margin: 0, isTextBox: true });
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

// Forest plot from shapes: rows [label, mean, lo, hi]; amber when the interval spans zero.
function forest(s, rows, o) {
  const { x, y, w, rowH, min, max, labW, title } = o;
  const px = (v) => x + labW + ((v - min) / (max - min)) * (w - labW - 0.95);
  if (title) T(s, title, { x, y: y - 0.36, w, h: 0.3, fontSize: 16, bold: true, color: MUTED });
  const top = y, bot = y + rows.length * rowH;
  s.addShape(pres.shapes.LINE, { x: px(0), y: top - 0.05, w: 0, h: bot - top + 0.05, line: { color: INK, width: 1, dashType: "dash" } });
  T(s, "0", { x: px(0) - 0.2, y: bot + 0.02, w: 0.4, h: 0.22, fontSize: 12, color: MUTED, align: "center" });
  rows.forEach(([lab, m, lo, hi], i) => {
    const cy = top + i * rowH + rowH / 2, col = lo > 0 ? TEAL : AMBER;
    T(s, lab, { x, y: cy - rowH / 2, w: labW - 0.1, h: rowH, fontSize: 15, valign: "middle" });
    s.addShape(pres.shapes.LINE, { x: px(lo), y: cy, w: px(hi) - px(lo), h: 0, line: { color: col, width: 2.5 } });
    s.addShape(pres.shapes.OVAL, { x: px(m) - 0.08, y: cy - 0.08, w: 0.16, h: 0.16, fill: { color: col }, line: { color: "FFFFFF", width: 1 } });
    T(s, `${m > 0 ? "+" : ""}${m.toFixed(1)}%`, { x: x + w - 0.85, y: cy - rowH / 2, w: 0.85, h: rowH, fontSize: 16, bold: true, color: col, align: "right", valign: "middle" });
  });
}

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
  T(s, [{ text: "รอบนี้: ", options: { bold: true, color: TEAL } },
        { text: "ปรับปรุงอะไร จึงเปลี่ยนจาก \"ชนะแบบคร่อม 0\" เป็น \"ชนะทิ้งห่าง\"" }],
    { x: 0.8, y: 2.9, w: 8.4, h: 0.65, fontSize: 22, valign: "middle" });
  T(s, "นายวิศรุต ไตรยวงศ์ 663380363-7\nอาจารย์ที่ปรึกษา ศ. ดร.จักรชัย โสอินทร์\nวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น",
    { x: 0.6, y: 3.95, w: 8.8, h: 1.1, fontSize: 18, color: MUTED, lineSpacingMultiple: 0.95 });
  s.addNotes("รอบนี้เล่าว่าปรับปรุงอะไรบ้าง ผลจึงเปลี่ยนจากชนะแบบคร่อม 0 มาเป็นชนะทิ้งห่าง แล้วจึงดูผลภาพรวมว่าแต่ละวิธีได้เท่าไรและเราชนะแต่ละวิธีได้อย่างไร");
}

// 2. The whole journey on one slide
{
  const s = slide("ภาพรวม: จากชนะคร่อม 0 สู่ชนะทิ้งห่าง", "จุดคือค่าเฉลี่ย เส้นคือช่วงความเชื่อมั่น 95%  สีส้ม = คร่อม 0 (ยังสรุปไม่ได้)  สีเขียว = ชนะชัด");
  forest(s, [
    ["ก่อนปรับเทียบ sim", 1.5, -11.5, 14.6],
    ["หลังปรับเทียบ (มี GRU)", 10.7, -0.9, 22.2],
    ["รุ่นสุดท้าย (ไม่มี GRU)", 23.1, 15.5, 30.7],
  ], { x: 0.45, y: 1.6, w: 4.5, rowH: 0.62, min: -15, max: 35, labW: 1.75, title: "เทียบ Ko บน ns-3 ระดับแพ็กเก็ต" });
  forest(s, [
    ["มี GRU", 0.7, -4.1, 5.5],
    ["ไม่มี GRU", 8.5, 4.3, 12.8],
  ], { x: 5.1, y: 1.6, w: 4.45, rowH: 0.62, min: -15, max: 35, labW: 1.4, title: "เทียบ Yuan+H (คู่เทียบที่แข็งที่สุด)" });
  card(s, 0.45, 3.85, 9.1, 1.2, GRAYT);
  T(s, [
    { text: "รอบก่อน (GAT+RAES, single-hop): ", options: { bold: true, color: MUTED } },
    { text: "ชนะ Ko เพียง +0.893 จุด SAT ชนะ 6 จาก 12 คู่ ไม่มีช่วงความเชื่อมั่น", options: { color: MUTED, breakLine: true } },
    { text: "รอบนี้ (topology ยืนยันที่ล็อกเกณฑ์ไว้ก่อน): ", options: { bold: true, color: TEAL } },
    { text: "ชนะ Ko +24.6% [+21.6, +27.6] และชนะทุกวิธีอย่างมีนัยสำคัญ ทั้งใน sim และ ns-3" },
  ], { x: 0.65, y: 3.97, w: 8.7, h: 1.0, fontSize: 17, paraSpaceAfter: 4 });
  s.addNotes("สไลด์นี้คือเรื่องทั้งหมดในหน้าเดียว เส้นสีส้มคือผลที่คร่อม 0 ยังสรุปไม่ได้ว่าชนะ เส้นสีเขียวคือชนะชัด แต่ละแถวคือผลหลังการปรับปรุงหนึ่งขั้น สไลด์ต่อไปจะอธิบายทีละขั้นว่าทำอะไรและทำไมช่วย");
}

// 3. Why the old result spanned zero, and what was changed
{
  const s = slide("จุดเริ่มต้น: ทำไมรอบก่อนคร่อม 0 และแก้อย่างไร", "แต่ละสาเหตุมีการปรับปรุงที่ตอบโดยตรง");
  table(s, [
    ["สาเหตุในรอบก่อน", "การปรับปรุงรอบนี้", "ขั้น"],
    ["single-hop 15 คู่ ไม่ใช่ mesh จริง และไม่มีต้นทุนการสลับ", "mesh หลาย hop 20 โหนด เลือกช่องและเส้นทางร่วมกัน การสลับทำให้ลิงก์หยุดและคิวหาย", "ปัญหา"],
    ["ตัวเรียนรู้ชนะ Q-neutral แค่ 0.833 จุด", "แบบจำลองใหม่ที่ใช้ข้อมูลลิงก์ และเริ่มจากความรู้ของ Ko", "1"],
    ["สลับบ่อยจนเสียคิว", "hysteresis: สลับเมื่อดีกว่าเกิน margin เท่านั้น", "2"],
    ["sim ไม่ตรงกับเครือข่ายจริง", "ปรับเทียบ sim กับ ns-3 แล้วฝึกใหม่ทั้งหมด", "3"],
    ["แบบจำลองยังเสมอคู่เทียบที่แข็งที่สุด", "ตัด GRU ที่ไม่ช่วย ยืนยันด้วยเกณฑ์ที่ล็อกไว้ก่อน", "4"],
    ["2 seed ที่เคยเห็น ไม่มีช่วงความเชื่อมั่น", "5 seed × 20 topology, CI 95%, เกณฑ์ 4 ข้อ, topology ใหม่ 2 ชุด", "ทุกขั้น"],
  ], { y: 1.3, colW: [3.5, 4.8, 0.8], rowH: 0.48, fontSize: 16 });
  s.addNotes("อาจารย์ติว่าผลรอบก่อนอาจเป็นความคลาดเคลื่อน ผมจึงไล่หาสาเหตุ แล้วแก้ทีละข้อ ขั้น 1 ถึง 4 คือการปรับปรุงที่ทำให้ส่วนต่างขยับ ส่วนแถวสุดท้ายคือการวัดผลที่ทำให้เชื่อถือได้");
}

// 4. Step 1: new model
{
  const s = slide("ขั้น 1  แบบจำลองใหม่: EGAT-MAPPO", "แต่ละโหนดตัดสินใจเองขณะใช้งาน ฝึกด้วย critic ส่วนกลาง");
  s.addImage({ path: path.join(FIG, "fig_architecture.png"), x: 0.45, y: 1.25, w: 6.6, h: 3.39 });
  const pts = [
    ["ดูคุณภาพลิงก์", "GATv2 ใช้ SNR อัตราส่ง ภาระ ให้น้ำหนักเพื่อนบ้าน คู่เทียบ GAT ไม่ใช้"],
    ["เริ่มจากความรู้ของ Ko", "BC เลียนแบบ Ko ก่อน และ Ko prior ในหัวตัดสินใจ ablation: ส่วนนี้ช่วยมากที่สุด"],
    ["เลือกช่อง + เส้นทางร่วมกัน", "หัวเส้นทางทุกขั้น หัวช่องทุก 10 ขั้น"],
  ];
  pts.forEach(([h, b], i) => {
    const y = 1.25 + i * 1.2;
    card(s, 7.2, y, 2.35, 1.08, TINT);
    T(s, h, { x: 7.32, y: y + 0.06, w: 2.15, h: 0.34, fontSize: 17, bold: true, color: TEAL });
    T(s, b, { x: 7.32, y: y + 0.42, w: 2.15, h: 0.62, fontSize: 13 });
  });
  T(s, "ผล: เทียบ Ko ใน sim เกิน +15% แล้ว แต่ยังสลับบ่อย (ราว 200 ครั้งต่อ episode) จึงแทบไม่ต่างจาก Li ในคะแนนรวม",
    { x: 0.45, y: 4.7, w: 6.6, h: 0.45, fontSize: 15, color: AMBER });
  s.addNotes("รอบก่อนตัวเรียนรู้แทบไม่ต่างจาก Q-neutral รอบนี้เปลี่ยนแบบจำลองใหม่ ให้รู้คุณภาพลิงก์ และเริ่มจากความรู้ของ Ko ผลคือชนะ Ko ได้แล้ว แต่ยังสลับบ่อยเกินไป จึงเป็นที่มาของขั้นที่ 2");
}

// 5. Step 2: hysteresis
{
  const s = slide("ขั้น 2  ลดการสลับที่ไม่จำเป็นด้วย hysteresis", "สลับเมื่อทางเลือกใหม่ดีกว่าเกิน margin m = 2 (เลือกบน validation ไม่ดูชุดทดสอบ)");
  table(s, [
    ["", "ไม่มี hysteresis", "มี hysteresis"],
    ["การสลับต่อ episode", "204", ok("42")],
    ["throughput (Mb/s)", "53.0", ok("53.2")],
    ["เทียบ Ko (คะแนนรวม)", "+21.5%", ok("+26.0%")],
  ], { x: 0.45, y: 1.35, w: 4.4, colW: [1.9, 1.25, 1.25], rowH: 0.5, fontSize: 17 });
  card(s, 0.45, 3.5, 4.4, 1.55, TINT);
  T(s, [{ text: "ทำไมช่วย", options: { bold: true, color: TEAL, breakLine: true } },
        { text: "ทุกการสลับทำให้ลิงก์หยุด 1 ขั้นและคิวหาย การสลับที่ได้ประโยชน์น้อยจึงขาดทุน ลดการสลับ 5 เท่าโดย throughput ไม่ลด" }],
    { x: 0.65, y: 3.6, w: 4.0, h: 1.4, fontSize: 16 });
  s.addImage({ path: path.join(FIG, "fig_switching.png"), x: 5.05, y: 1.3, w: 4.5, h: 2.95 });
  T(s, "มุมบนซ้าย = throughput สูงและสลับน้อย ซึ่งเป็นจุดที่ดีที่สุด", { x: 5.05, y: 4.35, w: 4.5, h: 0.6, fontSize: 15, color: MUTED });
  s.addNotes("hysteresis เป็นกฎตอนใช้งาน ไม่ได้เปลี่ยนการฝึก ลดการสลับจาก 204 เหลือ 42 ครั้ง throughput ไม่ลด คะแนนเทียบ Ko เพิ่มจาก 21.5 เป็น 26.0% ผลใน sim ดีมากแล้ว แต่ต้องถามต่อว่าเป็นจริงนอก sim หรือเปล่า");
}

// 6. Step 3: calibration against ns-3
{
  const s = slide("ขั้น 3  ปรับเทียบ sim กับ ns-3 แล้วฝึกใหม่", "ns-3 = โปรแกรมจำลองระดับแพ็กเก็ตมาตรฐาน ใช้ตรวจว่าผลไม่ได้เกิดจากข้อสมมติของ sim");
  const steps = [
    ["พบปัญหา", "sim บอกชนะ Ko +21.2%\nแต่ ns-3 ได้แค่ +1.5%\n[−11.5, +14.6] คร่อม 0", GRAYT, AMBER],
    ["สาเหตุ", "sim แบ่ง airtime เท่ากันตามจำนวนลิงก์\nประเมินช่องแออัดเกินจริง 22–28%\nนโยบายจึงเรียนรวมทราฟฟิกไว้ช่องเดียว", GRAYT, AMBER],
    ["แก้", "เปลี่ยนเป็นแบบจำลองโดเมนการรบกวน\nปรับ η = 0.65 กับ ns-3\nความคลาด 23.5% → 7.4%", TINT, TEAL],
    ["ผล", "ฝึกทุกวิธีใหม่บน sim ที่ปรับแล้ว\nns-3 เทียบ Ko ขึ้นเป็น +10.7%\nแต่ยังคร่อม 0 [−0.9, +22.2]", AMBERT, AMBER],
  ];
  steps.forEach(([h, b, fill, col], i) => {
    const x = 0.45 + i * 2.31;
    card(s, x, 1.35, 2.15, 2.75, fill);
    T(s, h, { x: x + 0.18, y: 1.47, w: 1.85, h: 0.42, fontSize: 21, bold: true, color: col });
    T(s, b, { x: x + 0.18, y: 1.98, w: 1.9, h: 2.05, fontSize: 15 });
    if (i < 3) T(s, "›", { x: x + 2.12, y: 2.35, w: 0.22, h: 0.5, fontSize: 28, bold: true, color: MUTED, align: "center" });
  });
  T(s, "บทเรียน: ต้องปรับเทียบ sim ก่อนฝึก มิฉะนั้นนโยบายจะเรียนเอาเปรียบจุดที่ sim ผิด ขั้นนี้ทำให้ผลบน ns-3 ขยับจาก +1.5% เป็น +10.7% แต่ยังไม่พอ",
    { x: 0.45, y: 4.3, w: 9.1, h: 0.75, fontSize: 16, color: MUTED });
  s.addNotes("ขั้นนี้สำคัญที่สุดในแง่ความน่าเชื่อถือ ถ้าไม่ตรวจกับ ns-3 เราจะรายงาน +21% ที่ไม่เป็นจริง หลังปรับเทียบผลขยับขึ้นแต่ยังคร่อม 0 และยังเสมอ Yuan+H จึงต้องมีขั้นที่ 4");
}

// 7. Step 4: drop the GRU, confirmed under a locked protocol
{
  const s = slide("ขั้น 4  ตัด GRU ออก และยืนยันด้วยเกณฑ์ที่ล็อกไว้ก่อน", "ablation พบว่า GRU (หน่วยความจำข้ามเวลา) ไม่ช่วย และอาจทำให้แย่ลง");
  table(s, [
    ["topology ทดสอบ 0–19", "มี GRU", "ไม่มี GRU"],
    ["throughput (Mb/s)", "51.35", ok("53.20")],
    ["p5 (Mb/s)", "1.442", ok("1.533")],
    ["อัตราสูญเสีย", "0.193", ok("0.172")],
    ["การสลับ / episode", "51.7", ok("42.2")],
    ["เทียบ Yuan+H", no("+0.7% (คร่อม 0)"), ok("+8.5%")],
    ["ns-3 เทียบ Ko", no("+10.7% (คร่อม 0)"), ok("+23.1%")],
  ], { x: 0.45, y: 1.3, w: 5.0, colW: [1.9, 1.55, 1.55], rowH: 0.44, fontSize: 16 });
  card(s, 5.65, 1.3, 3.9, 1.85, TINT);
  T(s, [{ text: "ทำไมต้องล็อกเกณฑ์ก่อน", options: { bold: true, color: TEAL, breakLine: true } },
        { text: "ผล ablation ได้หลังเห็นชุดทดสอบแล้ว จึงเขียนเกณฑ์ P1–P3 ไว้ก่อนรัน (7 พ.ย.) แล้วทดสอบบน topology 40–59 ที่ไม่เคยใช้ ถ้าไม่ผ่านจะคงแบบเดิม" }],
    { x: 5.85, y: 1.4, w: 3.5, h: 1.7, fontSize: 15 });
  card(s, 5.65, 3.3, 3.9, 1.75, AMBERT);
  T(s, [{ text: "ผลบน 40–59: ผ่านครบ", options: { bold: true, color: TEAL, breakLine: true } },
        { text: "P1 เทียบ Yuan+H +6.7% [+3.1, +10.2]\nP2 ขอบล่าง > 0 ทุกวิธี\nP3 ผ่านเกณฑ์ 4 ข้อ" }],
    { x: 5.85, y: 3.4, w: 3.5, h: 1.6, fontSize: 15 });
  s.addNotes("ตัด GRU แล้วดีขึ้นทุกตัวชี้วัด น่าจะเพราะข้อมูลแต่ละขั้นเพียงพอแล้ว และ GRU ทำให้ฝึกยากขึ้นในงบ 100,000 ขั้น ขั้นนี้เปลี่ยนผลเทียบ Yuan+H จากเสมอเป็นชนะ และ ns-3 เทียบ Ko จากคร่อม 0 เป็นชนะชัด");
}

// 8. Overall: what each method achieves
{
  const s = slide("ผลภาพรวม: แต่ละวิธีทำได้แค่ไหน", "ค่าจริงเฉลี่ย 11 สถานการณ์, setting (a), topology ทดสอบ 0–19 (ลูกศรบอกทิศที่ดี)");
  table(s, [
    ["วิธี", "throughput ↑", "p5 ↑", "สูญเสีย ↓", "Jain ↑", "สลับ ↓", "เราดีกว่า"],
    ["M0  ช่องคงที่", "26.87", "0.858", "0.516", "0.893", "0.2", "+31.1%"],
    ["Ko  heuristic (2007)", "47.32", "1.410", "0.248", "0.777", "615.5", "+26.0%"],
    ["Li  GAT-A2C (2025)", "47.06", "1.220", "0.245", "0.790", "169.1", "+20.3%"],
    ["Yuan  GAT+DQN (2022)", "49.38", "1.296", "0.221", "0.767", "351.5", "+24.1%"],
    ["Li+H", "47.00", "1.278", "0.244", "0.802", "43.3", "+13.3%"],
    ["Yuan+H", "49.70", "1.362", "0.213", "0.787", "43.3", "+8.5%"],
    [ok("EGAT+H (เรา)"), ok("53.20"), ok("1.533"), ok("0.172"), "0.786", ok("42.2"), "—"],
  ], { y: 1.25, colW: [2.3, 1.15, 0.95, 1.05, 0.95, 1.05, 1.65], rowH: 0.38, fontSize: 16 }, [7]);
  T(s, "เราดีที่สุดใน throughput, p5, อัตราสูญเสีย และการสลับ ส่วน Jain ใกล้เคียงคนอื่น (M0 สูงสุดเพราะทุก flow ได้น้อยเท่าๆ กัน)  Q-neutral แทบส่งไม่ได้ (0.17 Mb/s) จึงไม่แสดง",
    { x: 0.45, y: 4.4, w: 9.1, h: 0.7, fontSize: 15, color: MUTED });
  s.addNotes("ตารางนี้คือภาพรวมของทุกวิธี เราได้ throughput สูงสุด p5 สูงสุด สูญเสียต่ำสุด และสลับน้อยที่สุด ตัวเดียวที่ไม่ได้นำคือ Jain ซึ่งใกล้เคียงคนอื่น ช่องขวาสุดคือคะแนนรวมที่เราดีกว่าแต่ละวิธี");
}

// 9. How we beat each method
{
  const s = slide("เราชนะแต่ละวิธีด้วยอะไร", "จุดอ่อนของแต่ละวิธี และส่วนที่เราได้คะแนนเหนือกว่ามากที่สุด (topology ทดสอบ 0–19)");
  const cards = [
    ["M0", "+31.1%", "ไม่สลับเลย แต่ช่องคงที่ทำให้ช่องแออัด สูญเสีย 52%", "throughput เกือบ 2 เท่า สูญเสียเหลือ 17%"],
    ["Ko", "+26.0%", "ตัดสินใจจากต้นทุนเฉพาะหน้า สลับ 616 ครั้ง เสียคิวทุกครั้ง", "สลับน้อยกว่า 15 เท่า และ throughput +12.4%"],
    ["Yuan", "+24.1%", "DQN สลับ 352 ครั้ง ไม่ใช้ข้อมูลลิงก์", "ความเสถียร + สูญเสียต่ำกว่า"],
    ["Li", "+20.3%", "A2C แปรปรวนสูง (seed 4 ฝึกล้มเหลว) p5 ต่ำสุด", "p5 สูงกว่า 26% และสูญเสียต่ำกว่า"],
    ["Li+H / Yuan+H", "+13.3% / +8.5%", "ใส่ hysteresis แล้ว สลับเท่าเรา (43 ครั้ง)", "ชนะจากการเลือกช่องและเส้นทางที่ดีกว่า: สูญเสีย p5 throughput"],
    ["Q-neutral", "+73.6%", "Q-table เรียนไม่ทันกับ state ที่ใหญ่ สลับ 8,000 ครั้ง", "ส่งข้อมูลได้จริง"],
  ];
  cards.forEach(([name, val, weak, win], i) => {
    const x = 0.45 + (i % 3) * 3.08, y = 1.25 + Math.floor(i / 3) * 1.95;
    card(s, x, y, 2.9, 1.8, i === 4 ? AMBERT : TINT);
    T(s, name, { x: x + 0.15, y: y + 0.08, w: 1.55, h: 0.36, fontSize: 18, bold: true });
    T(s, val, { x: x + 1.3, y: y + 0.08, w: 1.45, h: 0.36, fontSize: 18, bold: true, color: TEAL, align: "right" });
    T(s, [{ text: "จุดอ่อน: ", options: { bold: true, color: MUTED } }, { text: weak, options: { color: MUTED, breakLine: true } },
          { text: "เราชนะด้วย: ", options: { bold: true, color: TEAL } }, { text: win }],
      { x: x + 0.15, y: y + 0.5, w: 2.62, h: 1.25, fontSize: 13.5, paraSpaceAfter: 3 });
  });
  s.addNotes("แต่ละวิธีแพ้ด้วยเหตุผลต่างกัน M0 ช่องแออัด Ko สลับบ่อย Yuan กับ Li ไม่ใช้ข้อมูลลิงก์และสลับบ่อย กรณีที่สำคัญที่สุดคือ Li+H และ Yuan+H ที่สลับเท่าเรา ส่วนต่างตรงนี้จึงมาจากคุณภาพการตัดสินใจล้วนๆ");
}

// 10. Results on all topology sets + criteria
{
  const s = slide("ผลยืนยัน: ชนะทุกวิธีบน topology ที่ไม่เคยใช้", "คะแนนรวมที่ดีกว่า (%), setting (a), topology ยืนยัน 40–59 ที่ล็อกเกณฑ์ไว้ก่อนรัน");
  s.addChart(pres.charts.BAR, [
    { name: "topology ยืนยัน 40–59", labels: ["M0", "Yuan", "Ko", "Li", "Li+H", "Yuan+H"], values: [31.6, 24.8, 24.6, 17.9, 10.7, 6.7] },
  ], {
    x: 0.45, y: 1.2, w: 5.4, h: 3.9, barDir: "col", barGapWidthPct: 55, chartColors: [TEAL],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: F, dataLabelFontSize: 15, dataLabelColor: INK, dataLabelFormatCode: "0.0",
    catAxisLabelFontFace: F, catAxisLabelFontSize: 16, catAxisLabelColor: INK,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 12, valAxisLabelColor: MUTED, valAxisMinVal: 0, valAxisMaxVal: 36,
    valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  });
  table(s, [
    ["เกณฑ์ที่ล็อกไว้ก่อน", "ผล"],
    ["(1) เฉลี่ย ≥ +10% ทุกคู่เทียบจากงานเผยแพร่", ok("ผ่าน")],
    ["(2) นัยสำคัญ ≥ 60% ของสถานการณ์", ok("ผ่าน")],
    ["(3) ไม่แพ้ Ko เกิน 5% ในสถานการณ์ใด", ok("ผ่าน")],
    ["(4) ทนการปรับน้ำหนัก ±50%", ok("ผ่าน")],
  ], { x: 6.05, y: 1.3, w: 3.5, colW: [2.75, 0.75], rowH: 0.42, fontSize: 14 });
  T(s, "ผ่านครบทั้ง 2 setting และ 3 ชุด topology ช่วงความเชื่อมั่นไม่คร่อม 0 ทุกวิธี  เครือข่าย 50 / 80 โหนดที่ไม่เคยฝึกยังชนะ Ko +47%",
    { x: 6.05, y: 3.6, w: 3.5, h: 1.45, fontSize: 15, color: MUTED });
  s.addNotes("topology 40–59 ไม่เคยใช้ก่อน ผลใกล้เคียงชุดทดสอบ แปลว่าไม่ได้ overfit เกณฑ์ 4 ข้อเขียนไว้ก่อน ผ่านครบทุกชุด ส่วน Yuan+H +6.7% ผ่านเป้า 5% แต่ยังไม่ถึง 10%");
}

// 11. ns-3
{
  const s = slide("ยืนยันบน ns-3 ระดับแพ็กเก็ต: ชนะทุกวิธี", "802.11ax, S2 / B1 / B3, topology 0–4, นโยบายเดิมไม่ฝึกใหม่, 2 การตั้งค่า MAC");
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
  T(s, "throughput เท่ากับหรือสูงกว่า Ko (S2 68.1 vs 67.1, B3 49.8 vs 43.6 Mb/s) และสลับน้อยกว่า 18 เท่า  ข้อยกเว้น: ไม่ใช้ RTS/CTS ใน S2 Ko ได้ throughput สูงกว่า (66.4 vs 63.8)",
    { x: 0.45, y: 4.3, w: 9.1, h: 0.8, fontSize: 16, color: MUTED });
  s.addNotes("ns-3 เป็นผลประกอบ ไม่ได้ใช้ตัดสิน แต่ไปทางเดียวกับ sim ขอบล่างเทียบ Yuan+H เฉียด 0 เพราะมีแค่ 5 topology");
}

// 12. Conclusion
{
  const s = slide("สรุป", "การปรับปรุง 4 ขั้นเปลี่ยนผลจากคร่อม 0 เป็นชนะทิ้งห่าง");
  card(s, 0.45, 1.3, 4.45, 3.75, TINT);
  card(s, 5.1, 1.3, 4.45, 3.75, AMBERT);
  T(s, "ปรับปรุงอะไร ได้อะไร", { x: 0.7, y: 1.42, w: 4, h: 0.45, fontSize: 22, bold: true, color: TEAL });
  T(s, [
    { text: "แบบจำลองใช้ข้อมูลลิงก์ + ความรู้ Ko: ชนะ Ko ได้", options: { bullet: true, breakLine: true } },
    { text: "hysteresis: สลับน้อยลง 5 เท่า throughput ไม่ลด", options: { bullet: true, breakLine: true } },
    { text: "ปรับเทียบกับ ns-3: ผลที่เห็นใน sim เป็นจริงนอก sim", options: { bullet: true, breakLine: true } },
    { text: "ตัด GRU: ชนะ Yuan+H และชนะ Ko บน ns-3 ชัดเจน", options: { bullet: true, breakLine: true } },
    { text: "ผล: ชนะทุกวิธีอย่างมีนัยสำคัญ ทั้ง sim และ ns-3", options: { bullet: true, bold: true } },
  ], { x: 0.7, y: 1.92, w: 4.05, h: 3.0, fontSize: 18, paraSpaceAfter: 5 });
  T(s, "ข้อจำกัดและงานต่อ", { x: 5.35, y: 1.42, w: 4, h: 0.45, fontSize: 22, bold: true, color: AMBER });
  T(s, [
    { text: "เหนือ Yuan+H +6.7% ยังไม่ถึง 10%: ลองเรียน hysteresis ระหว่างฝึก", options: { bullet: true, breakLine: true } },
    { text: "การตัด GRU เลือกหลังเห็น ablation (ยืนยันด้วยเกณฑ์ที่ล็อกไว้แล้ว)", options: { bullet: true, breakLine: true } },
    { text: "ns-3 มี 5 topology: ช่วงเทียบ Yuan+H ยังกว้าง", options: { bullet: true, breakLine: true } },
    { text: "ยังไม่ได้ทดสอบบนอุปกรณ์จริง", options: { bullet: true } },
  ], { x: 5.35, y: 1.92, w: 4.05, h: 3.0, fontSize: 18, paraSpaceAfter: 6 });
  s.addNotes("ข้อติเรื่องผลคร่อม 0 แก้แล้วด้วยการปรับปรุง 4 ขั้น ทั้งใน sim และ ns-3 และแสดงข้อจำกัดที่เหลือไว้ตรงๆ");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
