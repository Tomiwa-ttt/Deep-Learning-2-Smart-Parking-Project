const pptxgen = require("pptxgenjs");

const BG = "14161A";
const PANEL = "24282F";
const GREEN = "4FAE6D";
const AMBER = "E0A338";
const RED = "E0523A";
const MUTED = "8B909C";
const WHITE = "F4F3EF";

function newSlide(pres, opts = {}) {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

function titleText(s, text, opts = {}) {
  s.addText(text, {
    x: 0.6, y: opts.y || 0.5, w: 12.1, h: opts.h || 0.9,
    fontFace: "Arial", fontSize: opts.size || 32, bold: true, color: WHITE,
    margin: 0,
  });
}

function kicker(s, text, y = 0.35) {
  s.addText(text.toUpperCase(), {
    x: 0.6, y, w: 12.1, h: 0.35,
    fontFace: "Arial", fontSize: 12, color: AMBER, charSpacing: 2, bold: true, margin: 0,
  });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5

// ---------------- Slide 1: Title ----------------
{
  const s = newSlide(pres);
  s.addShape(pres.ShapeType.ellipse, { x: 6.45, y: 2.55, w: 0.14, h: 0.14, fill: { color: GREEN }, line: { type: "none" } });
  s.addText("SMART PARKING SYSTEM", {
    x: 0.8, y: 2.85, w: 11.7, h: 1.0, align: "center",
    fontFace: "Arial", fontSize: 44, bold: true, color: WHITE, margin: 0,
  });
  s.addText("USING OBJECT DETECTION", {
    x: 0.8, y: 3.55, w: 11.7, h: 0.7, align: "center",
    fontFace: "Arial", fontSize: 44, bold: true, color: WHITE, margin: 0,
  });
  s.addText("CNN-Based Real-Time Occupancy Detection — Trained & Validated on Real-World Data", {
    x: 0.8, y: 4.4, w: 11.7, h: 0.5, align: "center",
    fontFace: "Arial", fontSize: 16, italic: true, color: AMBER, margin: 0,
  });
  s.addText("Project Presentation  ·  July 2026", {
    x: 0.8, y: 6.6, w: 11.7, h: 0.4, align: "center",
    fontFace: "Courier New", fontSize: 13, color: MUTED, margin: 0,
  });
}

// ---------------- Slide 2: Problem ----------------
{
  const s = newSlide(pres);
  kicker(s, "The Problem");
  titleText(s, "Drivers waste real time and money\nsearching for parking", { size: 30, h: 1.3 });

  const stats = [
    { n: "66%", d: "of U.S. drivers spend up to 15\nminutes per trip searching for parking" },
    { n: "17 hrs / $345", d: "average American driver loses\nper year to parking search" },
    { n: "107 hrs / $2,243", d: "per year for drivers in dense\ncities like New York" },
  ];
  stats.forEach((st, i) => {
    const x = 0.6 + i * 4.13;
    s.addShape(pres.ShapeType.roundRect, { x, y: 2.7, w: 3.85, h: 3.6, rectRadius: 0.08, fill: { color: PANEL }, line: { type: "none" } });
    s.addText(st.n, { x: x + 0.25, y: 3.0, w: 3.35, h: 1.0, fontFace: "Arial", fontSize: 30, bold: true, color: GREEN, margin: 0 });
    s.addText(st.d, { x: x + 0.25, y: 4.1, w: 3.35, h: 1.9, fontFace: "Arial", fontSize: 15, color: MUTED, margin: 0 });
  });
  s.addText("Source: T2 Systems driver survey, Jan 2026", {
    x: 0.6, y: 6.75, w: 12, h: 0.4, fontFace: "Arial", fontSize: 11, italic: true, color: MUTED, margin: 0,
  });
}

// ---------------- Slide 3: Solution + architecture ----------------
{
  const s = newSlide(pres);
  kicker(s, "The Solution");
  titleText(s, "Classify every parking spot, live,\nfrom a camera feed", { size: 28, h: 1.3 });
  s.addText([
    { text: "Camera photo\n", options: { bold: true, color: WHITE } },
    { text: "→ crop each marked spot\n", options: { color: MUTED } },
    { text: "→ CNN classifies Empty / Occupied\n", options: { color: MUTED } },
    { text: "→ geometry check flags misparked cars\n", options: { color: MUTED } },
    { text: "→ REST API → mobile app", options: { color: MUTED } },
  ], { x: 0.6, y: 2.7, w: 5.6, h: 3.6, fontFace: "Arial", fontSize: 17, lineSpacingMultiple: 1.5, margin: 0 });
  s.addImage({ path: "report_assets/architecture.png", x: 7.2, y: 1.6, w: 2.6, h: 5.73 });
}

// ---------------- Slide 4: Data ----------------
{
  const s = newSlide(pres);
  kicker(s, "Data");
  titleText(s, "Synthetic first, then real PKLot data", { size: 28 });

  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 2.3, w: 5.9, h: 4.3, rectRadius: 0.08, fill: { color: PANEL }, line: { type: "none" } });
  s.addText("1. SYNTHETIC (sanity check)", { x: 1.0, y: 2.6, w: 5.1, h: 0.4, fontFace: "Arial", fontSize: 15, bold: true, color: MUTED, margin: 0 });
  s.addText("60 generated lot images\n600 labeled spot crops\nValidated the full pipeline\nbefore touching real data", {
    x: 1.0, y: 3.1, w: 5.1, h: 2.9, fontFace: "Arial", fontSize: 17, color: WHITE, lineSpacingMultiple: 1.4, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, { x: 6.8, y: 2.3, w: 5.9, h: 4.3, rectRadius: 0.08, fill: { color: PANEL }, line: { type: "none" } });
  s.addText("2. REAL (PKLot dataset)", { x: 7.2, y: 2.6, w: 5.1, h: 0.4, fontFace: "Arial", fontSize: 15, bold: true, color: AMBER, margin: 0 });
  s.addText("1,242 real lot photographs\n70,684 labeled real parking spaces\nCOCO annotations → cropped\nper-spot patches for training", {
    x: 7.2, y: 3.1, w: 5.1, h: 2.9, fontFace: "Arial", fontSize: 17, color: WHITE, lineSpacingMultiple: 1.4, margin: 0,
  });
}

// ---------------- Slide 5: Model ----------------
{
  const s = newSlide(pres);
  kicker(s, "Model");
  titleText(s, "A small, fast CNN — one spot at a time", { size: 28 });

  const rows = [
    ["Input", "64×64×3 spot crop"],
    ["Conv2D (32) + MaxPool", "32×32×32"],
    ["Conv2D (64) + MaxPool", "16×16×64"],
    ["Conv2D (128) + MaxPool", "8×8×128"],
    ["Dense (128) + Dropout", "128"],
    ["Dense (1, sigmoid)", "Empty / Occupied"],
  ];
  s.addTable(
    [
      [{ text: "Layer", options: { bold: true, color: WHITE, fill: { color: PANEL } } },
       { text: "Output", options: { bold: true, color: WHITE, fill: { color: PANEL } } }],
      ...rows.map(r => r.map(c => ({ text: c, options: { color: MUTED, fill: { color: "1B1E24" } } }))),
    ],
    { x: 0.6, y: 2.5, w: 6.2, colW: [3.6, 2.6], fontFace: "Courier New", fontSize: 13, border: { type: "solid", color: "333842", pt: 0.5 }, autoPage: false, rowH: 0.55 }
  );
  s.addText([
    { text: "Why per-spot classification, not full-scene detection?\n\n", options: { bold: true, color: WHITE, fontSize: 16 } },
    { text: "Each spot is a small, independent decision — much cheaper to run in real time than detecting cars anywhere in a full frame. This is the same approach used in the PKLot / CNRPark-EXT research.", options: { color: MUTED, fontSize: 15 } },
  ], { x: 7.2, y: 2.5, w: 5.5, h: 4, fontFace: "Arial", lineSpacingMultiple: 1.3, margin: 0 });
}

// ---------------- Slide 6: Results chart ----------------
{
  const s = newSlide(pres);
  kicker(s, "Results");
  titleText(s, "97.04% validation accuracy on real data", { size: 28 });

  s.addChart(pres.ChartType.bar, [
    {
      name: "Accuracy",
      labels: ["Synthetic\n(sanity check)", "Real PKLot\n(full val. set)", "Real PKLot\n(held-out spot-check)"],
      values: [96.0, 97.04, 93.9],
    },
  ], {
    x: 0.8, y: 2.4, w: 11.7, h: 4.4,
    barDir: "col",
    chartColors: [GREEN],
    showTitle: false,
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelColor: WHITE,
    dataLabelFontSize: 14,
    catAxisLabelColor: MUTED,
    catAxisLabelFontSize: 13,
    valAxisLabelColor: MUTED,
    valAxisMinVal: 0,
    valAxisMaxVal: 100,
    valGridLine: { color: "333842", size: 0.75 },
    catGridLine: { style: "none" },
    showLegend: false,
    plotArea: { fill: { color: BG } },
    chartArea: { fill: { color: BG } },
  });
}

// ---------------- Slide 7: Real photo qualitative ----------------
{
  const s = newSlide(pres);
  kicker(s, "Real Results");
  titleText(s, "Genuine model output on real photos", { size: 28 });
  s.addImage({ path: "demo/video_assets/lot1_annotated.jpg", x: 1.15, y: 2.3, w: 4.2, h: 4.2 });
  s.addImage({ path: "demo/video_assets/lot0_annotated.jpg", x: 5.85, y: 2.3, w: 4.2, h: 4.2 });
  s.addText("40/40 correct", { x: 1.15, y: 6.65, w: 4.2, h: 0.4, align: "center", fontFace: "Courier New", fontSize: 14, color: GREEN, margin: 0 });
  s.addText("24/28 correct", { x: 5.85, y: 6.65, w: 4.2, h: 0.4, align: "center", fontFace: "Courier New", fontSize: 14, color: AMBER, margin: 0 });
  s.addText("Green = empty   ·   Orange = occupied", {
    x: 0.9, y: 7.1, w: 9.6, h: 0.35, align: "center", fontFace: "Arial", fontSize: 12, color: MUTED, margin: 0,
  });
}

// ---------------- Slide 8: App ----------------
{
  const s = newSlide(pres);
  kicker(s, "Product");
  titleText(s, "Live in a mobile app", { size: 28 });
  s.addImage({ path: "report_assets/app_screenshot.png", x: 5.4, y: 1.6, w: 2.53, h: 4.93 });
  s.addText([
    { text: "CampusPark\n\n", options: { bold: true, fontSize: 20, color: WHITE } },
    { text: "• Live per-spot status grid\n", options: { color: MUTED, fontSize: 16 } },
    { text: "• Tap any stall for confidence %\n", options: { color: MUTED, fontSize: 16 } },
    { text: "• Backed by the real-trained CNN\n", options: { color: MUTED, fontSize: 16 } },
    { text: "• Browser-based — no install needed\n  to demo today", options: { color: MUTED, fontSize: 16 } },
  ], { x: 8.3, y: 2.2, w: 4.3, h: 4, fontFace: "Arial", lineSpacingMultiple: 1.4, margin: 0 });
}

// ---------------- Slide 9: Limitations ----------------
{
  const s = newSlide(pres);
  kicker(s, "Honest Limitations");
  titleText(s, "What's not solved yet", { size: 28 });

  const items = [
    ["\"Properly parked\" on real photos", "Works on synthetic data (97.3%). On real, tightly-packed lots, classic CV can't reliably isolate one car from its neighbors — needs a trained detector (e.g. YOLO)."],
    ["Cross-dataset generalization", "Trained & validated on PKLot only. A 5-image pilot test on CNRPark-EXT (different cameras/site) scored just 2/5 (40%) \u2014 a real, measured generalization gap. Larger sample needed for a precise number."],
    ["Installable mobile app", "Current app is a browser-based prototype by design (fast iteration). Native packaging is a scoped, known next step."],
  ];
  items.forEach((it, i) => {
    const y = 2.3 + i * 1.65;
    s.addShape(pres.ShapeType.roundRect, { x: 0.6, y, w: 12.1, h: 1.35, rectRadius: 0.06, fill: { color: PANEL }, line: { type: "none" } });
    s.addText(it[0], { x: 0.9, y: y + 0.12, w: 3.6, h: 1.1, fontFace: "Arial", fontSize: 15, bold: true, color: RED, valign: "top", margin: 0 });
    s.addText(it[1], { x: 4.6, y: y + 0.12, w: 7.8, h: 1.1, fontFace: "Arial", fontSize: 13, color: MUTED, valign: "top", margin: 0 });
  });
}

// ---------------- Slide 10: Conclusion ----------------
{
  const s = newSlide(pres);
  s.addShape(pres.ShapeType.ellipse, { x: 6.45, y: 2.2, w: 0.14, h: 0.14, fill: { color: GREEN }, line: { type: "none" } });
  s.addText("A working, real-data-validated\nparking detection system", {
    x: 0.8, y: 2.5, w: 11.7, h: 1.5, align: "center",
    fontFace: "Arial", fontSize: 34, bold: true, color: WHITE, margin: 0,
  });
  s.addText("97.04% accuracy on real PKLot data  ·  live mobile demo  ·  clearly scoped next steps", {
    x: 0.8, y: 4.1, w: 11.7, h: 0.6, align: "center",
    fontFace: "Arial", fontSize: 16, color: AMBER, margin: 0,
  });
}

pres.writeFile({ fileName: "report_assets/SmartPark_Slides.pptx" }).then(() => {
  console.log("Slides written.");
});
