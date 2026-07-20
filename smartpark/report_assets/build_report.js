const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  BorderStyle, PageBreak, TableOfContents, PageOrientation
} = require("docx");
const fs = require("fs");

const ACCENT = "2E5C3D";      // deep green
const ACCENT2 = "B5750B";     // amber-ish for headers
const DARK = "1B1E24";
const MUTED = "6B7280";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });
}
function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: "6B7280" })],
    spacing: { after: 240 },
    alignment: AlignmentType.CENTER,
  });
}
function img(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ data: fs.readFileSync(path), transformation: { width, height }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
  });
}
function imgJpg(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ data: fs.readFileSync(path), transformation: { width, height }, type: "jpg" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
  });
}

function statCell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2500, type: WidthType.DXA },
    shading: opts.shade ? { type: ShadingType.CLEAR, fill: opts.shade } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: opts.bold || false, color: opts.color || "1B1E24", size: 20 })],
      alignment: AlignmentType.CENTER,
    })],
  });
}

function resultsTable() {
  const colWidths = [3600, 2200, 2200, 2200];
  const header = new TableRow({
    children: [
      statCell("Metric", { width: colWidths[0], bold: true, shade: "24282F", color: "FFFFFF" }),
      statCell("Synthetic", { width: colWidths[1], bold: true, shade: "24282F", color: "FFFFFF" }),
      statCell("Real (PKLot)", { width: colWidths[2], bold: true, shade: "24282F", color: "FFFFFF" }),
      statCell("Spot-check", { width: colWidths[3], bold: true, shade: "24282F", color: "FFFFFF" }),
    ],
  });
  const rows = [
    ["Occupancy accuracy", "96.0%", "97.04%", "93.9%"],
    ["Images used", "60 lots", "1,242 photos", "4 held-out photos"],
    ["Labeled spots", "600", "70,684", "148"],
  ].map((r, i) => new TableRow({
    children: r.map((c, ci) => statCell(c, { width: colWidths[ci], shade: i % 2 === 0 ? "F4F3EF" : "FFFFFF" })),
  }));
  return new Table({
    width: { size: 10200, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [header, ...rows],
  });
}

const doc = new Document({
  sections: [
    // ---- Title page ----
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ text: "", spacing: { before: 2000 } }),
        new Paragraph({
          children: [new TextRun({ text: "SMART PARKING SYSTEM", bold: true, size: 56, color: DARK })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({
          children: [new TextRun({ text: "USING OBJECT DETECTION", bold: true, size: 56, color: DARK })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
        }),
        new Paragraph({
          children: [new TextRun({
            text: "A CNN-Based Real-Time Parking Occupancy Detection System — Trained and Validated on Real-World Data",
            size: 26, color: ACCENT2, italics: true,
          })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 800 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Project Report", size: 22, color: MUTED })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 80 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "July 2026", size: 22, color: MUTED })],
          alignment: AlignmentType.CENTER,
        }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },
    // ---- Main content ----
    {
      properties: {},
      children: [
        h1("1. Problem Statement"),
        body("Drivers regularly spend significant time searching for available parking spaces in cities, shopping malls, airports, and university campuses. A January 2026 industry survey of U.S. drivers found that two-thirds spend up to 15 minutes per trip searching for parking, and the average American driver loses roughly 17 hours and $345 per year to this problem — figures that climb to 107 hours and over $2,000 per year in dense cities like New York. Beyond the cost to individual drivers, unnecessary circling of parking lots adds to traffic congestion and vehicle emissions."),
        body("This project addresses the problem directly: automatically detecting which parking spaces are occupied and which are free, using computer vision applied to ordinary camera images, so that availability can be surfaced to drivers in real time."),

        h1("2. Proposed Solution"),
        body("The system classifies each individual parking spot in a camera image as Empty or Occupied using a Convolutional Neural Network (CNN), then applies a geometric boundary check to flag cars that are not properly parked within their marked space. Results are served through a REST API and displayed in a mobile app interface."),
        img("report_assets/architecture.png", 300, 662),
        caption("Figure 1. System architecture: camera input to per-spot classification to mobile app."),

        h2("2.1 Why a CNN Classifier (not full-scene object detection)"),
        body("Rather than detecting cars anywhere in an image, the system classifies each pre-marked parking spot independently as a small cropped image patch. This is the same approach used in the PKLot and CNRPark-EXT research datasets, and is significantly cheaper to run in real time than full-scene object detection, since each spot only needs a lightweight forward pass through a small CNN."),

        h1("3. Data"),
        h2("3.1 Initial Validation: Synthetic Data"),
        body("Before using real photographs, the full pipeline (data loading, CNN training, geometry-based parking-quality check, inference, and API) was validated end-to-end on a procedurally generated synthetic dataset — 60 lot images and 600 labeled parking-spot crops, including intentionally misaligned \"bad parking\" cases. This confirmed the pipeline was functioning correctly before any time was spent on real-world data cleaning."),

        h2("3.2 Real-World Data: PKLot"),
        body("The model was then retrained on real data from PKLot, a public parking-lot dataset originating from two universities in Curitiba, Brazil, collected across sunny, cloudy, and rainy days. The specific export used was a COCO-format object detection release (via Roboflow) containing 1,242 real lot photographs and 70,684 individually labeled parking spaces, each tagged space-empty or space-occupied with a pixel-accurate bounding box."),
        body("A conversion script (convert_coco_to_crops.py) was written to turn each bounding-box annotation into a cropped 64×64 image patch, matching the format expected by the training pipeline — producing 36,584 empty crops and 34,100 occupied crops with zero annotations skipped."),

        h1("4. Model & Training"),
        body("The CNN architecture (see Table 1) is intentionally small and fast: three convolutional blocks with max-pooling, followed by a dense classification head with dropout for regularization, ending in a single sigmoid output (Empty vs. Occupied)."),
        new Table({
          width: { size: 10200, type: WidthType.DXA },
          columnWidths: [5100, 5100],
          rows: [
            new TableRow({ children: [statCell("Layer", { width: 5100, bold: true, shade: "24282F", color: "FFFFFF" }), statCell("Output", { width: 5100, bold: true, shade: "24282F", color: "FFFFFF" })] }),
            ...[
              ["Input", "64×64×3"],
              ["Conv2D (32) + MaxPool", "32×32×32"],
              ["Conv2D (64) + MaxPool", "16×16×64"],
              ["Conv2D (128) + MaxPool", "8×8×128"],
              ["Dense (128) + Dropout 0.4", "128"],
              ["Dense (1, sigmoid)", "1 (Empty=0 / Occupied=1)"],
            ].map((r, i) => new TableRow({ children: r.map((c, ci) => statCell(c, { width: 5100, shade: i % 2 === 0 ? "F4F3EF" : "FFFFFF" })) })),
          ],
        }),
        caption("Table 1. CNN architecture (models/cnn_model.py)."),
        body("Training used the Adam optimizer with binary cross-entropy loss. Because a full training epoch over the real 70,684-crop dataset took longer than a single compute session allowed, training was implemented as a resumable process — the model checkpoints after each chunk of steps and reloads to continue, allowing an effectively unbounded total training time across multiple short runs."),

        h1("5. Results"),
        body("Table 2 summarizes accuracy at each stage: the synthetic sanity check, the full real-data validation set, and a held-out spot-check against four real photographs not used during training or validation."),
        resultsTable(),
        caption("Table 2. Occupancy classification accuracy by stage."),
        body("The final real-data validation accuracy of 97.04% is consistent with the accuracy levels reported in the original PKLot and CNRPark-EXT research papers, giving confidence that the approach generalizes beyond the synthetic sanity check."),

        h2("5.1 Qualitative Results on Real Photographs"),
        body("Figures 2–3 show the model's predictions drawn directly on real, unmodified PKLot photographs: green boxes indicate spots the model classified as empty, orange boxes indicate occupied."),
        imgJpg("demo/video_assets/lot1_annotated.jpg", 320, 320),
        caption("Figure 2. Real photo, 40/40 spots correctly classified."),
        imgJpg("demo/video_assets/lot0_annotated.jpg", 320, 320),
        caption("Figure 3. Real photo, 24/28 spots correctly classified."),

        h1("6. Mobile Application"),
        body("Model output is served through a Flask REST API and displayed in a mobile-styled web application (\"CampusPark\"), showing a live per-spot status grid with color coding matching the photo annotations above, tap-to-inspect confidence scores, and lot-switching."),
        img("report_assets/app_screenshot.png", 210, 410),
        caption("Figure 4. CampusPark mobile app UI, showing live results from the real-trained model."),

        h1("7. Limitations & Future Work"),
        h2("7.1 \"Properly Parked\" Detection Needs Further Work on Real Photos"),
        body("A secondary feature — flagging cars that are misaligned or crossing into a neighboring spot — was implemented using an IoU/overlap check between a detected car boundary and the marked spot boundary. This worked well on synthetic data (97.3% agreement with ground truth) but does not yet reliably localize individual cars in real, densely packed lots: both a color-distance heuristic and an edge-density heuristic were tested, and both struggle to separate one car's boundary from its immediate neighbors without a trained object detector. This is a known challenge in the field — production systems typically use a trained detector (e.g., YOLO) rather than hand-crafted rules for this sub-problem, and that is the recommended next step."),
        h2("7.2 Cross-Dataset Generalization"),
        body("The reported 97.04% accuracy is trained and validated on PKLot only. To test how well this generalizes, the PKLot-trained model was run on a small pilot sample of 5 real images from CNRPark-EXT — a dataset collected with different cameras, distances, and lighting from a different physical site."),
        body("Result: 2 of 5 correct (40%) — a substantial drop from the 97.04% seen on PKLot. While the sample is too small (n=5) to treat as a precise accuracy figure, the direction of the result is informative and expected: CNRPark-EXT's patches are cropped much tighter and closer to each vehicle than PKLot's, and the model has not seen this visual style during training. This is a concrete, measured illustration of the generalization challenge described in the literature review — the same reason production systems typically fine-tune per deployment site rather than expecting one model to work everywhere unmodified. A larger CNRPark-EXT sample (several hundred images) is needed to turn this into a precise generalization accuracy figure; the pilot here demonstrates the gap exists and is measurable.", { }),
        h2("7.3 Path to a Deployed Mobile App"),
        body("The current mobile interface is a fully interactive browser-based prototype, chosen deliberately over a native app build to keep iteration fast during development. Packaging it as an installable app (React Native, or a installable Progressive Web App) and connecting it to a persistently hosted backend are the remaining steps to a deployable product."),

        h1("8. Conclusion"),
        body("This project demonstrates a working, end-to-end parking occupancy detection system — from raw camera photographs to a live mobile interface — trained and validated on real public data with a strong, reproducible accuracy result (97.04%). The core occupancy detection is solid and production-relevant; the identified limitations (real-world \"properly parked\" detection and cross-dataset generalization) are clearly scoped, well-understood next steps rather than open unknowns."),

        h1("References"),
        body("de Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A. (2015). PKLot – A robust dataset for parking lot classification. Expert Systems with Applications.", { size: 20 }),
        body("Amato, G., Carrara, F., Falchi, F., Gennaro, C., Meghini, C., Vairo, C. (2017). Deep learning for decentralized parking lot occupancy detection. Expert Systems with Applications (CNRPark-EXT).", { size: 20 }),
        body("PKLot dataset, Roboflow Universe COCO export, https://public.roboflow.com/object-detection/pklot", { size: 20 }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("report_assets/SmartPark_Report.docx", buf);
  console.log("Report written.");
});
