/*
 * Builds the final Embedded Systems submission .docx from docs/WRITEUP.md.
 * Edits applied (per instructor brief + user instructions):
 *  - title page (authors, course, instructor, group blank)
 *  - abstract trimmed to ~250 words, keywords kept (10)
 *  - literature survey table condensed to 8 rows ([1]-[7] + old [10] -> [8]),
 *    References re-ordered so every [n] still resolves; all 12 entries retained
 *  - figures 1-6 embedded at 5.1 / 5.3 / 10.2 (x3) / 10.5 with captions + cross-refs
 *  - ASCII wiring diagram (6.2) kept as monospace block; code fences kept
 *  - [TABLE PENDING - MEASUREMENT NOT YET RUN] placeholders in 10.3 / 10.4 and
 *    the measured-inference-time row of the 10.5 table
 *  - Appendix A drive-link placeholder replaced with UCI + GitHub links
 * Format: single column, TNR 11pt body, bold numbered headings, 1.5 spacing,
 *         page numbers, three-line tables with bold header rows.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, PageNumber, NumberFormat, AlignmentType,
  HeadingLevel, WidthType, BorderStyle, ShadingType, ExternalHyperlink,
  UnderlineType, SectionType,
} = require("docx");

const ROOT = "C:/Users/DELL/threatwatch";
const OUT = path.join(ROOT, "ThreatWatch_Embedded_Systems_Writeup.docx");
const md = fs.readFileSync(path.join(ROOT, "docs/WRITEUP.md"), "utf8");

// ---------------- constants ----------------
const BODY = 22, H1S = 28, H2S = 24, CAP = 20, CODES = 18, REF = 21;
const LINE = 360; // 1.5x line spacing
const FONT = "Times New Roman";
const CODE_FONT = "Courier New";
const RED = "C00000";

// ---------------- inline markdown -> TextRun[] ----------------
function inlineRuns(text, opts = {}) {
  const size = opts.size ?? BODY;
  const runs = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|_[^_]+_)/g;
  let last = 0, m;
  const push = (t, o) => { if (t) runs.push(new TextRun({ text: t, size, font: FONT, color: "000000", ...(o || {}) })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) runs.push(new TextRun({ text: tok.slice(1, -1), size: Math.max(size - 2, 15), font: CODE_FONT, color: "000000" }));
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  if (runs.length === 0) runs.push(new TextRun({ text: " ", size, font: FONT }));
  return runs;
}

// header cell: strip markdown, force bold
function headerRuns(text, size) {
  const plain = text.replace(/\*\*|\*|`|_/g, "");
  return [new TextRun({ text: plain, bold: true, size, font: FONT, color: "000000" })];
}

// ---------------- markdown block parser ----------------
function parseMd(src) {
  const lines = src.split(/\r?\n/);
  const blocks = [];
  let cur = null;
  const flush = () => { if (cur) { blocks.push(cur); cur = null; } };
  let k = 0;
  while (k < lines.length) {
    const line = lines[k];
    if (line.startsWith("```")) {
      flush();
      const lang = line.slice(3).trim();
      const code = [];
      k++;
      while (k < lines.length && !lines[k].startsWith("```")) { code.push(lines[k]); k++; }
      blocks.push({ type: "code", lang, lines: code });
      k++; continue;
    }
    if (/^\|/.test(line)) {
      flush();
      const rows = [];
      while (k < lines.length && /^\|/.test(lines[k])) {
        const cells = lines[k].split("|").slice(1, -1).map(s => s.trim());
        if (!cells.every(c => /^:?-{2,}:?$/.test(c) || c === "")) rows.push(cells);
        k++;
      }
      blocks.push({ type: "table", rows });
      continue;
    }
    if (/^###\s/.test(line)) { flush(); blocks.push({ type: "h2", text: line.replace(/^###\s+/, "").trim() }); k++; continue; }
    if (/^##\s/.test(line)) { flush(); blocks.push({ type: "h1", text: line.replace(/^##\s+/, "").trim() }); k++; continue; }
    if (/^#\s/.test(line)) { flush(); k++; continue; } // doc title -> title page
    if (/^\*\*Author:\*\*/.test(line)) { flush(); k++; continue; }
    if (/^---\s*$/.test(line)) { flush(); k++; continue; }
    if (/^\*\s+/.test(line)) { flush(); cur = { type: "bullet", text: line.replace(/^\*\s+/, "") }; k++; continue; }
    const nm = line.match(/^(\d{1,2}[a-z]?)\.\s+(.*)$/);
    if (nm) { flush(); cur = { type: "num", marker: nm[1] + ".", text: nm[2] }; k++; continue; }
    if (/^\s*$/.test(line)) { flush(); k++; continue; }
    // paragraph / continuation
    if (line.startsWith("_Formatted for the provided template")) { // production note -> drop
      while (k < lines.length && !/^\s*$/.test(lines[k])) k++;
      continue;
    }
    if (!cur) cur = { type: "p", text: line };
    else {
      const t = line.trim();
      if (/[\u00AD-]$/.test(cur.text) && /^[a-z]/.test(t)) cur.text += t;
      else cur.text += " " + t;
    }
    k++;
  }
  flush();
  return blocks;
}

const blocks = parseMd(md);

// ---------------- edited content (new strings) ----------------
const ABSTRACT =
  "Autonomous safety wearables must detect distress without the wearer pressing anything. " +
  "Recent work reports 98\u201399.95 % accuracy for stress detection on the WESAD dataset using machine learning, " +
  "including a base paper that reaches 99.7 % with SVM and Naive Bayes [1]. Those figures, however, are obtained " +
  "with window-level cross-validation that places recordings from the same participant in both training and test " +
  "sets, and frequently with chest-mounted ECG, EMG and respiration sensors that a wrist wearable cannot " +
  "reproduce. This project builds a complete wearable-to-alert system and measures the gap between that protocol " +
  "and an honest one. A MAX30102 (PPG), MPU6050 (IMU) and SSD1306 OLED are interfaced to an ESP32 over " +
  "I\u00B2C; physiological features are fused with a camera-based YOLOv8-pose analyser; a five-state alert machine " +
  "dispatches a real notification through a cooldown- and retry-managed dispatcher. Five deep architectures " +
  "(1-D CNN, BiLSTM, Transformer, CNN+Transformer and a patch-based Vision Transformer) are compared against the " +
  "base paper\u2019s SVM and Naive Bayes under identical conditions on real WESAD wrist signals. On a " +
  "subject-disjoint test set the 1-D CNN with per-window normalisation reaches **0.617** macro-F1 and **0.709** " +
  "accuracy, against **0.486** for SVM and **0.375** for Naive Bayes, a +0.131 margin over SVM from the " +
  "normalisation change alone. The compact network is then quantised to int8 and executed **on the ESP32 " +
  "itself**, with the C implementation verified numerically against PyTorch. We conclude that the ceiling " +
  "reported in this literature is largely an artefact of evaluation protocol and sensor placement rather than a " +
  "solved problem.";
const abstractWords = ABSTRACT.replace(/\*\*/g, "").split(/\s+/).filter(Boolean).length;
if (abstractWords < 230 || abstractWords > 255) throw new Error("Abstract word count out of range: " + abstractWords);

const LEAD1 = "Figure 1 shows the high-level block diagram of the complete system, from the three I\u00B2C sensor modules through the ESP32 and the host pipeline to the vision, fusion and alert stages.";
const LEAD2 = "Figure 2 shows the software pipeline as implemented.";
const LEAD3 = "Figures 3\u20135 show the confusion matrices of the per-window 1-D CNN alongside the base paper\u2019s SVM and Naive Bayes classifiers, measured under the identical subject-disjoint protocol.";
const LEAD4 = "Figure 6 shows the confusion matrix of the compact width-32 CNN variant prepared for on-device deployment (cnn_edge32, 13\u00A0411 parameters).";

const FIGS = {
  f1: { file: "docs/figures/system_blocks.png", w: 1080, h: 780, disp: 580,
        cap: "Figure 1: High-level system block diagram \u2014 I\u00B2C sensor modules, ESP32, host inference pipeline, and the vision, fusion and alert stages." },
  f2: { file: "docs/figures/pipeline.png", w: 1275, h: 390, disp: 580,
        cap: "Figure 2: Software pipeline as implemented, from SensorSource through SensorPacket and WindowBuffer to P(stress)." },
  f3: { file: "results/cnn_wnorm_confusion.png", w: 780, h: 660, disp: 430,
        cap: "Figure 3: Confusion matrix \u2014 1-D CNN with per-window normalisation (subject-disjoint test set)." },
  f4: { file: "results/baseline_svm_wnorm_confusion.png", w: 780, h: 660, disp: 430,
        cap: "Figure 4: Confusion matrix \u2014 SVM (the base paper\u2019s classifier) with per-window normalisation (subject-disjoint test set)." },
  f5: { file: "results/baseline_naive_bayes_wnorm_confusion.png", w: 780, h: 660, disp: 430,
        cap: "Figure 5: Confusion matrix \u2014 Naive Bayes (the base paper\u2019s classifier) with per-window normalisation (subject-disjoint test set)." },
  f6: { file: "results/cnn_edge32_confusion.png", w: 780, h: 660, disp: 430,
        cap: "Figure 6: Confusion matrix \u2014 compact width-32 CNN (cnn_edge32, 13\u00A0411 parameters) prepared for on-device deployment (subject-disjoint test set)." },
};

// survey table: keep old refs 1-7 and 10; renumber to 1-8
const KEEP_MAP = { 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 10: 8 };
// references re-order: new list = old [1,2,3,4,5,6,7,10,8,9,11,12]
const REF_ORDER = [1, 2, 3, 4, 5, 6, 7, 10, 8, 9, 11, 12];

const UCI_URL = "https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection";
const GH_URL = "https://github.com/WJMatthew/WESAD";

// ---------------- table layout registry (by header first cell + column count) ----------------
function tableStyle(rows) {
  const hdr = rows[0];
  const n = hdr.length;
  const key = hdr[0] + "|" + n;
  const reg = {
    "#|5":        { widths: [6, 19, 26, 20, 29], size: 18 },   // survey
    "Component|4":{ widths: [26, 46, 13, 15], size: 20 },      // 6.1
    "Module|3":   { widths: [24, 38, 38], size: 20 },          // 6.2
    "Layer|2":    { widths: [20, 80], size: 20 },              // 7.1
    "Split|3":    { widths: [18, 41, 41], size: 20 },          // 7.4
    "Model|3":    { widths: [22, 56, 22], size: 20 },          // 7.5
    "Setting|2":  { widths: [26, 74], size: 20 },              // 7.6
    "Protocol|3": { widths: [20, 38, 42], size: 20 },          // 8.5
    "Metric|2":   { widths: [40, 60], size: 20 },              // 9.3
    "model|7":    { widths: [16, 15, 13, 13, 15, 12, 16], size: 18 }, // 10.1 sweep
    "model|4":    { widths: [32, 21, 23, 24], size: 20 },      // 10.2 comparison
    "Quantity|2": { widths: [58, 42], size: 20 },              // 10.5 edge
    "Aspect|3":   { widths: [18, 38, 44], size: 20 },          // 10.6
  };
  return reg[key] || { widths: null, size: 20 };
}

// ---------------- docx builders ----------------
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function bodyP(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: LINE },
    keepNext: opts.keepNext || undefined,
    children: inlineRuns(text),
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200, line: LINE },
    keepNext: true,
    children: [new TextRun({ text, bold: true, size: H1S, font: FONT, color: "000000" })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 320, after: 160, line: LINE },
    keepNext: true,
    children: [new TextRun({ text, bold: true, size: H2S, font: FONT, color: "000000" })],
  });
}

function bulletP(text, level = 0, size = BODY) {
  return new Paragraph({
    bullet: { level },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 80, line: LINE },
    children: inlineRuns(text, { size }),
  });
}

function numP(marker, text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { left: 620, hanging: 620 },
    spacing: { after: 80, line: LINE },
    children: [new TextRun({ text: marker + "  ", size: BODY, font: FONT, color: "000000" }), ...inlineRuns(text)],
  });
}

function codeBlock(codeLines) {
  const paras = codeLines.map(l => new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { after: 0, line: 240 },
    children: [new TextRun({ text: l.length ? l : " ", size: CODES, font: CODE_FONT, color: "000000" })],
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" },
      insideHorizontal: NB, insideVertical: NB,
    },
    rows: [new TableRow({
      cantSplit: false,
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: "FAFAFA" },
        margins: { top: 100, bottom: 100, left: 160, right: 160 },
        width: { size: 100, type: WidthType.PERCENTAGE },
        children: paras,
      })],
    })],
  });
}

function mdTable(rows) {
  const { widths, size } = tableStyle(rows);
  const nCols = rows[0].length;
  const colW = widths || Array(nCols).fill(Math.floor(100 / nCols));
  const PENDING_CELL = "@@PENDING@@";
  const mkCell = (text, isHeader, w) => new TableCell({
    width: { size: w, type: WidthType.PERCENTAGE },
    margins: { top: 40, bottom: 40, left: 100, right: 100 },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "F2F2F2" } : undefined,
    borders: isHeader ? { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000" }, top: NB, left: NB, right: NB } : { top: NB, bottom: NB, left: NB, right: NB },
    children: [new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { after: 0, line: 240 },
      children: text === PENDING_CELL
        ? [new TextRun({ text: "[PENDING \u2014 MEASUREMENT NOT YET RUN]", bold: true, size, font: FONT, color: RED })]
        : (isHeader ? headerRuns(text, size) : inlineRuns(text, { size })),
    })],
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 8, color: "000000" },
      bottom: { style: BorderStyle.SINGLE, size: 8, color: "000000" },
      left: NB, right: NB, insideHorizontal: NB, insideVertical: NB,
    },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: rows[0].map((t, i) => mkCell(t, true, colW[i])) }),
      ...rows.slice(1).map(r => new TableRow({ cantSplit: true, children: r.map((t, i) => mkCell(t, false, colW[i])) })),
    ],
  });
}

let figCount = 0;
function figure(spec) {
  figCount++;
  const buf = fs.readFileSync(path.join(ROOT, spec.file));
  const w = spec.disp;
  const h = Math.round(spec.h / spec.w * w);
  const numLabel = spec.cap.split(":")[0];
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 160, after: 60, line: 240 },
      children: [new ImageRun({ data: buf, transformation: { width: w, height: h }, type: "png" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240, line: 300 },
      children: [
        new TextRun({ text: numLabel + ": ", bold: true, size: CAP, font: FONT, color: "000000" }),
        ...inlineRuns(spec.cap.slice(numLabel.length + 2), { size: CAP }),
      ],
    }),
  ];
}

function pendingPlaceholder(label) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 240, line: LINE },
    border: {
      top: { style: BorderStyle.SINGLE, size: 6, color: RED, space: 6 },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: RED, space: 6 },
      left: { style: BorderStyle.SINGLE, size: 6, color: RED, space: 10 },
      right: { style: BorderStyle.SINGLE, size: 6, color: RED, space: 10 },
    },
    children: [new TextRun({ text: label, bold: true, size: BODY, font: FONT, color: RED })],
  });
}

function refP(n, text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { left: 480, hanging: 480 },
    spacing: { after: 60, line: LINE },
    children: [new TextRun({ text: "[" + n + "]  ", size: REF, font: FONT, color: "000000" }), ...inlineRuns(text, { size: REF })],
  });
}

// ---------------- walk blocks, apply edits, emit children ----------------
const children = [];
let h1t = null, h2t = null;
let inRefs = false;
const refEntries = {}; // oldNum -> text
let abstractInserted = false;
let pendingDone = { s103: false, s104: false };

for (let k = 0; k < blocks.length; k++) {
  const b = blocks[k];
  if (b.type === "h1") {
    h1t = b.text; h2t = null;
    inRefs = (h1t === "References");
    if (!inRefs) children.push(h1(h1t));
    else children.push(h1("References"));
    continue;
  }
  if (b.type === "h2") { h2t = b.text; children.push(h2(h2t)); continue; }

  // --- section 1: replace abstract, keep keywords ---
  if (h1t === "1. Abstract") {
    if (b.type === "p" && !abstractInserted) {
      children.push(bodyP(ABSTRACT));
      abstractInserted = true;
      continue;
    }
    if (b.type === "p" && b.text.startsWith("**Keywords:**")) { children.push(bodyP(b.text)); continue; }
    continue; // drop original abstract paragraph(s)
  }

  // --- references: collect + reorder ---
  if (inRefs) {
    if (b.type === "num") refEntries[parseInt(b.marker, 10)] = b.text;
    continue;
  }

  // --- survey table: condense to 8 rows, renumber ---
  if (h1t === "3. Literature survey" && b.type === "table") {
    const kept = [];
    for (const row of b.rows.slice(1)) {
      const m = row[0].match(/^\[(\d+)\]$/);
      if (!m) continue;
      const old = parseInt(m[1], 10);
      if (KEEP_MAP[old] !== undefined) {
        const nr = row.slice();
        nr[0] = "[" + KEEP_MAP[old] + "]";
        kept.push(nr);
      }
    }
    if (kept.length !== 8) throw new Error("Survey table must have exactly 8 rows, got " + kept.length);
    children.push(mdTable([b.rows[0], ...kept]));
    continue;
  }

  // --- 5.1 / 5.3: replace ASCII diagram with figures ---
  if (b.type === "code" && h2t && h2t.startsWith("5.1")) {
    children.push(bodyP(LEAD1));
    children.push(...figure(FIGS.f1));
    continue;
  }
  if (b.type === "code" && h2t && h2t.startsWith("5.3")) {
    children.push(bodyP(LEAD2));
    children.push(...figure(FIGS.f2));
    continue;
  }

  // --- 10.2: append comparison figures after "The comparison the brief asks for" ---
  if (h2t && h2t.startsWith("10.2") && b.type === "p" && b.text.startsWith("**The comparison the brief asks for.**")) {
    children.push(bodyP(b.text));
    children.push(bodyP(LEAD3));
    children.push(...figure(FIGS.f3));
    children.push(...figure(FIGS.f4));
    children.push(...figure(FIGS.f5));
    continue;
  }

  // --- 10.3: placeholder after the paragraph ---
  if (h2t && h2t.startsWith("10.3") && b.type === "p") {
    children.push(bodyP(b.text));
    children.push(pendingPlaceholder("[TABLE PENDING \u2014 MEASUREMENT NOT YET RUN]"));
    continue;
  }

  // --- 10.4: placeholder after the first paragraph ---
  if (h2t && h2t.startsWith("10.4") && b.type === "p" && !pendingDone.s104) {
    children.push(bodyP(b.text));
    children.push(pendingPlaceholder("[TABLE PENDING \u2014 MEASUREMENT NOT YET RUN]"));
    pendingDone.s104 = true;
    continue;
  }

  // --- 10.5: patch inference-time row + append Figure 6 after closing paragraph ---
  if (h2t && h2t.startsWith("10.5") && b.type === "table") {
    const rows = b.rows.map(r => {
      if (r[0].startsWith("Measured inference time")) return [r[0], "@@PENDING@@"];
      return r;
    });
    children.push(mdTable(rows));
    continue;
  }
  if (h2t && h2t.startsWith("10.5") && b.type === "p" && b.text.startsWith("The flash and SRAM figures")) {
    children.push(bodyP(b.text));
    children.push(bodyP(LEAD4));
    children.push(...figure(FIGS.f6));
    continue;
  }

  // --- Appendix A: replace drive-link bullet ---
  if (h1t && h1t.startsWith("Appendix A") && b.type === "bullet" && b.text.startsWith("Download link used:")) {
    children.push(bulletP("Download links used:"));
    children.push(new Paragraph({
      bullet: { level: 1 },
      spacing: { after: 60, line: LINE },
      children: [
        new TextRun({ text: "UCI Machine Learning Repository \u2014 official dataset page (external download hosted by the University of Siegen): ", size: BODY, font: FONT, color: "000000" }),
        new ExternalHyperlink({
          children: [new TextRun({ text: UCI_URL, size: BODY, font: FONT, color: "0563C1", underline: { type: UnderlineType.SINGLE } })],
          link: UCI_URL,
        }),
      ],
    }));
    children.push(new Paragraph({
      bullet: { level: 1 },
      spacing: { after: 120, line: LINE },
      children: [
        new TextRun({ text: "GitHub \u2014 public mirror and dataset loading utilities (WJMatthew/WESAD): ", size: BODY, font: FONT, color: "000000" }),
        new ExternalHyperlink({
          children: [new TextRun({ text: GH_URL, size: BODY, font: FONT, color: "0563C1", underline: { type: UnderlineType.SINGLE } })],
          link: GH_URL,
        }),
      ],
    }));
    continue;
  }

  // --- default render ---
  if (b.type === "p") {
    const next = blocks[k + 1];
    children.push(bodyP(b.text, { keepNext: next && next.type === "table" }));
  } else if (b.type === "bullet") {
    children.push(bulletP(b.text));
  } else if (b.type === "num") {
    children.push(numP(b.marker, b.text));
  } else if (b.type === "code") {
    children.push(codeBlock(b.lines));
    children.push(new Paragraph({ spacing: { after: 120, line: 240 }, children: [] }));
  } else if (b.type === "table") {
    children.push(mdTable(b.rows));
    children.push(new Paragraph({ spacing: { after: 120, line: 240 }, children: [] }));
  }
}

// flush reordered references
const refNums = Object.keys(refEntries).map(Number).sort((a, b) => a - b);
if (refNums.length !== 12) throw new Error("Expected 12 reference entries, got " + refNums.length);
REF_ORDER.forEach((oldN, idx) => {
  if (!refEntries[oldN]) throw new Error("Missing old ref " + oldN);
  children.push(refP(idx + 1, refEntries[oldN]));
});

// ---------------- title page ----------------
function titlePage() {
  const C = AlignmentType.CENTER;
  const line24 = { line: LINE };
  return [
    new Paragraph({ alignment: C, spacing: { before: 2600, after: 900, line: 400, lineRule: "atLeast" },
      children: [new TextRun({ text: "Embedded Systems \u2014 Final Project Write-Up", size: 26, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 120, line: 480, lineRule: "atLeast" },
      children: [new TextRun({ text: "IoT-Based Multimodal Distress Detection", bold: true, size: 40, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 120, line: 480, lineRule: "atLeast" },
      children: [new TextRun({ text: "on an ESP32 Wearable with", bold: true, size: 40, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 1000, line: 480, lineRule: "atLeast" },
      children: [new TextRun({ text: "Edge-Deployed Deep Learning", bold: true, size: 40, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 100, ...line24 }, children: [new TextRun({ text: "Pratik Ambastha (25BCE0337)", size: 24, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 100, ...line24 }, children: [new TextRun({ text: "Adarsh Kumar Singh (25BCE0401)", size: 24, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 700, ...line24 }, children: [new TextRun({ text: "Arunya Srivastava (25BCE0344)", size: 24, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 120, ...line24 }, children: [new TextRun({ text: "Course: Embedded Systems", size: 24, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 120, ...line24 }, children: [new TextRun({ text: "Instructor: Prof. Vivek Rajpoot", size: 24, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { after: 120, ...line24 }, children: [new TextRun({ text: "Group: ____________", size: 24, font: FONT, color: "000000" })] }),
    new Paragraph({ alignment: C, spacing: { before: 1300, ...line24 }, children: [new TextRun({ text: "October 2026", size: 24, font: FONT, color: "000000" })] }),
  ];
}

// ---------------- document ----------------
const doc = new Document({
  title: "IoT-Based Multimodal Distress Detection on an ESP32 Wearable with Edge-Deployed Deep Learning",
  creator: "Pratik Ambastha, Adarsh Kumar Singh, Arunya Srivastava",
  description: "Embedded Systems final project write-up",
  styles: {
    default: {
      document: {
        run: { font: FONT, size: BODY, color: "000000" },
        paragraph: { spacing: { line: LINE } },
      },
      heading1: {
        run: { font: FONT, size: H1S, bold: true, color: "000000" },
        paragraph: { spacing: { before: 400, after: 200, line: LINE } },
      },
      heading2: {
        run: { font: FONT, size: H2S, bold: true, color: "000000" },
        paragraph: { spacing: { before: 320, after: 160, line: LINE } },
      },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, bottom: 1440, left: 1701, right: 1417 },
        },
      },
      children: titlePage(),
    },
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, bottom: 1440, left: 1701, right: 1417, footer: 992 },
          pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
        },
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ children: [PageNumber.CURRENT], size: 20, font: FONT, color: "000000" })],
          })],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, buf.length, "bytes");
  console.log("abstract words:", abstractWords, "| figures:", figCount, "| body children:", children.length);
});
