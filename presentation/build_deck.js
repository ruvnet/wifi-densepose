// RuView (WiFi-DensePose) 簡報產生器
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const FA = require("react-icons/fa");

// ---------- palette (RF / signal dark theme) ----------
const DARK = "0A1A2F";   // deep midnight navy
const DARK2 = "112A45";  // panel navy
const CYAN = "22D3EE";   // bright signal cyan
const TEAL = "0EA5B5";
const BLUE = "3B82F6";
const INK = "1E293B";    // body on white
const MUTED = "64748B";
const AMBER = "F59E0B";  // honest-data callout
const GREEN = "10B981";  // verified
const LIGHT = "F1F5F9";
const CARD = "F8FAFC";
const LINE = "E2E8F0";
const WHITE = "FFFFFF";

const FONT = "Microsoft JhengHei";
const MONO = "Consolas";
const ASSET = "D:/RuView/assets/";

const W = 13.333, H = 7.5, MX = 0.6, CW = W - 2 * MX; // content width 12.13

const mkShadow = () => ({ type: "outer", color: "0A1A2F", blur: 7, offset: 3, angle: 135, opacity: 0.16 });

let pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: W, height: H });
pres.layout = "WIDE";
pres.author = "RuView";
pres.title = "RuView — WiFi-DensePose 專題簡報";

// ---------- icon helper ----------
async function icon(Comp, color = "#0A1A2F", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
const I = {};
async function loadIcons() {
  const map = {
    wifi: FA.FaWifi, videoSlash: FA.FaVideoSlash, battery: FA.FaBatteryQuarter, dot: FA.FaCircleNotch,
    lock: FA.FaLock, cube: FA.FaCube, moon: FA.FaMoon, dollar: FA.FaDollarSign,
    user: FA.FaUserAlt, lungs: FA.FaLungs, heart: FA.FaHeartbeat, run: FA.FaRunning,
    home: FA.FaHome, bed: FA.FaBed, users: FA.FaUsers, check: FA.FaCheckCircle,
    warn: FA.FaExclamationTriangle, chip: FA.FaMicrochip, server: FA.FaServer, brain: FA.FaBrain,
    db: FA.FaDatabase, cam: FA.FaVideo, github: FA.FaGithub, python: FA.FaPython, docker: FA.FaDocker,
    npm: FA.FaNpm, robot: FA.FaRobot, book: FA.FaBook, link: FA.FaLink, bolt: FA.FaBolt,
    layer: FA.FaLayerGroup, wave: FA.FaBroadcastTower, cog: FA.FaCog, flask: FA.FaFlask,
  };
  // colors decided at call site; render default navy + we re-render where needed
  for (const k of Object.keys(map)) I[k] = map[k];
}
async function ic(name, color) { return await icon(I[name], color); }

// ---------- shared chrome ----------
function signalRings(slide, cx, cy, color = CYAN) {
  // WiFi-signal motif: concentric circle outlines (decorative, topic-specific)
  [3.4, 2.4, 1.5, 0.7].forEach((r, i) => {
    slide.addShape(pres.shapes.OVAL, {
      x: cx - r, y: cy - r, w: r * 2, h: r * 2,
      fill: { type: "solid", color: DARK, transparency: 100 },
      line: { color, width: 1.25, transparency: 35 + i * 12 },
    });
  });
  slide.addShape(pres.shapes.OVAL, { x: cx - 0.09, y: cy - 0.09, w: 0.18, h: 0.18, fill: { color } });
}

function contentHeader(slide, num, title, sub) {
  slide.background = { color: WHITE };
  // number badge
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: MX, y: 0.46, w: 0.92, h: 0.92, rectRadius: 0.12, fill: { color: CYAN }, shadow: mkShadow(),
  });
  slide.addText(num, { x: MX, y: 0.46, w: 0.92, h: 0.92, align: "center", valign: "middle",
    fontFace: MONO, fontSize: num.length > 3 ? 18 : 22, bold: true, color: DARK, margin: 0 });
  // title
  slide.addText(title, { x: 1.72, y: 0.44, w: CW - 1.2, h: sub ? 0.66 : 0.96, valign: "middle",
    fontFace: FONT, fontSize: 30, bold: true, color: DARK, margin: 0 });
  if (sub) slide.addText(sub, { x: 1.74, y: 1.08, w: CW - 1.2, h: 0.38, valign: "middle",
    fontFace: FONT, fontSize: 13.5, color: TEAL, margin: 0 });
}

function footer(slide, n) {
  slide.addText("RuView · WiFi-DensePose", { x: MX, y: 7.06, w: 6, h: 0.3, fontFace: FONT, fontSize: 9, color: MUTED, margin: 0 });
  slide.addText(`${String(n).padStart(2, "0")} / 17`, { x: W - 2.1, y: 7.06, w: 1.5, h: 0.3, align: "right", fontFace: MONO, fontSize: 9, color: MUTED, margin: 0 });
}

function styledTable(slide, head, rows, opts) {
  const o = Object.assign({ x: MX, y: 1.6, w: CW, colW: null, fontSize: 12, rowH: 0.42, headFill: DARK }, opts);
  const headRow = head.map(t => ({ text: t, options: { fill: { color: o.headFill }, color: WHITE, bold: true, fontFace: FONT, fontSize: o.fontSize + 0.5, align: "left", valign: "middle" } }));
  const body = rows.map((r, ri) => r.map((c, ci) => {
    const cell = (typeof c === "object") ? c : { text: String(c) };
    return { text: cell.text, options: Object.assign({
      fill: { color: ri % 2 ? CARD : WHITE }, color: cell.color || INK,
      bold: cell.bold || false, fontFace: cell.mono ? MONO : FONT, fontSize: o.fontSize,
      align: cell.align || "left", valign: "middle",
    }, cell.opt || {}) };
  }));
  slide.addTable([headRow, ...body], {
    x: o.x, y: o.y, w: o.w, colW: o.colW, border: { type: "solid", pt: 0.75, color: LINE },
    rowH: o.rowH, valign: "middle", autoPage: false,
  });
}

function statCard(slide, x, y, w, h, number, label, col) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
  slide.addShape(pres.shapes.RECTANGLE, { x, y: y + h - 0.07, w, h: 0.07, fill: { color: col || CYAN } });
  slide.addText(number, { x: x + 0.1, y: y + 0.12, w: w - 0.2, h: h * 0.52, align: "center", valign: "middle", fontFace: MONO, fontSize: 27, bold: true, color: col || DARK, margin: 0 });
  slide.addText(label, { x: x + 0.1, y: y + h * 0.6, w: w - 0.2, h: h * 0.32, align: "center", valign: "top", fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 });
}

async function iconChip(slide, x, y, dia, name, col) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: dia, h: dia, fill: { color: col } });
  const pad = dia * 0.26;
  slide.addImage({ data: await ic(name, "#FFFFFF"), x: x + pad, y: y + pad, w: dia - 2 * pad, h: dia - 2 * pad });
}

// ===================================================================
async function build() {
  await loadIcons();

  // ---------- Slide 0 : Cover ----------
  {
    const s = pres.addSlide();
    s.background = { color: DARK };
    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: DARK } });
    signalRings(s, 11.7, 1.2);
    signalRings(s, 0.4, 7.3);
    // image panel
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.05, y: 1.5, w: 4.7, h: 4.5, rectRadius: 0.1, fill: { color: DARK2 }, line: { color: CYAN, width: 1.25 }, shadow: mkShadow() });
    s.addImage({ path: ASSET + "v2-screen.png", x: 8.2, y: 1.64, w: 4.4, h: 4.4 * (1653 / 1996), sizing: { type: "contain", w: 4.4, h: 3.64 } });
    s.addText("即時 WiFi CSI 姿態骨架", { x: 8.05, y: 5.62, w: 4.7, h: 0.34, align: "center", fontFace: FONT, fontSize: 10.5, italic: true, color: CYAN, margin: 0 });
    // left text
    s.addImage({ data: await ic("wifi", "#22D3EE"), x: MX, y: 1.55, w: 0.62, h: 0.62 });
    s.addText("RuView", { x: MX, y: 2.25, w: 7.2, h: 1.15, fontFace: FONT, fontSize: 60, bold: true, color: WHITE, margin: 0 });
    s.addText("用 WiFi 訊號看見人", { x: MX + 0.02, y: 3.42, w: 7.2, h: 0.7, fontFace: FONT, fontSize: 30, bold: true, color: CYAN, margin: 0 });
    s.addText("WiFi-DensePose:基於通道狀態資訊 (CSI) 的人體感測系統", { x: MX + 0.02, y: 4.18, w: 7.2, h: 0.5, fontFace: FONT, fontSize: 15, color: "CADCFC", margin: 0 });
    s.addText([
      { text: "無攝影機", options: { color: WHITE, fontFace: FONT } },
      { text: "  ·  ", options: { color: TEAL } },
      { text: "無穿戴裝置", options: { color: WHITE, fontFace: FONT } },
      { text: "  ·  ", options: { color: TEAL } },
      { text: "可穿牆", options: { color: WHITE, fontFace: FONT } },
      { text: "  ·  ", options: { color: TEAL } },
      { text: "隱私友善", options: { color: WHITE, fontFace: FONT } },
    ], { x: MX + 0.02, y: 5.0, w: 7.2, h: 0.4, fontSize: 14, bold: true, margin: 0 });
    s.addText("專題報告  ·  2026", { x: MX + 0.02, y: 6.15, w: 7, h: 0.4, fontFace: MONO, fontSize: 12, color: MUTED, margin: 0 });
    s.addNotes("開場:大家好,今天報告 RuView——用 WiFi 訊號看見人。不需攝影機、不需穿戴、能穿牆。靠的是 WiFi 電波加上物理。整套系統把每天都在身邊的 WiFi,變成空間感知能力。");
  }

  // ---------- Slide 1 : 研究動機 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "1", "研究動機:為什麼用 WiFi 感測人體?", "如何在不侵犯隱私下,持續、可靠地感測室內的人?");
    // pain cards
    const pains = [
      ["videoSlash", "攝影機", "隱私疑慮、怕黑、不能穿牆,每點 $200–$2000", "EF4444"],
      ["battery", "穿戴裝置", "需配戴、要充電,長者/病患配合度低", "F59E0B"],
      ["dot", "紅外 / PIR", "只知「有沒有動」,測不到呼吸與姿態", "8B5CF6"],
    ];
    s.addText("現有方案都有硬傷", { x: MX, y: 1.62, w: 5.6, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: MUTED, margin: 0 });
    let py = 2.12;
    for (const [ico, t, d, col] of pains) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: py, w: 5.75, h: 1.34, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
      await iconChip(s, MX + 0.24, py + 0.34, 0.66, ico, "#" + col);
      s.addText(t, { x: MX + 1.12, y: py + 0.18, w: 4.4, h: 0.42, fontFace: FONT, fontSize: 17, bold: true, color: DARK, margin: 0 });
      s.addText(d, { x: MX + 1.12, y: py + 0.62, w: 4.5, h: 0.62, fontFace: FONT, fontSize: 12.5, color: INK, margin: 0 });
      py += 1.52;
    }
    // right: opportunity + value
    const rx = 6.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.62, w: CW - rx + MX, h: 1.55, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    signalRings(s, rx + 5.4, 1.5, CYAN);
    s.addText("機會:WiFi 無所不在", { x: rx + 0.3, y: 1.82, w: 5.5, h: 0.4, fontFace: FONT, fontSize: 17, bold: true, color: CYAN, margin: 0 });
    s.addText("人移動、呼吸,甚至靜止,都會擾動電波。只要一台 Raspberry Pi 4 搭配 nexmon_csi 讀取 WiFi 的「通道狀態資訊 (CSI)」就能量測。", { x: rx + 0.3, y: 2.28, w: 5.45, h: 0.85, fontFace: FONT, fontSize: 13.5, color: "E2E8F0", margin: 0 });
    const vals = [
      ["lock", "無影像", "天生規避 GDPR / HIPAA 影像隱私法規"],
      ["cube", "可穿牆", "穿透牆面、家具、瓦礫;全黑亦可運作"],
      ["dollar", "成本低", "一台樹莓派約 $50,遠低於攝影機 $200–2000"],
    ];
    let vy = 3.45;
    for (const [ico, t, d] of vals) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: vy, w: CW - rx + MX, h: 1.04, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 } });
      await iconChip(s, rx + 0.22, vy + 0.27, 0.5, ico, TEAL);
      s.addText(t, { x: rx + 0.92, y: vy + 0.12, w: 1.6, h: 0.8, valign: "middle", fontFace: FONT, fontSize: 16, bold: true, color: TEAL, margin: 0 });
      s.addText(d, { x: rx + 2.5, y: vy + 0.12, w: 3.2, h: 0.8, valign: "middle", fontFace: FONT, fontSize: 12.5, color: INK, margin: 0 });
      vy += 1.16;
    }
    footer(s, 2);
    s.addNotes("問題:在隱私前提下持續感測室內的人。現有方案各有硬傷——攝影機怕黑、不能穿牆、貴、有隱私問題;穿戴要戴要充電;PIR 只知有沒有動。我們的切入點:WiFi 無所不在,人一動就擾動電波,一台約 $50 的 Raspberry Pi 4 加 nexmon_csi 讀 CSI 即可。三大價值:無影像規避法規、能穿牆全黑、成本遠低於攝影機。");
  }

  // ---------- Slide 2 : 系統簡介 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2", "系統簡介:RuView 系統總覽", "一句話:把一台普通的 WiFi 變成非接觸式感測器");
    const caps = [
      ["user", "存在 / 人數", "穿牆偵測、計數、進出", CYAN],
      ["lungs", "生命徵象", "呼吸 6–30 BPM、心率 40–120 BPM", TEAL],
      ["run", "姿態 / 活動", "17 關鍵點骨架、走路、跌倒", BLUE],
      ["home", "室內地圖", "RF 指紋辨識房間、物件變動", "8B5CF6"],
      ["bed", "睡眠品質", "睡眠分期、呼吸中止篩檢", "EC4899"],
    ];
    const cw = 2.27, gap = 0.18, cx0 = MX, cy = 1.72, ch = 2.2;
    for (let i = 0; i < caps.length; i++) {
      const [ico, t, d, col] = caps[i];
      const x = cx0 + i * (cw + gap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: cy, w: cw, h: ch, rectRadius: 0.09, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
      s.addShape(pres.shapes.RECTANGLE, { x, y: cy, w: cw, h: 0.09, fill: { color: col } });
      await iconChip(s, x + cw / 2 - 0.42, cy + 0.3, 0.84, ico, col);
      s.addText(t, { x: x + 0.1, y: cy + 1.2, w: cw - 0.2, h: 0.4, align: "center", fontFace: FONT, fontSize: 15, bold: true, color: DARK, margin: 0 });
      s.addText(d, { x: x + 0.12, y: cy + 1.6, w: cw - 0.24, h: 0.55, align: "center", valign: "top", fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 });
    }
    // bottom info band
    const by = 4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: by, w: CW, h: 2.25, rectRadius: 0.09, fill: { color: DARK }, shadow: mkShadow() });
    signalRings(s, 12.4, 6.9, CYAN);
    s.addText([
      { text: "完全在邊緣運算", options: { bold: true, color: CYAN, fontFace: FONT } },
      { text: "  —  無雲端、無攝影機,連網路斷了也能運作。", options: { color: "E2E8F0", fontFace: FONT } },
    ], { x: MX + 0.4, y: by + 0.32, w: CW - 0.8, h: 0.5, fontSize: 16, valign: "middle", margin: 0 });
    s.addText([
      { text: "整合生態:", options: { bold: true, color: WHITE, fontFace: FONT } },
      { text: "一鍵接 Home Assistant / Matter(Apple Home、Google Home、Alexa);可開「隱私模式」只輸出語意狀態。", options: { color: "CBD5E1", fontFace: FONT } },
    ], { x: MX + 0.4, y: by + 0.92, w: CW - 1.6, h: 0.5, fontSize: 13.5, valign: "middle", margin: 0 });
    s.addText([
      { text: "雙程式碼庫:", options: { bold: true, color: WHITE, fontFace: FONT } },
      { text: "Python v1(原型)  +  ", options: { color: "CBD5E1", fontFace: FONT } },
      { text: "Rust v2(主力,效能約 810×)", options: { color: CYAN, bold: true, fontFace: FONT } },
    ], { x: MX + 0.4, y: by + 1.5, w: CW - 1.6, h: 0.5, fontSize: 13.5, valign: "middle", margin: 0 });
    footer(s, 3);
    s.addNotes("一句話:普通 WiFi 變成非接觸感測器。五大能力:存在與人數、呼吸心率、姿態與跌倒、室內地圖、睡眠品質。重點:全部在邊緣運算,不需雲端攝影機,斷網也能跑,還能接 Apple、Google Home。系統有兩套碼:Python 原型,加上作為主力、Rust 重寫、快約八百倍的 v2。");
  }

  // ---------- Slide 3 : 2.1 硬體 (Raspberry Pi 4 單機) ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2.1", "軟硬體架構(一):Raspberry Pi 4 單機感測", "nexmon_csi 直接擷取 WiFi CSI;擷取與推論在同一台 Pi 4 完成");
    // left: Pi 4 spec table
    styledTable(s, ["項目", "規格"], [
      [{ text: "裝置", bold: true }, "Raspberry Pi 4(4 / 8 GB RAM)"],
      [{ text: "WiFi 晶片", bold: true }, "Broadcom BCM43455c0(1×1,802.11ac)"],
      [{ text: "CSI 擷取", bold: true }, { text: "nexmon_csi 韌體修補 · monitor mode", color: TEAL }],
      [{ text: "子載波", bold: true }, "逐子載波複數 CSI,最高 80 MHz / 256"],
      [{ text: "運算", bold: true }, "Cortex-A72 四核 @ 1.5 GHz"],
      [{ text: "角色", bold: true }, { text: "擷取 + 推論一機完成(本地 daemon)", color: TEAL, bold: true }],
    ], { y: 1.66, w: 7.2, colW: [1.45, 5.75], rowH: 0.46, fontSize: 11.5 });
    s.addText([
      { text: "平台亦相容  ", options: { color: MUTED, fontFace: FONT } },
      { text: "ESP32-S3 · Intel 5300 · Atheros AR9580", options: { color: INK, fontFace: FONT } },
      { text: "(本實作採用 Pi 4)", options: { color: MUTED, fontFace: FONT } },
    ], { x: MX, y: 4.95, w: 7.2, h: 0.36, fontSize: 11, margin: 0 });
    // single-machine end-to-end pipeline box
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.4, w: 7.2, h: 1.25, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    s.addText("單機端到端管線(全在一台 Pi 4)", { x: MX + 0.3, y: 5.52, w: 6.7, h: 0.36, fontFace: FONT, fontSize: 13.5, bold: true, color: CYAN, margin: 0 });
    s.addText("WiFi → nexmon CSI 擷取 → rvCSI 正規化 → 訊號處理 → 模型推論 → 輸出", { x: MX + 0.3, y: 5.92, w: 6.7, h: 0.36, fontFace: FONT, fontSize: 11.5, color: "E2E8F0", margin: 0 });
    s.addText("無需額外感測節點,也無需外部伺服器", { x: MX + 0.3, y: 6.26, w: 6.7, h: 0.32, fontFace: FONT, fontSize: 11, italic: true, color: "94A3B8", margin: 0 });
    // right: screenshot + spec cards + deployment
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.0, y: 1.66, w: 4.73, h: 2.25, rectRadius: 0.1, fill: { color: DARK2 }, line: { color: CYAN, width: 1.25 }, shadow: mkShadow() });
    s.addImage({ path: ASSET + "screenshot.png", x: 8.13, y: 1.84, w: 4.47, h: 1.62, sizing: { type: "contain", w: 4.47, h: 1.62 } });
    s.addText("系統即時感測畫面", { x: 8.0, y: 3.56, w: 4.73, h: 0.3, align: "center", fontFace: FONT, fontSize: 10, italic: true, color: CYAN, margin: 0 });
    s.addText("重點規格", { x: 8.0, y: 4.04, w: 4.73, h: 0.34, fontFace: FONT, fontSize: 13, bold: true, color: MUTED, margin: 0 });
    statCard(s, 8.0, 4.46, 1.5, 1.0, "256", "子載波上限", CYAN);
    statCard(s, 9.62, 4.46, 1.5, 1.0, "$35-55", "Pi 4 單機", TEAL);
    statCard(s, 11.24, 4.46, 1.49, 1.0, "1 台", "擷取+推論", BLUE);
    s.addText([
      { text: "部署:", options: { bold: true, color: DARK, fontFace: FONT } },
      { text: "單台 Raspberry Pi 4 — nexmon_csi 擷取 + Candle 本地推論,無 ESP32 mesh、無外部伺服器。", options: { color: INK, fontFace: FONT } },
    ], { x: 8.0, y: 5.62, w: 4.73, h: 1.0, valign: "top", fontSize: 11.5, margin: 0 });
    footer(s, 4);
    s.addNotes("我們的實作裝置是一台 Raspberry Pi 4。它內建的 Broadcom BCM43455c0 WiFi 晶片,透過 nexmon_csi 韌體修補,可在 monitor mode 下直接擷取逐子載波的複數 CSI,而且是 802.11ac、最高 80 MHz、最多 256 個子載波,比 ESP32 的 64 個多很多。最重要的是:CSI 擷取跟模型推論都在同一台 Pi 上完成——nexmon 擷取、rvCSI 正規化、訊號處理、Candle 本地推論,整條管線單機跑完,不需額外感測節點或外部伺服器。單台 Pi 4 約三十五到五十五美金。平台本身也相容 ESP32-S3、Intel 5300、Atheros 等 CSI 來源,並可擴展為多基地台 mesh,但本次我們用單台 Pi 4 驗證核心管線。");
  }

  // ---------- Slide 4 : 2.1 軟體 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2.1", "軟硬體架構(二):15 個 Rust Crate 分層", "嚴格分層的模組化架構,從韌體到應用一氣呵成");
    const layers = [
      ["應用層", "mat(災難搜救) · wasm(瀏覽器) · sensing-server(Axum) · cli", BLUE],
      ["訊號 / AI 層", "signal(SOTA DSP + RuvSense 14 模組) · nn(ONNX/Candle) · ruvector · train", TEAL],
      ["基礎層", "core(型別 / CSI frame) · vitals · wifiscan · hardware(TDM / 封包解析)", DARK],
    ];
    let ly = 1.72;
    for (const [t, d, col] of layers) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: ly, w: 7.5, h: 1.18, rectRadius: 0.08, fill: { color: col }, shadow: mkShadow() });
      s.addText(t, { x: MX + 0.3, y: ly + 0.16, w: 7.0, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: CYAN, margin: 0 });
      s.addText(d, { x: MX + 0.3, y: ly + 0.58, w: 7.0, h: 0.5, fontFace: MONO, fontSize: 11.5, color: "E2E8F0", margin: 0 });
      ly += 1.32;
    }
    s.addText("依賴方向:基礎層 → 訊號/AI 層 → 應用層", { x: MX, y: ly + 0.0, w: 7.5, h: 0.34, align: "center", fontFace: FONT, fontSize: 11, italic: true, color: MUTED, margin: 0 });
    // right: tech stack + perf
    const rx = 8.35, rw = CW - rx + MX;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.72, w: rw, h: 2.95, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
    s.addText("關鍵技術棧", { x: rx + 0.28, y: 1.9, w: rw - 0.5, h: 0.4, fontFace: FONT, fontSize: 14, bold: true, color: DARK, margin: 0 });
    s.addText([
      { text: "Rust(主力)+ Python(pip install ruview)", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "推論:ONNX / PyTorch / Candle 三後端", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "AI 骨幹:RuVector(注意力 / 圖論 / 壓縮)", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "模型:RVF 單檔簽章格式 · Server:Axum", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "整合:MQTT + Matter + 隱私模式", options: { bullet: { code: "2022" }, fontFace: FONT } },
    ], { x: rx + 0.3, y: 2.34, w: rw - 0.55, h: 2.2, fontSize: 12, color: INK, margin: 0, paraSpaceAfter: 6 });
    s.addText("Rust 重寫的效能成果", { x: rx, y: 4.85, w: rw, h: 0.36, fontFace: FONT, fontSize: 13, bold: true, color: MUTED, margin: 0 });
    statCard(s, rx, 5.28, (rw - 0.2) / 2, 1.3, "100 MB", "記憶體 (vs 500 MB)", TEAL);
    statCard(s, rx + (rw - 0.2) / 2 + 0.2, 5.28, (rw - 0.2) / 2, 1.3, "132 MB", "Docker (vs 569 MB)", BLUE);
    footer(s, 5);
    s.addNotes("十五個 Rust 模組分三層:基礎層是核心型別、CSI 影格、硬體溝通;中間是訊號與 AI 層,含 SOTA 訊號處理、十四個感測模組、神經網路、RuVector 骨幹跟訓練;最上是應用。技術棧:Rust 加 Python、Candle 推論、RVF 簽章模型、Axum、MQTT/Matter 加隱私模式。Rust 讓記憶體只用一百 MB、Docker 一百三十二 MB。");
  }

  // ---------- Slide 5 : 2.2 CSI 原理 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2.2", "工作原理(一):CSI 為何能「看見」人", "Channel State Information — 通道狀態資訊");
    // left text
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 1.72, w: 6.6, h: 2.3, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
    s.addText("CSI 是什麼?", { x: MX + 0.28, y: 1.9, w: 6.0, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: DARK, margin: 0 });
    s.addText([
      { text: "WiFi 用 OFDM 把訊號拆成數十個「子載波」", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "每個子載波記錄振幅衰減 + 相位偏移(複數值)", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "vs RSSI 只有 1 個純量 → CSI 有 56–256 子載波 × 複數相位,資訊量大幾個數量級", options: { bullet: { code: "2022" }, fontFace: FONT } },
    ], { x: MX + 0.3, y: 2.34, w: 6.1, h: 1.6, fontSize: 13, color: INK, margin: 0, paraSpaceAfter: 6 });
    // challenge box
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 4.2, w: 6.6, h: 2.35, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    await iconChip(s, MX + 0.28, 4.42, 0.56, "bolt", AMBER);
    s.addText("人為什麼會擾動訊號?", { x: MX + 1.0, y: 4.42, w: 5.4, h: 0.56, valign: "middle", fontFace: FONT, fontSize: 15, bold: true, color: CYAN, margin: 0 });
    s.addText("人體是反射 / 吸收體,改變電波的多重路徑傳播。", { x: MX + 0.3, y: 5.05, w: 6.0, h: 0.4, fontFace: FONT, fontSize: 12.5, color: "E2E8F0", margin: 0 });
    s.addText([
      { text: "🫁 呼吸:胸腔位移 1–5 mm @ 0.1–0.5 Hz", options: { breakLine: true, color: "E2E8F0", fontFace: FONT } },
      { text: "💓 心跳:體表僅 0.1–0.5 mm @ 0.8–2.0 Hz(更難)", options: { color: "FBBF24", fontFace: FONT } },
    ], { x: MX + 0.3, y: 5.5, w: 6.0, h: 0.95, fontSize: 12.5, margin: 0, paraSpaceAfter: 4 });
    // right: signal rings illustration + challenge note
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.5, y: 1.72, w: CW - 7.5 + MX, h: 3.05, rectRadius: 0.1, fill: { color: DARK2 }, shadow: mkShadow() });
    signalRings(s, 8.7, 3.25, CYAN);
    s.addShape(pres.shapes.OVAL, { x: 10.7, y: 2.95, w: 0.6, h: 0.6, fill: { color: AMBER } });
    s.addText("人體", { x: 10.55, y: 3.55, w: 0.9, h: 0.3, align: "center", fontFace: FONT, fontSize: 10, color: "FBBF24", margin: 0 });
    s.addText("TX", { x: 8.5, y: 3.1, w: 0.5, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, bold: true, color: WHITE, margin: 0 });
    s.addText("電波經人體散射 → CSI 改變", { x: 7.7, y: 4.3, w: 5.4, h: 0.34, align: "center", fontFace: FONT, fontSize: 11, italic: true, color: CYAN, margin: 0 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.5, y: 4.95, w: CW - 7.5 + MX, h: 1.6, rectRadius: 0.08, fill: { color: "FEF3C7" }, line: { color: AMBER, width: 1.25 }, shadow: mkShadow() });
    await iconChip(s, 7.74, 5.18, 0.5, "warn", AMBER);
    s.addText("工程挑戰", { x: 8.4, y: 5.18, w: 4.4, h: 0.5, valign: "middle", fontFace: FONT, fontSize: 14, bold: true, color: "92400E", margin: 0 });
    s.addText("商用晶片的原始相位含隨機硬體誤差(CFO / SFO),每個封包都不同 → 必須先做訊號清理才能用。", { x: 7.74, y: 5.72, w: 5.0, h: 0.75, fontFace: FONT, fontSize: 12, color: "92400E", margin: 0 });
    footer(s, 6);
    s.addNotes("關鍵在 CSI。WiFi 用 OFDM 拆成數十個子載波,CSI 記錄每個子載波的振幅衰減跟相位偏移,是複數值;相比 RSSI 只有一個數字,資訊量大好幾個數量級。人會擾動它,是因為身體是反射吸收體,改變多徑。呼吸胸腔起伏一到五毫米;心跳只動零點幾毫米,所以更難。挑戰:便宜晶片原始相位有硬體誤差,每包都不同,要先清理。");
  }

  // ---------- Slide 6 : 2.2 流程 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2.2", "工作原理(二):端到端訊號處理流程", "nexmon 擷取 → 正規化 → 把關 → 6 大 SOTA 演算法 → 推論");
    // flow row
    const steps = ["nexmon\nCSI 擷取", "rvCSI\n正規化", "品質\n相干閘", "DSP × 6\nSOTA", "模型推論\nCandle", "輸出\n姿態+生命徵象"];
    const bw = 1.78, bgap = 0.27, fy = 1.62, bh = 0.82;
    for (let i = 0; i < steps.length; i++) {
      const x = MX + i * (bw + bgap);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: fy, w: bw, h: bh, rectRadius: 0.08, fill: { color: i === 3 ? CYAN : DARK }, shadow: mkShadow() });
      s.addText(steps[i], { x: x + 0.04, y: fy, w: bw - 0.08, h: bh, align: "center", valign: "middle", fontFace: FONT, fontSize: 10.5, bold: true, color: i === 3 ? DARK : WHITE, margin: 0, lineSpacingMultiple: 0.9 });
      if (i < steps.length - 1) s.addText("›", { x: x + bw - 0.02, y: fy, w: bgap + 0.04, h: bh, align: "center", valign: "middle", fontFace: MONO, fontSize: 20, bold: true, color: TEAL, margin: 0 });
    }
    // algorithm table
    styledTable(s, ["訊號處理演算法", "作用", "出處論文"], [
      [{ text: "共軛相乘 (CSI Ratio)", bold: true }, "消除 CFO / SFO 硬體相位誤差", { text: "SpotFi", color: TEAL, mono: true }],
      [{ text: "Hampel 濾波", bold: true }, "抗 50% 污染的離群值移除", { text: "WiDance", color: TEAL, mono: true }],
      [{ text: "Fresnel 區模型", bold: true }, "依 TX-人體-RX 幾何預測呼吸", { text: "FarSense", color: TEAL, mono: true }],
      [{ text: "STFT 頻譜圖", bold: true }, "時頻分析(呼吸帶 / 走路帶)", { text: "—", color: MUTED, mono: true }],
      [{ text: "子載波敏感度選擇", bold: true }, "選最佳子載波,SNR +6–10 dB", { text: "WiDance", color: TEAL, mono: true }],
      [{ text: "身體速度分布 (BVP)", bold: true }, "域獨立 → 跨環境辨識基礎", { text: "Widar 3.0", color: TEAL, mono: true }],
    ], { y: 2.78, w: CW, colW: [3.3, 6.4, 2.43], rowH: 0.55, fontSize: 12.5 });
    s.addText([
      { text: "輕量管線:", options: { bold: true, color: DARK, fontFace: FONT } },
      { text: "nexmon 擷取 → rvCSI → 訊號處理 → Candle 推論,", options: { color: INK, fontFace: FONT } },
      { text: "全在單台 Pi 4 上即時完成(無多節點融合開銷)", options: { color: TEAL, bold: true, fontFace: FONT } },
    ], { x: MX, y: 6.5, w: CW, h: 0.4, fontSize: 12.5, align: "center", margin: 0 });
    footer(s, 7);
    s.addNotes("核心流程:WiFi 電波經人體散射,由 Pi 4 用 nexmon 擷取 CSI;rvCSI 正規化後,經品質相干閘把關,再進六個 SOTA 演算法做訊號處理,最後 Candle 本地推論輸出。六個演算法各對應經典論文:共軛相乘消硬體誤差來自 SpotFi、Hampel 去離群、Fresnel 預測呼吸來自 FarSense、頻譜圖、子載波選擇提升六到十 dB、身體速度分布跟房間無關來自 Widar 3.0。整條管線都在這一台樹莓派上即時跑完。");
  }

  // ---------- Slide 7 : 2.3 資料集 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2.3", "資料集建立方式", "公開資料集 bootstrap + 相機自監督配對 + 合成資料");
    styledTable(s, ["", "MM-Fi(主)", "Wi-Pose(次)"], [
      [{ text: "出處", bold: true }, "NeurIPS 2023", "MDPI Entropy 2023"],
      [{ text: "規模", bold: true }, "40 人 × 27 動作 × ~320K 幀", "12 人 × 12 動作 × 166,600 封包"],
      [{ text: "模態", bold: true }, "CSI + mmWave + LiDAR + RGB-D", "僅 CSI"],
      [{ text: "CSI 規格", bold: true }, "1T×3R,114 子載波", "3T×3R,30 子載波"],
      [{ text: "標註", bold: true }, "17 COCO + DensePose UV", "18 keypoint"],
    ], { y: 1.72, w: 7.3, colW: [1.3, 3.3, 2.7], rowH: 0.5, fontSize: 11.5 });
    // right: self-built process
    const rx = 8.05, rw = CW - rx + MX;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.72, w: rw, h: 2.55, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    await iconChip(s, rx + 0.26, 1.94, 0.54, "cam", CYAN);
    s.addText("自建:相機 Ground-Truth 配對", { x: rx + 0.92, y: 1.94, w: rw - 1.1, h: 0.54, valign: "middle", fontFace: FONT, fontSize: 13.5, bold: true, color: CYAN, margin: 0 });
    s.addText([
      { text: "webcam → MediaPipe(33 點)→ 17 COCO 當「教師訊號」", options: { breakLine: true, color: "E2E8F0", fontFace: FONT, bullet: { code: "2022" } } },
      { text: "200 ms 時間窗對齊 CSI 與相機", options: { breakLine: true, color: "E2E8F0", fontFace: FONT, bullet: { code: "2022" } } },
      { text: "相機只用於訓練,部署時完全移除", options: { breakLine: true, color: CYAN, bold: true, fontFace: FONT, bullet: { code: "2022" } } },
      { text: "🔒 原始影像永不存檔,只存關節座標", options: { color: "E2E8F0", fontFace: FONT, bullet: { code: "2022" } } },
    ], { x: rx + 0.3, y: 2.5, w: rw - 0.55, h: 1.7, fontSize: 11.5, margin: 0, paraSpaceAfter: 5 });
    // bottom band: self-supervised + numbers
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 4.95, w: 7.3, h: 1.65, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
    s.addText("自監督 / 合成資料", { x: MX + 0.28, y: 5.1, w: 6.7, h: 0.36, fontFace: FONT, fontSize: 13, bold: true, color: DARK, margin: 0 });
    s.addText([
      { text: "整夜真實採集 → 60,630 樣本 → 610K 對比三元組", options: { breakLine: true, fontFace: FONT, bullet: { code: "2022" } } },
      { text: "確定性合成 CSI(seed=42)→ SHA-256 可重現驗證", options: { breakLine: true, fontFace: FONT, bullet: { code: "2022" } } },
      { text: "前處理:逐幀 z-score、子載波統一內插到 56、相位淨化", options: { fontFace: FONT, bullet: { code: "2022" } } },
    ], { x: MX + 0.3, y: 5.5, w: 6.9, h: 1.05, fontSize: 11.5, color: INK, margin: 0, paraSpaceAfter: 4 });
    statCard(s, 8.05, 4.95, rw / 2 - 0.1, 1.65, "60,630", "真實樣本", TEAL);
    statCard(s, 8.05 + rw / 2 + 0.1, 4.95, rw / 2 - 0.1, 1.65, "610K", "對比三元組", BLUE);
    footer(s, 8);
    s.addNotes("三個來源。公開資料集做初始訓練:MM-Fi 是 NeurIPS 2023 最大的 WiFi CSI 加姿態資料集,四十人、二十七動作、約三十二萬幀、多模態;Wi-Pose 的三乘三天線剛好對上我們硬體。自建是最有特色的:收資料時用 webcam 透過 MediaPipe 偵測關鍵點當正確答案教 WiFi,兩百毫秒對齊,相機只訓練不部署,影像不存檔只留座標。自監督:整夜採集六萬樣本、衍生六十一萬三元組。前處理做正規化、子載波內插到五十六。");
  }

  // ---------- Slide 8 : 2.4 模型架構 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2.4", "模型選用與訓練(一):模型架構", "AETHER 對比式 CSI 編碼器 — 一次前向、雙輸出");
    // architecture diagram
    const dy = 2.0, bh = 1.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: dy + 0.5, w: 1.9, h: bh, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    s.addText([{ text: "CSI 輸入", options: { breakLine: true, bold: true, color: WHITE, fontFace: FONT } }, { text: "56 維", options: { color: CYAN, fontFace: MONO, fontSize: 11 } }], { x: MX, y: dy + 0.5, w: 1.9, h: bh, align: "center", valign: "middle", fontSize: 13, margin: 0 });
    s.addText("›", { x: MX + 1.9, y: dy + 0.5, w: 0.5, h: bh, align: "center", valign: "middle", fontFace: MONO, fontSize: 24, bold: true, color: TEAL, margin: 0 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 2.45, y: dy + 0.5, w: 3.5, h: bh, rectRadius: 0.08, fill: { color: TEAL }, shadow: mkShadow() });
    s.addText([{ text: "共享 Backbone", options: { breakLine: true, bold: true, color: WHITE, fontFace: FONT, fontSize: 14 } }, { text: "Transformer + 交叉注意力 + GNN", options: { color: "ECFEFF", fontFace: FONT, fontSize: 10.5 } }], { x: MX + 2.45, y: dy + 0.5, w: 3.5, h: bh, align: "center", valign: "middle", margin: 0 });
    s.addText("›", { x: MX + 5.95, y: dy + 0.5, w: 0.5, h: bh, align: "center", valign: "middle", fontFace: MONO, fontSize: 24, bold: true, color: TEAL, margin: 0 });
    // two heads
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 6.55, y: dy, w: 3.0, h: 0.92, rectRadius: 0.08, fill: { color: BLUE }, shadow: mkShadow() });
    s.addText([{ text: "🦴 姿態頭  ", options: { bold: true, color: WHITE, fontFace: FONT } }, { text: "→ 17 關鍵點 (x,y,z,信心)", options: { color: "EFF6FF", fontFace: FONT, fontSize: 11 } }], { x: MX + 6.65, y: dy, w: 2.85, h: 0.92, valign: "middle", fontSize: 12.5, margin: 0 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 6.55, y: dy + 1.08, w: 3.0, h: 0.92, rectRadius: 0.08, fill: { color: "8B5CF6" }, shadow: mkShadow() });
    s.addText([{ text: "🧬 嵌入頭  ", options: { bold: true, color: WHITE, fontFace: FONT } }, { text: "→ 128 維環境指紋", options: { color: "F5F3FF", fontFace: FONT, fontSize: 11 } }], { x: MX + 6.65, y: dy + 1.08, w: 2.85, h: 0.92, valign: "middle", fontSize: 12.5, margin: 0 });
    // connectors to heads
    s.addShape(pres.shapes.LINE, { x: MX + 6.45, y: dy + 0.46, w: 0.1, h: 0.54, line: { color: TEAL, width: 1.5 } });
    s.addShape(pres.shapes.LINE, { x: MX + 6.45, y: dy + 1.0, w: 0.1, h: 0.54, line: { color: TEAL, width: 1.5 } });
    s.addText("同一次前向運算同時產生兩種輸出(< 2 ms)", { x: MX, y: dy + 1.7, w: 9.5, h: 0.34, fontFace: FONT, fontSize: 11, italic: true, color: MUTED, margin: 0 });
    // stat cards right
    statCard(s, 10.05, 2.0, 1.32, 1.0, "55K", "參數量", CYAN);
    statCard(s, 11.45, 2.0, 1.28, 1.0, "55 KB", "INT8 大小", TEAL);
    // bottom: pose model + design rationale
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 4.45, w: 5.9, h: 2.1, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
    s.addText("姿態模型 WiFlow", { x: MX + 0.28, y: 4.6, w: 5.3, h: 0.36, fontFace: FONT, fontSize: 14, bold: true, color: DARK, margin: 0 });
    s.addText([
      { text: "TCN + 軸向注意力 + 姿態解碼器", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "1.8M 參數 → 4-bit 量化僅 881 KB", options: { breakLine: true, bullet: { code: "2022" }, fontFace: FONT } },
      { text: "輸出 DensePose 24 區 UV 座標", options: { bullet: { code: "2022" }, fontFace: FONT } },
    ], { x: MX + 0.3, y: 5.0, w: 5.5, h: 1.45, fontSize: 12, color: INK, margin: 0, paraSpaceAfter: 6 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.7, y: 4.45, w: CW - 6.7 + MX, h: 2.1, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    s.addText("為何這樣設計", { x: 6.98, y: 4.6, w: 5.6, h: 0.36, fontFace: FONT, fontSize: 14, bold: true, color: CYAN, margin: 0 });
    s.addText([
      { text: "128 維是 SimCLR / CLIP 的標準嵌入維度", options: { breakLine: true, bullet: { code: "2022" }, color: "E2E8F0", fontFace: FONT } },
      { text: "共享骨幹 → 一次運算同得姿態 + 指紋", options: { breakLine: true, bullet: { code: "2022" }, color: "E2E8F0", fontFace: FONT } },
      { text: "遵循 CMU「DensePose From WiFi」架構家族", options: { bullet: { code: "2022" }, color: "E2E8F0", fontFace: FONT } },
    ], { x: 7.0, y: 5.0, w: CW - 6.7 + MX - 0.55, h: 1.45, fontSize: 12, margin: 0, paraSpaceAfter: 6 });
    footer(s, 9);
    s.addNotes("核心模型 AETHER 是對比式 CSI 編碼器:輸入五十六維 CSI 後經共享骨幹——Transformer、交叉注意力、圖神經網路——再分兩個頭,在同一次前向同時輸出:姿態頭給十七個關鍵點,嵌入頭給一百二十八維指紋。只有五萬五千參數,八位元才五十五 KB。姿態 WiFlow 一百八十萬參數,四位元才八百八十一 KB。設計理由:一百二十八維是 CLIP 標準、共享骨幹省時、源自 CMU DensePose From WiFi。");
  }

  // ---------- Slide 9 : 2.4 訓練 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "2.4", "模型選用與訓練(二):訓練策略與量化", "三種訓練模式 + 邊緣量化,塞進 8 KB");
    styledTable(s, ["訓練模式", "需要的資料", "得到的能力"], [
      [{ text: "自監督對比學習", bold: true }, "僅原始 WiFi", "懂 CSI 結構的 backbone"],
      [{ text: "監督式微調", bold: true }, "WiFi + 姿態標籤", "全身姿態 + 環境指紋"],
      [{ text: "跨模態", bold: true }, "WiFi + 相機", "與視覺對齊的指紋"],
    ], { y: 1.72, w: 7.3, colW: [2.5, 2.4, 2.4], rowH: 0.5, fontSize: 12 });
    s.addText([
      { text: "訓練技巧:", options: { bold: true, color: DARK, fontFace: FONT } },
      { text: "Hard-negative mining(只挑最難負樣本)+ 課程學習(由易到難)", options: { color: INK, fontFace: FONT } },
    ], { x: MX, y: 3.62, w: 7.3, h: 0.4, fontSize: 12, margin: 0 });
    // speed cards
    s.addText("訓練 / 適應速度", { x: MX, y: 4.12, w: 7.3, h: 0.34, fontFace: FONT, fontSize: 13, bold: true, color: MUTED, margin: 0 });
    statCard(s, MX, 4.5, 2.3, 1.05, "2.1 s", "RTX 5080 · 400 epoch", CYAN);
    statCard(s, MX + 2.5, 4.5, 2.3, 1.05, "84 s", "M4 Pro 從零訓練", TEAL);
    statCard(s, MX + 5.0, 4.5, 2.3, 1.05, "<30 s", "換新房間適應", BLUE);
    // environment adaptation
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.78, w: 7.3, h: 0.85, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    s.addText([
      { text: "環境適應:", options: { bold: true, color: CYAN, fontFace: FONT } },
      { text: "每房間僅 1,792 參數 MicroLoRA(較重訓省 93%)+ EWC++ 防遺忘", options: { color: "E2E8F0", fontFace: FONT } },
    ], { x: MX + 0.3, y: 5.78, w: 6.8, h: 0.85, valign: "middle", fontSize: 12.5, margin: 0 });
    // right: quantization
    const rx = 8.05, rw = CW - rx + MX;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.72, w: rw, h: 3.55, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
    s.addText("邊緣量化(縮小模型)", { x: rx + 0.28, y: 1.9, w: rw - 0.5, h: 0.4, fontFace: FONT, fontSize: 14, bold: true, color: DARK, margin: 0 });
    const q = [["FP32 safetensors", "48 KB", MUTED], ["q8 (8-bit)", "16 KB", INK], ["q4 (4-bit) ★", "8 KB", TEAL], ["q2 (2-bit)", "4 KB", INK]];
    let qy = 2.42;
    for (const [lab, val, col] of q) {
      const hi = lab.includes("★");
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.28, y: qy, w: rw - 0.56, h: 0.6, rectRadius: 0.06, fill: { color: hi ? "ECFEFF" : WHITE }, line: { color: hi ? TEAL : LINE, width: hi ? 1.5 : 1 } });
      s.addText(lab, { x: rx + 0.45, y: qy, w: 1.95, h: 0.6, valign: "middle", fontFace: FONT, fontSize: 12.5, bold: hi, color: hi ? TEAL : INK, margin: 0 });
      if (hi) s.addText("邊緣推論首選", { x: rx + 2.05, y: qy, w: rw - 3.9, h: 0.6, valign: "middle", align: "center", fontFace: FONT, fontSize: 9.5, italic: true, color: TEAL, margin: 0 });
      s.addText(val, { x: rx + rw - 1.8, y: qy, w: 1.3, h: 0.6, valign: "middle", align: "right", fontFace: MONO, fontSize: 15, bold: true, color: col, margin: 0 });
      qy += 0.69;
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 5.42, w: rw, h: 1.2, rectRadius: 0.08, fill: { color: "ECFEFF" }, line: { color: TEAL, width: 1.25 } });
    s.addText("預訓練規模(已上 Hugging Face)", { x: rx + 0.28, y: 5.54, w: rw - 0.5, h: 0.34, fontFace: FONT, fontSize: 12, bold: true, color: "0E7490", margin: 0 });
    s.addText("12.2M 訓練步  ·  60K 幀  ·  610K 三元組", { x: rx + 0.28, y: 5.9, w: rw - 0.5, h: 0.6, fontFace: MONO, fontSize: 13, bold: true, color: TEAL, margin: 0 });
    footer(s, 10);
    s.addNotes("三種訓練模式:自監督對比學習先學 CSI 結構;監督式微調得到完整姿態;跨模態讓指紋跟視覺對齊。技巧:hard-negative mining 加課程學習。速度很快:RTX 5080 跑四百 epoch 兩點一秒、M4 從零八十四秒、換房間不到三十秒。量化:全精度四十八 KB,四位元只有八 KB,適合邊緣部署、載入更快。換環境只訓一千七百九十二參數的 LoRA、省九成三,加 EWC 防遺忘。預訓練一千兩百二十萬步已上 Hugging Face。");
  }

  // ---------- Slide 10 : 3.1 成果展示 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "3.1", "實驗結果 — 成果展示", "線上互動 Demo,可現場操作");
    // demo list
    const demos = [
      ["Live Observatory", "即時 CSI 姿態骨架主展示", "ruvnet.github.io/RuView/"],
      ["Dual-Modal Fusion", "webcam + CSI 雙模態融合", "…/pose-fusion.html"],
      ["Live 3D Point Cloud", "相機 + CSI + 雷達點雲", "…/pointcloud/"],
      ["three.js Demos (5)", "漸進式 3D 場景", "…/three.js/"],
    ];
    let dyy = 1.78;
    for (const [t, d, url] of demos) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: dyy, w: 6.5, h: 0.92, rectRadius: 0.07, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y: dyy, w: 0.08, h: 0.92, fill: { color: CYAN } });
      s.addText(t, { x: MX + 0.28, y: dyy + 0.1, w: 3.0, h: 0.4, fontFace: FONT, fontSize: 14, bold: true, color: DARK, margin: 0 });
      s.addText(d, { x: MX + 0.28, y: dyy + 0.5, w: 4.0, h: 0.36, fontFace: FONT, fontSize: 11, color: MUTED, margin: 0 });
      s.addText(url, { x: MX + 3.4, y: dyy + 0.1, w: 3.0, h: 0.4, valign: "middle", align: "right", fontFace: MONO, fontSize: 10.5, color: TEAL, margin: 0 });
      dyy += 1.04;
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.98, w: 6.5, h: 0.62, rectRadius: 0.07, fill: { color: DARK } });
    s.addText([
      { text: "30 秒上手:  ", options: { bold: true, color: CYAN, fontFace: FONT } },
      { text: "docker run -p 3000:3000 ruvnet/wifi-densepose:latest", options: { color: "E2E8F0", fontFace: MONO, fontSize: 11 } },
    ], { x: MX + 0.3, y: 5.98, w: 6.0, h: 0.62, valign: "middle", fontSize: 12, margin: 0 });
    // right: screenshot
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.4, y: 1.78, w: 5.33, h: 3.7, rectRadius: 0.1, fill: { color: DARK2 }, line: { color: CYAN, width: 1.25 }, shadow: mkShadow() });
    s.addImage({ path: ASSET + "screenshot.png", x: 7.54, y: 1.92, w: 5.05, h: 5.05 / 1.94, sizing: { type: "contain", w: 5.05, h: 2.6 } });
    s.addText("即時感測畫面:姿態骨架 + 訊號視覺化", { x: 7.4, y: 4.75, w: 5.33, h: 0.34, align: "center", fontFace: FONT, fontSize: 10.5, italic: true, color: CYAN, margin: 0 });
    s.addText([
      { text: "可現場演示:", options: { bold: true, color: DARK, fontFace: FONT } },
      { text: "即時骨架、呼吸/心率讀數、RF 房間掃描、跌倒警報、穿牆偵測", options: { color: INK, fontFace: FONT } },
    ], { x: 7.4, y: 5.55, w: 5.33, h: 1.0, valign: "top", fontSize: 12.5, margin: 0 });
    footer(s, 11);
    s.addNotes("我們有好幾個線上就能互動的 Demo:Observatory 即時看到 WiFi 還原的人體骨架;雙模態融合左邊攝影機右邊 WiFi 推論可對照;即時 3D 點雲;還有五個 three.js 場景。想自己試完全不用硬體,一行 Docker 三十秒上手。接上感測硬體(如 Pi 4)就能現場演示:即時骨架、呼吸心率、房間掃描、跌倒、穿牆。等一下播放錄屏。");
  }

  // ---------- Slide 11 : 3.2 指標 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "3.2", "測試與比較(一):已驗證指標", "以下全部為實測驗證數據");
    const stats = [
      ["100%", "存在偵測準確率", GREEN], ["24 / 24", "多人計數正確", GREEN], ["164,183", "嵌入 / 秒 (M4 Pro)", CYAN],
      ["810×", "Rust vs Python 加速", CYAN], ["8 KB", "模型大小 (4-bit)", TEAL], ["1,463", "測試通過 · 0 失敗", TEAL],
    ];
    const sw = 3.85, sg = 0.28, sh = 1.42;
    for (let i = 0; i < stats.length; i++) {
      const col = i % 3, row = Math.floor(i / 3);
      const x = MX + col * (sw + sg), y = 1.74 + row * (sh + 0.22);
      statCard(s, x, y, sw, sh, stats[i][0], stats[i][1], stats[i][2]);
    }
    // semantic F1 line
    s.addText([
      { text: "語意偵測 F1:", options: { bold: true, color: DARK, fontFace: FONT } },
      { text: "浴室占用 0.98  ·  房間活動 0.95  ·  無動作 0.92  ·  離床 0.91  ·  睡眠 0.84", options: { color: INK, fontFace: FONT } },
    ], { x: MX, y: 5.18, w: CW, h: 0.4, fontSize: 12.5, align: "center", margin: 0 });
    // honest note
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.66, w: CW, h: 1.0, rectRadius: 0.08, fill: { color: "FEF3C7" }, line: { color: AMBER, width: 1.5 }, shadow: mkShadow() });
    await iconChip(s, MX + 0.26, 5.88, 0.56, "warn", AMBER);
    s.addText([
      { text: "誠實標註:", options: { bold: true, color: "92400E", fontFace: FONT } },
      { text: "17 關鍵點姿態 PCK@20 目前 3.0%(目標 ≥35%)— 瓶頸是資料量(僅 1,077 樣本),非架構問題。早期文件的 92.9% 已被官方稽核更正,本報告採用真實數字。", options: { color: "92400E", fontFace: FONT } },
    ], { x: MX + 1.0, y: 5.72, w: CW - 1.3, h: 0.9, valign: "middle", fontSize: 12, margin: 0 });
    footer(s, 12);
    s.addNotes("這頁全部是已驗證的數字:存在偵測百分之百、計數二十四個全對、每秒十六萬四千個嵌入、Rust 快八百一十倍、模型八 KB、一千四百六十三個測試全過。語意層級浴室占用 F1 零點九八。這裡誠實交代:十七關鍵點姿態目前 PCK@20 只有百分之三,離百分之三十五目標還有差距,但原因是資料量不夠、只有一千零七十七個樣本,不是架構問題。早期文件出現過百分之九十二點九,後來團隊自己稽核發現是誇大、主動更正,我用真實數字報告。");
  }

  // ---------- Slide 12 : 3.2 比較 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "3.2", "測試與比較(二):與相機 / 傳統方案", "WiFi 感測的本質優勢 + 學術指標對照");
    styledTable(s, ["維度", "RuView (WiFi)", "攝影機"], [
      [{ text: "成本 / 點位", bold: true }, { text: "~$50(Pi 4)", color: TEAL, bold: true }, "$200–$2000"],
      [{ text: "隱私", bold: true }, { text: "無影像、規避法規", color: TEAL }, "需同意 / 告示"],
      [{ text: "穿牆", bold: true }, { text: "✓", color: GREEN, bold: true, align: "center" }, { text: "✕", color: "EF4444", align: "center" }],
      [{ text: "黑暗", bold: true }, { text: "✓", color: GREEN, bold: true, align: "center" }, { text: "✕", color: "EF4444", align: "center" }],
      [{ text: "多人", bold: true }, { text: "~3–5 / AP(可疊加)", color: TEAL }, "視角受限"],
    ], { y: 1.78, w: 6.4, colW: [1.7, 2.85, 1.85], rowH: 0.56, fontSize: 12.5 });
    // chart
    s.addText("DensePose-From-WiFi 文獻基準(CMU,本架構家族)", { x: 7.1, y: 1.72, w: 5.6, h: 0.4, fontFace: FONT, fontSize: 12.5, bold: true, color: DARK, margin: 0 });
    s.addChart(pres.charts.BAR, [{ name: "AP", labels: ["WiFi 同佈局", "相機 同佈局", "WiFi 跨佈局"], values: [87.2, 94.4, 27.3] }], {
      x: 7.0, y: 2.1, w: 5.75, h: 2.95, barDir: "col",
      chartColors: [CYAN, BLUE, AMBER], varyColors: true,
      chartArea: { fill: { color: WHITE } }, showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontFace: MONO, dataLabelFontSize: 12, dataLabelFontBold: true,
      catAxisLabelColor: MUTED, catAxisLabelFontFace: FONT, catAxisLabelFontSize: 11,
      valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      showLegend: false, showTitle: false, valAxisMaxVal: 100,
    });
    s.addText("AP@50 指標 · 數值越高越好", { x: 7.0, y: 5.05, w: 5.75, h: 0.3, align: "center", fontFace: FONT, fontSize: 10, italic: true, color: MUTED, margin: 0 });
    // insight band
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.5, w: CW, h: 1.1, rectRadius: 0.08, fill: { color: DARK }, shadow: mkShadow() });
    signalRings(s, 12.6, 6.9, CYAN);
    s.addText([
      { text: "關鍵觀察:", options: { bold: true, color: CYAN, fontFace: FONT } },
      { text: "同佈局下 WiFi(87.2)已逼近相機(94.4);但跨佈局掉到 27.3 → 「跨環境泛化」是領域公認難題,正是本專案 MERIDIAN 方向所解。", options: { color: "E2E8F0", fontFace: FONT } },
    ], { x: MX + 0.35, y: 5.5, w: CW - 2.0, h: 1.1, valign: "middle", fontSize: 13, margin: 0 });
    footer(s, 13);
    s.addNotes("先跟攝影機比:成本約五十美金(Pi 4)vs 兩百到兩千、無影像規避法規、能穿牆全黑、多人。學術指標上,我們架構家族來自 CMU 的 DensePose From WiFi:同一房間佈局下 WiFi 的 AP@50 可到八十七點二,逼近相機的九十四點四;但換到沒看過的佈局掉到二十七點三,這點出跨環境泛化的難題,正是我們 MERIDIAN 方向在解。Rust 快八百一十倍、Pi 5 推論八點四毫秒。");
  }

  // ---------- Slide 13 : 結論 ----------
  {
    const s = pres.addSlide();
    s.background = { color: DARK };
    signalRings(s, 0.6, 0.8);
    signalRings(s, 12.8, 7.0);
    slideBadgeDark(s, "4", "結論與未來工作");
    // conclusion
    s.addText("結論", { x: MX, y: 1.85, w: 6.0, h: 0.45, fontFace: FONT, fontSize: 20, bold: true, color: CYAN, margin: 0 });
    const concl = [
      "用一台 Raspberry Pi 4,實現非接觸、可穿牆、隱私友善的人體感測",
      "完整端到端系統:nexmon 擷取 → SOTA 訊號處理 → 對比學習模型 → 單機推論",
      "已驗證:存在 100% · 164K 嵌入/秒 · 8 KB 模型 · 1,463 測試通過",
      "工程嚴謹:確定性可重現驗證 + 見證鏈,誠實區分「實測 vs 目標」",
    ];
    s.addText(concl.map((t, i) => ({ text: t, options: { breakLine: true, bullet: { code: "2022" }, color: "E2E8F0", fontFace: FONT, paraSpaceAfter: 9 } })), { x: MX + 0.1, y: 2.4, w: 6.1, h: 3.2, fontSize: 14, margin: 0 });
    // future
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 1.85, w: CW - 7.0 + MX, h: 4.0, rectRadius: 0.1, fill: { color: DARK2 }, line: { color: TEAL, width: 1 }, shadow: mkShadow() });
    s.addText("未來工作", { x: 7.3, y: 2.05, w: 5.4, h: 0.45, fontFace: FONT, fontSize: 18, bold: true, color: CYAN, margin: 0 });
    const fut = [
      ["補多房間相機配對資料", "姿態 PCK@20 由 3% → 35%+"],
      ["MERIDIAN 跨環境泛化", "縮小跨佈局落差"],
      ["完成 HF 模型 live 載入", "+ 樹莓派端推論優化"],
    ];
    let fy = 2.65;
    for (const [t, d] of fut) {
      await iconChip(s, 7.3, fy, 0.5, "check", TEAL);
      s.addText(t, { x: 7.95, y: fy - 0.04, w: 4.7, h: 0.34, fontFace: FONT, fontSize: 14, bold: true, color: WHITE, margin: 0 });
      s.addText(d, { x: 7.95, y: fy + 0.3, w: 4.7, h: 0.34, fontFace: FONT, fontSize: 11.5, color: "94A3B8", margin: 0 });
      fy += 1.05;
    }
    // one-liner
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 6.1, w: CW, h: 0.85, rectRadius: 0.1, fill: { color: CYAN }, shadow: mkShadow() });
    s.addText("讓「空間感知」變成一台樹莓派、隱私優先、隨處可部署的能力", { x: MX, y: 6.1, w: CW, h: 0.85, align: "center", valign: "middle", fontFace: FONT, fontSize: 17, bold: true, color: DARK, margin: 0 });
    s.addNotes("總結:證明只用一台 Raspberry Pi 4,就能做非接觸、能穿牆、又保護隱私的人體感測,而且擷取跟推論都在這台 Pi 上完成,是完整的端到端系統。已驗證成果:存在百分之百、每秒十六萬嵌入、八 KB 模型、一千四百六十三測試全過。工程上有可重現驗證跟見證鏈,誠實區分實測跟目標。未來:補資料把姿態三趴拉到三十五趴、MERIDIAN 解跨環境泛化、優化樹莓派端推論。一句話:讓空間感知變成一台樹莓派、隱私優先、隨處部署。謝謝大家。");
  }

  // ---------- Slide 14 : 5.1 源碼 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "5.1", "參考文獻:推論源碼與測試資料集", null);
    const rows = [
      ["github", "主程式碼", "github.com/ruvnet/RuView", BLUE],
      ["python", "推論套件", "PyPI: ruview / wifi-densepose · crates.io: wifi-densepose-*", TEAL],
      ["npm", "套件 (npm)", "@ruvnet/rvagent (MCP) · @ruv/rvcsi", "EF4444"],
      ["docker", "容器映像", "hub.docker.com/r/ruvnet/wifi-densepose", BLUE],
      ["robot", "預訓練模型", "huggingface.co/ruvnet/wifi-densepose-pretrained", AMBER],
      ["db", "測試資料集", "確定性 proof: archive/v1/data/proof/  ·  MM-Fi  ·  Wi-Pose", "8B5CF6"],
    ];
    let y = 1.72;
    for (const [ico, t, url, col] of rows) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: CW, h: 0.78, rectRadius: 0.07, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
      await iconChip(s, MX + 0.2, y + 0.16, 0.46, ico, col);
      s.addText(t, { x: MX + 0.86, y, w: 2.2, h: 0.78, valign: "middle", fontFace: FONT, fontSize: 13.5, bold: true, color: DARK, margin: 0 });
      s.addText(url, { x: MX + 3.1, y, w: CW - 3.3, h: 0.78, valign: "middle", fontFace: MONO, fontSize: 12, color: INK, margin: 0 });
      y += 0.86;
    }
    footer(s, 14);
    s.addNotes("推論源碼:主程式碼在 GitHub ruvnet/RuView;套件發布在 PyPI、crates.io、npm、Docker Hub 都能直接安裝;預訓練模型在 Hugging Face;測試資料集有自帶的確定性 proof,跟外部公開的 MM-Fi、Wi-Pose;邊緣 runtime 是 rvCSI。");
  }

  // ---------- Slide 15 : 5.2 文檔 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "5.2", "參考文獻:公開文檔連結", null);
    const groups = [
      ["核心使用文檔", ["User Guide / Build Guide / Troubleshooting", "WiFi-Mat 災難搜救指南", "Extended Documentation(訊號/訓練/CLI/部署)"], CYAN],
      ["架構文檔", ["96 份 ADR(架構決策記錄)", "8 個 DDD 領域模型", "BFLD 研究 dossier(11 份)"], TEAL],
      ["線上 Demo / 整合", ["ruvnet.github.io/RuView/(4 個 Demo)", "Home Assistant + Matter 整合", "Cognitum Seed · Docker Hub"], BLUE],
    ];
    const gw = (CW - 0.6) / 3;
    for (let i = 0; i < groups.length; i++) {
      const [t, items, col] = groups[i];
      const x = MX + i * (gw + 0.3);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: gw, h: 4.6, rectRadius: 0.1, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
      s.addShape(pres.shapes.RECTANGLE, { x, y: 1.75, w: gw, h: 0.1, fill: { color: col } });
      await iconChip(s, x + gw / 2 - 0.4, 2.05, 0.8, ["book", "layer", "wave"][i], col);
      s.addText(t, { x: x + 0.1, y: 2.98, w: gw - 0.2, h: 0.45, align: "center", fontFace: FONT, fontSize: 15.5, bold: true, color: DARK, margin: 0 });
      s.addText(items.map(it => ({ text: it, options: { breakLine: true, bullet: { code: "2022" }, color: INK, fontFace: FONT, paraSpaceAfter: 10 } })), { x: x + 0.32, y: 3.55, w: gw - 0.6, h: 2.6, fontSize: 12, margin: 0, valign: "top" });
    }
    footer(s, 15);
    s.addNotes("文檔很完整:使用面有 User Guide、Build Guide、故障排除、災難搜救指南;架構面有九十六份 ADR、八個 DDD 模型;還有線上 Demo、Home Assistant 加 Matter 整合、Cognitum Seed、Docker Hub。連結都在投影片上。");
  }

  // ---------- Slide 16 : 5.3 論文 ----------
  {
    const s = pres.addSlide();
    contentHeader(s, "5.3", "參考文獻:相關論文", null);
    const cats = [
      ["核心 / 基礎", [["DensePose From WiFi (CMU)", "arXiv:2301.00250"], ["MM-Fi (NeurIPS 2023)", "arXiv:2305.10345"], ["Person-in-WiFi 3D (CVPR 2024)", "—"]], CYAN],
      ["訊號處理方法", [["SpotFi — 相位校正 / 定位", "SIGCOMM 2015"], ["FarSense — Fresnel 呼吸模型", "MobiCom 2019"], ["Widar 3.0 — 身體速度分布", "MobiSys 2019"]], TEAL],
      ["學習方法", [["SimCLR / VICReg — 對比學習", "2002.05709 / 2105.04906"], ["LoRA — 環境適應微調", "arXiv:2106.09685"], ["EWC / HNSW — 防遺忘 / 搜尋", "1612.00796 / 1603.09320"]], BLUE],
    ];
    let y = 1.7;
    for (const [cat, papers, col] of cats) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: CW, h: 1.55, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 }, shadow: mkShadow() });
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 0.1, h: 1.55, fill: { color: col } });
      s.addText(cat, { x: MX + 0.32, y: y + 0.12, w: 3.0, h: 1.3, valign: "middle", fontFace: FONT, fontSize: 15, bold: true, color: col === CYAN ? "0E7490" : col, margin: 0 });
      let py = y + 0.16;
      for (const [name, ref] of papers) {
        s.addText([
          { text: name, options: { color: INK, fontFace: FONT, fontSize: 12.5 } },
          { text: "   " + ref, options: { color: MUTED, fontFace: MONO, fontSize: 10.5 } },
        ], { x: 3.6, y: py, w: CW - 3.2, h: 0.4, valign: "middle", margin: 0 });
        py += 0.42;
      }
      y += 1.67;
    }
    footer(s, 16);
    s.addNotes("相關論文分三類。核心:CMU 的 DensePose From WiFi、MM-Fi、Person-in-WiFi 3D。訊號處理對應 SpotFi、FarSense、Widar 3.0。學習方法對應 SimCLR、VICReg、LoRA、EWC、HNSW,都是各領域代表作。報告到此結束,謝謝各位,進入提問時間。");
  }

  await pres.writeFile({ fileName: "D:/RuView/presentation/RuView-簡報.pptx" });
  console.log("OK wrote RuView-簡報.pptx");
}

// dark-slide header badge helper (defined after use is hoisted via function declaration)
function slideBadgeDark(slide, num, title) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.5, w: 0.92, h: 0.92, rectRadius: 0.12, fill: { color: CYAN }, shadow: mkShadow() });
  slide.addText(num, { x: MX, y: 0.5, w: 0.92, h: 0.92, align: "center", valign: "middle", fontFace: MONO, fontSize: 22, bold: true, color: DARK, margin: 0 });
  slide.addText(title, { x: 1.72, y: 0.5, w: CW - 1.2, h: 0.92, valign: "middle", fontFace: FONT, fontSize: 30, bold: true, color: WHITE, margin: 0 });
}

build().catch(e => { console.error(e); process.exit(1); });
