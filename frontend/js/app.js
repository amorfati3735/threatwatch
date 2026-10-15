/**
 * Personal Safety Monitor Dashboard Controller.
 * Manages WebSocket telemetry, dual source switching (WESAD vs Arduino Live),
 * USB serial connection management, audio alarms, and live UI updates.
 */

// Web Audio API Alarm Synthesizer
class SoundEffects {
  constructor() {
    this.audioCtx = null;
    this.isMuted = false;
    this.lastAlarmTime = 0;
  }

  init() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) this.audioCtx = new AudioContext();
    }
  }

  playBeep(freq = 880, type = "sine", duration = 0.15) {
    if (this.isMuted) return;
    this.init();
    if (!this.audioCtx) return;

    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);

      gain.gain.setValueAtTime(0.2, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start();
      osc.stop(this.audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio play error:", e);
    }
  }

  playThreatAlarm() {
    const now = Date.now();
    if (now - this.lastAlarmTime > 2000) {
      this.lastAlarmTime = now;
      this.playBeep(920, "sawtooth", 0.25);
      setTimeout(() => this.playBeep(1240, "sawtooth", 0.35), 250);
    }
  }
}

const sfx = new SoundEffects();

// Charts
let edaChart, respChart, accChart, threatChart;

// App State
let socket = null;
let activeSource = "WESAD_REPLAY";
let arduinoConnected = false;
let currentTheme = "dark";

// Threat Score Smoothing & Hold State
let displayedThreatPct = 0.0;
let lastThreatTextUpdate = 0;
let telemetryFrameCount = 0;

// Serial Monitor State
let smFormat = "decoded"; // "decoded" or "csv"
let smAutoScroll = true;
const smBuffer = [];
const SM_MAX_LINES = 120;

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initTheme();
  initReviewMode();
  initSerialMonitorControls();
  initPanelDropdown();
  connectWebSocket();
  fetchSubjects();
  fetchArduinoPorts();
  setupEventListeners();
});

function initCharts() {
  edaChart = new RealtimeWaveformChart("edaCanvas", 100, "#00f0ff", "EDA / PPG");
  respChart = new RealtimeWaveformChart("respCanvas", 100, "#ccff00", "Resp / Gyro");
  accChart = new RealtimeWaveformChart("accCanvas", 100, "#ffb700", "Acc Mag (g)");
  threatChart = new ThreatHistoryChart("threatCanvas", 120);
}

function initTheme() {
  document.documentElement.setAttribute("data-theme", currentTheme);
  const themeBtn = document.getElementById("themeToggleBtn");
  if (themeBtn) {
    themeBtn.textContent = currentTheme === "dark" ? "LIGHT" : "DARK";
    themeBtn.addEventListener("click", () => {
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", currentTheme);
      themeBtn.textContent = currentTheme === "dark" ? "LIGHT" : "DARK";
    });
  }
}

function initReviewMode() {
  const btn = document.getElementById("reviewModeBtn");
  if (!btn) return;
  const apply = (on) => {
    document.body.classList.toggle("review-mode", on);
    btn.textContent = on ? "Review Mode: ON" : "Review Mode";
    btn.classList.toggle("btn-neon", on);
    try { localStorage.setItem("tw_review_mode", on ? "1" : "0"); } catch (e) { /* private mode */ }
  };
  let on = false;
  try { on = localStorage.getItem("tw_review_mode") === "1"; } catch (e) { /* private mode */ }
  apply(on);
  btn.addEventListener("click", () => apply(!document.body.classList.contains("review-mode")));
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  const statusBadge = document.getElementById("wsStatusBadge");

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    if (statusBadge) {
      statusBadge.textContent = "Live";
      statusBadge.className = "badge-uni badge-neon";
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "TELEMETRY") {
        updateDashboard(data);
      }
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  socket.onclose = () => {
    if (statusBadge) {
      statusBadge.textContent = "Disconnected";
      statusBadge.className = "badge-uni badge-alert";
    }
    setTimeout(connectWebSocket, 2000);
  };
}

function updateDashboard(data) {
  // 1. Video Feed
  const feed = document.getElementById("webcamFeed");
  if (feed && data.camera_frame) {
    feed.src = data.camera_frame;
  }

  // 2. Header Stats & Active Source
  setText("streamFps", `${data.fps} FPS`);
  setText("activeScenarioDisplay", data.demo_scenario || "Live Free");

  const currentSource = data.active_source_type || "WESAD_REPLAY";
  if (currentSource !== activeSource) {
    activeSource = currentSource;
    updateSourceUI();
  }

  // Arduino connection status badge
  const ardState = data.arduino || {};
  arduinoConnected = ardState.connected || false;
  const ardBadge = document.getElementById("arduinoStatusBadge");
  const btnConnect = document.getElementById("btnArduinoConnect");

  if (ardBadge && btnConnect) {
    if (arduinoConnected) {
      ardBadge.textContent = `ESP32: Connected (${ardState.port || 'USB'})`;
      ardBadge.className = "badge-uni badge-neon";
      btnConnect.textContent = "Disconnect";
      btnConnect.className = "btn-uni btn-dark text-xs py-1 px-2.5 border border-uni-alert text-uni-alert";
    } else {
      ardBadge.textContent = ardState.error ? `Error: ${ardState.error.slice(0, 15)}` : "Hardware: Standby";
      ardBadge.className = "badge-uni text-uni-muted";
      btnConnect.textContent = "Connect";
      btnConnect.className = "btn-uni btn-dark text-xs py-1 px-2.5 border border-uni-border hover:border-uni-neon";
    }
  }

  // 3. Raw Sensor Indicators
  const sp = data.sensor_packet || {};
  setText("valHR", sp.heart_rate ? `${Math.round(sp.heart_rate)} BPM` : "-- BPM");
  setText("valAcc", sp.acc_mag !== undefined ? `${sp.acc_mag.toFixed(2)} g` : "--");

  if (activeSource === "HARDWARE_LIVE" || activeSource === "ARDUINO_LIVE") {
    setText("sensorCardTitle", "Wearable Sensors (ESP32 Live)");
    setText("groundTruthLabel", "Hardware: MAX30102 + MPU6050");
    setText("lblSlot2", "IMU Die Temp (not skin)");
    setText("valSlot2", sp.temperature !== undefined ? `${sp.temperature.toFixed(1)} °C` : "--");
    setText("lblSlot3", "Gyro Rotation");
    setText("valSlot3", sp.gyro_mag !== undefined ? `${Math.round(sp.gyro_mag)} °/s` : "--");
    setText("lblSlot4", "Finger Sensor");
    setText("valSlot4", sp.finger_detected ? "Contact OK" : "No Finger");

    setText("chartLabel1", "Pulse / HR Tracking");
    setText("chartLabel2", "Gyroscope Energy (°/s)");
    setText("chartLabel3", "Accelerometer Magnitude (g)");

    if (sp.heart_rate) edaChart.push(sp.heart_rate);
    if (sp.gyro_mag !== undefined) respChart.push(sp.gyro_mag);
    if (sp.acc_mag !== undefined) accChart.push(sp.acc_mag);
  } else {
    setText("sensorCardTitle", "Wearable Sensors (WESAD Stream)");
    setText("groundTruthLabel", `Subject: ${data.replay?.subject_id || 'S2'}`);
    setText("lblSlot2", "EDA (Sweat)");
    setText("valSlot2", sp.eda !== undefined ? `${sp.eda.toFixed(2)} µS` : "--");
    setText("lblSlot3", "Respiration");
    setText("valSlot3", sp.respiration !== undefined ? `${sp.respiration.toFixed(2)}` : "--");
    setText("lblSlot4", "Skin Temp (WESAD)");
    setText("valSlot4", sp.temperature !== undefined ? `${sp.temperature.toFixed(1)} °C` : "--");

    setText("chartLabel1", "Electrodermal Activity (EDA)");
    setText("chartLabel2", "Respiration Waveform");
    setText("chartLabel3", "Acceleration Magnitude (g)");

    if (sp.eda !== undefined) edaChart.push(sp.eda);
    if (sp.respiration !== undefined) respChart.push(sp.respiration);
    if (sp.acc_mag !== undefined) accChart.push(sp.acc_mag);
  }

  // 4. Sensor ML Breakdown
  const sm = data.sensor_ml || {};
  const probs = sm.probabilities || { baseline: 0, stress: 0, amusement: 0 };
  setText("sensorDistressScore", `${Math.round((sm.distress_score || 0) * 100)}%`);

  if (activeSource === "HARDWARE_LIVE" || activeSource === "ARDUINO_LIVE") {
    setText("modelCardTitle", "ESP32 Distress Inference (CNN)");
    setText("lblProb1", "Normal Baseline");
    setText("lblProb2", "Tachycardia / Elevated HR");
    setText("lblProb3", "Impact Shock / Struggle Motion");
  } else {
    setText("modelCardTitle", "Sensor Sequence Model (LSTM)");
    setText("lblProb1", "Baseline State (Calm)");
    setText("lblProb2", "Stress / Distress State");
    setText("lblProb3", "Amusement State");
  }

  setText("probBaseline", `${Math.round((probs.baseline || 0) * 100)}%`);
  setText("probStress", `${Math.round((probs.stress || 0) * 100)}%`);
  setText("probAmusement", `${Math.round((probs.amusement || 0) * 100)}%`);
  setBar("barBaseline", (probs.baseline || 0) * 100);
  setBar("barStress", (probs.stress || 0) * 100);
  setBar("barAmusement", (probs.amusement || 0) * 100);
  setText("sensorLatency", `${sm.latency_ms || 0} ms`);

  // 5. Vision ML Breakdown
  const vm = data.vision_ml || {};
  setText("visionDistressScore", `${Math.round((vm.distress_score || 0) * 100)}%`);
  setBar("visionDistressBar", (vm.distress_score || 0) * 100);
  setText("visionPosture", vm.posture || "Unknown");
  setText("visionExpression", vm.facial_expression || "Neutral");
  setText("visionMovement", vm.movement_level || "Normal");

  // 6. Multimodal Threat Score & State Machine (Asymmetric Smoothing & Persistence)
  const fusion = data.fusion || {};
  const rawTargetPct = fusion.threat_score_pct !== undefined ? Number(fusion.threat_score_pct) : 0.0;

  // Asymmetric Attack / Decay Filter:
  // Attack: responds rapidly when threat elevates (~0.4s to surge)
  // Decay: holds and gracefully glides down over ~3-5 seconds so transient drops don't flap the verdict
  if (rawTargetPct >= displayedThreatPct) {
    const attackAlpha = 0.20; // Fast response to legitimate rising threat
    displayedThreatPct += (rawTargetPct - displayedThreatPct) * attackAlpha;
  } else {
    const decayAlpha = 0.035; // Gentle ~3.5s - 5s decay hold
    displayedThreatPct += (rawTargetPct - displayedThreatPct) * decayAlpha;
  }

  if (Math.abs(rawTargetPct - displayedThreatPct) < 0.2) {
    displayedThreatPct = rawTargetPct;
  }

  // Rate-limit numerical text repaints to eliminate high-frequency digit vibration
  const now = Date.now();
  if (now - lastThreatTextUpdate >= 200) {
    setText("threatScoreText", `${Math.round(displayedThreatPct)}%`);
    lastThreatTextUpdate = now;
  }

  // Smooth CSS width transition for the meter fill
  setBar("threatMeterFill", displayedThreatPct, fusion.verdict_color);

  // Decimate canvas chart pushes to 5 Hz (every 4th sample) to show 24s of smooth history
  if (telemetryFrameCount % 4 === 0) {
    threatChart.push(displayedThreatPct / 100.0);
  }

  // State Banner & Styling
  const smState = data.state_machine || {};
  const currState = smState.current_state || fusion.verdict || "SAFE";
  const stateBadge = document.getElementById("stateBadge");

  if (stateBadge) {
    stateBadge.textContent = currState;
    if (currState.includes("THREAT") || currState.includes("ALERT")) {
      stateBadge.className = "btn-uni btn-alert text-sm px-4 py-2 font-display pulse-threat";
      sfx.playThreatAlarm();
    } else if (currState.includes("CAUTION") || currState.includes("SUSPICIOUS")) {
      stateBadge.className = "btn-uni bg-uni-dark text-uni-cyan border-uni-cyan text-sm px-4 py-2 font-display";
    } else {
      stateBadge.className = "btn-uni btn-neon text-sm px-4 py-2 font-display";
    }
  }

  const threatFb = document.getElementById("threatFeedback");
  if (threatFb && threatFb.textContent.includes("Raising threat")) {
    if (currState.includes("ALERT") || currState.includes("THREAT") || currState.includes("EMERGENCY")) {
      threatFb.textContent = "Threat active (" + currState + ")";
      threatFb.className = "text-[11px] font-mono text-uni-alert font-bold";
    }
  }

  setText("fusionConsensus", fusion.agreement_status || "Normal / Congruent");
  setText("sensorWeightDisp", `Sensor (${Math.round((fusion.sensor_weight || 0.6) * 100)}%)`);
  setText("visionWeightDisp", `Vision (${Math.round((fusion.vision_weight || 0.4) * 100)}%)`);

  // 7. Replay Telemetry
  const rep = data.replay || {};
  setText("elapsedTime", `${rep.elapsed_seconds || 0}s / ${rep.total_seconds || 0}s`);
  const progressBar = document.getElementById("replayProgressBar");
  if (progressBar && rep.progress_fraction !== undefined) {
    progressBar.style.width = `${rep.progress_fraction * 100}%`;
  }

  // 8. Update Serial Monitor / Live Sensor Stream
  updateSerialMonitor(data, fusion, currState);

  // 9. Event Log Updates
  if (data.recent_logs && data.recent_logs.length > 0) {
    renderEventLog(data.recent_logs);
  }
}

function updateSourceUI() {
  const btnWesad = document.getElementById("btnSourceWesad");
  const btnArd = document.getElementById("btnSourceArduino");

  if (btnWesad && btnArd) {
    if (activeSource === "HARDWARE_LIVE" || activeSource === "ARDUINO_LIVE") {
      btnArd.className = "btn-uni btn-neon text-xs py-1 px-3";
      btnWesad.className = "btn-uni btn-dark text-xs py-1 px-3";
    } else {
      btnWesad.className = "btn-uni btn-neon text-xs py-1 px-3";
      btnArd.className = "btn-uni btn-dark text-xs py-1 px-3";
    }
  }
}

function renderEventLog(logs) {
  const container = document.getElementById("eventLogContainer");
  if (!container) return;

  const html = logs.map(l => {
    let colorClass = "text-uni-muted";
    if (l.level === "CRITICAL" || l.level === "EMERGENCY") colorClass = "text-uni-alert font-bold";
    else if (l.level === "WARNING") colorClass = "text-uni-cyan";
    else if (l.category === "HARDWARE" || l.category === "SOURCE") colorClass = "text-uni-neon";

    return `
      <div class="flex items-start gap-2 py-1 border-b border-uni-border/40 text-xs font-mono">
        <span class="text-uni-muted shrink-0">[${l.timestamp_str}]</span>
        <span class="badge-uni shrink-0">${l.category}</span>
        <span class="${colorClass}">${l.message}</span>
      </div>
    `;
  }).join("");

  container.innerHTML = html;
}

function updateSerialMonitor(data, fusion, currState) {
  const sp = data.sensor_packet || {};

  // 1. Quick inspector values
  setText("smValHR", sp.heart_rate ? `${Math.round(sp.heart_rate)} BPM` : "-- BPM");

  if (activeSource === "HARDWARE_LIVE" || activeSource === "ARDUINO_LIVE") {
    setText("smLblTemp", "IMU DIE TEMP");
    setText("smValTemp", sp.temperature !== undefined ? `${sp.temperature.toFixed(1)} °C` : "-- °C");
  } else {
    setText("smLblTemp", "SKIN TEMP (WESAD)");
    setText("smValTemp", sp.temperature !== undefined ? `${sp.temperature.toFixed(1)} °C` : "-- °C");
  }

  const accStr = (sp.acc_x !== undefined && sp.acc_y !== undefined && sp.acc_z !== undefined)
    ? `(${sp.acc_x.toFixed(2)}, ${sp.acc_y.toFixed(2)}, ${sp.acc_z.toFixed(2)}) g`
    : "--";
  setText("smValAcc", accStr);

  const gyroStr = (sp.gyro_x !== undefined && sp.gyro_y !== undefined && sp.gyro_z !== undefined)
    ? `(${Math.round(sp.gyro_x)}, ${Math.round(sp.gyro_y)}, ${Math.round(sp.gyro_z)}) °/s`
    : "--";
  setText("smValGyro", gyroStr);

  let contactStr = "--";
  let contactClass = "font-bold text-uni-muted";
  if (sp.finger_detected === true) {
    contactStr = "Contact OK";
    contactClass = "font-bold text-uni-neon";
  } else if (sp.finger_detected === false) {
    contactStr = "No Finger";
    contactClass = "font-bold text-uni-alert";
  }
  const contactEl = document.getElementById("smValContact");
  if (contactEl) {
    contactEl.textContent = contactStr;
    contactEl.className = contactClass;
  }

  setText("smValPackets", data.packet_counter || telemetryFrameCount);
  setText("smSourceBadge", (activeSource === "HARDWARE_LIVE" || activeSource === "ARDUINO_LIVE") ? "Wearable Live" : "WESAD Replay");
  setText("smRateBadge", `${data.fps || 20} Hz`);

  // 2. Append formatted stream line every 4 ticks (~5 Hz for comfortable human viewing)
  if (telemetryFrameCount % 4 === 0) {
    const timeStr = new Date().toLocaleTimeString("en-GB", { hour12: false });
    let line = "";
    if (smFormat === "decoded") {
      const hr = sp.heart_rate ? `${Math.round(sp.heart_rate)}` : "--";
      const temp = sp.temperature !== undefined ? `${sp.temperature.toFixed(1)}C` : "--";
      const acc = (sp.acc_x !== undefined) ? `${sp.acc_x.toFixed(2)},${sp.acc_y.toFixed(2)},${sp.acc_z.toFixed(2)}` : "--,--,--";
      const gyro = (sp.gyro_x !== undefined) ? `${Math.round(sp.gyro_x)},${Math.round(sp.gyro_y)},${Math.round(sp.gyro_z)}` : "--,--,--";
      const finger = sp.finger_detected === true ? "FINGER" : (sp.finger_detected === false ? "NO-FINGER" : "--");
      const threat = `${Math.round(displayedThreatPct)}%`;
      const state = currState || "SAFE";
      line = `${timeStr} | HR=${hr.padStart(3)} | Temp=${temp.padStart(5)} | Acc=(${acc}) | Gyro=(${gyro}) | ${finger.padEnd(9)} | Threat=${threat.padStart(4)} | ${state}`;
    } else {
      // Raw CSV matching SensorPacket.to_csv() format
      const f = (v, d = 4) => (v === undefined || v === null) ? "" : Number(v).toFixed(d);
      const finger = (sp.finger_detected === true) ? "1" : (sp.finger_detected === false ? "0" : "");
      const ts = sp.timestamp !== undefined ? Number(sp.timestamp).toFixed(4) : (Date.now() / 1000).toFixed(4);
      line = `${ts},${f(sp.heart_rate, 1)},${f(sp.spo2, 1)},${f(sp.eda, 2)},${f(sp.temperature, 2)},${f(sp.acc_x, 3)},${f(sp.acc_y, 3)},${f(sp.acc_z, 3)},${f(sp.gyro_x, 2)},${f(sp.gyro_y, 2)},${f(sp.gyro_z, 2)},${finger},${sp.label !== undefined ? sp.label : "-1"}`;
    }

    smBuffer.push(line);
    if (smBuffer.length > SM_MAX_LINES) {
      smBuffer.shift();
    }
    renderSerialMonitor();
  }
}

function renderSerialMonitor() {
  const container = document.getElementById("smConsole");
  if (!container) return;

  const html = smBuffer.map(line => {
    let cls = "text-uni-contrast";
    if (line.includes("THREAT") || line.includes("ALERT") || line.includes("EMERGENCY")) {
      cls = "text-uni-alert font-bold";
    } else if (line.includes("CAUTION") || line.includes("SUSPICIOUS")) {
      cls = "text-uni-cyan";
    } else if (line.includes("NO-FINGER")) {
      cls = "text-uni-muted";
    }
    return `<div class="${cls} whitespace-pre font-mono">${escapeHtml(line)}</div>`;
  }).join("");

  container.innerHTML = html;
  if (smAutoScroll) {
    container.scrollTop = container.scrollHeight;
  }
  setText("smLineCount", `${smBuffer.length} lines`);
}

function initSerialMonitorControls() {
  const btnFormat = document.getElementById("btnSmFormat");
  if (btnFormat) {
    btnFormat.addEventListener("click", () => {
      smFormat = smFormat === "decoded" ? "csv" : "decoded";
      btnFormat.textContent = smFormat === "decoded" ? "Format: Decoded" : "Format: Raw CSV";
      const legend = document.getElementById("smColumnsLegend");
      if (legend) {
        legend.textContent = smFormat === "decoded"
          ? "Columns: TIME | HR | TEMP | ACC(X,Y,Z) | GYRO(X,Y,Z) | CONTACT | THREAT | STATE"
          : "CSV: ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label";
      }
      renderSerialMonitor();
    });
  }

  const btnScroll = document.getElementById("btnSmScroll");
  if (btnScroll) {
    btnScroll.addEventListener("click", () => {
      smAutoScroll = !smAutoScroll;
      btnScroll.textContent = smAutoScroll ? "Auto-Scroll: ON" : "Auto-Scroll: OFF";
      btnScroll.className = smAutoScroll
        ? "btn-uni btn-neon text-[10px] py-1 px-2.5"
        : "btn-uni btn-dark text-[10px] py-1 px-2.5 border border-uni-border";
      if (smAutoScroll) {
        const container = document.getElementById("smConsole");
        if (container) container.scrollTop = container.scrollHeight;
      }
    });
  }

  const btnCopy = document.getElementById("btnSmCopy");
  if (btnCopy) {
    btnCopy.addEventListener("click", async () => {
      const text = smBuffer.join("\n");
      try {
        await navigator.clipboard.writeText(text);
        btnCopy.textContent = "Copied!";
        btnCopy.className = "btn-uni btn-neon text-[10px] py-1 px-2.5";
        setTimeout(() => {
          btnCopy.textContent = "Copy";
          btnCopy.className = "btn-uni btn-dark text-[10px] py-1 px-2.5 border border-uni-border hover:border-uni-cyan";
        }, 1500);
      } catch (e) {
        console.warn("Clipboard copy failed:", e);
      }
    });
  }

  const btnClear = document.getElementById("btnSmClear");
  if (btnClear) {
    btnClear.addEventListener("click", () => {
      smBuffer.length = 0;
      renderSerialMonitor();
    });
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBar(id, pct, color = null) {
  const el = document.getElementById(id);
  if (el) {
    el.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    if (color) el.style.backgroundColor = color;
  }
}

async function fetchSubjects() {
  try {
    const res = await fetch("/api/replay/subjects");
    const data = await res.json();
    const select = document.getElementById("subjectSelect");
    if (select && data.subjects) {
      select.innerHTML = data.subjects.map(s => `<option value="${s}">${s}</option>`).join("");
    }
  } catch (e) {
    console.warn("Could not fetch subjects:", e);
  }
}

async function fetchArduinoPorts() {
  try {
    const res = await fetch("/api/arduino/ports");
    const data = await res.json();
    const select = document.getElementById("arduinoPortSelect");
    if (select && data.ports) {
      if (data.ports.length === 0) {
        select.innerHTML = `<option value="COM3">COM3 (Manual)</option>`;
      } else {
        select.innerHTML = data.ports.map(p => `<option value="${p.port}">${p.port} (${p.description})</option>`).join("");
      }
    }
  } catch (e) {
    console.warn("Could not fetch ports:", e);
  }
}

function setupEventListeners() {
  // Source Selection Tabs
  document.getElementById("btnSourceWesad")?.addEventListener("click", () => switchSource("WESAD_REPLAY"));
  document.getElementById("btnSourceArduino")?.addEventListener("click", () => switchSource("ARDUINO_LIVE"));

  // Arduino Connect / Disconnect
  document.getElementById("btnArduinoConnect")?.addEventListener("click", async () => {
    const port = document.getElementById("arduinoPortSelect")?.value || "COM3";
    const action = arduinoConnected ? "disconnect" : "connect";
    try {
      const res = await fetch("/api/arduino/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, port }),
      });
      const data = await res.json();
      if (!res.ok) alert(data.detail || "Connection failed");
    } catch (e) {
      alert("Error connecting to Arduino: " + e.message);
    }
  });

  // Replay Controls
  document.getElementById("btnPlay")?.addEventListener("click", () => sendReplayControl({ action: "play" }));
  document.getElementById("btnPause")?.addEventListener("click", () => sendReplayControl({ action: "pause" }));
  document.getElementById("btnReset")?.addEventListener("click", () => sendReplayControl({ action: "reset" }));

  // Preset Segment Jumps
  document.getElementById("jumpBaseline")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "BASELINE" }));
  document.getElementById("jumpStress")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "STRESS" }));
  document.getElementById("jumpAmusement")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "AMUSEMENT" }));

  // Speed buttons
  document.querySelectorAll("[data-speed]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const speed = parseFloat(e.currentTarget.getAttribute("data-speed"));
      sendReplayControl({ action: "speed", speed });
    });
  });

  // Subject Selector
  document.getElementById("subjectSelect")?.addEventListener("change", (e) => {
    sendReplayControl({ action: "subject", subject_id: e.target.value });
  });

  // Timeline scrubber
  const scrubber = document.getElementById("timelineScrubber");
  if (scrubber) {
    scrubber.addEventListener("click", (e) => {
      const rect = scrubber.getBoundingClientRect();
      const fraction = (e.clientX - rect.left) / rect.width;
      sendReplayControl({ action: "seek", seek_fraction: fraction });
    });
  }

  // Demo Scenarios
  document.querySelectorAll("[data-scenario]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const scenario = e.currentTarget.getAttribute("data-scenario");
      triggerScenario(scenario);
    });
  });

  // Mute Alarm
  const muteBtn = document.getElementById("muteBtn");
  if (muteBtn) {
    muteBtn.textContent = sfx.isMuted ? "MUTED" : "AUDIO ON";
    muteBtn.addEventListener("click", () => {
      sfx.isMuted = !sfx.isMuted;
      muteBtn.textContent = sfx.isMuted ? "MUTED" : "AUDIO ON";
      muteBtn.className = sfx.isMuted ? "btn-uni btn-alert text-xs py-1 px-2.5" : "btn-uni btn-dark text-xs py-1 px-2.5";
    });
  }

  // Simulate Threat
  const simBtn = document.getElementById("btnSimulateThreat");
  if (simBtn) {
    simBtn.addEventListener("click", async () => {
      const fb = document.getElementById("threatFeedback");
      if (fb) {
        fb.textContent = "Raising threat... (~4s countdown)";
        fb.className = "text-[11px] font-mono text-uni-alert animate-pulse";
      }
      await triggerScenario("MULTIMODAL_THREAT");
    });
  }

  // Reset Alarm
  document.getElementById("btnAlertReset")?.addEventListener("click", async () => {
    try {
      await fetch("/api/alert/reset", { method: "POST" });
      await triggerScenario("LIVE_FREE");
      const fb = document.getElementById("threatFeedback");
      if (fb) {
        fb.textContent = "Alarm reset (LIVE_FREE)";
        fb.className = "text-[11px] font-mono text-uni-cyan";
        setTimeout(() => { if (fb.textContent.includes("LIVE_FREE")) fb.textContent = ""; }, 3000);
      }
    } catch (e) {
      console.error("Alert reset error:", e);
    }
  });

  // Test Telegram
  const testTelegramBtn = document.getElementById("btnTestTelegram");
  if (testTelegramBtn) {
    testTelegramBtn.addEventListener("click", async () => {
      const fb = document.getElementById("threatFeedback");
      if (fb) {
        fb.textContent = "Testing Telegram alert...";
        fb.className = "text-[11px] font-mono text-uni-muted";
      }
      testTelegramBtn.disabled = true;
      try {
        const resp = await fetch("/api/alerts/test", { method: "POST" });
        const res = await resp.json();
        if (fb) {
          if (res.delivered) {
            fb.textContent = "Telegram: Delivered";
            fb.className = "text-[11px] font-mono text-uni-neon font-bold";
          } else {
            const notifiers = (res.notifiers || []).map(n => n.name).join(", ");
            fb.textContent = `Alert test failed (${notifiers || "null"} - alerts disabled in .env)`;
            fb.className = "text-[11px] font-mono text-uni-alert";
          }
          setTimeout(() => { fb.textContent = ""; }, 7000);
        }
      } catch (e) {
        if (fb) {
          fb.textContent = `Test error: ${e.message}`;
          fb.className = "text-[11px] font-mono text-uni-alert";
        }
      } finally {
        testTelegramBtn.disabled = false;
      }
    });
  }

  // Fusion Weight Slider
  const slider = document.getElementById("sensorWeightSlider");
  if (slider) {
    slider.addEventListener("input", (e) => {
      const sW = parseFloat(e.target.value);
      const vW = Math.round((1.0 - sW) * 100) / 100;
      setText("sensorWeightLabel", `${Math.round(sW * 100)}%`);
      setText("visionWeightLabel", `${Math.round(vW * 100)}%`);
      sendWeightConfig(sW, vW);
    });
  }
}

async function switchSource(sourceType) {
  try {
    const res = await fetch("/api/source/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_type: sourceType }),
    });
    if (res.ok) {
      activeSource = sourceType;
      updateSourceUI();
    }
  } catch (e) {
    console.error("Failed to switch source:", e);
  }
}

async function sendReplayControl(payload) {
  try {
    await fetch("/api/replay/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    console.error("Replay control error:", e);
  }
}

async function triggerScenario(scenario) {
  try {
    await fetch("/api/demo/scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario }),
    });
  } catch (e) {
    console.error("Demo trigger error:", e);
  }
}

async function sendWeightConfig(sW, vW) {
  try {
    await fetch("/api/fusion/weights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sensor_weight: sW, vision_weight: vW }),
    });
  } catch (e) {
    console.error("Weight config error:", e);
  }
}

// --------------------------------------------------------------------------- //
// Panel Visibility Dropdown & Layout Manager
// --------------------------------------------------------------------------- //
const DASHBOARD_PANELS = [
  { id: "camera", label: "Camera & Pose Tracking", elId: "panelCameraFeed" },
  { id: "sensors", label: "Wearable Sensors (Vitals)", elId: "panelWearableSensors" },
  { id: "sensor-ml", label: "Sensor Sequence Model (ML)", elId: "panelSensorML" },
  { id: "vision-ml", label: "Vision Distress Model", elId: "panelVisionML" },
  { id: "threat-decision", label: "Multimodal Threat Decision", elId: "panelThreatDecision" },
  { id: "demo-scenarios", label: "Demo Scenarios", elId: "panelDemoScenarios" },
  { id: "replay-deck", label: "Sensor Replay & Timeline", elId: "panelReplayDeck" },
  { id: "serial-monitor", label: "Serial Monitor & Stream", elId: "panelSerialMonitor" },
  { id: "event-log", label: "Event Audit Log", elId: "panelEventLog" },
];

let panelVisibilityState = {};

function loadPanelVisibility() {
  const initial = {};
  DASHBOARD_PANELS.forEach(p => { initial[p.id] = true; });
  try {
    const saved = localStorage.getItem("tw_panel_visibility");
    if (saved) {
      const parsed = JSON.parse(saved);
      Object.keys(initial).forEach(k => {
        if (parsed[k] !== undefined) initial[k] = Boolean(parsed[k]);
      });
    }
  } catch (e) {
    /* fallback to defaults */
  }
  return initial;
}

function applyPanelVisibility(states) {
  panelVisibilityState = states;

  // Toggle individual card elements
  DASHBOARD_PANELS.forEach(panel => {
    const el = document.getElementById(panel.elId);
    if (!el) return;
    const isVisible = states[panel.id] !== false;
    if (isVisible) {
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  });

  // Dynamically adapt Row 1 layout (Camera + Sensors)
  const row1 = document.getElementById("row1Container");
  const camEl = document.getElementById("panelCameraFeed");
  const senEl = document.getElementById("panelWearableSensors");
  if (row1 && camEl && senEl) {
    const camVis = states["camera"] !== false;
    const senVis = states["sensors"] !== false;
    row1.style.display = (!camVis && !senVis) ? "none" : "";
    if (camVis && !senVis) {
      camEl.classList.remove("lg:col-span-7");
      camEl.classList.add("lg:col-span-12");
    } else if (!camVis && senVis) {
      senEl.classList.remove("lg:col-span-5");
      senEl.classList.add("lg:col-span-12");
    } else if (camVis && senVis) {
      camEl.classList.remove("lg:col-span-12");
      camEl.classList.add("lg:col-span-7");
      senEl.classList.remove("lg:col-span-12");
      senEl.classList.add("lg:col-span-5");
    }
  }

  // Dynamically adapt Row 2 layout (Sensor ML + Vision ML)
  const row2 = document.getElementById("row2Container");
  const smlEl = document.getElementById("panelSensorML");
  const vmlEl = document.getElementById("panelVisionML");
  if (row2 && smlEl && vmlEl) {
    const smlVis = states["sensor-ml"] !== false;
    const vmlVis = states["vision-ml"] !== false;
    row2.style.display = (!smlVis && !vmlVis) ? "none" : "";
    if (smlVis && !vmlVis) {
      smlEl.classList.add("md:col-span-2");
    } else if (!smlVis && vmlVis) {
      vmlEl.classList.add("md:col-span-2");
    } else {
      smlEl.classList.remove("md:col-span-2");
      vmlEl.classList.remove("md:col-span-2");
    }
  }

  // Dynamically adapt Row 4 layout (Demo Scenarios + Replay Deck)
  const row4 = document.getElementById("row4Container");
  const demoEl = document.getElementById("panelDemoScenarios");
  const repEl = document.getElementById("panelReplayDeck");
  if (row4 && demoEl && repEl) {
    const demoVis = states["demo-scenarios"] !== false;
    const repVis = states["replay-deck"] !== false;
    row4.style.display = (!demoVis && !repVis) ? "none" : "";
    if (demoVis && !repVis) {
      demoEl.classList.remove("lg:col-span-6");
      demoEl.classList.add("lg:col-span-12");
    } else if (!demoVis && repVis) {
      repEl.classList.remove("lg:col-span-6");
      repEl.classList.add("lg:col-span-12");
    } else if (demoVis && repVis) {
      demoEl.classList.remove("lg:col-span-12");
      demoEl.classList.add("lg:col-span-6");
      repEl.classList.remove("lg:col-span-12");
      repEl.classList.add("lg:col-span-6");
    }
  }

  // Persist
  try {
    localStorage.setItem("tw_panel_visibility", JSON.stringify(states));
  } catch (e) {
    /* private mode */
  }

  // Update button label with active count
  const activeCount = DASHBOARD_PANELS.filter(p => states[p.id] !== false).length;
  const toggleBtn = document.getElementById("btnPanelToggle");
  if (toggleBtn) {
    toggleBtn.innerHTML = `<span>Panels (${activeCount}/${DASHBOARD_PANELS.length})</span><span class="text-[10px] text-uni-muted font-mono" id="panelDropdownArrow">[v]</span>`;
  }
}

function initPanelDropdown() {
  const container = document.getElementById("panelDropdownWrap");
  const toggleBtn = document.getElementById("btnPanelToggle");
  const menu = document.getElementById("panelDropdownMenu");
  const listContainer = document.getElementById("panelCheckboxesList");
  if (!container || !toggleBtn || !menu || !listContainer) return;

  const currentStates = loadPanelVisibility();

  // Render checkbox rows in dropdown
  listContainer.innerHTML = DASHBOARD_PANELS.map(panel => {
    const checked = currentStates[panel.id] !== false ? "checked" : "";
    return `
      <label class="flex items-center justify-between gap-2 p-1.5 hover:bg-uni-panel cursor-pointer rounded select-none border border-transparent hover:border-uni-border/50 transition-colors">
        <span class="text-uni-contrast text-[11px] truncate">${escapeHtml(panel.label)}</span>
        <input type="checkbox" data-panel-id="${panel.id}" ${checked} class="accent-uni-neon cursor-pointer h-3.5 w-3.5">
      </label>
    `;
  }).join("");

  applyPanelVisibility(currentStates);

  // Wire up checkbox toggles
  listContainer.querySelectorAll("input[type='checkbox']").forEach(input => {
    input.addEventListener("change", (e) => {
      const pid = e.target.getAttribute("data-panel-id");
      panelVisibilityState[pid] = e.target.checked;
      applyPanelVisibility(panelVisibilityState);
    });
  });

  // Toggle Dropdown Visibility
  const toggleMenu = (open) => {
    const isHidden = menu.classList.contains("hidden");
    const willOpen = open !== undefined ? open : isHidden;
    if (willOpen) {
      menu.classList.remove("hidden");
      const arrow = document.getElementById("panelDropdownArrow");
      if (arrow) arrow.textContent = "[^]";
    } else {
      menu.classList.add("hidden");
      const arrow = document.getElementById("panelDropdownArrow");
      if (arrow) arrow.textContent = "[v]";
    }
  };

  toggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleMenu();
  });

  // Close when clicking outside
  document.addEventListener("click", (e) => {
    if (!container.contains(e.target)) {
      toggleMenu(false);
    }
  });

  // Close on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      toggleMenu(false);
    }
  });

  // Quick Action Buttons
  const btnAll = document.getElementById("btnShowAllPanels");
  if (btnAll) {
    btnAll.addEventListener("click", (e) => {
      e.stopPropagation();
      DASHBOARD_PANELS.forEach(p => { panelVisibilityState[p.id] = true; });
      listContainer.querySelectorAll("input[type='checkbox']").forEach(cb => { cb.checked = true; });
      applyPanelVisibility(panelVisibilityState);
    });
  }

  const btnNone = document.getElementById("btnHideAllPanels");
  if (btnNone) {
    btnNone.addEventListener("click", (e) => {
      e.stopPropagation();
      DASHBOARD_PANELS.forEach(p => { panelVisibilityState[p.id] = false; });
      listContainer.querySelectorAll("input[type='checkbox']").forEach(cb => { cb.checked = false; });
      applyPanelVisibility(panelVisibilityState);
    });
  }

  const btnReset = document.getElementById("btnResetPanels");
  if (btnReset) {
    btnReset.addEventListener("click", (e) => {
      e.stopPropagation();
      try { localStorage.removeItem("tw_panel_visibility"); } catch (err) {}
      const resetStates = {};
      DASHBOARD_PANELS.forEach(p => { resetStates[p.id] = true; });
      listContainer.querySelectorAll("input[type='checkbox']").forEach(cb => { cb.checked = true; });
      applyPanelVisibility(resetStates);
    });
  }
}

